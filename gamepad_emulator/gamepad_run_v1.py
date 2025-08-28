import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

import socket
import numpy as np
import copy

# -----------------------------------------------------------------------------
# Networking (send UDP packets to Isaac-Sim)
# -----------------------------------------------------------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ADDR = ("127.0.0.1", 55001)  # IP:port of the TeleopEnv listener

# -----------------------------------------------------------------------------
# GUI constants
# -----------------------------------------------------------------------------
NUM_AXES = 23
STREAM_INTERVAL_MS = 10  # 100 Hz

SLIDER_MIN, SLIDER_MAX = -1.0, 1.0
EXTRA_SPEED_MIN, EXTRA_SPEED_MAX = 0.0, 1.0   # v_x ∈ [0,1]
EXTRA_LR_MIN, EXTRA_LR_MAX = -1.0, 1.0        # v_y
EXTRA_ANGLE_MIN, EXTRA_ANGLE_MAX = -1.0, 1.0  # ω_z ∈ [−1,1]

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]

# --- твоя нейтральная поза (нормализованная в [-1,1]) ---
INIT_BASELINE = np.array([-0.05084747076034546, -0.05084747076034546, 0.0, -0.7368420958518982, \
        0.736842155456543, 0.0625, 0.0625, 0.0, 0.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.2857142686843872, 0.2857142686843872, \
        0.7333333492279053, 0.7333333492279053, 0.0, 0.0, 0.0, 0.0
], dtype=np.float32)

assert INIT_BASELINE.shape == (NUM_AXES,)

# ================================== UTIL ===================================

def smoothstep(t: np.ndarray) -> np.ndarray:
    """3t^2 - 2t^3, монотонная S-кривая, нулевые производные на концах."""
    return t * t * (3.0 - 2.0 * t)

def resample_track(y_old: np.ndarray, new_T: int) -> np.ndarray:
    """Линейный ресэмпл в новый размер."""
    T_old = len(y_old)
    if T_old == new_T:
        return y_old.copy()
    x_old = np.linspace(0.0, 1.0, T_old, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, new_T, dtype=np.float32)
    return np.interp(x_new, x_old, y_old).astype(np.float32)

# =============================== PATTERN EDITOR ==============================

class PatternEditor(tk.Toplevel):
    """
    Редактор паттернов: 23 полосы-канваса, редактирование кликом/перетаскиванием.
    Между пинами строится монотонный плавный переход (smoothstep).
    Начало/конец — baseline. Состояние сохраняется между открытиями окна.
    """

    CANVAS_W = 1000
    ROW_H    = 100
    PAD_Y    = 8

    def __init__(self, parent, *,
                 baseline_vec,
                 on_save_callback,
                 initial_tracks=None,      # np.ndarray [T,23] или None
                 initial_pins=None,        # list[set] или None
                 initial_duration_s=None   # float или None
                 ):
        super().__init__(parent)
        self.title("Pattern Editor")
        self.transient(parent)
        self.grab_set()

        self.parent = parent
        self.on_save_callback = on_save_callback

        # baseline (23,)
        self.baseline = np.asarray(baseline_vec, dtype=np.float32)

        # state
        if initial_tracks is not None:
            self.tracks = initial_tracks.astype(np.float32).copy()         # [T,23]
            self.T = self.tracks.shape[0]
            self.duration_s = tk.DoubleVar(value=float(
                initial_duration_s if initial_duration_s is not None
                else self.T * STREAM_INTERVAL_MS / 1000.0
            ))
            if initial_pins is not None:
                self.pins = [set(p) for p in initial_pins]
                for a in range(NUM_AXES):
                    self.pins[a].add(0)
                    self.pins[a].add(self.T - 1)
            else:
                self.pins = [set([0, self.T - 1]) for _ in range(NUM_AXES)]
        else:
            self.duration_s = tk.DoubleVar(value=5.0)
            self.T = max(3, int(round(self.duration_s.get() * 1000.0 / STREAM_INTERVAL_MS)))
            self.tracks = np.tile(self.baseline[None, :], (self.T, 1)).astype(np.float32)
            self.pins = [set([0, self.T - 1]) for _ in range(NUM_AXES)]

        self._build_ui()
        self._redraw_all()

    # -------------------- UI --------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Duration (s):").pack(side="left")
        ttk.Entry(top, textvariable=self.duration_s, width=6).pack(side="left", padx=(4, 12))
        ttk.Button(top, text="Apply", command=self._apply_duration).pack(side="left")

        ttk.Button(top, text="Reset Axis", command=self._reset_selected).pack(side="left", padx=(10, 0))
        ttk.Button(top, text="Reset All", command=self._reset_all).pack(side="left", padx=(6, 0))

        ttk.Button(top, text="Save & Play", command=self._save_and_play).pack(side="right")

        # scroll area with 23 canvases
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self.scroll = tk.Canvas(outer, highlightthickness=0, height=NUM_AXES * (self.ROW_H + self.PAD_Y) // 2)
        self.scroll.pack(side="left", fill="both", expand=True)

        vs = ttk.Scrollbar(outer, orient="vertical", command=self.scroll.yview)
        vs.pack(side="right", fill="y")
        self.scroll.configure(yscrollcommand=vs.set)

        self.rows_holder = ttk.Frame(self.scroll)
        self.scroll.create_window((0, 0), window=self.rows_holder, anchor="nw")
        self.rows_holder.bind("<Configure>", lambda e: self.scroll.configure(scrollregion=self.scroll.bbox("all")))

        # per-axis canvases
        self.canvases = []
        self.active_axis = 0
        for a in range(NUM_AXES):
            row = ttk.Frame(self.rows_holder)
            row.pack(fill="x", pady=(0, self.PAD_Y))

            ttk.Label(row, text=JOINT_NAMES[a], width=28).pack(side="left", padx=(0, 8))
            c = tk.Canvas(row, width=self.CANVAS_W, height=self.ROW_H, bg="#0e0e10", highlightthickness=2)
            c.pack(side="left", fill="x", expand=True)
            c.configure(highlightbackground="#2a2a2d", highlightcolor="#4ea1ff")
            c.bind("<Enter>", lambda e, ax=a: self._set_active_axis(ax))
            c.bind("<Button-1>", lambda e, ax=a: self._on_click_drag(ax, e))
            c.bind("<B1-Motion>", lambda e, ax=a: self._on_click_drag(ax, e))
            c.bind("<ButtonRelease-1>", lambda e, ax=a: self._on_release(ax, e))
            self.canvases.append(c)

    # -------------------- helpers --------------------
    def _set_active_axis(self, a):
        self.active_axis = a

    def _apply_duration(self):
        new_d = max(0.1, float(self.duration_s.get()))
        self.duration_s.set(new_d)
        new_T = max(3, int(round(new_d * 1000.0 / STREAM_INTERVAL_MS)))
        if new_T == self.T:
            return
        old_T = self.T
        self.tracks = np.stack([resample_track(self.tracks[:, a], new_T) for a in range(NUM_AXES)], axis=1)
        new_pins = []
        for a in range(NUM_AXES):
            scaled = { int(round(p / (old_T - 1) * (new_T - 1))) for p in self.pins[a] }
            scaled.add(0); scaled.add(new_T - 1)
            new_pins.append(scaled)
        self.pins = new_pins
        self.T = new_T
        self._redraw_all()

    # координаты
    def _x_to_i(self, x):
        x = np.clip(x, 0, self.CANVAS_W - 1)
        i = int(round(x / (self.CANVAS_W - 1) * (self.T - 1)))
        return int(np.clip(i, 0, self.T - 1))

    def _y_to_val(self, y):
        y = np.clip(y, 0, self.ROW_H - 1)
        v = 1.0 - 2.0 * (y / (self.ROW_H - 1))
        return float(np.clip(v, -1.0, 1.0))

    def _val_to_y(self, v):
        v = float(np.clip(v, -1.0, 1.0))
        y = (1.0 - v) * 0.5 * (self.ROW_H - 1)
        return y

    # монотонный сегмент между двумя пинами
    def _fill_segment_monotone(self, axis, i0, i1):
        if i1 <= i0:
            return
        y0 = self.tracks[i0, axis]
        y1 = self.tracks[i1, axis]
        n = i1 - i0
        xs = np.arange(0, n + 1, dtype=np.float32) / float(n)
        s = smoothstep(xs)  # монотонно
        self.tracks[i0:i1 + 1, axis] = y0 + (y1 - y0) * s

    def _rebuild_axis_from_pins(self, axis):
        self.pins[axis].add(0); self.pins[axis].add(self.T - 1)
        self.tracks[0, axis]  = self.baseline[axis]
        self.tracks[-1, axis] = self.baseline[axis]
        pins_sorted = sorted(self.pins[axis])
        for k in range(len(pins_sorted) - 1):
            i0, i1 = pins_sorted[k], pins_sorted[k + 1]
            self._fill_segment_monotone(axis, i0, i1)

    # отрисовка
    def _draw_axis(self, a):
        c = self.canvases[a]
        c.delete("all")
        y0 = self._val_to_y(0.0)
        c.create_line(0, y0, self.CANVAS_W, y0, fill="#2a2a2d")
        yb = self._val_to_y(self.baseline[a])
        c.create_line(0, yb, self.CANVAS_W, yb, fill="#2d6cdf", dash=(4, 3))
        stride = max(1, self.T // 1000)
        pts = []
        for i in range(0, self.T, stride):
            x = i / (self.T - 1) * (self.CANVAS_W - 1)
            y = self._val_to_y(self.tracks[i, a])
            pts.extend([x, y])
        c.create_line(*pts, fill="#e5e5e5", width=2, smooth=True, splinesteps=12)
        for i in self.pins[a]:
            x = i / (self.T - 1) * (self.CANVAS_W - 1)
            y = self._val_to_y(self.tracks[i, a])
            r = 3
            c.create_oval(x - r, y - r, x + r, y + r, fill="#ffffff", outline="")
        c.configure(highlightbackground=("#4ea1ff" if a == self.active_axis else "#2a2a2d"))

    def _redraw_all(self):
        for a in range(NUM_AXES):
            self._draw_axis(a)

    # события
    def _on_click_drag(self, axis, event):
        self.active_axis = axis
        i = self._x_to_i(event.x)
        v = self._y_to_val(event.y)
        self.tracks[i, axis] = v
        self.pins[axis].add(i)
        self.pins[axis].add(0); self.pins[axis].add(self.T - 1)
        self.tracks[0, axis]  = self.baseline[axis]
        self.tracks[-1, axis] = self.baseline[axis]
        self._rebuild_axis_from_pins(axis)
        self._draw_axis(axis)

    def _on_release(self, axis, _event):
        pass

    # reset/save
    def _reset_selected(self):
        a = self.active_axis
        self.tracks[:, a] = self.baseline[a]
        self.pins[a] = set([0, self.T - 1])
        self._draw_axis(a)

    def _reset_all(self):
        for a in range(NUM_AXES):
            self.tracks[:, a] = self.baseline[a]
            self.pins[a] = set([0, self.T - 1])
        self._redraw_all()

    def _save_and_play(self):
        traj = self.tracks.copy()
        traj[0, :]  = self.baseline
        traj[-1, :] = self.baseline
        self.on_save_callback(traj, float(self.duration_s.get()), copy.deepcopy(self.pins), self.baseline.copy())
        self.destroy()

# ================================ MAIN APP ===================================

class AxisControlApp(tk.Tk):
    """Отправляет 49-элементный вектор каждые STREAM_INTERVAL_MS.
       Паттерн синхронизирует ползунки/галочки во время воспроизведения.
    """

    def __init__(self):
        super().__init__()
        self.title("Axis Control (23 DOF + XY speed + yaw)")
        self.geometry("1200x980")
        self.tk.call("tk", "scaling", 2.0)

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=16)
        self.option_add("*Font", default_font)

        # baseline для всего приложения — из INIT_BASELINE
        self.global_baseline = INIT_BASELINE.copy()

        # live vars
        self._make_vars()

        # pattern playback state
        self.pattern_active = False
        self.pattern_targets = None   # np.ndarray [T, 23]
        self.pattern_masks = None     # np.ndarray [T, 23]
        self.pattern_index = 0
        self.pattern_total = 0

        # editor persistent state между открытиями
        self.editor_tracks = None      # np.ndarray [T,23]
        self.editor_pins = None        # list[set] per axis
        self.editor_duration_s = None  # float

        self._build_ui()
        self.after(STREAM_INTERVAL_MS, self._stream_vector)

    # -------------------------- model vars --------------------------
    def _make_vars(self):
        self.switch_vars = [tk.IntVar(value=0) for _ in range(NUM_AXES)]
        # слайдеры стартуют в нейтральной позе
        self.slider_vars = [tk.DoubleVar(value=float(self.global_baseline[i])) for i in range(NUM_AXES)]
        self.speed_x_var = tk.DoubleVar(value=0.0)
        self.speed_y_var = tk.DoubleVar(value=0.0)
        self.angle_z_var = tk.DoubleVar(value=0.0)

    # -------------------------- UI --------------------------
    def _build_ui(self):
        header_font = ("Helvetica", 20, "bold")
        ttk.Label(self, text="Activate axis + set value", font=header_font).grid(
            row=0, column=0, columnspan=4, pady=(4, 12)
        )

        # pattern controls
        ctrl = ttk.Frame(self)
        ctrl.grid(row=1, column=0, columnspan=4, sticky="we", padx=6, pady=(0, 8))
        ttk.Button(ctrl, text="Open Pattern Editor", command=self._open_pattern_editor).pack(side="left")
        self.play_label = ttk.Label(ctrl, text="(idle)")
        self.play_label.pack(side="left", padx=12)

        # grid of axes
        for i in range(NUM_AXES):
            row = i + 2
            ttk.Checkbutton(self, variable=self.switch_vars[i]).grid(row=row, column=0, sticky="w")
            ttk.Label(self, text=JOINT_NAMES[i]).grid(row=row, column=1, sticky="w")
            ttk.Scale(
                self,
                from_=SLIDER_MIN,
                to=SLIDER_MAX,
                orient="horizontal",
                length=600,
                variable=self.slider_vars[i],
            ).grid(row=row, column=2, padx=10, pady=2, sticky="we")

        self.columnconfigure(2, weight=1)

        base = NUM_AXES + 3
        ttk.Label(self, text="Global motion", font=header_font).grid(
            row=base, column=0, columnspan=4, pady=(24, 8)
        )

        self._add_extra_slider(base + 1, "Speed X (0‥1)", self.speed_x_var, EXTRA_SPEED_MIN, EXTRA_SPEED_MAX)
        self._add_extra_slider(base + 2, "Speed Y (−1‥1)", self.speed_y_var, EXTRA_LR_MIN, EXTRA_LR_MAX)
        self._add_extra_slider(base + 3, "Yaw Z (−1‥1)", self.angle_z_var, EXTRA_ANGLE_MIN, EXTRA_ANGLE_MAX)

        # playback progress bar
        self.pb = ttk.Progressbar(self, orient="horizontal", mode="determinate", length=600)
        self.pb.grid(row=base + 4, column=0, columnspan=4, pady=(16, 8))

        ttk.Label(self, text=f"Vector sent every {STREAM_INTERVAL_MS} ms to {ADDR[0]}:{ADDR[1]}").grid(
            row=base + 5, column=0, columnspan=4, pady=(0, 6)
        )

    def _add_extra_slider(self, row, text, var, vmin, vmax):
        ttk.Label(self, text=text).grid(row=row, column=1, sticky="w")
        ttk.Scale(self, from_=vmin, to=vmax, orient="horizontal", length=600, variable=var).grid(
            row=row, column=2, padx=10, pady=2, sticky="we"
        )

    # -------------------------- pattern editor --------------------------
    def _open_pattern_editor(self):
        # baseline редактора = глобальная нейтральная поза
        baseline = self.global_baseline.copy()

        def on_save(traj_T23, duration_s, pins, baseline_out):
            """Принимаем траекторию и сохраняем состояние редактора."""
            total = traj_T23.shape[0]
            self.pattern_targets = traj_T23.astype(np.float32)
            # маска: 1, где отличается от baseline
            eps = 1e-6
            self.pattern_masks = (np.abs(self.pattern_targets - baseline[None, :]) > eps).astype(np.float32)
            self.pattern_total = int(total)
            self.pattern_index = 0
            self.pattern_active = True
            self.pb.configure(maximum=self.pattern_total, value=0)
            self.play_label.configure(text=f"PLAY {duration_s:.2f}s ({total} frames)")

            # запомним состояние редактора для следующего открытия
            self.editor_tracks = traj_T23.copy()
            self.editor_pins = pins
            self.editor_duration_s = float(duration_s)
            self.global_baseline = baseline_out.copy()  # на случай будущей смены

        PatternEditor(
            self,
            baseline_vec=baseline,
            on_save_callback=on_save,
            initial_tracks=self.editor_tracks,
            initial_pins=self.editor_pins,
            initial_duration_s=self.editor_duration_s,
        )

    # -------------------------- packet construction --------------------------
    def _collect_live_vector(self):
        targets = [v.get() for v in self.slider_vars]
        mask = [float(v.get()) for v in self.switch_vars]
        extra = [self.speed_x_var.get(), self.speed_y_var.get(), self.angle_z_var.get()]
        return np.asarray(targets + mask + extra, dtype=np.float32)

    def _collect_pattern_vector(self):
        t = self.pattern_index
        targets = self.pattern_targets[t]  # (23,)
        mask = self.pattern_masks[t]       # (23,)
        # sync UI sliders + checkboxes to current pattern sample
        for i in range(NUM_AXES):
            self.slider_vars[i].set(float(targets[i]))
            self.switch_vars[i].set(int(mask[i] > 0.5))

        extra = np.array([self.speed_x_var.get(), self.speed_y_var.get(), self.angle_z_var.get()], dtype=np.float32)
        vec = np.concatenate([targets, mask, extra]).astype(np.float32)
        return vec

    # -------------------------- main loop --------------------------
    def _stream_vector(self):
        try:
            if self.pattern_active and self.pattern_targets is not None:
                vec = self._collect_pattern_vector()
                sock.sendto(vec.tobytes(), ADDR)
                print(vec)  # always print vector
                # advance
                self.pattern_index += 1
                self.pb["value"] = self.pattern_index
                if self.pattern_index >= self.pattern_total:
                    self.pattern_active = False
                    self.play_label.configure(text="(idle)")
                    self.pb["value"] = 0
            else:
                vec = self._collect_live_vector()
                sock.sendto(vec.tobytes(), ADDR)
                print(vec)  # always print vector
        except OSError as ex:
            print(f"[UDP] send failed: {ex}")
        self.after(STREAM_INTERVAL_MS, self._stream_vector)

# ----------------------------------------------------------------------
def run_gui() -> None:
    AxisControlApp().mainloop()

if __name__ == "__main__":
    run_gui()


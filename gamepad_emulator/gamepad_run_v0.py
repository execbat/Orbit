import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

import socket
import numpy as np

# -----------------------------------------------------------------------------
# Networking (send UDP packets to Isaac‑Sim)
# -----------------------------------------------------------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ADDR = ("127.0.0.1", 55001)         # IP:port of the TeleopEnv listener

# -----------------------------------------------------------------------------
# GUI constants
# -----------------------------------------------------------------------------
NUM_AXES            = 23              # robot DOF
STREAM_INTERVAL_MS  = 10             # send every 100 ms

SLIDER_MIN, SLIDER_MAX           = -1.0, 1.0
EXTRA_SPEED_MIN,  EXTRA_SPEED_MAX =  0.0, 1.0   # v_x ∈ [0,1]
EXTRA_LR_MIN,     EXTRA_LR_MAX    = -1.0, 1.0   # v_y (disabled for now)
EXTRA_ANGLE_MIN,  EXTRA_ANGLE_MAX = -1.0, 1.0   # ω_z ∈ [−1,1]


class AxisControlApp(tk.Tk):
    """Send a 49‑float vector over UDP every 100 ms.

    Layout:
      • 23 check‑boxes (mask 0/1)
      • 23 sliders  (targets −1…1)
      • 3 extra sliders (vₓ, v_y, ω_z) – Y is disabled by default
    """

    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.title("Axis Control (23 DOF + XY speed + yaw)")
        self.geometry("1000x900")              # enlarge window
        self.tk.call("tk", "scaling", 2.0)    # ×2 widgets size

        # enlarge default font (global)
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=28)
        self.option_add("*Font", default_font)

        self._make_vars()
        self._build_ui()
        self.after(STREAM_INTERVAL_MS, self._stream_vector)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    def _make_vars(self):
        self.switch_vars = [tk.IntVar(value=0)   for _ in range(NUM_AXES)]
        self.slider_vars = [tk.DoubleVar(value=0) for _ in range(NUM_AXES)]
        self.speed_x_var = tk.DoubleVar(value=0.0)
        self.speed_y_var = tk.DoubleVar(value=0.0)
        self.angle_z_var = tk.DoubleVar(value=0.0)

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def _build_ui(self):
        header_font = ("Helvetica", 32, "bold")
        ttk.Label(self, text="Activate axis + set value", font=header_font).grid(
            row=0, column=0, columnspan=3, pady=(0, 20)
        )

        # -----------------------------------------------------------------
        # 0) Имена осей в нужной последовательности
        # -----------------------------------------------------------------
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

        NUM_AXES = len(JOINT_NAMES)        # 23  ← заменяет прежнюю константу

        # -----------------------------------------------------------------
        # 1) В _build_ui() вместо «Axis 00/01/…» используем JOINT_NAMES
        # -----------------------------------------------------------------
        for i in range(NUM_AXES):
            row = i + 1
            ttk.Checkbutton(self, variable=self.switch_vars[i]).grid(row=row, column=0, sticky="w")

            #   ↓ было: ttk.Label(self, text=f"Axis {i:02d}") …
            ttk.Label(self, text=JOINT_NAMES[i]).grid(row=row, column=1, sticky="w")

            ttk.Scale(
                self,
                from_=SLIDER_MIN,
                to=SLIDER_MAX,
                orient="horizontal",
                length=400,
                variable=self.slider_vars[i],
            ).grid(row=row, column=2, padx=10, pady=4, sticky="we")

        self.columnconfigure(2, weight=1)  # make right column stretch

        # extra motion sliders
        base = NUM_AXES + 2
        ttk.Label(self, text="Global motion", font=header_font).grid(
            row=base, column=0, columnspan=3, pady=(40, 10)
        )

        self._add_extra_slider(base + 1, "Speed X (0‥1)",
                               self.speed_x_var, EXTRA_SPEED_MIN, EXTRA_SPEED_MAX)
        self._add_extra_slider(base + 2, "Left / Right Y (disabled)",
                               self.speed_y_var, EXTRA_LR_MIN, EXTRA_LR_MAX,
                               disabled=True)
        self._add_extra_slider(base + 3, "Yaw Z (−1‥1)",
                               self.angle_z_var, EXTRA_ANGLE_MIN, EXTRA_ANGLE_MAX)

        ttk.Label(self, text="Vector sent every 100 ms to 127.0.0.1:55001").grid(
            row=base + 4, column=0, columnspan=3, pady=(20, 0))

    def _add_extra_slider(self, row, text, var, vmin, vmax, *, disabled=False):
        ttk.Label(self, text=text).grid(row=row, column=1, sticky="w")
        scale = ttk.Scale(self, from_=vmin, to=vmax, orient="horizontal",
                          length=400, variable=var)
        scale.grid(row=row, column=2, padx=10, pady=4, sticky="we")
        if disabled:
            scale.state(["disabled"])           # ttk way to disable widget

    # ------------------------------------------------------------------
    # Build + send vector
    # ------------------------------------------------------------------
    def _collect_vector(self):
        targets = [v.get() for v in self.slider_vars]
        mask    = [float(v.get()) for v in self.switch_vars]
        extra   = [self.speed_x_var.get(), self.speed_y_var.get(), self.angle_z_var.get()]
        return targets + mask + extra  # len == 49

    def _stream_vector(self):
        vec = np.asarray(self._collect_vector(), dtype=np.float32)
        try:
            sock.sendto(vec.tobytes(), ADDR)
            print(vec)
        except OSError as ex:
            print(f"[UDP] send failed: {ex}")
        self.after(STREAM_INTERVAL_MS, self._stream_vector)


# ----------------------------------------------------------------------
def run_gui() -> None:
    """Entry-point for multiprocessing.Process()."""
    AxisControlApp().mainloop()



if __name__ == "__main__":
    run_gui()


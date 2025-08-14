![Isaac Lab](docs/source/_static/isaaclab.jpg)

# Isaac Lab — **execbat** Contributor Build (Unitree G1 23-DOF Ballet Teleop)

> **This is a contributor build by `execbat`** that adds **two custom environments** for the **Unitree G1 (23 DOF)**:
>
> - `Math-Velocity-Flat-G1-v0` — **training**
> - `Math-Velocity-Flat-G1-Play-v0` — **teleoperation / testing**

## Project Goal

Teach a humanoid robot to **dance ballet** while remaining **responsive to external commands**:

- Accept live commands from a **hardware controller** or a **virtual pose source** (emulator).
- Make the **real robot mirror** the virtual scene in real time.
- Maintain **balance** and allow **locomotion** via additional controller commands.

---

## Quick Start

### Training

```bash
./isaaclab.sh \
  -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Math-Velocity-Flat-G1-v0 \
  --num_envs 128 \
  --headless
```

### Teleop / Testing

```bash
./isaaclab.sh \
  -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Math-Velocity-Flat-G1-Play-v0 \
  --num_envs 1 \
  --checkpoint ./logs/rsl_rl/g1_flat/<experiment folder>/<checkpoint_name>.pt \
  --rendering_mode performance
```

### Axis Controller Emulator

```bash
python ./gamepad_emulator/gamepad_run.py
```

The emulator streams commands over UDP; the teleop environment reads them and the robot reacts in real time.

---

## Teleop Command Protocol (UDP)

**Packet layout:** **49 × float32** (little-endian), total **196 bytes**:

| Range         | Count | Meaning                                      | Notes                                                    |
|---------------|:-----:|----------------------------------------------|----------------------------------------------------------|
| `0..22`       |  23   | **Joint targets**                            | Normalized to **[-1, 1]** (per-DOF soft limits)         |
| `23..45`      |  23   | **Joint mask**                               | `1.0` = active (track target), `0.0` = keep near init   |
| `46..48`      |   3   | **Base velocity**                            | `[vx, vy, yaw_rate]` in the **base frame**              |

**Default endpoint:** `127.0.0.1:55001` (configurable in the teleop env).

#### Minimal Python sender example

```python
import socket, numpy as np

AXES = 23
PKT_LEN = 49

targets = np.linspace(-0.5, 0.5, AXES).astype(np.float32)  # demo targets in [-1,1]
mask    = np.ones(AXES, dtype=np.float32)                  # all DOFs active
speed   = np.array([0.0, 0.0, 0.0], np.float32)            # vx, vy, yaw_rate

packet = np.concatenate([targets, mask, speed])
assert packet.size == PKT_LEN

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(packet.tobytes(), ("127.0.0.1", 55001))
```

---

## Environments Added in This Build

- **`Math-Velocity-Flat-G1-v0`**  
  RL training environment for Unitree G1 (23 DOF). Exposes joint-space targets, per-DOF masking, and base velocity commands. Targets and masks are generated internally (commands manager).

- **`Math-Velocity-Flat-G1-Play-v0`**  
  Teleop/testing environment. Receives **UDP** commands (**23 targets, 23 mask, 3 velocity**) and writes them directly into the command manager **every sim step**.

---

## Notes & Tips

- **Joint targets** must be normalized to **[-1, 1]** (using each joint’s **soft limits**).
- **Mask semantics**:
  - **1.0** → DOF is **active** and tracks the target,
  - **0.0** → DOF is **inactive** and is driven to the **saved init pose** (not the arbitrary reset pose).
- **Headless training**: use `--headless` to maximize environment count; tune `--num_envs` to your GPU memory.
- **Rendering** (teleop): pick a suitable mode, e.g. `--rendering_mode performance`.

---

## Requirements

- NVIDIA **Isaac Sim 4.5** (or the matching version for your Isaac Lab branch)
- Python **3.10**
- NVIDIA GPU with recent drivers (**RTX** recommended)
- (Optional) A controller or the included emulator to stream UDP commands

---

## Troubleshooting

- **Robot doesn’t react to UDP**  
  - Ensure the **teleop env** (`*Play-v0`) is running.  
  - Check sender and env **IP/port** (`127.0.0.1:55001` by default).  
  - Packet must contain **exactly 49 float32** values.

- **Targets saturate or behave oddly**  
  - Verify target **normalization** to **[-1, 1]** and the **mask** values.  
  - Confirm your joint ordering matches the env’s DOF order.

- **Low FPS**  
  - Reduce rendering quality or use `--rendering_mode performance`.  
  - For training, prefer `--headless` and adjust `--num_envs`.

---

## License & Upstream

This contributor build extends **Isaac Lab**. Please refer to upstream license files (BSD-3 / Apache-2.0 as applicable) and NVIDIA Isaac Sim licensing.

If you use this build academically, please also cite **Orbit** (the framework Isaac Lab originated from):

```
@article{mittal2023orbit,
  author  = {Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
  journal = {IEEE Robotics and Automation Letters},
  title   = {Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
  year    = {2023},
  volume  = {8},
  number  = {6},
  pages   = {3740-3747},
  doi     = {10.1109/LRA.2023.3270034}
}
```

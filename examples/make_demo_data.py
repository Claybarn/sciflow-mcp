"""Write a synthetic two-system session (wheel + widefield stack + sync pulses) to ./demo_data."""

from pathlib import Path

import numpy as np

out = Path(__file__).parent / "demo_data"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

# Reference system (e.g. Spike2): wheel speed at 1 kHz and sync pulses.
t_ref = np.arange(0, 120, 1e-3)
speed = np.where((t_ref % 20) < 6, 8.0, 0.0) + rng.normal(scale=0.3, size=t_ref.size)
pulses_ref = np.cumsum(rng.uniform(0.8, 1.2, 110))
np.savez(out / "wheel.npz", timestamps=t_ref, data=speed)
np.savez(out / "sync_ref.npz", times=pulses_ref)

# Imaging system on its own drifting clock: 10 Hz, 32 x 32 pixels, time on the last axis (MATLAB style).
drift, offset = 1.0003, -4.2
to_local = lambda t: (t - offset) / drift  # noqa: E731
t_img = np.arange(to_local(0), to_local(118), 0.1)
t_true = t_img * drift + offset
response = np.exp(-((t_true[None, None, :] % 20) - 1.0) ** 2)
yy, xx = np.mgrid[:32, :32]
footprint = np.exp(-((yy - 16) ** 2 + (xx - 10) ** 2) / 40)[..., None]
stack = 1.0 + 0.2 * footprint * response + rng.normal(scale=0.02, size=(32, 32, t_img.size))
np.savez(out / "widefield.npz", timestamps=t_img, data=stack)
np.savez(out / "sync_img.npz", times=to_local(pulses_ref[3:]))  # imaging started after 3 pulses
print(f"wrote demo data to {out}")

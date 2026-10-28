"""A synthetic 4-subject x 2-condition experiment, run as a sciflow pipeline.

Writes per-session files to ./dataset_data, registers them as a dataset, runs
examples/onset_pipeline.yaml, and exports a long-format table for statistics.
"""

from pathlib import Path

import numpy as np
import yaml

from sciflow import Store
from sciflow.dataset import register_dataset, run_any
from sciflow.io import export_table

here = Path(__file__).parent
root = here / "dataset_data"
store = Store(here / ".sciflow")
rng = np.random.default_rng(0)
fs, dur = 250.0, 120.0
t = np.arange(0, dur, 1 / fs)

sessions = []
for subj in ["m1", "m2", "m3", "m4"]:
    subject_gain = rng.normal(1.0, 0.2)                       # subject random effect
    for cond, effect in [("control", 1.0), ("sleep_dep", 1.6)]:
        for day in (1, 2):
            sid = f"{subj}-{cond}-d{day}"
            onsets = np.sort(rng.uniform(5, dur - 10, 8))
            running = np.any((t[:, None] >= onsets) & (t[:, None] < onsets + rng.uniform(2, 4, 8)), axis=1)
            lag = t[:, None] - onsets
            # 8 Hz burst after running onset on 4 channels with a depth gradient
            burst = np.sum(np.where((lag >= 0) & (lag < 1), np.sin(2 * np.pi * 8 * lag) * np.exp(-lag / 0.4), 0), 1)
            lfp = (effect * subject_gain * burst[:, None] * np.linspace(1, 0.4, 4)
                   + rng.normal(scale=0.3, size=(len(t), 4)) + 0.2 * np.sin(2 * np.pi * 0.05 * t)[:, None])
            d = root / sid
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "wheel.npz", timestamps=t, data=running * 6.0 + rng.normal(scale=0.2, size=len(t)))
            np.savez(d / "lfp.npz", timestamps=t, data=lfp)
            sessions.append({"id": sid, "subject": subj, "attributes": {"condition": cond, "day": day},
                             "inputs": {"wheel": {"path": str(d / "wheel.npz")},
                                        "lfp": {"path": str(d / "lfp.npz"), "options": {"axes": ["time", "channel"]}}}})

dataset_id = register_dataset({"name": "synthetic-sleep-dep", "sessions": sessions}, store)
pipeline = yaml.safe_load((here / "onset_pipeline.yaml").read_text())
pipeline["dataset"] = dataset_id
rec = run_any(pipeline, store)

mean = store.get(rec["outputs"]["condition_mean"])
sem = store.get(rec["outputs"]["condition_sem"])
ran = sum(not s["cached"] for s in rec["steps"])
print(f"dataset {dataset_id}: {len(sessions)} sessions, run {rec['run_id']} ({ran}/{len(rec['steps'])} steps computed)")
for g, label in enumerate(mean.coords["group"]):
    i = np.nanargmax(np.abs(mean.data[:, g, 0]))
    n = mean.meta["unit_attributes"]["group"]["n"][g]
    print(f"  {label:22s} peak z = {mean.data[i, g, 0]:.2f} ± {sem.data[i, g, 0]:.2f} (sem, n = {n} subjects) at lag {mean.stamps[i]:.3f} s")
perm = store.get(rec["outputs"]["permutation"])
lme = store.get(rec["outputs"]["lme"])
stat = list(perm.coords["statistic"])
p_corr = perm.data[:, stat.index("p_corrected"), :]
sig = p_corr < 0.05
print(f"stratified permutation (max-corrected, {perm.meta['test']['n_strata']} subjects, "
      f"{perm.meta['test']['n_observations']} sessions): {sig.sum()}/{sig.size} lag x channel bins significant, "
      f"lags {perm.stamps[sig.any(1)].min():.2f}-{perm.stamps[sig.any(1)].max():.2f} s")
term = list(lme.coords["term"]).index("condition[T.sleep_dep]")
lstat = list(lme.coords["statistic"])
best = np.nanargmin(lme.data[:, term, lstat.index("p"), 0])
print(f"mixed model value ~ condition + C(day) + (1|subject), channel 0 at lag {lme.stamps[best]:.2f} s: "
      f"sleep_dep effect = {lme.data[best, term, lstat.index('estimate'), 0]:.2f} "
      f"± {lme.data[best, term, lstat.index('se'), 0]:.2f}, "
      f"Holm p = {lme.data[best, term, lstat.index('p_corrected'), 0]:.1e}")
stacked = next(s["output"] for s in rec["steps"] if s["id"] == "responses")
print("long-format table:", export_table(store.get(stacked), here / "responses_long.csv"))

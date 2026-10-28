"""Import the demo data, run examples/running_response.yaml, and export the result for MATLAB."""

from pathlib import Path

import numpy as np
import yaml

from sciflow import Store, io, run

here = Path(__file__).parent
store = Store(here / ".sciflow")
d = here / "demo_data"
ids = {
    "wheel": store.put(io.import_file(d / "wheel.npz", units="cm/s"), {"source": "import"}),
    "widefield": store.put(io.import_file(d / "widefield.npz", parametric_axis=2, axes=["time", "y", "x"]), {"source": "import"}),
    "sync_ref": store.put(io.import_file(d / "sync_ref.npz", kind="intervals", times_key="times"), {"source": "import"}),
    "sync_img": store.put(io.import_file(d / "sync_img.npz", kind="intervals", times_key="times"), {"source": "import"}),
}
spec = yaml.safe_load((here / "running_response.yaml").read_text())
spec["inputs"] = ids
rec = run(spec, store)
resp = store.get(rec["outputs"]["response"])
peak = np.unravel_index(np.nanargmax(resp.data.max(0)), resp.sample_shape)
print(f"run {rec['run_id']}: response {resp.data.shape}, {resp.meta['n_valid']} bouts, peak pixel {peak}")
io.export_artifact(resp, here / "response.mat")

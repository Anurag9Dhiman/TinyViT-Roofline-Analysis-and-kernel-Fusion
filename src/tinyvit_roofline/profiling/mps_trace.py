"""Supplementary Metal-level GPU trace capture (.gputrace), for the handful
of representative configs where we need to see the true GPU kernel-dispatch
sequence underneath an opaque composite op like
aten::_scaled_dot_product_attention_math_for_mps (torch.profiler's
dispatcher-level view can't see inside it -- see torch_profile.py).

Requires the MTL_CAPTURE_ENABLED=1 environment variable to be set *before the
Python process starts* (Metal reads it at launch, so os.environ[...] = "1"
inside the script is too late). scripts/run_all.py sets this via subprocess
env. The resulting .gputrace bundle is opened manually in Xcode/Instruments
for the report's supplementary evidence -- it is not auto-parsed here.
"""

import os
from typing import Dict

import torch
import torch.mps.profiler as mps_profiler


def capture_gputrace(model, x, path: str, iters: int = 5) -> Dict:
    if os.path.exists(path):
        # metal_capture refuses to overwrite an existing bundle
        import shutil
        shutil.rmtree(path, ignore_errors=True)

    if not mps_profiler.is_metal_capture_enabled():
        return {
            "captured": False,
            "reason": (
                "MTL_CAPTURE_ENABLED=1 was not set before process start. "
                "Re-run this script with MTL_CAPTURE_ENABLED=1 python3 ... to capture a .gputrace."
            ),
        }

    with torch.no_grad():
        for _ in range(3):
            model(x)
        torch.mps.synchronize()
        with mps_profiler.metal_capture(path):
            for _ in range(iters):
                model(x)
            torch.mps.synchronize()

    return {"captured": True, "path": path, "iters": iters}

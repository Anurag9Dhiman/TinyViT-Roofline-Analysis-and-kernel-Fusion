#!/usr/bin/env python3
"""Phase 7: streamed ImageNet-1k accuracy -- pytorch_baseline vs
pytorch_optimized (whole-model torch.compile) vs mlx_fused_hybrid (attention
routed through MLX inside the real pretrained model, fusion/hybrid_model.py)
vs pytorch_conv_bn_fused (Phase 9's conv+BN fusion) vs
pytorch_conv_bn_fused_channels_last (Phase 11's best combination -- conv+BN
fusion stacked with channels_last memory format) vs
pytorch_conv_bn_fused_channels_last_fp16 (Phase 14's fp16 addition) -- real
accuracy+speed demonstration of the project's working optimizations, not
just synthetic-tensor benchmarks. fp16 is the one arm here that is NOT
numerically exact (unlike every other optimization in this project) --
measuring whether that actually changes classification accuracy at scale
is the point of including it.

Each arm runs as a SEPARATE subprocess, with a cooldown pause between them.
This is not paranoia: an earlier single-process, four-models-in-a-row version
of this script gave results that flatly contradicted the isolated,
order-independent benchmark in scripts/09_conv_fusion_benchmark.py (which
found pytorch_conv_bn_fused reliably ~1.3x *faster*) -- the in-process
version showed it and every other non-baseline arm ~3x *slower*, uniformly,
regardless of how different their actual implementations are. That
uniformity points at something external to the arms themselves. This
machine is a fanless MacBook Air: back-to-back full-dataset passes (each
several seconds of sustained GPU load) are a plausible thermal-throttling
trigger, and throttling would slow every later arm equally regardless of
its own efficiency. Separate processes with a cooldown between them is the
same fix already validated for conv+BN fusion specifically (see that
module's docstring) -- confirmed order-independent under isolation.

Streams the HF validation split (no local download, given limited disk) and
evaluates on a stratified subset (config.IMAGES_PER_CLASS per class) to
bound runtime. Requires the HF account to have accepted ILSVRC/imagenet-1k's
gated-access terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k.
"""

import argparse
import json
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinyvit_roofline.config import RAW_DIR, MODEL_REGISTRY, IMAGENET_HF_ID, IMAGES_PER_CLASS, NUM_CLASSES

SCRATCH_DIR = Path("/private/tmp/claude-501/-Users-anuragdhiman-Documents-tinyViT/ac31620d-7359-4957-aff3-df1814dc89cd/scratchpad")
SUBSET_CACHE = SCRATCH_DIR / "imagenet_subset_cache.pkl"
ARM_RESULT_PATH = SCRATCH_DIR / "phase7_arm_result.json"

ARMS = ("pytorch_baseline", "pytorch_optimized", "mlx_fused_hybrid", "pytorch_conv_bn_fused",
        "pytorch_conv_bn_fused_channels_last", "pytorch_conv_bn_fused_channels_last_fp16")
CHANNELS_LAST_ARMS = ("pytorch_conv_bn_fused_channels_last", "pytorch_conv_bn_fused_channels_last_fp16")
FP16_ARMS = ("pytorch_conv_bn_fused_channels_last_fp16",)
COOLDOWN_S = 10


def collect_stratified_subset(images_per_class: int, max_scan: int = 60000):
    if SUBSET_CACHE.exists():
        try:
            cached = pickle.loads(SUBSET_CACHE.read_bytes())
            if cached.get("images_per_class") == images_per_class:
                print(f"[07] using cached subset from {SUBSET_CACHE}")
                return cached["samples"], None
        except (EOFError, pickle.UnpicklingError):
            print(f"[07] cache at {SUBSET_CACHE} is corrupt/truncated (likely a prior disk-full write) -- ignoring it")

    from datasets import load_dataset
    from huggingface_hub import HfFolder

    token = HfFolder.get_token()
    try:
        ds = load_dataset(IMAGENET_HF_ID, split="validation", streaming=True, token=token)
    except Exception as e:
        return None, str(e)

    counts = defaultdict(int)
    collected = []
    try:
        for i, sample in enumerate(ds):
            if i >= max_scan:
                break
            if i % 5000 == 0 and i > 0:
                print(f"[07]   scanned {i}/{max_scan}, collected {len(collected)} so far "
                      f"({len(counts)}/{NUM_CLASSES} classes seen)", flush=True)
            label = sample["label"]
            if counts[label] >= images_per_class:
                continue
            img = sample["image"].convert("RGB")
            # Real ImageNet-1k validation images vary in size (commonly
            # 400-500px on a side); holding thousands of them at native
            # resolution is what pushed images_per_class=10/20 (10k/20k
            # images) into real memory pressure and a kernel OOM kill
            # (confirmed directly: exit code 137, SIGKILL -- not a
            # catchable Python exception, hence no traceback) on this
            # machine's shared 16GB, especially with other apps running.
            # The eval transform's own resize target here is ~236px
            # (input_size=224, crop_pct=0.95) -- downsizing to 256 on the
            # short side now, with the same bicubic filter the transform
            # itself uses, bounds memory per image (~280KB vs. 500KB-1MB+
            # at native resolution) without materially affecting the
            # transform's later resize+crop output.
            if min(img.size) > 256:
                scale = 256 / min(img.size)
                img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
            collected.append((img, label))
            counts[label] += 1
            if len(counts) >= NUM_CLASSES and all(c >= images_per_class for c in counts.values()):
                break
    except Exception as e:
        # A mid-stream network drop (seen in practice: DNS failures, connection
        # resets) must not crash the whole run with an unhandled traceback --
        # report it as a clean error so the caller can retry, instead of
        # leaving a half-finished process that may hang.
        return None, f"streaming interrupted after {len(collected)} images: {e}"
    try:
        SUBSET_CACHE.parent.mkdir(parents=True, exist_ok=True)
        # pickle.dump(..., file_handle) streams the serialized bytes directly to
        # disk, unlike write_bytes(pickle.dumps(...)) which first materializes
        # the ENTIRE serialized blob in memory -- a second, full-size copy of
        # the already-large `collected` list held simultaneously. At
        # images_per_class=20 (20,000 real PIL images, ~10GB+ raw) that doubling
        # was enough to exceed this machine's 16GB unified memory and trigger a
        # silent OOM kill (SIGKILL bypasses Python's exception handling
        # entirely -- no traceback, no error return, just a dead process) --
        # found the hard way, reproduced twice before this fix.
        with open(SUBSET_CACHE, "wb") as f:
            pickle.dump({"images_per_class": images_per_class, "samples": collected}, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as e:
        # Caching is a convenience for re-runs, not a requirement -- a full
        # disk (seen on this machine: an unrelated 64GB ~/.cache/huggingface
        # buildup) must not lose the already-collected in-memory samples.
        print(f"[07] WARNING: could not write subset cache ({e}); continuing without it")
    return collected, None


def preprocess_batches(transform, samples, batch_size):
    """Pre-transform every image ONCE, shared across all models. PIL resize/
    crop/normalize is real, non-trivial, single-threaded CPU work -- dominant
    enough that timing it together with inference (an earlier bug) made the
    cross-model comparison meaningless."""
    import torch

    batches = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        x_cpu = torch.stack([transform(img) for img, _ in batch])
        labels = torch.tensor([lbl for _, lbl in batch])
        batches.append((x_cpu, labels))
    return batches


def build_model(arm: str, variant: str):
    from tinyvit_roofline.models import load_baseline_model

    if arm == "pytorch_baseline":
        return load_baseline_model(variant)
    if arm == "pytorch_optimized":
        import torch
        return torch.compile(load_baseline_model(variant))
    if arm == "mlx_fused_hybrid":
        from tinyvit_roofline.fusion.hybrid_model import wrap_blocks_with_mlx_attention
        model = load_baseline_model(variant)
        wrap_blocks_with_mlx_attention(model)
        return model
    if arm == "pytorch_conv_bn_fused":
        import torch
        from tinyvit_roofline.fusion.conv_bn_fusion import fuse_conv_norms
        model = load_baseline_model(variant)
        fuse_conv_norms(model)
        model.to("mps")
        return model
    if arm == "pytorch_conv_bn_fused_channels_last":
        import torch
        from tinyvit_roofline.fusion.conv_bn_fusion import fuse_conv_norms
        model = load_baseline_model(variant)
        fuse_conv_norms(model)
        model.to("mps")
        return model.to(memory_format=torch.channels_last)
    if arm == "pytorch_conv_bn_fused_channels_last_fp16":
        import torch
        from tinyvit_roofline.fusion.conv_bn_fusion import fuse_conv_norms
        model = load_baseline_model(variant)
        fuse_conv_norms(model)
        model.to("mps")
        return model.to(memory_format=torch.channels_last).half()
    raise ValueError(arm)


def run_one_arm(arm: str, variant: str, batch_size: int, images_per_class: int = IMAGES_PER_CLASS,
                 warmup_batches: int = 5) -> dict:
    """Runs in its OWN process (invoked as a subprocess by run_all_arms).
    Loads samples from the shared cache, builds only this arm's model,
    times it in isolation.

    images_per_class must match whatever run_all_arms collected, or
    collect_stratified_subset's cache check (images_per_class mismatch)
    silently falls back to re-streaming from HF inside this subprocess --
    found the hard way: a bug here previously meant every --arm invocation
    ignored this parameter entirely and used the IMAGES_PER_CLASS constant,
    invisible as long as nothing ever overrode the default, but a genuine
    stuck-subprocess bug (re-streaming thousands of images with no
    visible progress) the first time something did.
    """
    import torch
    from timm.data import resolve_data_config, create_transform

    samples, err = collect_stratified_subset(images_per_class)
    if err is not None:
        return {"error": err}

    from tinyvit_roofline.models import load_baseline_model as _load_probe
    data_cfg = resolve_data_config({}, model=_load_probe(variant))
    transform = create_transform(**data_cfg, is_training=False)
    batches = preprocess_batches(transform, samples, batch_size)

    model = build_model(arm, variant)
    use_cl = arm in CHANNELS_LAST_ARMS
    use_fp16 = arm in FP16_ARMS

    with torch.no_grad():
        x_warm = batches[0][0].to("mps")
        if use_cl:
            x_warm = x_warm.to(memory_format=torch.channels_last)
        if use_fp16:
            x_warm = x_warm.half()
        for _ in range(warmup_batches):
            model(x_warm)
        torch.mps.synchronize()

        correct1, correct5, n = 0, 0, 0
        all_preds = []
        t0 = time.time()
        for x_cpu, labels in batches:
            x = x_cpu.to("mps")
            if use_cl:
                x = x.to(memory_format=torch.channels_last)
            if use_fp16:
                x = x.half()
            logits = model(x).float().to("cpu")
            top5 = logits.topk(5, dim=-1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            n += len(labels)
            all_preds.append(top5[:, 0])
        torch.mps.synchronize()
        compute_time = time.time() - t0

    preds = torch.cat(all_preds).tolist()
    return {
        "arm": arm, "top1_acc": correct1 / n, "top5_acc": correct5 / n, "n": n,
        "compute_time_s": compute_time, "preds": preds,
    }


def run_all_arms(variant: str, images_per_class: int, batch_size: int) -> dict:
    print(f"[07] streaming ImageNet-1k validation ({images_per_class} images/class target) ...")
    samples, err = collect_stratified_subset(images_per_class)
    if err is not None:
        return {"phase": "07_accuracy_eval", "error": err,
                "note": "Requires accepting the gated-access terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k"}
    print(f"[07] collected {len(samples)} images across {len(set(l for _, l in samples))} classes")

    results = {}
    for i, arm in enumerate(ARMS):
        print(f"[07] evaluating {arm} in a fresh subprocess ...")
        if ARM_RESULT_PATH.exists():
            ARM_RESULT_PATH.unlink()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--arm", arm,
             "--variant", variant, "--images-per-class", str(images_per_class),
             "--batch-size", str(batch_size)],
            capture_output=True, text=True,
        )
        if not ARM_RESULT_PATH.exists():
            print(f"[07] ERROR: {arm} subprocess produced no result. stderr tail:\n{proc.stderr[-2000:]}")
            return {"phase": "07_accuracy_eval", "error": f"{arm} subprocess failed", "stderr": proc.stderr[-4000:]}
        r = json.loads(ARM_RESULT_PATH.read_text())
        ms_img = r["compute_time_s"] / r["n"] * 1000
        print(f"     top1={r['top1_acc']*100:.2f}%  top5={r['top5_acc']*100:.2f}%  "
              f"(compute only: {r['compute_time_s']:.1f}s, {ms_img:.2f}ms/img)")
        results[arm] = r
        if i < len(ARMS) - 1:
            print(f"[07] cooling down {COOLDOWN_S}s before next arm ...")
            time.sleep(COOLDOWN_S)

    other_arms = tuple(a for a in ARMS if a != "pytorch_baseline")
    base_preds = results["pytorch_baseline"]["preds"]
    agreement = {}
    for name in other_arms:
        p = results[name]["preds"]
        agreement[name] = sum(a == b for a, b in zip(p, base_preds)) / len(base_preds)
        del results[name]["preds"]
    del results["pytorch_baseline"]["preds"]

    return {
        "phase": "07_accuracy_eval",
        "variant": variant,
        "num_images": len(samples),
        "images_per_class_target": images_per_class,
        "methodology_note": (
            "Each arm evaluated in a separate subprocess with a cooldown pause between them, "
            "to avoid both MPS allocator fragmentation and thermal throttling (this is a fanless "
            "MacBook Air) confounding the cross-model speed comparison -- see this script's "
            "module docstring for the specific contradiction that motivated the redesign."
        ),
        "results": results,
        "top1_prediction_agreement_vs_baseline": agreement,
        "accuracy_difference_top1_pp": {
            name: (results[name]["top1_acc"] - results["pytorch_baseline"]["top1_acc"]) * 100
            for name in other_arms
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny_vit_5m_224", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--images-per-class", type=int, default=IMAGES_PER_CLASS)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--arm", default=None, choices=list(ARMS),
                     help="Internal: run a single arm in this process and write ARM_RESULT_PATH. Used by subprocess invocations.")
    ap.add_argument("--output-suffix", default="",
                     help="Appended to the output filename (e.g. '_5x' for a larger-sample robustness run) "
                          "so it doesn't overwrite the default 1-image/class result.")
    args = ap.parse_args()

    if args.arm is not None:
        r = run_one_arm(args.arm, args.variant, args.batch_size, images_per_class=args.images_per_class)
        ARM_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARM_RESULT_PATH.write_text(json.dumps(r))
        sys.exit(0)

    result = run_all_arms(args.variant, args.images_per_class, args.batch_size)
    out_path = RAW_DIR / f"07_accuracy_eval_{args.variant}{args.output_suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[07] wrote {out_path}")
    if "error" in result:
        print(f"[07] ERROR: {result['error']}")

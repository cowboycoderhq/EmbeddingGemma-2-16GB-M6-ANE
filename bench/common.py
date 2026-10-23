"""Shared parts of the m6 ANE-vs-GPU benchmarks: host facts, compute units, the Core ML compute plan, the timing
loop, and power sampling with macmon. Each part is a separate function so tests can drive it alone."""
from __future__ import annotations

import json
import os
import platform
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

CU_NAMES = {"ane": "CPU_AND_NE", "gpu": "CPU_AND_GPU", "cpu": "CPU_ONLY", "all": "ALL"}


def host_info() -> dict:
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    # no host name: result files are meant to be shared
    return {"hw.model": sysctl("hw.model"), "cpu": sysctl("machdep.cpu.brand_string"), "macos": platform.mac_ver()[0],
            "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2**30)}


def compute_unit(name: str):
    import coremltools as ct
    return getattr(ct.ComputeUnit, CU_NAMES[name])


def compiled_path(package: str | Path) -> Path:
    """Compile an .mlpackage once, next to it, and return the .mlmodelc path."""
    package = Path(package)
    if package.suffix == ".mlmodelc":
        return package
    out = package.with_suffix(".mlmodelc")
    if not out.exists() or out.stat().st_mtime < package.stat().st_mtime:
        import coremltools as ct
        tmp = ct.models.utils.compile_model(str(package))
        if out.exists():
            shutil.rmtree(out)
        shutil.move(tmp, out)
    return out


def device_name(dev) -> str:
    n = type(dev).__name__ if dev is not None else "None"
    return {"MLNeuralEngineComputeDevice": "ANE", "MLGPUComputeDevice": "GPU", "MLCPUComputeDevice": "CPU"}.get(n, n)


def compute_plan_summary(mlmodelc: str | Path, cu: str) -> dict:
    """Preferred device of each ML Program operation, as Core ML plans it for these compute units.
    Weighted by Core ML's estimated cost, so cheap ops (reshapes, constants) do not hide where the work runs."""
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(path=str(mlmodelc), compute_units=compute_unit(cu))
    ops, cost = {}, {}

    def walk(block):
        for op in block.operations:
            if op.operator_name in ("const",):
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = device_name(usage.preferred_compute_device if usage else None)
            est = plan.get_estimated_cost_for_mlprogram_operation(op)
            w = est.weight if est is not None else 0.0
            ops[dev] = ops.get(dev, 0) + 1
            cost[dev] = cost.get(dev, 0.0) + w
            for b in op.blocks:
                walk(b)

    st = plan.model_structure
    if st.program is not None:
        for fn in st.program.functions.values():
            walk(fn.block)
    elif st.neuralnetwork is not None:  # older NeuralNetwork models: per-layer devices, no cost estimate
        for layer in st.neuralnetwork.layers:
            usage = plan.get_compute_device_usage_for_neuralnetwork_layer(layer)
            dev = device_name(usage.preferred_compute_device if usage else None)
            ops[dev] = ops.get(dev, 0) + 1
            cost[dev] = cost.get(dev, 0.0) + 1.0
    total = sum(cost.values()) or 1.0
    return {"compute_units": CU_NAMES[cu], "ops_by_device": ops,
            "cost_share_by_device": {k: round(v / total, 4) for k, v in cost.items()}}


def timed_runs(call, items_per_call: int, iters: int, runs: int, warmup: int) -> list[float]:
    """Run call() warmup times, then `runs` timed runs of `iters` calls. Returns items/s for each run."""
    for _ in range(warmup):
        call()
    rates = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(iters):
            call()
        dt = time.perf_counter() - t0
        rates.append(items_per_call * iters / dt)
    return rates


def summarize(values: list[float]) -> dict:
    return {"runs": [round(v, 4) for v in values], "median": round(statistics.median(values), 4),
            "min": round(min(values), 4), "max": round(max(values), 4)}


MACMON = os.environ.get("MACMON") or shutil.which("macmon") or os.path.expanduser("~/.local/bin/macmon")


class PowerSampler:
    """macmon (no sudo) in the background: raw JSON lines go to `log`; mean() averages a key over the window."""

    def __init__(self, log: str | Path | None, interval_ms: int = 250):
        self.log, self.interval, self.samples, self.proc = log, interval_ms, [], None

    def __enter__(self):
        if not os.path.exists(MACMON):
            print("macmon not found (brew install macmon, or set MACMON): no power numbers", flush=True)
            return self
        self.proc = subprocess.Popen([MACMON, "pipe", "-i", str(self.interval)], stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        time.sleep(self.interval / 1000 * 2)
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(5)
            self.thread.join(5)
        if self.log:
            with open(self.log, "a") as f:
                for s in self.samples:
                    f.write(json.dumps({k: s.get(k) for k in POWER_KEYS} | {"timestamp": s.get("timestamp")}) + "\n")

    def summary(self) -> dict:
        return power_summary(self.samples)


POWER_KEYS = ("ane_power", "gpu_power", "cpu_power", "all_power", "sys_power", "gpu_active_ratio", "gpu_freq_mhz")


def power_summary(samples: list[dict]) -> dict:
    if not samples:
        return {"samples": 0}
    out = {"samples": len(samples)}
    for k in POWER_KEYS:
        vals = [s[k] for s in samples if isinstance(s.get(k), (int, float))]
        if vals:
            out[k + "_mean"] = round(statistics.fmean(vals), 4)
            if k in ("ane_power", "gpu_power"):
                out[k + "_max"] = round(max(vals), 4)
    return out


def report(result: dict, path: str | Path | None):
    text = json.dumps(result, indent=1)
    print(text, flush=True)
    if path:
        Path(path).write_text(text)

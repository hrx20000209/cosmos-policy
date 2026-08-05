"""Per-stage latency profiler for the SO101 Cosmos async deployment.

Instruments the real deployment path (the same server object the gRPC service uses) and
breaks one action-chunk request down into its stages, so it is obvious where the ~2.5 s
per chunk actually goes.

Stages measured
---------------
  obs_prep        camera frames -> uint8 HWC + proprio extraction (server side)
  t5_lookup       text embedding fetch from the precomputed cache
  image_preproc   resize/normalise the 3 views into the 11-slot Cosmos layout
  vae_encode      Wan2.1 tokenizer encoding the conditioning frames to latents
  dit_denoise     the DiT forward passes (one per denoising step) -- usually dominant
  sampler_other   generate_samples_from_batch minus vae_encode minus dit_denoise
  action_decode   latent -> action chunk + unnormalisation
  safety_filter   the deployment safety clamps
  serialize       pickling the chunk for the gRPC reply

Everything is wrapped with torch.cuda.synchronize() so GPU work is attributed to the stage
that launched it rather than to whatever happens to sync later.

Usage (repo root, cosmos env):
    PYTHONPATH=$PWD:/home/hrx/Projects/lerobot/src python \
        cosmos_policy/scripts/profile_so101_latency.py --iters 12 --denoise 10
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import get_action
from cosmos_policy.experiments.robot.so101_async_deploy import (
    SO101_ACTION_NAMES,
    SO101CosmosAsyncPolicyServer,
    SO101CosmosAsyncServerConfig,
)

CAMERA_KEYS = {"front": "observation.images.front",
               "right": "observation.images.right",
               "wrist": "observation.images.wrist"}

STAGE_ORDER = ["obs_prep", "t5_lookup", "image_preproc", "vae_encode", "dit_denoise",
               "sampler_other", "action_decode", "safety_filter", "serialize"]

STAGE_NOTES = {
    "obs_prep": "3 views -> uint8 HWC",
    "t5_lookup": "cached embedding",
    "image_preproc": "resize + 11-slot layout",
    "vae_encode": "Wan2.1 tokenizer",
    "dit_denoise": "2B DiT, one fwd per step",
    "sampler_other": "scheduler + latent plumbing",
    "action_decode": "latent -> deg, unnormalise",
    "safety_filter": "deployment clamps",
    "serialize": "pickle for gRPC",
}


class Timers:
    """Accumulates per-stage timings for the current iteration."""

    def __init__(self):
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._cur: dict[str, float] = defaultdict(float)
        self.dit_calls: list[int] = []
        self._dit_n = 0

    def add(self, stage: str, ms: float):
        self._cur[stage] += ms

    def flush(self):
        for k, v in self._cur.items():
            self.samples[k].append(v)
        self.dit_calls.append(self._dit_n)
        self._cur = defaultdict(float)
        self._dit_n = 0


TIMERS = Timers()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class stage:
    """Context manager timing a stage, GPU-synchronised at both ends."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        TIMERS.add(self.name, (time.perf_counter() - self.t0) * 1000)
        return False


def instrument(model):
    """Monkeypatch the inference path so each stage reports its own time."""
    # --- VAE tokenizer ---
    tok = model.tokenizer
    for meth in ("encode", "decode"):
        if hasattr(tok, meth):
            orig = getattr(tok, meth)

            def wrapper(*a, _orig=orig, **kw):
                with stage("vae_encode"):
                    return _orig(*a, **kw)
            setattr(tok, meth, wrapper)

    # --- DiT forward passes (one per denoising step) ---
    cls = type(model)
    orig_x0 = cls.get_x0_fn_from_batch

    def _wrap_x0(fn):
        def timed_fn(*fa, **fkw):
            sync()
            t0 = time.perf_counter()
            out = fn(*fa, **fkw)
            sync()
            TIMERS.add("dit_denoise", (time.perf_counter() - t0) * 1000)
            TIMERS._dit_n += 1
            return out
        return timed_fn

    def patched_x0(self, *a, **kw):
        res = orig_x0(self, *a, **kw)
        # policy_text2world_model returns (x0_fn, orig_clean_latent_frames);
        # the base class returns just x0_fn.
        if isinstance(res, tuple):
            return (_wrap_x0(res[0]), *res[1:])
        return _wrap_x0(res)
    cls.get_x0_fn_from_batch = patched_x0

    # --- whole sampling call, to derive the residual ---
    orig_gen = cls.generate_samples_from_batch

    def patched_gen(self, *a, **kw):
        sync()
        t0 = time.perf_counter()
        out = orig_gen(self, *a, **kw)
        sync()
        total = (time.perf_counter() - t0) * 1000
        other = total - TIMERS._cur.get("dit_denoise", 0.0) - TIMERS._cur.get("vae_encode", 0.0)
        TIMERS.add("sampler_other", max(other, 0.0))
        return out
    cls.generate_samples_from_batch = patched_gen

    # --- image preprocessing + t5 + unnormalise (module-level helpers) ---
    for name, label in (("prepare_images_for_model", "image_preproc"),
                        ("get_t5_embedding_from_cache", "t5_lookup"),
                        ("unnormalize_actions", "action_decode")):
        if hasattr(cosmos_utils, name):
            orig_fn = getattr(cosmos_utils, name)

            def fn_wrapper(*a, _orig=orig_fn, _label=label, **kw):
                with stage(_label):
                    return _orig(*a, **kw)
            setattr(cosmos_utils, name, fn_wrapper)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="hrx2000/Three_Cubes_1")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--denoise", type=int, default=None)
    p.add_argument("--outdir", default="/home/hrx/Projects/cosmos-policy/outputs/latency")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = SO101CosmosAsyncServerConfig()
    denoise = args.denoise or cfg.num_denoising_steps_action
    server = SO101CosmosAsyncPolicyServer(cfg)
    instrument(server.model)

    ds = LeRobotDataset(args.repo_id)
    ep_from = int(ds.meta.episodes["dataset_from_index"][args.episode])
    task = ds[ep_from].get("task") or ""

    print(f"\nprofiling {args.iters} chunks (+{args.warmup} warmup), denoise={denoise}, "
          f"chunk_size={server.cosmos_cfg.chunk_size}\n")

    totals = []
    for i in range(args.warmup + args.iters):
        sample = ds[ep_from + (i * 10) % 400]
        state = np.asarray(sample["observation.state"], dtype=np.float32)

        sync()
        t_start = time.perf_counter()

        with stage("obs_prep"):
            raw = {c: sample[k] for c, k in CAMERA_KEYS.items()}
            for j, nm in enumerate(SO101_ACTION_NAMES):
                raw[nm] = float(state[j])
            cosmos_obs = server._build_cosmos_observation(raw)

        result = get_action(server.cosmos_cfg, server.model, server.dataset_stats,
                            cosmos_obs, task, seed=cfg.seed + i, randomize_seed=False,
                            num_denoising_steps_action=denoise,
                            generate_future_state_and_value_in_parallel=False)

        with stage("safety_filter"):
            chunk = np.asarray(result["actions"], dtype=np.float32)
            clamped = server._apply_safety_filters(chunk, cosmos_obs["proprio"])

        with stage("serialize"):
            payload = pickle.dumps([torch.from_numpy(a) for a in clamped])

        sync()
        total_ms = (time.perf_counter() - t_start) * 1000

        if i < args.warmup:
            TIMERS._cur.clear()
            TIMERS._dit_n = 0
            print(f"  warmup {i + 1}: {total_ms:.0f} ms")
            continue

        TIMERS.flush()
        totals.append(total_ms)
        print(f"  iter {i - args.warmup + 1:2d}: {total_ms:7.0f} ms  "
              f"(payload {len(payload) / 1024:.1f} KB)")

    stats = summarize(totals, denoise)
    report(stats, outdir, denoise, server.cosmos_cfg.chunk_size, cfg)


def summarize(totals, denoise):
    def pct(v, p):
        return statistics.quantiles(v, n=100)[p - 1] if len(v) > 2 else max(v)

    stats = {"total": {"mean": statistics.mean(totals), "p50": statistics.median(totals),
                       "p95": pct(totals, 95), "max": max(totals), "n": len(totals)},
             "denoise_steps": denoise,
             "dit_forwards_per_chunk": statistics.median(TIMERS.dit_calls) if TIMERS.dit_calls else 0,
             "stages": {}}
    for name in STAGE_ORDER:
        v = TIMERS.samples.get(name)
        if not v:
            continue
        stats["stages"][name] = {"mean": statistics.mean(v), "p50": statistics.median(v),
                                 "p95": pct(v, 95), "max": max(v),
                                 "share": statistics.mean(v) / statistics.mean(totals) * 100}
    return stats


def report(stats, outdir, denoise, chunk_size, cfg):
    print("\n=== per-stage latency (ms) ===")
    print(f"{'stage':16s} {'mean':>9s} {'p50':>9s} {'p95':>9s} {'max':>9s} {'share':>8s}")
    print("-" * 66)
    for name, s in stats["stages"].items():
        print(f"{name:16s} {s['mean']:9.1f} {s['p50']:9.1f} {s['p95']:9.1f} "
              f"{s['max']:9.1f} {s['share']:7.1f}%")
    t = stats["total"]
    print("-" * 66)
    print(f"{'TOTAL':16s} {t['mean']:9.1f} {t['p50']:9.1f} {t['p95']:9.1f} {t['max']:9.1f}")
    print(f"\nDiT forward passes per chunk: {stats['dit_forwards_per_chunk']:.0f} "
          f"(denoise={denoise})")
    accounted = sum(s["mean"] for s in stats["stages"].values())
    print(f"accounted for: {accounted:.0f} ms of {t['mean']:.0f} ms "
          f"({accounted / t['mean'] * 100:.1f}%)")

    sustainable_fps_full = chunk_size / (t["mean"] / 1000)
    print(f"\nsustainable client FPS executing the full {chunk_size}-step chunk: "
          f"{sustainable_fps_full:.1f}")
    print(f"sustainable client FPS at actions_per_chunk={cfg.actions_per_chunk}: "
          f"{cfg.actions_per_chunk / (t['mean'] / 1000):.1f}")

    stats["chunk_size"] = chunk_size
    stats["actions_per_chunk"] = cfg.actions_per_chunk
    (outdir / "latency.json").write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {outdir / 'latency.json'}")
    _plot(stats, outdir)
    _html(stats, outdir)


def _plot(stats, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(stats["stages"].keys())
    means = [stats["stages"][n]["mean"] for n in names]
    dominant = names[int(np.argmax(means))]
    colors = ["#bd7a10" if n == dominant else "#0f9c92" for n in names]

    fig, ax = plt.subplots(figsize=(11, 0.62 * len(names) + 2.2))
    y = np.arange(len(names))
    ax.barh(y, means, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n{STAGE_NOTES.get(n, '')}" for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("latency per chunk (ms, log scale)")
    ax.set_title(f"Cosmos SO101 per-stage latency on Thor — "
                 f"{stats['total']['mean']:.0f} ms/chunk total "
                 f"(denoise={stats['denoise_steps']})")
    ax.grid(axis="x", alpha=0.3, which="both", lw=0.6)
    for yi, m in zip(y, means):
        share = stats["stages"][names[yi]]["share"]
        ax.text(m * 1.08, yi, f"{m:.0f} ms ({share:.0f}%)", va="center", fontsize=9)
    ax.set_xlim(right=max(means) * 3)
    fig.tight_layout()
    out = outdir / "latency_breakdown.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


CSS = """
:root{
  --ground:#eff1f3; --surface:#fff; --surface-2:#f5f7f9; --line:#dce1e6;
  --ink:#111619; --muted:#586570; --faint:#8996a1;
  --accent:#0f9c92; --accent-soft:#0f9c9220;
  --warn:#b8730d; --warn-soft:#b8730d1c; --ok:#2d8b50;
  --track:#e4e8ec;
  --shadow:0 1px 2px rgba(17,30,40,.05),0 8px 24px rgba(17,30,40,.06);
}
@media (prefers-color-scheme:dark){:root{
  --ground:#0a0d10; --surface:#131a1f; --surface-2:#0f151a; --line:#232c34;
  --ink:#e8eef3; --muted:#8fa0ad; --faint:#68767f;
  --accent:#2ed6c6; --accent-soft:#2ed6c61c;
  --warn:#e2a63c; --warn-soft:#e2a63c18; --ok:#49cc7e;
  --track:#212a31;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}}
:root[data-theme="light"]{
  --ground:#eff1f3; --surface:#fff; --surface-2:#f5f7f9; --line:#dce1e6;
  --ink:#111619; --muted:#586570; --faint:#8996a1;
  --accent:#0f9c92; --accent-soft:#0f9c9220;
  --warn:#b8730d; --warn-soft:#b8730d1c; --ok:#2d8b50; --track:#e4e8ec;
  --shadow:0 1px 2px rgba(17,30,40,.05),0 8px 24px rgba(17,30,40,.06);
}
:root[data-theme="dark"]{
  --ground:#0a0d10; --surface:#131a1f; --surface-2:#0f151a; --line:#232c34;
  --ink:#e8eef3; --muted:#8fa0ad; --faint:#68767f;
  --accent:#2ed6c6; --accent-soft:#2ed6c61c;
  --warn:#e2a63c; --warn-soft:#e2a63c18; --ok:#49cc7e; --track:#212a31;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:940px;margin:0 auto;padding:clamp(20px,4vw,50px) clamp(16px,4vw,30px) 72px}
header{border-bottom:1px solid var(--line);padding-bottom:24px;margin-bottom:32px}
.eyebrow{font-size:11.5px;letter-spacing:.17em;text-transform:uppercase;color:var(--accent);
  font-weight:660;margin:0 0 11px}
h1{font-size:clamp(25px,4.4vw,37px);line-height:1.1;letter-spacing:-.02em;margin:0 0 13px;
  text-wrap:balance;font-weight:700}
.sub{color:var(--muted);font-size:15.5px;max-width:64ch;margin:0}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:19px}
.chip{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;padding:5px 11px;
  border:1px solid var(--line);border-radius:999px;background:var(--surface);
  color:var(--muted);font-weight:540}
.chip b{color:var(--ink);font-weight:640;font-variant-numeric:tabular-nums}
.dot{width:7px;height:7px;border-radius:50%;background:var(--accent)}
.dot.ok{background:var(--ok)}.dot.warn{background:var(--warn)}
section{margin:0 0 38px}
h2{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:var(--faint);
  font-weight:660;margin:0 0 15px;padding-bottom:9px;border-bottom:1px solid var(--line)}
p{margin:0 0 12px}p.dim{color:var(--muted);font-size:14.5px}
.k{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;
  background:var(--surface-2);border:1px solid var(--line);padding:1px 6px;border-radius:5px}
.chart{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:22px 22px 14px;box-shadow:var(--shadow)}
.chart-cap{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
  margin-bottom:19px;flex-wrap:wrap}
.chart-cap .big{font-size:15px;font-weight:640}
.chart-cap .scale{font-size:12px;color:var(--faint)}
.bars{display:flex;flex-direction:column;gap:12px}
.bar-row{display:grid;grid-template-columns:200px 1fr 96px;gap:14px;align-items:center}
@media(max-width:620px){.bar-row{grid-template-columns:1fr 92px;gap:5px 12px}
  .bar-label{grid-column:1/-1}}
.bar-label{font-size:13.5px;font-weight:560;display:flex;flex-direction:column;line-height:1.25}
.bar-note{font-size:11.5px;color:var(--faint);font-weight:440}
.bar-track{position:relative;height:25px;border-radius:5px;overflow:hidden;background:var(--track);
  background-image:repeating-linear-gradient(90deg,var(--line) 0 1px,transparent 1px 25%)}
.bar-fill{height:100%;border-radius:5px;background:var(--accent);opacity:.92}
.bar-fill.dominant{background:linear-gradient(90deg,var(--accent),var(--warn));opacity:1}
.bar-val{font-size:12.5px;font-family:ui-monospace,monospace;color:var(--ink);
  font-variant-numeric:tabular-nums;text-align:right;font-weight:600}
.bar-val small{display:block;color:var(--faint);font-weight:500;font-size:11px}
.axis{display:grid;grid-template-columns:200px 1fr 96px;gap:14px;margin-top:8px}
@media(max-width:620px){.axis{grid-template-columns:1fr 92px}}
.axis-ticks{position:relative;height:16px}
.axis-ticks span{position:absolute;transform:translateX(-50%);font-size:11px;color:var(--faint)}
.axis-ticks span:first-child{transform:translateX(0)}
.axis-ticks span:last-child{transform:translateX(-100%)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:640px){.grid2{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:17px 19px;box-shadow:var(--shadow)}
.panel.ok{border-left:3px solid var(--ok)}.panel.warn{border-left:3px solid var(--warn)}
.panel h3{font-size:15.5px;margin:0 0 6px;font-weight:640}
ul.tight{margin:9px 0 0;padding-left:18px;color:var(--muted);font-size:14.5px}
ul.tight li{margin:5px 0}
table{width:100%;border-collapse:collapse;font-size:13.5px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);
  border-radius:14px;box-shadow:var(--shadow)}
th,td{padding:9px 14px;text-align:right;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
th{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);font-weight:640}
tbody tr:last-child td{border-bottom:none}
tr.total td{font-weight:680;background:var(--surface-2)}
.callout{background:var(--warn-soft);border:1px solid var(--warn);border-radius:12px;
  padding:15px 19px;font-size:14.5px;color:var(--muted)}
.callout b{color:var(--warn)}
.finding{background:var(--surface-2);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:10px;padding:13px 17px;margin-top:14px;
  font-size:14px;color:var(--muted);line-height:1.6}
.finding strong{color:var(--ink)}
footer{margin-top:50px;padding-top:18px;border-top:1px solid var(--line);color:var(--faint);
  font-size:12.5px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
"""


def _html(stats, outdir):
    """Emit a standalone per-stage latency report page."""
    import math

    st = stats["stages"]
    total = stats["total"]
    names = list(st.keys())
    means = [st[n]["mean"] for n in names]
    dominant = names[int(np.argmax(means))]

    # Log scale so the sub-millisecond stages stay visible next to the 1.9 s one.
    lo, hi = 0.01, max(means) * 1.6

    def width(v):
        v = max(v, lo)
        return (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * 100

    rows = []
    for n in names:
        s = st[n]
        cls = " dominant" if n == dominant else ""
        rows.append(
            f'<div class="bar-row"><div class="bar-label">{n}'
            f'<span class="bar-note">{STAGE_NOTES.get(n, "")}</span></div>'
            f'<div class="bar-track"><div class="bar-fill{cls}" '
            f'style="width:{width(s["mean"]):.1f}%"></div></div>'
            f'<div class="bar-val">{s["mean"]:.0f} ms<small>{s["share"]:.1f}%</small></div></div>')

    trows = []
    for n in names:
        s = st[n]
        trows.append(f"<tr><td>{n}</td><td>{s['mean']:.1f}</td><td>{s['p50']:.1f}</td>"
                     f"<td>{s['p95']:.1f}</td><td>{s['max']:.1f}</td><td>{s['share']:.1f}%</td></tr>")
    trows.append(f"<tr class='total'><td>total</td><td>{total['mean']:.1f}</td>"
                 f"<td>{total['p50']:.1f}</td><td>{total['p95']:.1f}</td>"
                 f"<td>{total['max']:.1f}</td><td>100%</td></tr>")

    ticks = "".join(
        f'<span style="left:{width(v):.1f}%">{lbl}</span>'
        for v, lbl in [(0.01, "0.01 ms"), (1, "1 ms"), (100, "100 ms"), (max(means), "2 s")])

    chunk = stats.get("chunk_size", 30)
    apc = stats.get("actions_per_chunk", 10)
    fps_full = chunk / (total["mean"] / 1000)
    fps_apc = apc / (total["mean"] / 1000)
    dit = st.get("dit_denoise", {}).get("mean", 0)
    vae = st.get("vae_encode", {}).get("mean", 0)
    per_step = dit / max(stats["dit_forwards_per_chunk"], 1)
    floor = total["mean"] - dit  # what remains if denoising were free

    html = f"""<title>Cosmos SO101 on Thor — latency breakdown</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">Per-stage profile · measured on device</p>
  <h1>Where the 2.5 seconds go</h1>
  <p class="sub">One action-chunk request through the real SO101 deployment path on NVIDIA Thor:
    three camera views in, a 30-step joint trajectory out. Every stage is GPU-synchronised, so
    work is charged to the stage that launched it.</p>
  <div class="chips">
    <span class="chip"><b>{total['mean']:.0f} ms</b> per chunk</span>
    <span class="chip">Cosmos Predict2 <b>2B</b> · bf16</span>
    <span class="chip">chunk <b>{chunk}</b> · denoise <b>{stats['denoise_steps']}</b></span>
    <span class="chip"><span class="dot warn"></span>DiT is <b>{st['dit_denoise']['share']:.0f}%</b></span>
    <span class="chip">n = <b>{total['n']}</b> chunks</span>
  </div>
</header>

<section>
  <h2>Per-stage latency</h2>
  <div class="chart">
    <div class="chart-cap">
      <div class="big">One chunk · {total['mean']:.0f} ms total</div>
      <div class="scale">log scale · lower is better</div>
    </div>
    <div class="bars">{''.join(rows)}</div>
    <div class="axis"><div></div><div class="axis-ticks">{ticks}</div><div></div></div>
  </div>
  <p class="dim" style="margin-top:14px">The <strong>2B DiT dominates at
    {dit:.0f}&nbsp;ms across {stats['dit_forwards_per_chunk']:.0f} forward passes
    ({per_step:.0f}&nbsp;ms each)</strong>. The stages that are neither DiT nor VAE add up to
    about {total['mean'] - dit - vae:.0f}&nbsp;ms — camera conversion, the 11-slot image layout,
    unnormalisation, safety clamps and pickling are all effectively free.</p>
</section>

<section>
  <h2>The floor nobody can tune away</h2>
  <div class="callout">
    <b>VAE encode is {vae:.0f} ms and does not shrink with denoising steps.</b>
    Cutting <span class="k">num_denoising_steps_action</span> only attacks the {dit:.0f} ms DiT
    term. Even with denoising free, a chunk still costs about {floor:.0f} ms — so
    ~{chunk / (floor / 1000):.0f} FPS is the ceiling for this checkpoint on this device while it
    encodes three views every replan.
  </div>
  <div class="finding"><strong>What this means for the client.</strong> Async inference only
    overlaps while a chunk covers more wall-clock time than the next one takes to compute. At
    {total['mean']:.0f} ms, executing the full {chunk}-step chunk sustains
    <strong>{fps_full:.1f} FPS</strong>; executing only <span class="k">actions_per_chunk={apc}</span>
    sustains just <strong>{fps_apc:.1f} FPS</strong>. Running the client faster than that makes the
    arm move in bursts and then stall — which is exactly what "slow and stuttering" looks like.</div>
</section>

<section>
  <h2>Full statistics</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>stage</th><th>mean</th><th>p50</th><th>p95</th><th>max</th><th>share</th></tr></thead>
    <tbody>{''.join(trows)}</tbody>
  </table></div>
  <p class="dim" style="margin-top:12px">All values in milliseconds over {total['n']} profiled
    chunks after warm-up. Stages account for
    {sum(s['mean'] for s in st.values()) / total['mean'] * 100:.1f}% of measured wall time.</p>
</section>

<footer>
  <span>cosmos_policy/scripts/profile_so101_latency.py</span>
  <span>NVIDIA Thor · aarch64 · CUDA 13.0 · torch 2.10 · TE 2.16</span>
</footer>
</div>
"""
    out = outdir / "latency_report.html"
    out.write_text(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

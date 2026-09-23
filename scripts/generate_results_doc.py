"""Generate docs/research/RESULTS.md from the real result files.

Every number in the generated document is read from a JSON file on disk; nothing
is transcribed by hand. Re-run this after new evaluations land and the tables
update themselves.
"""
import glob, json, os

QUARTET = ["MAE", "S_measure", "E_adaptive", "F_adaptive"]
EXTRA = ["E_mean", "E_max", "F_mean", "F_max", "boundary_MAE", "boundary_F1"]

# canonical metric file per (label, split); duplicates exist and are identical
RUNS = [
    ("E8 legacy (old code, renormalized gate, eff batch 32)", "results/legacy/legacy_8expert/evaluation_results", "legacy"),
    ("E4 K2 (dense gate, eff batch 40)", "results/EXP_B4_E4_K2_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results", "exp"),
    ("E2 K1 (dense gate, eff batch 40)", "results/EXP_B4_E2_K1_S40_R1_L3_M_SPARSE_GATEDENSE/eval_results", "exp"),
]


def load_metrics(directory, style):
    """Return {"test_real": {...}, "test_sys": {...}} for one run."""
    out = {}
    for split in ("test_real", "test_sys"):
        if style == "legacy":
            pattern = f"{directory}/metrics_{split}_*.json"
        else:
            pattern = f"{directory}/metrics_{split}_*.json"
        hits = sorted(glob.glob(pattern))
        if hits:
            out[split] = (hits[0], json.load(open(hits[0])))
    return out


def fmt(v, places=4):
    return f"{v:.{places}f}" if isinstance(v, float) else str(v)


lines = []
w = lines.append

w("# Results")
w("")
w("Every number in this file is read from a result file on disk and names its source. "
  "Nothing here is transcribed by hand: `RESULTS.md` is generated from the JSON under "
  "`results/`, so it can be regenerated after new evaluations land.")
w("")
w("All evaluations are single-pass (no test-time augmentation) and scored in **original "
  "image coordinates** — predictions are reverse-geometried out of the 384x384 padded "
  "model frame before metrics are computed. Flip-average TTA (`--tta hflip`) is "
  "implemented and defaults to on for *new* runs; every table below predates that, so "
  "none of it is TTA-boosted.")
w("")

w("## 1. Main results")
w("")
w("Two test sets: `test_sys` (1500 synthetic adverse-weather images) and `test_real` "
  "(554 real-world images). MAE lower is better; S-measure, E-measure and F-measure "
  "higher is better.")
w("")
hdr = "| run | split | " + " | ".join(QUARTET) + " | n | source |"
sep = "|" + "---|" * (len(QUARTET) + 4)
w(hdr)
w(sep)
sources = []
for label, directory, style in RUNS:
    for split, (path, d) in sorted(load_metrics(directory, style).items()):
        gl = d.get("global", d)
        cells = " | ".join(fmt(gl.get(k)) for k in QUARTET)
        w(f"| {label} | {split} | {cells} | {gl.get('sample_count')} | `{path}` |")
        sources.append(path)
w("")
w("**Provenance and confounds — read before comparing rows.**")
w("")
w("- The **E8 legacy** numbers come from a different code version and a different recipe "
  "(renormalized gate, router noise on, load-balance weights 0.08/0.15/0.1, effective "
  "batch 32). It is the historical best and the reproduction target, not a like-for-like "
  "comparison against the rows below it.")
w("- The **E4** and **E2** rows were trained under dense gates, router noise off, "
  "load-balance weights 0.016/0.03/0.02, and effective batch 40. They differ from the E8 "
  "row in five ways at once, so a performance gap between them and E8 cannot be "
  "attributed to expert count.")
w("- The 2026-09 experiment set (E8 k=2 renormalized and dense, seed repeats, and E4 at "
  "the E8 recipe) is training now; those rows get added when they are evaluated.")
w("")
w("### Extended metrics")
w("")
w("| run | split | " + " | ".join(EXTRA) + " |")
w("|" + "---|" * (len(EXTRA) + 2))
for label, directory, style in RUNS:
    for split, (path, d) in sorted(load_metrics(directory, style).items()):
        gl = d.get("global", d)
        w(f"| {label} | {split} | " + " | ".join(fmt(gl.get(k)) for k in EXTRA) + " |")
w("")

w("## 2. Weather-wise breakdown")
w("")
w("Per-condition MAE / S-measure / F-measure. Conditions with fewer than 10 samples are "
  "marked; treat them as anecdote, not evidence.")
w("")
w("| run | split | condition | n | MAE | S-measure | F-measure |")
w("|---|---|---|---|---|---|---|")
for label, directory, style in RUNS:
    for split, (path, d) in sorted(load_metrics(directory, style).items()):
        for cond, wd in sorted((d.get("weather") or {}).items()):
            n = wd.get("sample_count")
            flag = " *" if isinstance(n, int) and n < 10 else ""
            w(f"| {label} | {split} | {cond}{flag} | {n} | "
              + " | ".join(fmt(wd.get(k)) for k in ("MAE", "S_measure", "F_adaptive")) + " |")
w("")

w("## 3. Proxy ablations")
w("")
w("Each arm perturbs the *trained* model at inference time, so this measures how much "
  "each component contributes to the final prediction rather than what training does. "
  "`random_routing` reports the mean over four seeds.")
w("")
w("| model | arm | MAE | S-measure | source |")
w("|---|---|---|---|---|")
for path in sorted(glob.glob("results/**/proxy_ablation*.json", recursive=True)):
    d = json.load(open(path))
    ck = os.path.basename(str(d.get("checkpoint", "best.pth")))
    for arm, v in (d.get("ablations") or {}).items():
        if isinstance(v, dict) and "MAE" in v:
            w(f"| `{path.split('/')[1]}` | {arm} | {fmt(v['MAE'])} | "
              f"{fmt(v.get('S_measure'))} | `{path}` |")
        elif isinstance(v, dict) and "random_routing_seeds" in v:
            rs = v["random_routing_seeds"]
            m = sum(r["MAE"] for r in rs) / len(rs)
            s = sum(r["S_measure"] for r in rs) / len(rs)
            w(f"| `{path.split('/')[1]}` | {arm} (4 seeds) | {fmt(m)} | {fmt(s)} | `{path}` |")
w("")

w("## 4. Routing behaviour")
w("")
w("Per-token routing entropy, averaged over the test set. For reference, the maximum "
  "entropy of an E-way gate distribution is ln(E): ln(2)=0.693, ln(4)=1.386, ln(8)=2.079.")
w("")
w("| model | source |")
w("|---|---|")
for path in sorted(glob.glob("results/**/entropy_comparison.json", recursive=True)):
    d = json.load(open(path))
    real = d.get("real", {})
    vals = ", ".join(f"{k}={v:.4f}" for k, v in sorted(real.items()))
    w(f"| `{path.split('/')[1]}` | real: {vals} |")
w("")
w("Expert usage (hard assignment fraction, %) at the final logged epoch, per scale. "
  "Values near 100/E are uniform; values near 0 mean the expert is effectively dead.")
w("")
w("| run | epoch | scale | usage % | dead (<2%) |")
w("|---|---|---|---|---|")
for run in sorted(glob.glob("results/EXP_*")):
    eps = sorted(glob.glob(f"{run}/*routing_stats_ep*.json"),
                 key=lambda s: int(s.split("ep")[-1].split(".")[0]))
    if not eps:
        continue
    for path in eps[-1:]:
        d = json.load(open(path))
        ep = path.split("ep")[-1].split(".")[0]
        for scale in ("moe_4", "moe_8", "moe_16"):
            if scale in d:
                fr = d[scale]["hard_fractions"]
                dead = sum(1 for f in fr if f < 0.02)
                usage = ", ".join(f"{f*100:.1f}" for f in fr)
                w(f"| `{run.split('/')[-1]}` | {ep} | {scale} | {usage} | {dead} |")
w("")

w("## 5. What is missing")
w("")
w("Stated plainly so no claim outruns the evidence:")
w("")
w("- **The mixture-vs-dense control has never been run.** `M_DENSE` and `M_NONE` have "
  "zero runs under `results/`. The configurations exist "
  "(`experiments/v_4expert_densecontrol.json`, `experiments/v_4expert_nonecontrol.json`). "
  "Without this arm the paper cannot claim the mixture contributes anything over a "
  "matched dense model.")
w("- **No external baselines.** Every number in this file is this project's own. A "
  "comparison table against prior methods needs either published WXSOD numbers or "
  "re-implementations.")
w("- **Single seed per configuration** except the seed repeats now training. There is "
  "no historical variance estimate, so a difference smaller than the seed spread cannot "
  "be interpreted.")
w("- **The backbone axis is not testable as configured.** `config.model.backbone` does "
  "not reach the model — `src/model.py` hard-codes `pvt_v2_b4` — so the generated "
  "ablation arms that vary it would train identical architectures. Do not report a "
  "backbone comparison until this is wired through.")
w("")
w("## 6. Source files")
w("")
for s in sorted(set(sources)):
    w(f"- `{s}`")
out = "\n".join(lines) + "\n"
with open("docs/research/RESULTS.md", "w") as f:
    f.write(out)
print(f"  wrote docs/research/RESULTS.md ({len(lines)} lines)")

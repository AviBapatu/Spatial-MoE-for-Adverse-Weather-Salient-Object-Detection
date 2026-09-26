# Documentation index and writing brief

This directory is the knowledge base the paper is written from. Two documents are
**canonical**: `RESEARCH_TRUTH.md` and `docs/research/RESULTS.md`. Nothing in the paper may
contradict them, and every number must be traceable to a file named in them.

## Writing brief

To have the paper written from these documents, give the writer:

- `docs/research/RESEARCH_TRUTH.md`
- everything in `docs/research/`
- `paper/FINAL_PAPER_PLAN.md` and `paper/main.tex`

Do **not** give the historical audits listed under "Historical — do not cite" as sources;
they are records of earlier revisions and their bodies contain superseded numbers.

Read in this order:

1. `RESEARCH_TRUTH.md` — what may be claimed, what is forbidden, and the evidence behind each claim.
2. `docs/research/RESULTS.md` — every measured number, each with its source file.
3. `docs/research/PROJECT_OVERVIEW.md` — problem, claimed novelty, system summary, repository map.
4. `docs/research/ARCHITECTURE.md` — the model as built (tensor shapes, router, decoder, heads).
5. `docs/research/TRAINING.md` — optimisation, checkpointing, distributed invariants, evaluation protocol.
6. `docs/research/DATASETS.md` — WXSOD splits, geometry, augmentation, targets.
7. `docs/research/EXPERIMENTS.md` — config system, experiment IDs, the run set, protocols, gaps.
8. `docs/research/ABLATIONS.md` — what has and has not been ablated.
9. `docs/research/RESULTS_NARRATIVE.md` — interpretation of the results.
10. `docs/research/RELATED_WORK.md` and `docs/research/LITERATURE_DATABASE.md` — positioning and citations.
11. `paper/FINAL_PAPER_PLAN.md` — the target structure, page budget, figures, tables, equations and references.

**Target paper:** SCOVA, **6 pages maximum including references**, two-column, 8 references.
The structure, float plan and page budget are in `paper/FINAL_PAPER_PLAN.md` §2–§7, and its
section numbering matches `paper/main.tex` (§I–§V, with §III-A–G and §IV-A–F). `paper/main.tex`
is the current manuscript and builds with `latexmk -pdf` to 6 pages.

**The two traps that produced every stale claim in this repository:**

- **The preset is not the recipe.** `experiments/baseline_v1.json` is a template
  (`ssim_weight`/`boundary_weight` = 0, `deep_supervision` = false). The reported recipe sets
  both loss weights to 1.0 without it and enables deep supervision, so the objective has eight
  active terms, not four. See `RESEARCH_TRUTH.md` §1.1.
- **Routing entropy is over the full $E$-way distribution**, ceiling `ln E = 2.0794` nats for
  `E = 8`, normalised by `log 8` — not the top-$k$ gate entropy. The older
  `results/legacy/legacy_8expert/eval_results/entropy_comparison.json` files (~0.69 nats) are a pre-fix
  measurement.

**Forbidden claims** are listed in `RESEARCH_TRUTH.md` §3; do not write them. In particular:
no measured comparison to prior methods, no claim that the mixture helps, no claim of
expert specialisation by weather, and no causal claim about any design choice.

## Canonical

| File | What it is |
|---|---|
| `docs/research/RESEARCH_TRUTH.md` | **Canonical.** What may be claimed, and what is forbidden. Every claim is marked as implementation-, experiment- or literature-supported, with the gaps stated. |
| `docs/research/RESULTS.md` | **Canonical.** Every measured number, each traceable to a result file. |

## Reference

| File | Covers |
|---|---|
| `docs/research/ARCHITECTURE.md` | The model as built, with the caveats a paper must not contradict |
| `docs/research/TRAINING.md` | Entrypoints, optimisation, checkpointing, distributed invariants, evaluation protocol |
| `docs/research/DATASETS.md` | WXSOD structure, splits, geometry, augmentation, targets |
| `docs/research/EXPERIMENTS.md` | Config system, ID rules, the current experiment set, protocols, what is missing |
| `docs/research/ABLATIONS.md` | What has been ablated and what has not, including comparisons that are currently meaningless |
| `docs/research/RESULTS_NARRATIVE.md` | Interpretation of the legacy results, plus a provenance note on which evaluation is canonical |
| `docs/research/PROJECT_OVERVIEW.md` | Problem, claimed novelty, system summary, repository map |

## Literature

| File | Covers |
|---|---|
| `docs/research/RELATED_WORK.md` | Closest prior work, where this project differs, which claims survive, which must be qualified, and the experiments each claim still needs |
| `docs/research/LITERATURE_DATABASE.md` | The underlying citation records, keyed `DB-nn` |

## Paper

| File | Covers |
|---|---|
| `paper/main.tex` | The manuscript (6 pages), and the reference for any rewrite |
| `paper/FINAL_PAPER_PLAN.md` | Structure, figure and table plan; matches `paper/main.tex` |
| `paper/REDESIGN_PLAN.md` | The (applied) 6-page redesign plan |

## Historical — do not cite

| File | Why |
|---|---|
| `paper/CLAIM_AUDIT.md`, `paper/MANUSCRIPT_AUDIT.md`, `docs/research/PAPER_CORRECTIONS.md` | Audits of an earlier manuscript revision. Their banners list the findings that are settled; their bodies still carry superseded numbers (66.27M, top-2 entropy, `log 2`). Take no number or fact from them. |
| `docs/research/BLUEPRINT_CODE_AUDIT.md` | Audit of the original blueprint against the code; same caveat. |
| `spatial-moe-adverse-weather-sod-blueprint.md` (repo root) | The original research blueprint. Aspirational, and wrong in the places `RELATED_WORK.md` lists. |

## Working rules

`AGENTS.md` (repository root) states how to work in this repository, including the rule that
no claim enters the documentation without support from the implementation, a result file or
a verified source.

## Removed

Four files that claimed to be the source of truth, three competing paper plans, the
literature search log and verification queue, the hypothesis and claim-evidence matrices, a
results inventory superseded by the generated `RESULTS.md`, and the early design transcripts
under `docs/elicit/`. Their content lives in the canonical documents above; everything is
recoverable from git history.

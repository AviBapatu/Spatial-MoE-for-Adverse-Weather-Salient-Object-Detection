# Documentation index

Two documents are canonical. Read them before writing anything.

| File | What it is |
|---|---|
| `RESEARCH_TRUTH.md` (repo root) | **Canonical.** What may be claimed, and what is forbidden. Every claim is marked as implementation-, experiment- or literature-supported, with the gaps stated. |
| `docs/research/RESULTS.md` | **Canonical.** Every measured number, each traceable to a result file. Generated from the JSON under `results/` — regenerate rather than edit. |

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
| `docs/research/LITERATURE_DATABASE.md` | The citation database |
| `docs/research/LITERATURE_CLAIMS.md` | Claims attributed to specific papers |
| `docs/research/LITERATURE_NOTES.md` | Reading notes |
| `docs/research/FINAL_RESEARCH_POSITION.md` | Closest competing methods and how this project differs |
| `docs/research/NOVELTY_MATRIX.md` | Novelty assessed axis by axis |

These five overlap and are the obvious next merge: one related-work document would replace
them.

## Paper

| File | Covers |
|---|---|
| `paper/main.tex` | The manuscript |
| `paper/FINAL_PAPER_PLAN.md` | Structure, figure and table plan |
| `paper/REDESIGN_PLAN.md` | Figure, table and reference plan for the 4-page format |
| `paper/CLAIM_AUDIT.md`, `paper/MANUSCRIPT_AUDIT.md`, `docs/research/PAPER_CORRECTIONS.md` | Audits of a specific manuscript revision; findings are issues to check, not current fact |
| `docs/research/BLUEPRINT_CODE_AUDIT.md` | Audit of the original blueprint against the code |
| `spatial-moe-adverse-weather-sod-blueprint.md` | The original research blueprint (repository root) |

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

# Documentation Index

The repository accumulated several overlapping documents. This file states which one to read
for what, and which files are superseded. It is the map; the sources of truth it points at
are marked **canonical**.

## Read these

| File | What it is |
|---|---|
| `RESEARCH_TRUTH.md` (repo root) | **Canonical.** The authoritative list of claims and whether each is supported by implementation, experiment or literature. Read before drafting any paper section. |
| `docs/research/RESULTS.md` | **Canonical.** Every measured number, each traceable to a result file. Generated from the JSON under `results/`. |
| `docs/research/ARCHITECTURE.md` | **Canonical.** The model, verified against `src/`, with the findings a paper must not contradict. |
| `docs/research/TRAINING.md` | **Canonical.** Entrypoints, optimisation, checkpointing, distributed invariants, evaluation protocol. |
| `docs/research/EXPERIMENTS.md` | **Canonical.** Config system, experiment ID rules, the run matrix, and the protocol for launching runs. |
| `docs/research/ABLATIONS.md` | **Canonical.** What has been ablated, what has not, and which comparisons are currently meaningless. |
| `docs/research/DATASETS.md` | WXSOD structure, splits, augmentation, geometry. |
| `docs/research/PAPER_PLAN.md` | Structure, figures and open questions for the manuscript. |
| `docs/research/LITERATURE_DATABASE.md` | The citation database. |
| `AGENTS.md` | Working rules for this repository. |
| `README.md` | Orientation and how to run things. |

## Superseded — safe to delete once you agree

These duplicate the canonical files above and were kept only because deleting documentation
is irreversible in spirit even when it is recoverable in git. Nothing in them is unique
unless noted.

| File | Superseded by / why |
|---|---|
| `docs/research/PAPER_SOURCE_OF_TRUTH.md` | `RESEARCH_TRUTH.md` (root). One of three files claiming to be the source of truth. |
| `paper/PAPER_SOURCE_OF_TRUTH.md` | `RESEARCH_TRUTH.md` (root). Third claimant. |
| `docs/research/FINAL_CLAIM_EVIDENCE_MATRIX.md` | Same content type as `RESEARCH_TRUTH.md`; keep whichever is more current. |
| `docs/research/FINAL_EXPERIMENTAL_RESULTS.md` | `RESULTS.md`, which is generated from the result files instead of transcribed. |
| `docs/research/RESULTS_NARRATIVE.md` | `RESULTS.md` for the numbers; the narrative belongs in the paper. |
| `docs/research/RESEARCH_HYPOTHESES.md` | Folded into `RESEARCH_TRUTH.md` and `PAPER_PLAN.md`. |
| `docs/research/FINAL_RESEARCH_POSITION.md` | `docs/research/PROJECT_OVERVIEW.md`. |
| `docs/research/NOVELTY_MATRIX.md` | `PAPER_PLAN.md`, unless the novelty analysis is still being maintained. |
| `docs/research/BLUEPRINT_CODE_AUDIT.md` | Its findings are re-verified in `ARCHITECTURE.md`; the blueprint itself is historical. |
| `docs/research/PAPER_CORRECTIONS.md` | A changelog of past corrections. Now that the canonical docs are current, it records history rather than guidance. |
| `docs/research/LITERATURE_SEARCH_LOG.md` | Process artifact — a log of searches, not reference material. |
| `docs/research/LITERATURE_VERIFICATION_QUEUE.md` | Process artifact — a to-do list from a past session. |
| `docs/research/LITERATURE_CLAIMS.md` | `LITERATURE_DATABASE.md`, unless claims are tracked separately. |
| `docs/research/LITERATURE_NOTES.md` | `LITERATURE_DATABASE.md`. |
| `paper/PAPER_PLAN.md`, `paper/FINAL_PAPER_PLAN.md` | `docs/research/PAPER_PLAN.md` — three plans for one paper. |
| `paper/CLAIM_AUDIT.md`, `paper/MANUSCRIPT_AUDIT.md` | Audits of drafts that are no longer the live plan. |
| `paper/REDESIGN_PLAN.md` | Superseded design notes. |
| `docs/elicit/*.md` | Transcripts of early design prompts. Historical, not reference. |
| `spatial-moe-adverse-weather-sod-blueprint.md` | The original blueprint. Historical; the code is authoritative where they disagree (see `ARCHITECTURE.md`). |

## Why deletions are listed rather than done

Each file above is deleted by one command, but not before someone has confirmed the item it
might uniquely hold is preserved somewhere canonical. The risky ones are named: the three
source-of-truth files disagree in places, and reconciling them is the job of
`RESEARCH_TRUTH.md` — not of a delete.

## Consolidation rules going forward

- One source of truth for claims: `RESEARCH_TRUTH.md`.
- One source of numbers: `RESULTS.md`, and it is generated, never hand-edited.
- One plan: `docs/research/PAPER_PLAN.md`.
- If two files disagree, the order of authority is: source code, then configs, then result
  files, then these documents. The documents are the last word only when they are quoting the
  layers above them.

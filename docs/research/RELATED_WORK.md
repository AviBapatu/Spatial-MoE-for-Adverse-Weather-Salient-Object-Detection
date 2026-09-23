# Related work and positioning

Synthesis of the project's literature analysis: what the closest prior work does, where this
project differs, which claims are defensible, and which experiments each claim still needs.
`LITERATURE_DATABASE.md` holds the underlying records under `DB-nn` keys; identifiers below
were taken from that database and the notes, and where a source is unverified that is stated.

## The position

Adverse-weather SOD is addressed today in one of three ways: condition on explicit weather
labels, add a dedicated weather branch, or apply mixture-of-experts to weather *restoration*
rather than to detection. No published method we have found combines **spatial token-level
MoE routing** with **SOD** under adverse weather **without a weather-specific component or
supervision**.

## Closest prior work

| Work | Task | Routing granularity | Routing signal | Weather labels | Per-scale routers | Entropy into decoder |
|---|---|---|---|---|---|---|
| WM-MoE (Luo et al., 2023) | restoration | token-level | content + weather features (contrastive) | no labels at test time, but a weather branch exists | no (one router) | no |
| MoFME (AAAI 2024) | restoration | token-level | uncertainty (MC dropout) | no | no | no |
| Complexity Experts (CVPR 2025) | restoration | image-level | computational complexity | no | no | no |
| NIFM (2025) | adverse-weather SOD | none (one-hot encoding) | explicit weather type | **yes** | no | no |
| WFANet (2025) | adverse-weather SOD | image-level branch | weather prediction | **yes** | no | no |
| M3KE (2026) | nighttime dehazing | multi-granularity, hierarchical | frequency-aware | no | no (hierarchical) | no |
| V-MoE (NeurIPS 2021) | classification | token-level top-k | learned router | n/a | no | no |
| Soft MoE (ICLR 2024) | classification | token-level, differentiable | learnable | n/a | no | no |
| M3ViT (NeurIPS 2022) | multi-task | token-level, task-conditioned | task embedding | n/a | no | no |
| SegMoTE (CVPR 2026) | medical segmentation | token-level | modality | n/a | no | no |
| Rule-based spatial MoE (2026) | edge detection | spatial | context vs boundary | n/a | no | no |
| Lee et al. (2025) | segmentation | token-level | learned | n/a | no | entropy used for OOD detection, not as input |

**This work** | adverse-weather SOD | token-level, per scale | spatial content only | **none** | **yes, three independent** | **yes**

## Novelty assessment

| Claim | Closest prior art | Classification | Confidence |
|---|---|---|---|
| A. Token-level MoE for SOD | MMSOD, CMFNet, CMoE, M4-SAM, PSOD all apply MoE to SOD, but at modality/task/scale level | likely distinct | high |
| B. Label-free routing | WM-MoE routes without weather labels at test time via a learned weather branch | **potentially distinct — the threatened claim** | medium |
| C. Independent per-scale routers | WM-MoE has one router; M3KE is hierarchical, not independent | likely distinct | high |
| D. Routing entropy as a decoder input | Lee et al. compute gate entropy for OOD detection; UGRAN uses prediction uncertainty, not routing entropy | likely distinct | high |
| E. MoE for adverse-weather SOD | NIFM and WFANet target the same task without MoE | likely distinct | high |

The defensible claim is the **combination** A + C + D + E. Individual elements have partial
precedents, and D is the strongest single element: no work we found feeds routing entropy as
an explicit spatial feature channel into a dense-prediction decoder. (A negative claim of
this kind cannot be proven exhaustively; the search is recorded in the database.)

## Claims that must be qualified or dropped

| Claim | Problem | Use instead |
|---|---|---|
| "First MoE for SOD" | several papers apply MoE to SOD | "first token-level spatial MoE routing for single-RGB SOD" |
| "First label-free weather routing" | WM-MoE also routes without weather labels at test time | "no weather-specific architectural component or supervision at any stage" |
| "Novel multi-scale MoE" | WM-MoE has multi-scale experts | "independent per-scale routers with separate expert pools" |
| "Novel routing-entropy contribution" | gate entropy is computed elsewhere, for other purposes | "no prior work feeds routing entropy into the decoder as a feature channel" |
| "WM-MoE routes on feature maps" | wrong — it routes per token | correct the blueprint |
| "WM-MoE needs weather labels at test time" | wrong — its weather branch runs automatically | correct the blueprint |
| "WM-MoE and MoWE are different papers" | same paper, arXiv:2303.13739, renamed | cite once |

## Blueprint corrections

The repository's blueprint predates the literature review and is wrong in eight places. It is
still cited by other documents, so the corrections are recorded here.

| # | Blueprint says | Correct |
|---|---|---|
| 1 | WM-MoE and MoWE are separate papers | the same paper |
| 2 | WM-MoE routes at feature-map level | token-level |
| 3 | WM-MoE requires weather labels at test time | it does not |
| 4 | M3ViT is multi-scale | multi-task |
| 5 | Routers in Vision MoE analyses failure modes | it is a comparative benchmark of six router designs |
| 6 | Zhu et al. (CVPR 2023) is a channel-level two-branch split | a two-stage parameter-expansion approach |
| 7 | GridFormer is 2023 | IJCV 2024 |
| 8 | M3ViT title omits the hardware co-design | full title includes it |

## Identifiers as recorded

WM-MoE arXiv:2303.13739 · MoFME doi:10.1609/aaai.v38i15.29622 · Soft MoE arXiv:2308.00951 ·
V-MoE arXiv:2106.05974 · Complexity Experts arXiv:2411.18466 · SMOE arXiv:2211.13491 ·
Controllable-LPMoE arXiv:2410.16076 · WXSOD arXiv:2508.12250 · NIFM arXiv:2512.10592 ·
MMSOD doi:10.1109/ijcnn64981.2025.11228833 · CMFNet doi:10.1109/tmm.2026.3668533 ·
CMoE doi:10.1609/aaai.v40i12.37959 · Multi-Scale MoE KAN doi:10.1016/j.neucom.2025.130349 ·
LDR arXiv:2312.01381 · DA2Diff arXiv:2504.05135.

Unverified and needing a check before citation: XWOD (described as extreme-weather
*detection*), and the venue details of any 2026 preprint.

## Experiments each claim still needs

All of these are unrun. Where an existing hook makes one cheap, that is noted.

| Claim | Experiment | Feasible today? |
|---|---|---|
| A | compare against modality/scale-level MoE SOD baselines | needs baselines |
| B | add a weather branch or weather supervision and show no gain | needs code |
| C | share one router across scales and compare | needs code |
| D | remove the entropy channel and compare — **`disable_entropy` already exists** | yes, one evaluation run |
| E | compare against NIFM and WFANet on WXSOD | needs baselines |
| MoE itself | the mixture-vs-dense control — **configs already exist** | yes, one training run |
| Specialisation | per-expert assignment statistics — the diagnostics already compute them | yes, already measured (no specialisation found) |

The last two rows are the point: the cheapest experiments that move the paper forward are
the mixture-vs-dense control and the entropy ablation, both of which are runnable now.

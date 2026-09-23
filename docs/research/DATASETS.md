# Datasets

Verified against `src/dataset.py` and `src/boundary.py`.

## WXSOD

| Split | Images | Role |
|---|---|---|
| `train_sys` | 12,891 | synthetic adverse weather, training |
| `test_sys` | 1,500 | synthetic adverse weather, held out |
| `test_real` | 554 | real-world adverse weather, held out |

Weather labels are carried in the filename as `{scene_id}_{weather}.ext` and parsed by
`get_weather_type`. `test_real` contains single conditions (dark, fog, light, rain, snow);
`test_sys` additionally contains compound conditions (rainafog, rainasnow, snowafog) and
clean images, which is why the synthetic set is compositionally harder.

## Train / validation split

The validation set is carved from `train_sys` with a scene-level `GroupShuffleSplit`, so no
scene appears in both halves, with an explicit disjointness assertion. The split is derived,
not shipped as a separate directory. Verified composition at 20%:

| Split | Weather counts |
|---|---|
| train | clean 508, dark 1230, fog 1228, light 1217, rain 1210, rainafog 1265, rainasnow 1197, snow 1232, snowafog 1255 |
| validation | clean 123, dark 305, fog 306, light 314, rain 314, rainafog 297, rainasnow 297, snow 315, snowafog 278 |

The counts come from the loader's own distribution report, printed at the start of every run.

## Geometry

`_aspect_preserving_resize_pad` resizes so the longest side reaches 384, then
**centre-pads** to 384x384 — reflect padding (`BORDER_REFLECT_101`) for the image, constant
zero for the mask and edge map. The original size and every pad amount are recorded in the
sample's `meta`, which is what lets evaluation undo the transform exactly
(`reverse_geometry`) before scoring.

Two consequences worth remembering:

- Predictions are scored in original image coordinates, never in the padded model frame.
- Because padding is centred and reflective, flipping a padded image at test time is
  geometrically equivalent to mirroring the original — so flip-average TTA is well posed
  here.

## Augmentation

| Split | Pipeline |
|---|---|
| train | `HorizontalFlip(0.5)`, `ColorJitter(0.5)`, normalize |
| val / test | normalize only |

Augmentation is applied *after* the geometry step and is deterministically seeded per sample
and epoch (an MD5 of base seed, epoch, index and name), so a run is reproducible while still
varying across epochs. Because the model is trained with horizontal flips, the flipped
forward pass in TTA is in-distribution rather than a probe.

## Targets

Each sample yields the image, the saliency mask, an edge/boundary map, and a validity mask
for the padded region. Boundary targets come from `src/boundary.py`: morphological
dilate-minus-erode with a 5x5 ellipse kernel.

## Where the data lives

The dataset is not in the repository. On Kaggle the notebook downloads and extracts it,
then resolves the real root by looking for the expected `train_sys/input`, `train_sys/gt`,
`test_sys/...`, `test_real/...` structure rather than trusting a hard-coded path. Configs
under `experiments/` point at that resolved location.

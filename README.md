# sargolini-toolkit

Load the grid-cell recordings of

> Sargolini F, Fyhn M, Hafting T, McNaughton BL, Witter MP, Moser M-B,
> Moser EI (2006). *Conjunctive representation of position, direction, and
> velocity in entorhinal cortex.* Science 312(5774): 758.
> doi:10.1126/science.1125572

and reproduce their classification into **grid**, **head-direction** and
**conjunctive** cells, with a circular-shift null for every score. A
running-speed score is computed and classified alongside them.

Rate maps and gridness use [`gridcan`](https://github.com/nihad-hashimov/gridcan): its NaN-tolerant
autocorrelogram and rotational gridness score (`gridcan.analysis`).

## Data

The recordings are **not** in this repository. They are the archive dataset
"Grid cell data Sargolini et al 2006", https://doi.org/10.11582/2014.00003.
The archive's metadata lists its licence as CC BY 4.0. Check the licence and
citation requirements on the landing page before using or redistributing
anything derived from it.

```bash
python scripts/download_data.py --toc table_of_contents.csv --dest data/sargolini
```

`table_of_contents.csv` is the archive's own file listing, saved from the
landing page. The script verifies each file's MD5 and can be re-run to resume.

## Pipeline

1. **Inventory** (`data_io.inventory`). Every `.mat` file is keyed by its
   lower-cased name, so `_T1C1.mat` and `_t1c1.mat` are never counted twice.
   Files are grouped into sessions. Two file families are recognised: cell
   files paired with a session `_POS.mat`, and self-contained cell files that
   carry their own position.
2. **Session loading** (`data_io.load_session`):
   - fills short tracking gaps;
   - takes position as the midpoint of the two LEDs;
   - computes speed and movement direction from a lightly smoothed copy;
   - drops impossible speeds;
   - takes head direction from the LED pair, deciding which LED is the front
     one from forward-motion agreement.
3. **Enclosure fit** (`data_io.fit_arena`). The fraction of the bounding box
   the animal visited decides square vs circle; a circle's centre and radius
   are then fitted to the outermost samples.
4. **Per-cell scores** (`cells`):
   - gridness of the rate-map autocorrelogram;
   - Watson's two-sample U² of head direction at spikes against head direction
     over the session;
   - speed score (Pearson r of binned rate against binned speed).
5. **Null and thresholds** (`run_classify`). Each spike train is shifted
   circularly in time 200 times and every score is recomputed. The 95th
   percentile of the shuffled scores **pooled over all cells** is the
   threshold. The speed score is two-tailed (2.5th and 97.5th percentiles).
   Per-cell percentiles are recorded as well.

## Parameters

Every preprocessing and classification choice, as set in the code:

| stage | parameter | value | where |
|---|---|---|---|
| loading | tracking gaps filled by linear interpolation | ≤ 1 s | `load_session(max_gap_s=1.0)` |
| loading | animal position | midpoint of LED 1 and LED 2 (LED 1 if only one) | `load_session` |
| loading | speed: Gaussian smoothing of x, y before `np.gradient` | σ = 0.1 s | `load_session(speed_sigma_s=0.1)` |
| loading | implausible speeds dropped (set to NaN, not clipped) | > 100 cm/s | `load_session(speed_max=100.0)` |
| loading | movement direction defined only above | 2 cm/s | `load_session(speed_min=2.0)` |
| loading | front-LED decision: samples used | speed > 5 cm/s, ≥ 100 samples | `_resolve_led_order(speed_min=5.0)` |
| enclosure | bounding box | 0.2th–99.8th percentile of x and y | `fit_arena` |
| enclosure | square if visited fraction of bounding box (40 × 40 grid, holes filled) | ≥ 0.88 | `fit_arena(nb=40, circle_max_frac=0.88)` |
| enclosure | circle centre fitted to | outer 10% of samples by radius (Kasa fit) | `fit_arena` |
| enclosure | circle radius | 99.5th percentile of distance from centre | `fit_arena` |
| all scores | "running" samples | speed ≥ 2.5 cm/s | `cells.SPEED_MIN_CMS` |
| rate map | bin | 5 cm | `run_classify.BIN_CM` |
| rate map | Gaussian smoothing (spikes and dwell smoothed separately, then divided) | σ = 5 cm | `run_classify.SIGMA_CM` |
| rate map | minimum dwell per bin | 0.15 s | `MapMaker(min_dwell_s=0.15)` |
| rate map | bins kept | inside fitted enclosure grown by one bin | `MapMaker` |
| rate map | spike position | nearest position sample in time | `cells.spike_indices` |
| grid | score | min(r60, r120) − max(r30, r90, r150) on the autocorrelogram annulus | `gridcan.analysis.gridness_score` |
| head direction | tuning-curve bins | 60 (6°), bins with ≥ 0.5 s dwell | `HDTuner(bins=60, min_dwell_s=0.5)` |
| head direction | Watson U² behavioural sample | every 10th running frame, ≥ 200 frames | `cells.THIN = 10` |
| speed | rate and speed binned in time | 0.4 s bins | `SpeedTuner(bin_s=0.4)` |
| speed | bins used | mean speed ≤ 60 cm/s, occupancy > half a bin, ≥ 50 bins | `SpeedTuner(speed_max=60.0)` |
| null | circular shift offset | uniform in [20 s, T − 20 s] | `cells.MIN_SHIFT_S = 20.0` |
| null | shuffles per cell | 200 | `run_classify --shuffles 200` |
| null | RNG | `numpy.random.default_rng(1)` | `run_classify --seed 1` |
| classification | cells scored | ≥ 100 spikes | `run_classify.MIN_SPIKES` |
| classification | threshold | 95th percentile of the pooled null (speed: 2.5th / 97.5th) | `run_classify.PCT` |
| classification | head direction requires | two LEDs in the session | `run_classify` |

## Results

One run, seed 1, 200 shuffles per cell (`results/logs/run_classify.log`,
`results/figures/classification.png`):

| | value |
|---|---|
| sessions / rats / cells | 119 / 15 / 626 |
| enclosures (as fitted) | square: 96 sessions, 485 cells; circle: 23 sessions, 141 cells |
| sessions with two LEDs (head direction available) | 72 sessions, 459 cells |
| cells with ≥ 100 spikes | 626 / 626 |
| gridness threshold (pooled 95th pct, 125,168 shuffled values) | > 0.391 |
| Watson U² threshold (pooled 95th pct, 91,800 shuffled values) | > 1.3986 |
| speed-score thresholds (pooled 2.5th / 97.5th pct) | < −0.089 or > +0.047 |
| **grid cells** | **183** (29.2% of 626) |
| **head-direction cells** | **326** (71.0% of the 459 with two LEDs) |
| **conjunctive (grid and head direction)** | **84** |
| speed-modulated | 212 (33.9%) |
| grid spacing (autocorrelogram ring radius), grid cells | median 60 cm, range 35–120 cm |

The tabulated 5% critical value of Watson U² (0.187) is exceeded by 444 of
the 459 cells that have head direction (96.7%). It assumes independent samples,
which 50 Hz tracking and bursting spikes are not. That is why the threshold
comes from the shuffled null instead.

Full run time: 365 s on the author's workstation.

### Ours vs Sargolini et al. 2006

To be filled from the paper.

| quantity | Sargolini et al. 2006 | this toolkit | notes |
|---|---|---|---|
| cells analysed | | 626 | |
| grid cells | | 183 | |
| head-direction cells | | 326 | |
| conjunctive cells | | 84 | |
| gridness threshold | | 0.391 | |
| head-direction threshold / test | | Watson U² > 1.3986 | |
| shuffle procedure | | circular shift ≥ 20 s, 200 per cell, pooled 95th pct | |

## How to run

Python 3.14 (results produced with CPython 3.14.6). Installing the package
also installs its dependency [`gridcan`](https://github.com/nihad-hashimov/gridcan)
from GitHub:

```bash
pip install -e ".[test]"
```
```bash
python -m sargolini_toolkit.run_classify --data-dir data/sargolini --shuffles 200 --seed 1
```

Add `--limit 3 --shuffles 10` for a quick check. The per-cell table
(`cell_classification.csv`) and the pooled nulls (`shuffle_nulls.npz`) are
written to `results/tables/`. That folder is not committed until the dataset
licence has been checked.

```bash
pytest
```

## Validation and limitations

- **No automated tests yet.** `tests/` holds stubs that state their claim and
  tolerance and are currently skipped:
  - the loader on synthetic fixtures of both file families;
  - LED-order resolution on synthetic traces;
  - shuffle reproducibility and offset bounds.
- **The comparison with the paper is not done yet.** The table above is empty.
- **One seed, one run.** The pooled thresholds depend on the shuffle draw;
  how much has not been measured.
- **The speed score is a whole-session Pearson correlation** of binned rate
  against binned speed. Whether that matches the paper's own speed analysis
  has not been checked.
- **Enclosure shape is inferred from the path.** A session in which the animal
  covered its enclosure poorly could be misclassified. The visited fraction is
  available on the fitted `ArenaGeometry` (`area_frac`) for inspection.
- **Pinned to one environment** (numpy 2.5.2, scipy 1.18.0, matplotlib 3.11.1,
  Python 3.14); other versions are untested.

## License

MIT -- see `LICENSE`. Copyright (c) 2026 Nihad Hashimov. The licence covers the
code only; the data are covered by
the archive's own terms.

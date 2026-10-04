"""Reproduce the cell classifications of Sargolini et al. 2006.

Grid cells, head-direction cells, conjunctive (grid and head-direction) cells,
and cells whose firing rate is significantly correlated with running speed.

THE CRITERION

Each cell gets three scores and each is tested against a null built by
circularly shifting that cell's spike train in time, which destroys the pairing
with behaviour while preserving both the spike train's burst structure and the
animal's.

    grid          gridness score of the rate-map autocorrelogram
    head direction  **Watson's U^2**, following Sargolini et al.: the spike
                  head-direction distribution against the behavioural one
    speed         correlation between binned firing rate and running speed

Head direction is tested with Watson's U^2 rather than a mean vector length.
U^2 is an omnibus test -- it responds to any difference between the two
circular distributions, not only to a single preferred direction -- and it needs
neither binning nor division by occupancy, since the behavioural sample IS the
occupancy correction.  Its tabulated critical value is unusable here; see
`cells.HDTuner` for the measurement that shows why, and why the shuffled
threshold is the one used.

Sargolini et al. pooled the shuffled scores from every cell into one
distribution and took its 95th percentile as a single dataset-wide threshold.
That is the primary criterion here, because it is what reproduces their
classification.  The per-cell percentile is computed too and written to the
table: it is the stricter test for any individual cell, and the disagreement
between the two is worth being able to look at.

    python -m sargolini_toolkit.run_classify --data-dir <folder of .mat files> --shuffles 200
"""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt

from sargolini_toolkit import cells as C
from sargolini_toolkit import common
from sargolini_toolkit import data_io as dio

BIN_CM = 5.0
SIGMA_CM = 5.0
MIN_SPIKES = 100
PCT = 95.0


def measure_cell(mm, hdt, spt, spike_t, bin_cm):
    rate = mm.rate(spike_t)
    grid, ring_cm, _ = C.grid_measures(rate, bin_cm)
    _, u2, pref, peak = hdt.measures(spike_t)
    sscore, sp_p, n_bins = spt.measures(spike_t)
    return grid, ring_cm, u2, pref, peak, sscore, sp_p, n_bins


def main():
    p = common.base_parser(__doc__.splitlines()[0])
    p.add_argument("--shuffles", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="stop after this many sessions (for a quick check)")
    args = common.parse_args(p)

    inv = dio.inventory(args.data_dir)
    keys = [k for k in sorted(inv) if inv[k]["cells"]]
    if args.limit:
        keys = keys[:args.limit]
    rng = np.random.default_rng(args.seed)

    rows, null_grid, null_u2, null_speed = [], [], [], []
    t_start = time.time()

    for n_done, key in enumerate(keys, 1):
        rec = dio.load_session(inv[key], key)
        mm = C.MapMaker(rec, bin_cm=BIN_CM, sigma_cm=SIGMA_CM)
        hdt = C.HDTuner(rec)
        spt = C.SpeedTuner(rec)

        for cid, st in sorted(rec.spikes.items()):
            n_sp = int(np.sum((st >= rec.t[0]) & (st <= rec.t[-1])))
            g, ring, u2, pref, peak, ss, sp_p, nb = measure_cell(
                mm, hdt, spt, st, BIN_CM)

            sg = sm = sv = np.zeros(0)
            if n_sp >= MIN_SPIKES:
                offs = C.shuffle_offsets(rec, args.shuffles, rng)
                sg = np.empty(offs.size)
                sm = np.empty(offs.size)
                sv = np.empty(offs.size)
                for i, o in enumerate(offs):
                    sh = C.shift_spikes(st, o, rec.t[0], rec.duration_s)
                    sg[i] = C.grid_measures(mm.rate(sh), BIN_CM)[0]
                    sm[i] = hdt.u2(sh)
                    sv[i] = spt.measures(sh)[0]
                null_grid.append(sg)
                null_u2.append(sm)
                null_speed.append(sv)

            def pct_of(obs, null):
                n = null[np.isfinite(null)]
                if not np.isfinite(obs) or n.size < 10:
                    return np.nan
                return float(100.0 * np.mean(n < obs))

            rows.append(dict(
                session=key, rat=key.split("-")[0], cell=cid,
                cell_uid=f"{key}_{cid}",
                arena_shape=rec.arena.shape, side_cm=rec.side_cm,
                duration_s=rec.duration_s, has_hd=float(rec.has_hd),
                led_margin=rec.led_margin,
                n_spikes=n_sp, mean_rate_hz=n_sp / rec.duration_s,
                peak_rate_hz=peak,
                gridness=g, ring_cm=ring,
                hd_u2=u2, hd_pref_deg=pref,
                speed_score=ss, speed_p=sp_p,
                speed_bins=nb,
                pct_grid=pct_of(g, sg), pct_u2=pct_of(u2, sm),
                pct_speed=pct_of(ss, sv),
                n_shuffles=float(sg.size),
            ))

        el = time.time() - t_start
        print(f"  [{n_done:3d}/{len(keys)}] {key:26s} "
              f"{len(rec.spikes):3d} cells  {rec.arena.shape:6s} "
              f"{rec.side_cm:5.0f} cm   {el:6.0f}s elapsed", flush=True)

    # ---- pooled nulls ----------------------------------------------------
    def pooled(chunks):
        v = np.concatenate(chunks) if chunks else np.zeros(0)
        return v[np.isfinite(v)]

    pg, pm, pv = pooled(null_grid), pooled(null_u2), pooled(null_speed)

    def thr(v, q):
        # A null can be empty -- e.g. the head-direction null when no session in
        # the selection has a second LED.  Return NaN so every comparison
        # against it is False, rather than crashing or silently passing.
        return float(np.percentile(v, q)) if v.size else np.nan

    thr_grid = thr(pg, PCT)
    thr_u2 = thr(pm, PCT)
    thr_speed_hi = thr(pv, 97.5)
    thr_speed_lo = thr(pv, 2.5)

    n = len(rows)
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    enough = arr["n_spikes"].astype(float) >= MIN_SPIKES
    is_grid = enough & (arr["gridness"].astype(float) > thr_grid)
    is_hd = enough & (arr["has_hd"].astype(float) > 0) & \
        (arr["hd_u2"].astype(float) > thr_u2)
    ss = arr["speed_score"].astype(float)
    is_speed = enough & ((ss > thr_speed_hi) | (ss < thr_speed_lo))

    arr["is_grid"] = is_grid.astype(float)
    arr["is_hd"] = is_hd.astype(float)
    arr["is_speed"] = is_speed.astype(float)
    arr["is_conjunctive"] = (is_grid & is_hd).astype(float)
    common.save_table(arr, "cell_classification.csv")
    np.savez_compressed(common.table_path("shuffle_nulls.npz"),
                        grid=pg, u2=pm, speed=pv)

    print("\n=== THRESHOLDS (pooled shuffled null, %gth percentile) ===" % PCT)
    print(f"  gridness           > {thr_grid:+.3f}   "
          f"({pg.size} shuffled values)")
    print(f"  Watson U^2         > {thr_u2:.4f}  ({pm.size} shuffled values)")
    n_tab = int(np.sum(arr["hd_u2"].astype(float) > 0.187))
    n_jud = int(np.sum(np.isfinite(arr["hd_u2"].astype(float))))
    print(f"      the TABULATED 5% value is 0.187, which {n_tab}/{n_jud} cells "
          f"({n_tab / max(n_jud, 1):.1%}) exceed.")
    print("      It assumes independent samples, and neither 50 Hz tracking "
          "nor bursting")
    print("      spikes are, so the shuffled threshold is the usable one.")
    print(f"  speed score        > {thr_speed_hi:+.3f} or "
          f"< {thr_speed_lo:+.3f}   ({pv.size}, two-tailed at 95%)")

    print("\n=== CLASSIFICATION ===")
    print(f"  cells with >= {MIN_SPIKES} spikes : {int(enough.sum())}/{n}")
    print(f"  grid cells                : {int(is_grid.sum()):4d}"
          f"  ({is_grid.sum() / max(enough.sum(), 1):.1%} of usable)")
    hd_pool = enough & (arr["has_hd"].astype(float) > 0)
    print(f"  head-direction cells      : {int(is_hd.sum()):4d}"
          f"  ({is_hd.sum() / max(hd_pool.sum(), 1):.1%} of the "
          f"{int(hd_pool.sum())} with a second LED)")
    print(f"  conjunctive (grid AND HD) : {int((is_grid & is_hd).sum()):4d}")
    print(f"  speed-modulated           : {int(is_speed.sum()):4d}"
          f"  ({is_speed.sum() / max(enough.sum(), 1):.1%})")

    print("\n=== DATASET ===")
    sess_arr = np.asarray(arr["session"])
    print(f"  sessions with cells : {len(set(sess_arr))}")
    print(f"  rats                : {len(set(np.asarray(arr['rat'])))}")
    print(f"  cells               : {n}")
    shape_arr = np.asarray(arr["arena_shape"])
    for sh in sorted(set(shape_arr)):
        sel = shape_arr == sh
        print(f"  {sh:6s} enclosures   : {len(set(sess_arr[sel]))} sessions, "
              f"{int(sel.sum())} cells")
    two_led = arr["has_hd"].astype(float) > 0
    print(f"  two LEDs (head direction available): "
          f"{len(set(sess_arr[two_led]))} sessions, {int(two_led.sum())} cells")

    gr = arr["ring_cm"].astype(float)[is_grid]
    gr = gr[np.isfinite(gr)]
    if gr.size:
        print("\n=== GRID SPACING (first-ring radius of the autocorrelogram) ===")
        print(f"  n = {gr.size}   median {np.median(gr):.1f} cm   "
              f"range {gr.min():.0f}-{gr.max():.0f} cm")

    # ---- figure ----------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, obs, null, thr, lab, log in [
            (axs[0], arr["gridness"].astype(float)[enough], pg, thr_grid,
             "gridness", False),
            # Watson U^2 spans 0.05 to 94 while its null sits below 1.4, so a
            # linear axis squashes the null into a single invisible bar.  The
            # separation between them is the whole point of the panel.
            (axs[1], arr["hd_u2"].astype(float)[hd_pool], pm, thr_u2,
             "Watson $U^2$", True),
            (axs[2], ss[enough], pv, thr_speed_hi, "speed score", False)]:
        o = obs[np.isfinite(obs)]
        if o.size == 0 or null.size == 0:
            ax.set_axis_off()
            ax.set_title(f"{lab}: no data", fontsize=10)
            continue
        if log:
            lo = max(min(null[null > 0].min(), o[o > 0].min()), 1e-4)
            bins = np.geomspace(lo, max(null.max(), o.max()) * 1.05, 45)
            ax.set_xscale("log")
        else:
            bins = np.linspace(min(np.percentile(null, 0.1), o.min()),
                               max(np.percentile(null, 99.9), o.max()), 45)
        ax.hist(null, bins=bins, density=True, color=common.GREY, alpha=0.55,
                label="shuffled")
        ax.hist(o, bins=bins, density=True, histtype="step", lw=2,
                color=common.BLUE, label="observed")
        ax.axvline(thr, color=common.RED, ls="--", lw=1.5, label="95th pct")
        if lab == "speed score":
            ax.axvline(thr_speed_lo, color=common.RED, ls="--", lw=1.5)
        ax.set_xlabel(lab)
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)

    ax = axs[3]
    ax.hist(gr, bins=np.arange(0, 210, 10), color=common.GREEN, alpha=0.85)
    ax.set_xlabel("grid spacing, AC first-ring radius (cm)")
    ax.set_ylabel("grid cells")
    ax.set_title("grid cells: autocorrelogram ring radius", fontsize=10)

    fig.suptitle("Sargolini et al. 2006 -- cell classification against a "
                 "circular-shift null", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    common.save(fig, "classification.png")
    print(f"\ntotal {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()

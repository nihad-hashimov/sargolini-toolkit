"""Cell-level measurements: rate maps, gridness, head direction, speed tuning.

This module computes the four numbers each cell is classified on, and the
shuffled null distribution each is tested against.  It does not decide anything
-- `run_classify.py` applies the thresholds and writes the table.

THE SHUFFLE, AND WHY IT IS A CIRCULAR SHIFT

Every score here (gridness, Watson U^2, speed score) can be large by
accident when a cell simply fires in bursts while the rat happens to be doing
something structured.  The null has to destroy the *pairing* between spikes and
behaviour while preserving the temporal structure of both, so the spike train is
shifted circularly in time by a random offset and everything is recomputed.  A
plain reshuffle of spike times would destroy burst structure too and give an
optimistically narrow null.

The offset is drawn from [MIN_SHIFT_S, T - MIN_SHIFT_S] so that no shuffle is
close to the identity.  Sargolini et al. used a 20 s floor, which is kept.

POOLED VERSUS PER-CELL NULLS

Sargolini et al. pooled the shuffled scores from *all* cells and took the 95th
percentile of that single distribution as one threshold for the whole dataset.
That is what `run_classify.py` reports as the primary criterion, because it is
what reproduces their classification.  The per-cell percentile is computed and
stored alongside it, because it is the stricter and more defensible test for a
single cell, and because the difference between the two is worth being able to
see rather than argue about.

SPEED SCORE

`speed_score` is the Pearson correlation between a cell's firing rate and the
animal's running speed, both averaged into short time bins over the whole
session (`SpeedTuner`).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

from gridcan import analysis as an   # NaN-tolerant autocorrelogram, gridness

__all__ = ["spike_indices", "rate_map", "hd_tuning", "speed_tuning",
           "grid_measures", "shuffle_offsets", "shift_spikes", "watson_u2",
           "MapMaker", "HDTuner", "SpeedTuner"]

MIN_SHIFT_S = 20.0             # Sargolini et al.'s floor on the circular shift
SPEED_MIN_CMS = 2.5            # samples slower than this are not "running"
HD_BINS = 60                   # 6-degree bins, for the tuning curve
THIN = 10                      # keep 1 behavioural frame in 10 for Watson U^2
SPEED_BIN_CMS = 2.5


def spike_indices(spike_t, t):
    """Index of the position sample nearest in time to each spike.

    `searchsorted` then a one-sided correction, rather than interpolation: a
    spike belongs to the sample the animal actually was at, and interpolating
    position at spike times would invent sub-sample precision the 50 Hz tracker
    does not have.
    """
    spike_t = np.asarray(spike_t, float)
    spike_t = spike_t[(spike_t >= t[0]) & (spike_t <= t[-1])]
    i = np.searchsorted(t, spike_t)
    i = np.clip(i, 1, t.size - 1)
    left = (spike_t - t[i - 1]) < (t[i] - spike_t)
    return np.where(left, i - 1, i)


def _smooth_nan(m, mask, sigma_bins):
    """Gaussian smoothing that ignores unvisited bins instead of zeroing them.

    Smoothing a map with NaN holes by zero-filling drags every bin near a hole
    toward zero.  Smoothing the numerator and the mask separately and dividing
    is the standard fix and keeps the result an average of real data only.
    """
    if sigma_bins <= 0:
        return np.where(mask, m, np.nan)
    num = ndimage.gaussian_filter(np.where(mask, m, 0.0), sigma_bins,
                                  mode="constant")
    den = ndimage.gaussian_filter(mask.astype(float), sigma_bins,
                                  mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 1e-6, out, np.nan)


class MapMaker:
    """Rate maps for one session, with everything behaviour-only cached.

    Occupancy, the arena mask and the smoothed dwell depend on the trajectory
    alone, so they are identical for every cell in the session AND for every
    circular shuffle of every cell.  Computing them once turns each shuffled
    map into one 2-D histogram plus one smoothing, which is what makes a
    100-shuffle null over 600 cells affordable.

    Samples below `speed_min` are dropped from both spikes and occupancy: a rat
    grooming in a corner otherwise contributes a large dwell at one place and
    depresses the rate there, which reads as a spatial feature.
    """

    def __init__(self, rec, *, bin_cm=5.0, sigma_cm=5.0, min_dwell_s=0.15,
                 speed_min=SPEED_MIN_CMS):
        x0, x1, y0, y1 = rec.arena.bbox
        self.rec, self.bin_cm, self.extent = rec, bin_cm, (x0, x1, y0, y1)
        self.nx = max(int(round((x1 - x0) / bin_cm)), 1)
        self.ny = max(int(round((y1 - y0) / bin_cm)), 1)
        self.range = ((x0, x1), (y0, y1))

        self.ok = (np.isfinite(rec.x) & np.isfinite(rec.y)
                   & np.isfinite(rec.speed) & (rec.speed >= speed_min))
        dwell, xe, ye = np.histogram2d(rec.x[self.ok], rec.y[self.ok],
                                       bins=(self.nx, self.ny), range=self.range)
        self.dwell = dwell * rec.dt

        cx, cy = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
        CX, CY = np.meshgrid(cx, cy, indexing="ij")
        # Bins inside the fitted enclosure (grown by one bin) with enough dwell.
        inside = rec.arena.contains(CX, CY, margin=bin_cm)
        self.mask = inside & (self.dwell >= min_dwell_s)
        self.sigma_bins = sigma_cm / bin_cm
        # Smooth spike counts and dwell separately, then divide.  Smoothing the
        # ratio instead would weight a bin holding 0.2 s of dwell as heavily as
        # one holding 5 s.
        self.dwell_s = _smooth_nan(self.dwell, self.mask, self.sigma_bins)

    def spike_counts(self, spike_t):
        si = spike_indices(spike_t, self.rec.t)
        si = si[self.ok[si]]
        n, _, _ = np.histogram2d(self.rec.x[si], self.rec.y[si],
                                 bins=(self.nx, self.ny), range=self.range)
        return n

    def rate(self, spike_t):
        """Occupancy-normalised, smoothed firing-rate map in Hz (NaN outside)."""
        n_s = _smooth_nan(self.spike_counts(spike_t), self.mask, self.sigma_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = n_s / self.dwell_s
        return np.where(np.isfinite(r) & self.mask, r, np.nan)


def rate_map(rec, spike_t, **kw):
    """One-off rate map.  Use `MapMaker` when more than one map is wanted."""
    mm = MapMaker(rec, **kw)
    return mm.rate(spike_t), np.where(mm.mask, mm.dwell, np.nan), mm.extent


def grid_measures(rate, bin_cm):
    """Gridness score and first-ring radius, in cm, from a rate map."""
    ac = an.autocorrelogram(rate)
    score, ring_bins = an.gridness_score(ac)
    return float(score), float(ring_bins * bin_cm), ac


def watson_u2(x, y):
    """Two-sample Watson U^2 for circular data, x and y in degrees.

    The circular analogue of the Cramer-von Mises statistic: the integrated
    squared difference between the two empirical CDFs, with the mean difference
    removed so the answer does not depend on where the circle is cut.
    """
    n1, n2 = x.size, y.size
    if n1 < 5 or n2 < 5:
        return np.nan
    N = n1 + n2
    lab = np.concatenate([np.ones(n1), np.zeros(n2)])
    lab = lab[np.argsort(np.concatenate([x, y]), kind="mergesort")]
    d = np.cumsum(lab) / n1 - np.cumsum(1.0 - lab) / n2
    return float((n1 * n2 / N ** 2) * (np.sum(d * d) - np.sum(d) ** 2 / N))


class HDTuner:
    """Head-direction analysis for one session, with the behaviour cached.

    THE TEST IS WATSON'S U^2, following Sargolini et al.

    It compares the distribution of head direction **at spike times** against
    the distribution of head direction **over the whole session**.  If the cell
    is not directional its spikes are just a random sample of wherever the head
    happened to point, so the two distributions coincide.  It is an omnibus
    test -- it responds to any difference between them, not only to a single
    preferred direction -- and it needs neither binning nor division by
    occupancy, because the behavioural sample IS the occupancy correction.

    THE TABULATED CRITICAL VALUE IS UNUSABLE HERE, AND THIS IS NOT A DETAIL.

    U^2 > 0.187 at alpha = 0.05 assumes independent samples.  Neither sample is
    independent: head direction at 50 Hz is enormously autocorrelated
    (consecutive frames are 20 ms apart and nearly identical) and spikes come in
    bursts, so far more cells exceed 0.187 than are directional.
    `run_classify.py` prints how many cells exceed the tabulated value next to
    the shuffle-derived threshold, so the size of the problem is measured on
    every run rather than asserted here.

    So two things are done: the behavioural sample is thinned to one frame in
    `THIN`, removing most of the frame-to-frame redundancy, and significance
    comes from the same circular-shift null used for every other score here.
    That null inherits whatever autocorrelation the data have, so it stays
    valid regardless.

    The tuning curve is still computed, because a preferred direction is worth
    reporting and because the curve is what one looks at to check a cell by eye.
    Its *length* -- the Rayleigh mean vector length -- is no longer used as a
    criterion anywhere.
    """

    def __init__(self, rec, *, bins=HD_BINS, speed_min=SPEED_MIN_CMS,
                 min_dwell_s=0.5, thin=THIN):
        self.rec, self.bins, self.min_dwell_s = rec, bins, min_dwell_s
        self.edges = np.linspace(0, 360, bins + 1)
        self.centres = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.ok = (np.isfinite(rec.hd) & np.isfinite(rec.speed)
                   & (rec.speed >= speed_min))
        occ, _ = np.histogram(rec.hd[self.ok], bins=self.edges)
        self.occ = occ * rec.dt
        self.usable = self.occ >= min_dwell_s
        self.behaviour = rec.hd[self.ok][::thin] if rec.has_hd else np.zeros(0)

    def curve(self, spike_t):
        """Occupancy-corrected tuning curve, Hz per direction bin."""
        if not self.rec.has_hd:
            return np.full(self.bins, np.nan)
        si = spike_indices(spike_t, self.rec.t)
        si = si[self.ok[si]]
        n_sp, _ = np.histogram(self.rec.hd[si], bins=self.edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.usable, n_sp / self.occ, np.nan)

    def u2(self, spike_t):
        """Watson U^2 of this cell's spike directions against the behaviour."""
        if not self.rec.has_hd or self.behaviour.size < 200:
            return np.nan
        si = spike_indices(spike_t, self.rec.t)
        return watson_u2(self.rec.hd[si[self.ok[si]]], self.behaviour)

    def measures(self, spike_t):
        """(curve, Watson U^2, preferred direction deg, peak Hz).

        The preferred direction is the angle of the tuning curve's resultant.
        That is the *direction* of the mean vector, not its length -- it is a
        descriptive readout, and nothing is classified on it.
        """
        rate = self.curve(spike_t)
        u2 = self.u2(spike_t)
        good = np.isfinite(rate)
        if good.sum() < self.bins // 2 or not np.nansum(rate) > 0:
            return rate, u2, np.nan, np.nan
        z = np.sum(rate[good] * np.exp(1j * np.radians(self.centres[good])))
        return (rate, u2, float(np.degrees(np.angle(z)) % 360.0),
                float(np.nanmax(rate)))


class SpeedTuner:
    """Rate-vs-speed for one session, with the binned speed cached.

    Rate and speed are both averaged into `bin_s` bins before correlating.  At
    the raw 20 ms sample a cell's "rate" is 0 or 50 Hz and the correlation would
    be dominated by whether a spike happened to land in a sample rather than by
    any tuning.

    `speed_max` bounds the correlation: above roughly 60 cm/s a rat spends
    almost no time, so those bins hold a handful of samples with enormous
    leverage.
    """

    def __init__(self, rec, *, bin_s=0.4, speed_min=SPEED_MIN_CMS,
                 speed_max=60.0):
        self.rec, self.bin_s = rec, bin_s
        t = rec.t
        nb = max(int(np.floor((t[-1] - t[0]) / bin_s)), 2)
        self.edges = t[0] + np.arange(nb + 1) * bin_s

        good = np.isfinite(rec.speed) & (rec.speed >= speed_min)
        cnt, _ = np.histogram(t[good], bins=self.edges)
        ssum, _ = np.histogram(t[good], bins=self.edges, weights=rec.speed[good])
        with np.errstate(invalid="ignore", divide="ignore"):
            self.sp = np.where(cnt > 0, ssum / cnt, np.nan)
        self.occ = cnt * rec.dt
        self.valid = (np.isfinite(self.sp) & (self.sp <= speed_max)
                      & (self.occ > 0.5 * bin_s))

    def measures(self, spike_t):
        """(speed score r, p, n bins used)."""
        n_sp, _ = np.histogram(np.asarray(spike_t, float), bins=self.edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(self.occ > 0, n_sp / self.occ, np.nan)
        m = self.valid & np.isfinite(rate)
        if m.sum() < 50 or np.std(rate[m]) == 0 or np.std(self.sp[m]) == 0:
            return np.nan, np.nan, 0
        r, p = stats.pearsonr(self.sp[m], rate[m])
        return float(r), float(p), int(m.sum())


def hd_tuning(rec, spike_t, **kw):
    """One-off head-direction tuning.  Use `HDTuner` for many cells."""
    tun = HDTuner(rec, **kw)
    rate, u2, pref, peak = tun.measures(spike_t)
    return tun.centres, rate, u2, pref, peak


def speed_tuning(rec, spike_t, **kw):
    """One-off speed tuning.  Use `SpeedTuner` for many cells."""
    return SpeedTuner(rec, **kw).measures(spike_t)


def shuffle_offsets(rec, n, rng):
    """`n` circular time shifts, each at least MIN_SHIFT_S from the identity."""
    T = rec.duration_s
    if T <= 2 * MIN_SHIFT_S + 1:
        return np.zeros(0)
    return rng.uniform(MIN_SHIFT_S, T - MIN_SHIFT_S, size=n)


def shift_spikes(spike_t, offset, t0, T):
    """Circularly shift spike times within the session window."""
    return t0 + np.mod(np.asarray(spike_t, float) - t0 + offset, T)



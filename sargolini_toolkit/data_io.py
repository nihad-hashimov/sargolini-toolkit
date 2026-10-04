"""Reading the Sargolini et al. 2006 recordings, and normalising them.

Named `data_io` rather than `io` so it can never shadow the standard library.

WHAT IS IN THE ARCHIVE

The release is not one file format.  The `.mat` files fall into four
variable sets:

    ('cellTS',)                             spike times only
    ('EEG', 'Fs')                           LFP, unused here
    ('post','posx','posx2','posy','posy2')  session position
    ('t','ts','x1','x2','y1','y2')          self-contained cell + position

So there are two families.  In the **session family** a cell file carries only
its spike train and must be paired with the session's `_POS.mat`; in the
**self-contained family** (the one the dataset readme describes) each cell file
carries its own copy of the position trace with both LEDs.

`posx2`/`posy2` are frequently *empty arrays* rather than absent: some
session-position files have only one LED, and for those head direction cannot
be computed at all.  That has to be screened for, not assumed.

WINDOWS CASE-INSENSITIVITY

Cell files appear as both `_T2C1.mat` and `_t2c1.mat` across the archive.  On a
case-insensitive filesystem a glob for each pattern returns *the same files
twice*, which silently doubles every count.  The inventory therefore keys on
the lower-cased basename and never globs case variants separately.

HEAD DIRECTION AND THE SIGN AMBIGUITY

Head direction is the bearing of the LED pair, but the archive does not say
which LED is the front one, and the answer is not guaranteed constant across
rats.  Guessing costs a 180-degree error, which would invert every directional
result.  It is resolved from the data instead: over a whole session a rat moves
predominantly forwards, so the circular mean of (head direction - movement
direction) sits near 0 for the correct assignment and near 180 for the flipped
one.  `_resolve_led_order` measures that and records both the choice and the
margin, so a session where the evidence is weak can be seen rather than
trusted.

THE ENCLOSURE IS NOT ALWAYS A SQUARE

The archive mixes square and circular enclosures, and nothing in the filenames
says which is which.  `fit_arena` decides from the tracked path: a square fills
its bounding box, an inscribed circle covers only pi/4 = 0.785 of it, so the
visited-area fraction of the bounding box separates the two.  Getting this
wrong matters for rate maps: a circle treated as its bounding box has four
unvisited corners that are not part of the arena at all.

POSITION, SPEED, AND WHY THEY ARE SMOOTHED SEPARATELY

The archive's position is already moving-mean smoothed for tracking jitter, but
differentiating it at 50 Hz still amplifies what is left.  Speed is therefore
taken from a lightly Gaussian-smoothed copy of the trace, while the position
used for rate maps stays as supplied.  The two uses have different noise
requirements and sharing one smoothing level would compromise both.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio
from scipy import ndimage


__all__ = ["inventory", "SessionRecord", "load_session", "circmean_deg",
           "ArenaGeometry", "fit_arena"]

# Cell files are named <rat>-<session>_t<tetrode>c<cluster>.mat, in either case.
_CELL_RE = re.compile(r"^(?P<session>.+)_t(?P<tetrode>\d+)c(?P<cluster>\d+)$",
                      re.IGNORECASE)


def _col(m, key):
    """One .mat variable as a 1-D float array; empty variables become empty."""
    a = np.asarray(m[key])
    if a.size == 0:
        return np.zeros(0)
    return np.asarray(a, float).ravel()


# --------------------------------------------------------------------------
# 1.  Inventory
# --------------------------------------------------------------------------
def inventory(raw_dir) -> dict:
    """Map every file in the archive to its session, without counting it twice.

    Returns ``{session_key: {"pos": path|None, "cells": {cell_id: path},
    "self_contained": bool}}``.  Sessions are keyed by the filename stem before
    the final underscore, so ``11207-21060501+02`` groups its own cells.
    """
    by_lower = {}
    for p in glob.glob(os.path.join(str(raw_dir), "*.mat")):
        by_lower[os.path.basename(p).lower()] = p     # dedup case variants

    sessions: dict[str, dict] = {}

    def slot(key):
        return sessions.setdefault(
            key, {"pos": None, "cells": {}, "self_contained": False})

    for low, path in sorted(by_lower.items()):
        stem = low[:-4]                                # drop ".mat"
        if stem.endswith(("_eeg", "_egf")):
            continue
        if stem.endswith("_pos"):
            slot(stem[:-4])["pos"] = path
            continue
        m = _CELL_RE.match(stem)
        if m is None:
            continue                                   # readme, stray files
        s = slot(m.group("session"))
        s["cells"][f"t{int(m.group('tetrode'))}c{int(m.group('cluster'))}"] = path

    # A cell file from the self-contained family carries its own position, so
    # its session needs no _POS file.  Detect it by reading one cell's keys.
    for s in sessions.values():
        if s["pos"] is None and s["cells"]:
            probe = next(iter(s["cells"].values()))
            keys = {k for k in sio.loadmat(probe) if not k.startswith("__")}
            s["self_contained"] = keys.issuperset({"x1", "y1", "t", "ts"})

    return sessions


# --------------------------------------------------------------------------
# 2.  Circular helpers
# --------------------------------------------------------------------------
def circmean_deg(a_deg, weights=None):
    """Circular mean of angles in degrees, returned in [0, 360)."""
    a = np.radians(np.asarray(a_deg, float))
    ok = np.isfinite(a)
    w = ok.astype(float) if weights is None else np.where(
        ok, np.asarray(weights, float), 0.0)
    z = np.sum(w * np.exp(1j * np.where(ok, a, 0.0)))
    return float(np.degrees(np.angle(z)) % 360.0)


# --------------------------------------------------------------------------
# 3.  Arena geometry
# --------------------------------------------------------------------------
@dataclass
class ArenaGeometry:
    """The enclosure, fitted from the path.

    `shape` is "circle" or "square".  `area_frac` is the discriminator that
    chose it (visited area over bounding-box area) and is kept so a marginal
    session can be spotted instead of silently classified.
    """
    shape: str
    bbox: tuple                   # (x0, x1, y0, y1)
    centre: tuple                 # (cx, cy)
    radius: float                 # circle only; NaN for a square
    area_frac: float

    @property
    def side_cm(self):
        """Diameter for a circle, mean side for a square."""
        if self.shape == "circle":
            return 2.0 * self.radius
        x0, x1, y0, y1 = self.bbox
        return 0.5 * ((x1 - x0) + (y1 - y0))

    @property
    def area_cm2(self):
        if self.shape == "circle":
            return float(np.pi * self.radius ** 2)
        x0, x1, y0, y1 = self.bbox
        return float((x1 - x0) * (y1 - y0))

    def contains(self, x, y, margin=0.0):
        """True where (x, y) lies inside the enclosure, grown by `margin` cm."""
        if self.shape == "circle":
            cx, cy = self.centre
            return np.hypot(x - cx, y - cy) <= self.radius + margin
        x0, x1, y0, y1 = self.bbox
        return ((x >= x0 - margin) & (x <= x1 + margin)
                & (y >= y0 - margin) & (y <= y1 + margin))

    def describe(self):
        if self.shape == "circle":
            return (f"circle d={self.side_cm:.0f} cm "
                    f"(centre {self.centre[0]:.1f}, {self.centre[1]:.1f})")
        x0, x1, y0, y1 = self.bbox
        return f"square {x1 - x0:.0f} x {y1 - y0:.0f} cm"


def _fit_circle(x, y):
    """Kasa algebraic circle fit: minimises sum (x^2+y^2 + Dx + Ey + F)^2."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = D / 2.0, E / 2.0
    return float(cx), float(cy), float(np.sqrt(F + cx ** 2 + cy ** 2))


def fit_arena(x, y, *, nb=40, circle_max_frac=0.88) -> ArenaGeometry:
    """Fit the enclosure to the tracked path.

    The rat presses its whole boundary over ten minutes, so the path's support
    *is* the arena.  Shape is decided by how much of the bounding box that
    support fills; the circle's centre and radius are then fitted to the
    outermost samples only, because interior samples carry no information about
    where the boundary is.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    bbox = (float(np.nanpercentile(x, 0.2)), float(np.nanpercentile(x, 99.8)),
            float(np.nanpercentile(y, 0.2)), float(np.nanpercentile(y, 99.8)))

    H, _, _ = np.histogram2d(x, y, bins=nb, range=((bbox[0], bbox[1]),
                                                   (bbox[2], bbox[3])))
    area_frac = float(ndimage.binary_fill_holes(H > 0).mean())

    if area_frac >= circle_max_frac:
        cx, cy = 0.5 * (bbox[0] + bbox[1]), 0.5 * (bbox[2] + bbox[3])
        return ArenaGeometry("square", bbox, (cx, cy), np.nan, area_frac)

    # Refine on the outer shell: fit once from the bbox centre, then use the
    # points near that radius, which are the ones actually at the boundary.
    cx0, cy0 = 0.5 * (bbox[0] + bbox[1]), 0.5 * (bbox[2] + bbox[3])
    rad = np.hypot(x - cx0, y - cy0)
    shell = rad >= np.percentile(rad, 90)
    cx, cy, R = _fit_circle(x[shell], y[shell])
    # The boundary is where the animal stops, so take a high percentile of the
    # refined radius rather than the algebraic fit's own R, which is pulled in
    # by the shell's inner edge.
    R = float(np.percentile(np.hypot(x - cx, y - cy), 99.5))
    return ArenaGeometry("circle", bbox, (cx, cy), R, area_frac)


# --------------------------------------------------------------------------
# 4.  A loaded session
# --------------------------------------------------------------------------
@dataclass
class SessionRecord:
    key: str
    t: np.ndarray                 # (T,) seconds
    x: np.ndarray                 # (T,) cm, animal position (LED midpoint)
    y: np.ndarray
    speed: np.ndarray             # (T,) cm/s
    move_dir: np.ndarray          # (T,) deg in [0,360), NaN when near-stationary
    hd: np.ndarray                # (T,) deg in [0,360), NaN if no second LED
    spikes: dict                  # {cell_id: (n,) spike times in seconds}
    arena: ArenaGeometry          # fitted enclosure: shape and bounds
    dt: float
    has_hd: bool
    led_flipped: bool
    led_margin: float             # |mean cos(hd - move_dir)|; low = unreliable
    nan_frac: float
    fast_frac: float              # samples rejected as impossible running speed
    notes: list = field(default_factory=list)

    @property
    def duration_s(self):
        return float(self.t[-1] - self.t[0])

    @property
    def side_cm(self):
        return self.arena.side_cm


def _interp_short_gaps(v, max_gap):
    """Linearly fill NaN runs of at most `max_gap` samples; leave longer ones.

    Tracking dropouts of a few samples are noise; long ones mean the animal was
    genuinely untracked, and inventing a path across them would place spikes at
    fictitious positions.
    """
    v = np.array(v, float)
    bad = ~np.isfinite(v)
    if not bad.any() or bad.all():
        return v
    lab, _ = ndimage.label(bad)
    idx = np.arange(v.size)
    good = ~bad
    fill = np.interp(idx, idx[good], v[good])
    for s in ndimage.find_objects(lab):
        seg = s[0]
        if (seg.stop - seg.start) <= max_gap:
            v[seg] = fill[seg]
    return v


def _resolve_led_order(hd_raw, move_dir, speed, speed_min=5.0):
    """Decide whether (x1,y1) is the front LED, from forward-motion agreement.

    Returns ``(flipped, margin)``.  `margin` is the absolute mean cosine of the
    angle between head and movement direction: near 1 means head and path agree
    strongly, near 0 means the evidence is weak and the session's directional
    data should be treated with suspicion.
    """
    ok = np.isfinite(hd_raw) & np.isfinite(move_dir) & (speed > speed_min)
    if ok.sum() < 100:
        return False, np.nan
    c = float(np.mean(np.cos(np.radians(hd_raw[ok] - move_dir[ok]))))
    return (c < 0), abs(c)


def load_session(entry, key="", *, speed_sigma_s=0.1, speed_min=2.0,
                 speed_max=100.0, max_gap_s=1.0) -> SessionRecord:
    """Load one session into a normalised record.

    `entry` is one value from `inventory()`.  Both file families are handled;
    the self-contained family's position is read from its first cell file.
    """
    notes = []
    spikes = {}

    if entry["self_contained"]:
        paths = list(entry["cells"].values())
        m0 = sio.loadmat(paths[0])
        t = _col(m0, "t")
        x1, y1, x2, y2 = (_col(m0, k) for k in ("x1", "y1", "x2", "y2"))
        for cid, p in entry["cells"].items():
            m = sio.loadmat(p)
            spikes[cid] = _col(m, "ts")
            if _col(m, "t").size != t.size:
                notes.append(f"{cid}: position length differs; used first cell's")
    else:
        if entry["pos"] is None:
            raise FileNotFoundError(f"session {key!r} has no position file")
        m = sio.loadmat(entry["pos"])
        t = _col(m, "post")
        x1, y1 = _col(m, "posx"), _col(m, "posy")
        x2, y2 = _col(m, "posx2"), _col(m, "posy2")
        for cid, p in entry["cells"].items():
            # 17 cell files use the self-contained layout (`ts` plus their own
            # position) even though their session also ships a _POS file, so
            # the spike-time key cannot be assumed from the session's family.
            mc = sio.loadmat(p)
            spikes[cid] = _col(mc, "cellTS" if "cellTS" in mc else "ts")

    has_hd = x2.size == x1.size and x2.size > 0
    if not has_hd:
        notes.append("no second LED: head direction unavailable")

    dt = float(np.median(np.diff(t)))
    max_gap = max(1, int(round(max_gap_s / dt)))
    x1, y1 = _interp_short_gaps(x1, max_gap), _interp_short_gaps(y1, max_gap)
    if has_hd:
        x2, y2 = _interp_short_gaps(x2, max_gap), _interp_short_gaps(y2, max_gap)
        # The animal sits between its LEDs; taking one LED as "the position"
        # would offset every rate map by half a head length, in a
        # direction-dependent way.
        x, y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    else:
        x, y = x1, y1

    nan_frac = float(np.mean(~np.isfinite(x) | ~np.isfinite(y)))

    # Speed and movement direction from a lightly smoothed copy (see module
    # docstring).  NaNs are bridged for the filter then re-masked, so a gap
    # cannot leak a fictitious velocity into its neighbours.
    def smooth(v):
        f = _interp_short_gaps(v, v.size)
        s = ndimage.gaussian_filter1d(f, max(speed_sigma_s / dt, 0.5),
                                      mode="nearest")
        return np.where(np.isfinite(v), s, np.nan)

    xs, ys = smooth(x), smooth(y)
    vx, vy = np.gradient(xs, t), np.gradient(ys, t)
    speed = np.hypot(vx, vy)

    # A rat does not run at 4 m/s.  Raw traces reach that because a tracking
    # dropout teleports the LED, and a single such sample would otherwise
    # dominate the speed score.  Implausible samples are dropped, not clipped:
    # clipping would pile fictitious mass at the ceiling of the speed range.
    too_fast = np.isfinite(speed) & (speed > speed_max)
    fast_frac = float(np.mean(too_fast))
    speed = np.where(too_fast, np.nan, speed)

    move_dir = np.degrees(np.arctan2(vy, vx)) % 360.0
    move_dir[~np.isfinite(speed) | (speed < speed_min)] = np.nan

    led_flipped, led_margin = False, np.nan
    if has_hd:
        hd = np.degrees(np.arctan2(y1 - y2, x1 - x2)) % 360.0
        led_flipped, led_margin = _resolve_led_order(hd, move_dir, speed)
        if led_flipped:
            hd = (hd + 180.0) % 360.0
    else:
        hd = np.full_like(x, np.nan)

    return SessionRecord(
        key=key, t=t, x=x, y=y, speed=speed, move_dir=move_dir, hd=hd,
        spikes=spikes, arena=fit_arena(x, y), dt=dt, has_hd=has_hd,
        led_flipped=led_flipped, led_margin=led_margin, nan_frac=nan_frac,
        fast_frac=fast_frac, notes=notes)

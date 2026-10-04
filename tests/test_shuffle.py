"""Circular-shift null: reproducibility and constraints."""

import pytest


def test_seeded_shuffle_null_is_reproducible():
    """CLAIM: running the per-cell shuffle (`shuffle_offsets` +
    `shift_spikes` + gridness / Watson U^2 / speed score) twice with
    `np.random.default_rng(1)` on the same synthetic session gives
    bit-identical null distributions; a different seed gives different ones.

    TOLERANCE: `np.array_equal` for the same seed; not equal for seed 2.
    """
    pytest.skip("TODO")


def test_offsets_respect_the_minimum_shift():
    """CLAIM: every offset from `shuffle_offsets` lies in
    [MIN_SHIFT_S, T - MIN_SHIFT_S] (20 s floor), and a session shorter than
    2 * MIN_SHIFT_S + 1 s yields no offsets.

    TOLERANCE: exact bounds over 10,000 draws.
    """
    pytest.skip("TODO")


def test_shift_preserves_spike_count_and_window():
    """CLAIM: `shift_spikes` returns the same number of spikes, all inside
    [t0, t0 + T), with inter-spike intervals preserved except across the one
    wrap point.

    TOLERANCE: counts exact; intervals equal to 1e-9 s.
    """
    pytest.skip("TODO")

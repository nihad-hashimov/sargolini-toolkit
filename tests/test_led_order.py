"""Front/back LED resolution on synthetic traces."""

import pytest


def test_flipped_leds_are_detected_and_corrected():
    """CLAIM: for a synthetic 600 s, 50 Hz trajectory moving forwards (speed
    10-40 cm/s, smooth heading), with LED 1 placed 3 cm BEHIND the body and
    LED 2 3 cm in FRONT, `_resolve_led_order` returns flipped = True, and
    `load_session` returns head direction equal to the true heading.

    TOLERANCE: flipped exact; margin >= 0.9; circular error of the returned
    head direction <= 2 deg (median over samples moving faster than 5 cm/s).
    """
    pytest.skip("TODO")


def test_correct_led_order_is_left_alone():
    """CLAIM: the same trajectory with LED 1 in front returns flipped = False
    and the same head direction.

    TOLERANCE: as above.
    """
    pytest.skip("TODO")


def test_weak_evidence_is_reported_as_a_low_margin():
    """CLAIM: when head direction is independent of movement direction
    (heading drawn uniformly at random each sample), the margin is small.

    TOLERANCE: margin <= 0.1.
    """
    pytest.skip("TODO")

"""``nuzline.text``: the display rounding (a win chance down, a loss chance and expected
deaths up)."""

from nuzline.text import decimals, tenths


def test_a_win_chance_rounds_down_and_a_loss_chance_up():
    assert tenths(0.9996) == "99.9" and tenths(1.0) == "100.0" and tenths(0.0) == "0.0"
    assert tenths(0.999) == "99.9"  # 0.999 * 1000 = 998.9999999999999 is taken as 999
    assert tenths(0.0004, up=True) == "0.1" and tenths(0.05, up=True) == "5.0"


def test_expected_deaths_round_up():
    assert decimals(0.0011, 2, up=True) == "0.01"
    assert decimals(0.0123, 3, up=True) == "0.013"
    assert decimals(0.25, 2, up=True) == "0.25" and decimals(0.0, 2, up=True) == "0.00"
    assert decimals(1.239, 2) == "1.23"

"""D3-8b on the reference: the per-enemy-battler participation mask (``_grant_exp`` reads and
clears entry ``(battler >> 1) & 1``; ``_send_out_exp`` flags and clears per entry) equals the
engine's ``_pay_all`` / ``_flag_exp`` (``tests/engine/test_d3_exp_mask.py`` has the rules)."""

from __future__ import annotations

import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.format import DOUBLES, MULTI, TWO_TRAINERS, format_of, party
from nuzline.engine.levelup import pay_exp
from nuzline.engine.step import _flag_exp, _pay_all
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import _send_out_exp, ref_grant_exp
from tests.engine.d2_states import four
from tests.engine.test_d3_exp_mask import HOLDER, both_foes_faint, foe_faints, holding


def ref_pay_all(tables, before, after):
    """The reference's payout of battler 1 then battler 3, each on the previous result."""
    np_tables = to_numpy(tables)
    r = to_ref(after)
    for b in (1, 3):
        r = ref_grant_exp(np_tables, r, int(after.active[b]), battler=b)
    return from_ref(r)


def assert_payout_equal(want, got):
    for name in ("exp", "evs", "level", "hp", "exp_mask"):
        np.testing.assert_array_equal(
            np.asarray(getattr(want, name)), np.asarray(getattr(got, name)), err_msg=name
        )


def test_ref_payouts_equal_the_engine_per_format(tables, names):
    for fmt in (DOUBLES, TWO_TRAINERS, MULTI):
        s = four(tables, names, fmt)
        if fmt != DOUBLES:  # battler 3's own party: a level that tells the payouts apart
            s = s._replace(level=s.level.at[party(fmt, 3), 0].set(80))
        before, after = both_foes_faint(s)
        assert_payout_equal(_pay_all(tables, before, after), ref_pay_all(tables, before, after))


def test_ref_send_outs_equal_the_engine(tables, names):
    """Each case: battler ``b`` has just taken party slot ``slot`` (``active`` set), then the
    flag step runs on both engines from the same ``exp_mask``."""
    cases = (
        (DOUBLES, 0, 2, None),
        (DOUBLES, 3, 2, None),
        (DOUBLES, 2, 2, "foe3_down"),
        (DOUBLES, 1, 2, "player2_absent"),
        (MULTI, 2, 1, "zeroed"),
        (TWO_TRAINERS, 0, 2, None),
        (TWO_TRAINERS, 3, 1, None),
    )
    for fmt, b, slot, tweak in cases:
        s = four(tables, names, fmt, bench=3 if tweak == "player2_absent" else 6)
        if tweak == "foe3_down":
            s = s._replace(hp=s.hp.at[party(fmt, 3), int(s.active[3])].set(0))
        if tweak == "player2_absent":
            s = s._replace(hp=s.hp.at[0, 1].set(0).at[0, 2].set(0))
        if tweak in ("zeroed", "foe3_down"):
            s = s._replace(exp_mask=s.exp_mask * 0)
        s = s._replace(active=s.active.at[b].set(slot))
        want = _flag_exp(s, b)
        r = to_ref(s)
        _send_out_exp(r, b)
        got = from_ref(r)
        assert format_of(got) == fmt
        assert np.asarray(want.exp_mask).tolist() == np.asarray(got.exp_mask).tolist(), (
            fmt,
            b,
            tweak,
        )


def test_ref_knocked_off_items_are_read_from_the_party(tables, names):
    """D9: the reference's payout reads ``knocked_item`` as the party item, like the engine."""
    for slot, item in ((0, "Lucky Egg"), (HOLDER, "Exp. Share"), (1, "Macho Brace")):
        s = holding(tables, names, slot, item, True)
        after = foe_faints(s)
        want = pay_exp(tables, s, after, battler=1)
        got = from_ref(ref_grant_exp(to_numpy(tables), to_ref(after), int(s.active[1]), battler=1))
        assert_payout_equal(want, got)
        assert int(got.exp[slot]) > int(s.exp[slot])

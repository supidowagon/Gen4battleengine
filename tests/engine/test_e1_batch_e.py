"""E1 Batch E (rulings G-62 onward, docs/kaizo/e1-mechanics-ledger.md): hand cases.

G-62: a bag item's Guard Spec param sets Mist on the user's side, its crit-stages param
Focus Energy (``BattleSystem_UseBagItem``, battle_system.c:563-573, :617-626).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import FIRST_ITEM_ACTION
from nuzline.engine.bag import bag_item_category, refused_bag_params, use_bag_item
from nuzline.engine.ids import ItemAiCategory, ItemParam
from nuzline.engine.outcome import empty_outcome, side_crit_odds
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import _use_item
from tests.engine.helpers import assert_matches_reference, roark_state

SWITCH_TO_1 = 4 + 1  # order.MOVE_ACTIONS + party slot


def _bag(s, names, item):
    return s._replace(
        trainer_items=s.trainer_items.at[1].set(jnp.asarray([names.item(item), 0, 0, 0])),
        trainer_item_count=s.trainer_item_count.at[1].set(1),
    )


def _use_both(tables, s, item):
    """Side 1 uses bag item ``item`` in both engines (``use_bag_item``, reference
    ``_use_item``); asserts every field equal and returns the JAX state."""
    out = use_bag_item(tables, s, 1, item, True)
    ref = to_ref(s)
    _use_item(to_numpy(tables), ref, 1, item)
    for name, x, y in zip(out._fields, out, from_ref(ref), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    return out


def _same(a, b):
    for name, x, y in zip(a._fields, a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def test_guard_spec_and_dire_hit_are_not_refused(tables, kaizo_tables, names, kaizo_names):
    # G-62 lifts G-43's Mist and Focus Energy refusals. Vanilla's Guard Spec. and Dire Hit are
    # modelled; a Poke Ball (every param its dummy 3) and Kaizo's Guard Spec. (every param its
    # dummy 128) are still refused, only for the friendship wrap at 255.
    params = np.asarray(tables.item_party_param)
    for item in ("Guard Spec.", "Dire Hit"):
        assert refused_bag_params(params[names.item(item)], names.item(item)) == []
    params = np.asarray(kaizo_tables.item_party_param)
    for item in ("Poké Ball", "Guard Spec."):
        assert int(params[kaizo_names.item(item)][ItemParam.GUARD_SPEC]) != 0
        reasons = refused_bag_params(params[kaizo_names.item(item)], kaizo_names.item(item))
        assert reasons and all("friendship" in r for r in reasons)


def test_ai_guard_spec_sets_mist_on_its_side(tables, names):
    s = roark_state(tables, names)
    item = names.item("Guard Spec.")
    assert int(np.asarray(tables.item_party_param)[item][ItemParam.GUARD_SPEC]) != 0
    # TrainerAI_ShouldUseItem's Guard Spec branch (trainer_ai.c:4181-4182): no Mist on side 1.
    assert int(bag_item_category(tables, s, 1, item)) == ItemAiCategory.GUARD_SPEC
    out = _use_both(tables, s, item)
    assert int(out.mist[1]) == 5 and int(out.mist[0]) == 0  # battle_system.c:569
    # The AI's own side now has Mist: the branch no longer uses it.
    assert int(bag_item_category(tables, out, 1, item)) == ItemAiCategory.NONE
    # A second use with Mist up changes nothing (mistTurns != 0, :565-566; result FALSE).
    misty = s._replace(mist=s.mist.at[1].set(3))
    _same(_use_both(tables, misty, item), misty)
    # The whole turn: the item action, then the end-of-turn countdown (5 -> 4).
    acts = [SWITCH_TO_1, FIRST_ITEM_ACTION]
    out = assert_matches_reference(tables, _bag(s, names, "Guard Spec."), acts, empty_outcome())
    assert int(out.mist[1]) == 4


def test_dire_hit_sets_focus_energy_and_its_crit_stages(tables, names):
    s = roark_state(tables, names)
    item = names.item("Dire Hit")
    assert int(np.asarray(tables.item_party_param)[item][ItemParam.CRIT_STAGES]) != 0
    move = s.moves[1, s.active[1], 0]
    assert int(side_crit_odds(tables, s, 1, move)) == 16  # stage 0 (battle_lib.c:7097-7103)
    out = _use_both(tables, s, item)
    assert int(out.focus_energy[1]) == 1 and int(out.focus_energy[0]) == 0
    # Focus Energy's +2 (battle_lib.c:7123): stage 2, 1 in 4.
    assert int(side_crit_odds(tables, out, 1, move)) == 4
    # Set already: nothing changes (:620-621; result FALSE, so no friendship either).
    _same(_use_both(tables, out, item), out)
    acts = [SWITCH_TO_1, FIRST_ITEM_ACTION]
    out = assert_matches_reference(tables, _bag(s, names, "Dire Hit"), acts, empty_outcome())
    assert int(out.focus_energy[1]) == 1

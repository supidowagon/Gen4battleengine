"""The outcome laws follow a faster foe's Transform (G-69).

``BtlCmd_Transform`` (external/pokeplatinum src/battle/battle_script.c:5330-5373) copies the
defender's BattleMon up to and including ``ability`` (include/battle/battle_mon.h:57-84): the
stats, ``statBoosts`` and ``ability`` among them, not the gender. Each case below has a faster
foe transform into the player before the player's hit or the turn's end, so a draw the turn-start
abilities and speeds would leave unread is read: the laws must keep it relevant, and the two
values of the draw step to different states in both engines.
"""

from __future__ import annotations

import numpy as np

from nuzline.engine.ids import AbilityId, Stat, Status
from nuzline.engine.outcome import armored, empty_outcome
from tests.engine.helpers import A, assert_matches_reference
from tests.engine.test_damage_classes import _state
from tests.engine.test_enumerate import laws_of


def _two(tables, s, acts, field, index):
    """The two states ``field[index]`` = 0 and 1 step to (both engines)."""
    out = []
    for v in (0, 1):
        o = empty_outcome()
        arr = getattr(o, field)
        o = o._replace(**{field: arr.at[index].set(v)})
        out.append(assert_matches_reference(tables, s, acts, o))
    return out


def test_copied_armor_lost_keeps_the_crit(tables, names):
    # A Battle Armor foe (battle_lib.c:7135-7136) transforms into the player, whose ability is
    # no armor: the player's Slash can crit it.
    s = _state(tables, names, "Slash", "Transform", "opp")
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.BATTLE_ARMOR))
    acts = A("Slash", "Transform", s, names)
    assert not bool(armored(tables, s, acts)[0])
    assert not bool(laws_of(tables, s, acts).no_crit[0])
    plain, crit = _two(tables, s, acts, "crit", (0, 0))
    assert int(crit.hp[1, 0]) < int(plain.hp[1, 0])


def test_copied_static_is_a_contact_proc(tables, names):
    # The transformed foe holds the player's Static (effects._contact_ability): the player's
    # Tackle reads its 30 % roll (battle_lib.c:4222).
    s = _state(tables, names, "Tackle", "Transform", "opp")
    s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.STATIC))
    acts = A("Tackle", "Transform", s, names)
    assert bool(laws_of(tables, s, acts).relevant.contact_proc[1, 0])
    calm, para = _two(tables, s, acts, "contact_proc", (1, 0))
    assert int(calm.status[0, 0]) == Status.NONE
    assert int(para.status[0, 0]) == Status.PARALYSIS


def test_copied_speed_ties_the_turn_end(tables, names):
    # The foe (one Spe faster) copies the player's Spe: the end-of-turn order (step.
    # end_of_turn_order) is a tie, and it decides which poisoned 1-HP Pokemon faints first.
    s = _state(tables, names, "Growl", "Transform", "opp")
    s = s._replace(
        status=s.status.at[:, 0].set(Status.POISON),
        hp=s.hp.at[:, 0].set(1),
        stages=s.stages.at[:, Stat.SPE].set(0),
    )
    acts = A("Growl", "Transform", s, names)
    assert bool(laws_of(tables, s, acts).relevant.end_speed_tie)
    first, second = _two(tables, s, acts, "end_speed_tie", ())
    assert any(
        not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(first, second, strict=True)
    )


def test_copied_shed_skin_rolls(tables, names):
    # The poisoned foe copies the player's Shed Skin: its turn-end cure roll is read.
    s = _state(tables, names, "Growl", "Transform", "opp")
    s = s._replace(
        ability=s.ability.at[0, 0].set(AbilityId.SHED_SKIN),
        status=s.status.at[1, 0].set(Status.POISON),
    )
    acts = A("Growl", "Transform", s, names)
    assert bool(laws_of(tables, s, acts).relevant.shed_skin[1])
    kept, cured = _two(tables, s, acts, "shed_skin", (1,))
    assert int(kept.status[1, 0]) == Status.POISON
    assert int(cured.status[1, 0]) == Status.NONE

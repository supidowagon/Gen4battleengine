"""The AI reads a battler's current ability, not its species' (E1 batch 6).

``LoadBattlerAbility`` returns ``battleMons[battler].ability`` for the AI's own Pokemon and for
an opponent whose ability it knows (trainer_ai.c:1170-1209), and a change in battle such as
Trace's copy writes it and tells the AI (``BattleAI_SetAbility``, battle_script.c:3401-3403,
:12106-12109). Only the guess for an unknown player ability reads the species' two abilities
(:1191-1192). The engine's current ability is ``engine.damage.decomp_abilities``; these tests
stand a changed ability in for it.
"""

import pytest

from nuzline.ai import commands as c
from nuzline.ai import policy, switching
from nuzline.engine.state import make_state

LEVITATE, MOLD_BREAKER = c.ABILITY_IDS["Levitate"], c.ABILITY_IDS["Mold Breaker"]
NORMALIZE = c.ABILITY_IDS["Normalize"]


def _changed(monkeypatch, side: int, ability: int):
    """``damage.decomp_abilities`` with ``side``'s active Pokemon's ability ``ability``, as
    after a Trace copy, in every AI module that reads it."""
    from nuzline.engine.damage import decomp_abilities

    def changed(tables, state):
        out = decomp_abilities(tables, state)
        return out.at[side, state.active[side]].set(ability)

    for module in (c, policy, switching):
        monkeypatch.setattr(module, "decomp_abilities", changed)


@pytest.fixture
def state(kaizo_tables, kaizo_roark_party):
    return make_state(kaizo_tables, [kaizo_roark_party[2]], list(kaizo_roark_party))


def test_attacker_reads_its_current_ability(monkeypatch, kaizo_tables, state):
    t = kaizo_tables
    _changed(monkeypatch, 1, MOLD_BREAKER)
    assert int(c.attacker_ability(t, state)) == MOLD_BREAKER


@pytest.mark.parametrize("known", [0, 1])
def test_defender_reads_its_current_ability_when_known(monkeypatch, kaizo_tables, state, known):
    """Known: the current ability. Unknown: the guess from the species' two abilities, never
    the changed one (Gible: Sand Veil / Rough Skin)."""
    t = kaizo_tables
    s = state._replace(ai_ability_known=state.ai_ability_known.at[0].set(known and LEVITATE))
    _changed(monkeypatch, 0, LEVITATE)
    got = {int(c.defender_ability(t, s, g)) for g in (0, 1)}
    if known:
        assert got == {LEVITATE}
    else:
        assert LEVITATE not in got


@pytest.mark.fullset
def test_refusals_read_the_current_ability(monkeypatch, kaizo_tables, kaizo_names, state):
    """A player's Pokemon whose current ability is a refused one (Normalize, switching
    ``_ANY_SIDE_ABILITIES``) is refused although its species has none."""
    t, n = kaizo_tables, kaizo_names
    assert not any("Normalize" in r for r in policy.ai_refusals(t, state, n))
    import jax

    _changed(monkeypatch, 0, NORMALIZE)
    with jax.disable_jit():  # the patched function, not a program compiled before the patch
        assert any("Normalize" in r for r in policy.ai_refusals(t, state, n))


@pytest.mark.fullset
def test_switching_reads_the_current_ability(monkeypatch, kaizo_tables, state):
    """The switch logic refuses on the current ability too (``switching._layout`` and its
    message, ``_refusals``): a player's Pokemon whose current ability is Normalize
    (``_ANY_SIDE_ABILITIES``) is refused, though its species has none (batch 6 review M4)."""
    import jax

    sites = 44
    assert not switching._refusals(
        switching._layout(kaizo_tables, state, sites), state, kaizo_tables
    )
    _changed(monkeypatch, 0, NORMALIZE)
    with jax.disable_jit():
        lay = switching._layout(kaizo_tables, state, sites)
        lines = switching._refusals(lay, state, kaizo_tables)
    assert any("Normalize" in line for line in lines), lines

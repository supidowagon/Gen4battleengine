# Mechanics quirks reported by the user, each checked against the decomp

## Focus Sash survives every hit of a multi-hit move — CONFIRMED, modelled
battle_controller_player.c UpdateHP ~3386-3404: a full-HP hit sets selfTurnFlags.focusItemActivated; while set, any lethal hit is capped to 1 HP; the item is removed after the move.

## Magic Guard: never fully paralysed, but keeps the paralysis speed drop — CONFIRMED, modelled
- battle_controller_player.c:2653-2654: CHECK_STATUS_STATE_PARALYSIS requires `Battler_Ability(attacker) != ABILITY_MAGIC_GUARD`, so the full-paralysis roll is skipped, with no RandNext draw.
- battle_lib.c:1259-1260 / 1325-1326: the speed /4 has no Magic Guard exemption.
- The user relies on this in Kaizo's permanent Trick Room sections.
- Every other Magic Guard site: bcp.c:1381, 1488 (end-of-turn damage), 2654 (full paralysis), 4760; battle_lib.c:4259 (Rough Skin does not hurt a Magic Guard attacker), 4360, 4946, 5331, 5402, 5413, 7358, 7370; subscript_weather_continues.s (weather chip).

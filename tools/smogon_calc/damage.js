// Damage oracle for tests/test_damage_oracle.py: @smogon/calc's Gen 4 (DPP) damage
// calculation. Reads a JSON array of matchups on stdin and writes one JSON result per
// matchup to stdout:
//   {damage: [16 ints], attacker: {stats}, defender: {stats}, move: {bp, type, category}}
// A matchup is {attacker, defender, move, overrides, crit, weather, reflect, lightScreen}
// (overrides: the move's {basePower, type}, e.g. Hidden Power's from the IVs), where
// attacker/defender are {species, level, nature, ivs, evs, gender, ability, boosts,
// status, curHP} in @smogon/calc's own option names. Install with
// `npm install --prefix tools/smogon_calc`.
'use strict';
const {calculate, Generations, Pokemon, Move, Field} = require('@smogon/calc');

const gen = Generations.get(4);

// Decomp names whose @smogon/calc id differs (beyond case, spaces and punctuation).
const MOVE_ALIASES = {vicegrip: 'visegrip', faintattack: 'feintattack', hijumpkick: 'highjumpkick'};

const toID = (name) => name.toLowerCase().replace(/[^a-z0-9]/g, '');

function lookup(table, name, what, aliases = {}) {
  const id = aliases[toID(name)] || toID(name);
  const entry = table.get(id);
  if (!entry) throw new Error(`@smogon/calc has no ${what} ${JSON.stringify(name)}`);
  return entry.name;
}

function mon(spec) {
  return new Pokemon(gen, lookup(gen.species, spec.species, 'species'), {
    level: spec.level,
    nature: spec.nature,
    ivs: spec.ivs,
    evs: spec.evs,
    gender: spec.gender,
    ability: spec.ability,
    boosts: spec.boosts,
    status: spec.status,
    curHP: spec.curHP,
  });
}

function one(m) {
  const attacker = mon(m.attacker);
  const defender = mon(m.defender);
  // One hit: multi-hit moves are compared per hit.
  const move = new Move(gen, lookup(gen.moves, m.move, 'move', MOVE_ALIASES), {
    isCrit: m.crit,
    hits: 1,
    overrides: m.overrides,
  });
  const field = new Field({
    gameType: 'Singles',
    weather: m.weather || undefined,
    defenderSide: {isReflect: m.reflect, isLightScreen: m.lightScreen},
  });
  const result = calculate(gen, attacker, defender, move, field);
  let damage = result.damage;
  if (typeof damage === 'number') damage = Array(16).fill(damage);
  // A fixed-count multi-hit move (Double Kick) ignores `hits` and gives one roll array per
  // hit; every hit is the same calculation, so the first is the per-hit damage.
  if (Array.isArray(damage[0])) {
    if (damage.some((hit) => JSON.stringify(hit) !== JSON.stringify(damage[0]))) {
      throw new Error(`${m.move}: the hits differ`);
    }
    damage = damage[0];
  }
  return {
    damage,
    attacker: attacker.rawStats,
    defender: defender.rawStats,
    move: {bp: move.bp, type: move.type, category: move.category},
  };
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => (input += chunk));
process.stdin.on('end', () => {
  process.stdout.write(JSON.stringify(JSON.parse(input).map(one)));
});

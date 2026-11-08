-- Check the battle-RAM addresses that tools/trace_battle.lua relies on, in one battle.
--
-- DeSmuME 0.9.x (Lua 5.1 with the `bit` library): Tools > Lua Scripting > New Lua Script
-- Window > Browse > this file > Run, then start any battle. The script finds the battle
-- structures exactly as trace_battle.lua does, and at every turn it writes one line per
-- check with the address, the value it read and what that value should be. Lines the
-- script can judge itself end in OK or CHECK; weather and the party lines are compared by
-- hand (or by Claude, against the save). It records no trace. Output: traces/probe.txt in the project folder
-- (the Lua console prints the full path). Player guide: docs/tracing_kaizo.md; what each
-- check means: docs/tracing.md, "Verifying the addresses".

-- Known issue (2026-11-08): run beside trace_battle.lua in a second Lua window, in a Kaizo
-- Roark fight, this script wrote only its header line, while trace_battle.lua (same locate
-- code and loop) located and recorded. No bug was found. Likely causes: two windows each
-- running a long blocking scan at once, or this window stopping on an unreported error.
-- Run it alone and read the Lua console (docs/tracing.md, "Why the probe logged nothing").

---------------------------------------------------------------------------- config

-- The player's lead, used to find its BattleMon in RAM: species id, level.
-- 27 = Sandshrew, level 16: the lead of the pre-Roark Kaizo save.
LOCATE_SPECIES = LOCATE_SPECIES or 27
LOCATE_LEVEL = LOCATE_LEVEL or 16
-- nil: write traces/probe.txt in the project folder (found from this script's path).
PROBE_PATH = PROBE_PATH or nil
LOCATE_EVERY = 120 -- frames between scans while not located

local RAM_START, RAM_END = 0x02000000, 0x02400000

-- The same tables as tools/trace_battle.lua (checked by `python -m tools.rom.offsets`).
local SYS = {
  battleCtx = 0x30,
  parties = 0x68,
  seedRandNext = 0x2448,
}
local CTX = {
  command = 0x08,
  commandNext = 0x0C,
  attacker = 0x64,
  defender = 0x6C,
  totalTurns = 0x150,
  fieldConditionsMask = 0x180,
  damage = 0x2144,
  selectedPartySlot = 0x219C,
  battleMons = 0x2D40,
}
local MON = {
  size = 0xC0,
  species = 0x00,
  moves = 0x0C,
  level = 0x34,
  curHP = 0x4C,
  maxHP = 0x50,
}
local PARTY = { currentCount = 0x04, pokemon = 0x08, stride = 0xEC }
local PKM = { personality = 0x00, flags = 0x04, checksum = 0x06, blocks = 0x08, party = 0x88 }
local CMD = {
  INIT_COMMAND_SELECTION = 4,
  COMMAND_SELECTION_INPUT = 5,
  UPDATE_HP = 27,
  MOVE_END = 39,
  RESULT = 41,
}
local SHUFFLE = {
  [0] = { 0, 1, 2, 3 }, { 0, 1, 3, 2 }, { 0, 2, 1, 3 }, { 0, 3, 1, 2 },
  { 0, 2, 3, 1 }, { 0, 3, 2, 1 }, { 1, 0, 2, 3 }, { 1, 0, 3, 2 },
  { 2, 0, 1, 3 }, { 3, 0, 1, 2 }, { 2, 0, 3, 1 }, { 3, 0, 2, 1 },
  { 1, 2, 0, 3 }, { 1, 3, 0, 2 }, { 2, 1, 0, 3 }, { 3, 1, 0, 2 },
  { 2, 3, 0, 1 }, { 3, 2, 0, 1 }, { 1, 2, 3, 0 }, { 1, 3, 2, 0 },
  { 2, 1, 3, 0 }, { 3, 1, 2, 0 }, { 2, 3, 1, 0 }, { 3, 2, 1, 0 },
}

------------------------------------------------------------------------- memory

local function u8(a) return memory.readbyte(a) end
local function u16(a) return memory.readword(a) end
local function u32(a)
  local v = memory.readdword(a)
  if v < 0 then v = v + 4294967296 end
  return v
end
local function s32(a)
  local v = u32(a)
  if v >= 2147483648 then v = v - 4294967296 end
  return v
end

local function lcrng_next(seed)
  local s_hi, s_lo = math.floor(seed / 65536), seed % 65536
  local mid = (s_hi * 0x4E6D + s_lo * 0x41C6) % 65536
  return (s_lo * 0x4E6D + mid * 65536 + 24691) % 4294967296
end

local function decode(addr, count, seed, plain)
  local out = {}
  for i = 0, count - 1 do
    seed = lcrng_next(seed)
    local hw = u16(addr + 2 * i)
    out[i] = plain and hw or bit.bxor(hw, math.floor(seed / 65536))
  end
  return out
end

local function party_mon(addr)
  local pid = u32(addr + PKM.personality)
  local flags = u16(addr + PKM.flags)
  local box = decode(addr + PKM.blocks, 64, u16(addr + PKM.checksum), bit.band(flags, 2) ~= 0)
  local party = decode(addr + PKM.party, 50, pid, bit.band(flags, 1) ~= 0)
  local order = SHUFFLE[math.floor((pid % 0x40000) / 0x2000) % 24]
  local a, b = order[1] * 16, order[2] * 16
  return {
    species = box[a], moves = { box[b], box[b + 1], box[b + 2], box[b + 3] },
    level = party[2] % 256, hp = party[3], max_hp = party[4],
  }
end

local function hex(v) return string.format("0x%04X%04X", math.floor(v / 65536), v % 65536) end

------------------------------------------------------------------------- locate

-- Exactly as tools/trace_battle.lua: a BattleMon of the player's lead, then the
-- BattleSystem that points at its context.
local function locate()
  for addr = RAM_START, RAM_END - MON.size, 4 do
    if u16(addr + MON.species) == LOCATE_SPECIES and u8(addr + MON.level) == LOCATE_LEVEL then
      local hp, max_hp = s32(addr + MON.curHP), u32(addr + MON.maxHP)
      if max_hp > 0 and max_hp <= 999 and hp >= 0 and hp <= max_hp then
        local ctx = addr - CTX.battleMons
        for p = RAM_START, RAM_END - 4, 4 do
          if u32(p) == ctx then return p - SYS.battleCtx end
        end
      end
    end
  end
  return nil
end

------------------------------------------------------------------------- output

local function default_path()
  local info = debug and debug.getinfo(1, "S")
  local src = info and info.source or ""
  local dir = src:sub(1, 1) == "@" and src:sub(2):match("^(.*)[/\\]tools[/\\][^/\\]*$")
  if not dir then return "probe.txt" end
  return dir .. "\\traces\\probe.txt", dir .. "\\traces"
end

local path, folder = PROBE_PATH, nil
if not path then path, folder = default_path() end
local out = io.open(path, "w")
if not out and folder then
  os.execute('mkdir "' .. folder .. '"')
  out = io.open(path, "w")
end
assert(out, "probe: cannot write " .. path)
print("probe: writing to " .. path)

local function line(text)
  out:write(text .. "\n")
  out:flush()
end

local function verdict(ok) return ok and "OK" or "CHECK" end

line(string.format("probe_battle.lua: LOCATE_SPECIES %d, LOCATE_LEVEL %d", LOCATE_SPECIES, LOCATE_LEVEL))

------------------------------------------------------------------------- checks

local function battle_mon(ctx, battler) return ctx + CTX.battleMons + battler * MON.size end

local turns_seen, last_rng = 0, nil
local commands = {}

local function turn_checks(sys, ctx)
  local turn = s32(ctx + CTX.totalTurns)
  line("")
  line(string.format("== turn boundary %d (commands seen since the last one: %s)",
    turns_seen, table.concat(commands, " ")))
  line(string.format("totalTurns           %08X = %d   should be: %d (the turn number, from 0)   %s",
    ctx + CTX.totalTurns, turn, turns_seen, verdict(turn == turns_seen)))
  local field = u32(ctx + CTX.fieldConditionsMask)
  line(string.format("fieldConditionsMask  %08X = %s   should be: nonzero in sandstorm (or other weather), 0 otherwise",
    ctx + CTX.fieldConditionsMask, hex(field)))
  local rng = u32(sys + SYS.seedRandNext)
  line(string.format("seedRandNext         %08X = %s   should be: a different value every turn   %s",
    sys + SYS.seedRandNext, hex(rng), verdict(last_rng == nil or rng ~= last_rng)))
  last_rng = rng
  for battler = 0, 1 do
    local slot = u8(ctx + CTX.selectedPartySlot + battler)
    local party = u32(sys + SYS.parties + 4 * battler)
    local mon = battle_mon(ctx, battler)
    local live = u16(mon + MON.species)
    local count = u32(party + PARTY.currentCount)
    local stored = slot < count and party_mon(party + PARTY.pokemon + slot * PARTY.stride).species or -1
    local who = battler == 0 and "your" or "the opponent's"
    line(string.format(
      "selectedPartySlot[%d] %08X = %d   should be: %s party slot now out (0 = first); party slot %d species %d, battle species %d   %s",
      battler, ctx + CTX.selectedPartySlot + battler, slot, who, slot, stored, live,
      verdict(stored == live)))
  end
  local party = u32(sys + SYS.parties)
  local count = math.min(u32(party + PARTY.currentCount), 6) -- 6: a bad address stays short
  for slot = 0, count - 1 do
    local m = party_mon(party + PARTY.pokemon + slot * PARTY.stride)
    line(string.format("party[%d] %08X: species %d, level %d, HP %d/%d, moves %d %d %d %d   should be: the summary screen",
      slot, party + PARTY.pokemon + slot * PARTY.stride, m.species, m.level, m.hp, m.max_hp,
      m.moves[1], m.moves[2], m.moves[3], m.moves[4]))
  end
  turns_seen = turns_seen + 1
  commands = {}
end

local pending_hit = nil -- the UPDATE_HP being followed to its MOVE_END

local function hit_start(ctx)
  local damage = s32(ctx + CTX.damage)
  if damage == 0 then return end
  local defender = s32(ctx + CTX.defender)
  pending_hit = { damage = damage, defender = defender,
    hp = s32(battle_mon(ctx, defender) + MON.curHP), addr = ctx + CTX.damage }
end

local function hit_end(ctx)
  if not pending_hit then return end
  local h = pending_hit
  local now = s32(battle_mon(ctx, h.defender) + MON.curHP)
  local lost = h.hp - now
  local ko = now == 0
  line(string.format(
    "damage               %08X = %d at UPDATE_HP   should be: minus the HP lost; battler %d HP %d -> %d at MOVE_END (lost %d)%s   %s",
    h.addr, h.damage, h.defender, h.hp, now, lost, ko and ", a KO" or "",
    verdict(lost == -h.damage or (ko and lost <= -h.damage))))
  pending_hit = nil
end

------------------------------------------------------------------------- main

local sys = nil
local frame, prev_cmd, done = 0, -1, false

while true do
  frame = frame + 1
  if not sys and frame % LOCATE_EVERY == 0 then
    sys = locate()
    if sys then
      local msg = string.format("probe: located BattleSystem 0x%08X, BattleContext 0x%08X",
        sys, u32(sys + SYS.battleCtx))
      print(msg)
      line(msg)
    end
  end
  if sys and not done then
    local ctx = u32(sys + SYS.battleCtx)
    if ctx >= RAM_START and ctx < RAM_END then
      local cmd = u32(ctx + CTX.command)
      if cmd ~= prev_cmd then
        commands[#commands + 1] = tostring(cmd)
        if cmd == CMD.INIT_COMMAND_SELECTION or cmd == CMD.COMMAND_SELECTION_INPUT then
          -- Every entry to the command menu is a turn boundary, so a totalTurns that does
          -- not move shows as CHECK. INPUT counts only before the first check, for a
          -- probe located after INIT had passed.
          if cmd == CMD.INIT_COMMAND_SELECTION or turns_seen == 0 then
            turn_checks(sys, ctx)
            print("probe: turn " .. (turns_seen - 1))
          end
        elseif cmd == CMD.UPDATE_HP then
          hit_start(ctx)
        elseif cmd == CMD.MOVE_END then
          hit_end(ctx)
        elseif cmd == CMD.RESULT then
          hit_end(ctx)
          line("")
          line("== battle over (commands seen: " .. table.concat(commands, " ") .. ")")
          out:close()
          done = true
          print("probe: battle over, wrote " .. path)
        end
        prev_cmd = cmd
      end
    end
  end
  emu.frameadvance()
end

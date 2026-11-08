-- Record a Pokemon Platinum (US) single battle from DeSmuME's RAM as a raw trace.
--
-- DeSmuME 0.9.13+ (Lua 5.1 with the `bit` library): Tools > Lua Scripting > New Lua
-- Script Window > Browse > this file > Run, then start the battle. The output is JSON
-- lines (docs/tracing.md, "Raw trace format"):
--   {"kind":"turn",...}      at every command selection, and once more at the result
--   {"kind":"hit",...}       when a move's damage is about to be applied, and when a
--                            confused battler hurts itself (attacker == defender)
--   {"kind":"move_end",...}  at the end of every move
-- Convert the trace with tools/trace_to_state.py. Every SYS/CTX/MON offset below is
-- checked against the machine code of both ROMs (vanilla and Kaizo) by
-- `python -m tools.rom.offsets` (docs/tracing.md, "RAM layout"); the VERIFY notes are
-- what only the emulator can confirm (tools/probe_battle.lua).

---------------------------------------------------------------------------- config

TRACE_PATH = TRACE_PATH or "traces/roark_01.jsonl"

-- Address of the BattleSystem struct, or nil to locate it (docs/tracing.md,
-- "Locating the battle structures"). Paste the address the first run prints to skip
-- the scan when replaying from the same save state.
BATTLE_SYSTEM = BATTLE_SYSTEM or nil
-- The player's lead, used to find its BattleMon in RAM: species id, level.
LOCATE_SPECIES = LOCATE_SPECIES or 391 -- Monferno
LOCATE_LEVEL = LOCATE_LEVEL or 14
LOCATE_EVERY = 120 -- frames between scans while not located

local RAM_START, RAM_END = 0x02000000, 0x02400000

-- struct BattleSystem (include/struct_defs/battle_system.h)
local SYS = {
  battleCtx = 0x30,
  parties = 0x68, -- Party *parties[4]; battler 0 = player, 1 = opponent
  seedRandNext = 0x2448, -- BattleSystem_RandNext's literal; the decomp's unread_2440 name is 4 off
}

-- struct BattleContext (include/battle/battle_context.h)
local CTX = {
  command = 0x08,
  commandNext = 0x0C,
  attacker = 0x64,
  defender = 0x6C,
  totalTurns = 0x150,
  fieldConditionsMask = 0x180,
  damage = 0x2144, -- int, negative at BATTLE_CONTROL_UPDATE_HP
  criticalMul = 0x2150,
  hpCalcTemp = 0x215C, -- int, minus the confusion self-hit damage
  moveStatusFlags = 0x216C,
  selectedPartySlot = 0x219C, -- u8[4]; VERIFY (probe) that it indexes the battle's Party
  battleMons = 0x2D40, -- BattleMon[4]
  moveCur = 0x3044,
}

-- BattleMon (include/battle/battle_mon.h)
local MON = {
  size = 0xC0,
  species = 0x00,
  moves = 0x0C,
  statBoosts = 0x18,
  ppCur = 0x2C,
  level = 0x34,
  curHP = 0x4C,
  maxHP = 0x50,
  status = 0x6C,
  heldItem = 0x78,
}

-- Party (include/party.h) and Pokemon (include/struct_defs/pokemon.h)
local PARTY = { currentCount = 0x04, pokemon = 0x08, stride = 0xEC }
local PKM = { personality = 0x00, flags = 0x04, checksum = 0x06, blocks = 0x08, party = 0x88 }

-- enum BattleControlSequence (include/battle/battle_controller_player.h)
local CMD = {
  INIT_COMMAND_SELECTION = 4,
  COMMAND_SELECTION_INPUT = 5,
  EXEC_SCRIPT = 21,
  UPDATE_HP = 27,
  LOOP_FAINTED = 33,
  MOVE_END = 39,
  RESULT = 41,
}

-- BoxPokemon_GetDataBlock (src/pokemon.c:4861): storage index of blocks A, B, C, D by
-- (personality & 0x3E000) >> 13; 24-31 repeat 0-7.
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
local function s8(a)
  local v = u8(a)
  if v >= 128 then v = v - 256 end
  return v
end

-- LCRNG_NextFrom (src/math_util.c): seed = seed * 1103515245 + 24691 (mod 2^32),
-- split into 16-bit halves so that Lua's doubles stay exact.
local function lcrng_next(seed)
  local s_hi, s_lo = math.floor(seed / 65536), seed % 65536
  local mid = (s_hi * 0x4E6D + s_lo * 0x41C6) % 65536
  return (s_lo * 0x4E6D + mid * 65536 + 24691) % 4294967296
end

-- DecodeData (src/math_util.c:225): halfword i ^= the (i + 1)-th LCRNG output >> 16.
-- Returns halfwords 0-based; `plain` skips the XOR (the decrypted flag is set).
local function decode(addr, count, seed, plain)
  local out = {}
  for i = 0, count - 1 do
    seed = lcrng_next(seed)
    local hw = u16(addr + 2 * i)
    out[i] = plain and hw or bit.bxor(hw, math.floor(seed / 65536))
  end
  return out
end

local function lo(hw) return hw % 256 end
local function hi(hw) return math.floor(hw / 256) end

------------------------------------------------------------------------- json

local function is_array(t)
  return type(t) == "table" and (t[1] ~= nil or next(t) == nil)
end

local function to_json(v, keys)
  if v == nil then return "null" end
  if type(v) == "boolean" then return v and "true" or "false" end
  -- "%.0f", not "%d": Lua 5.1's %d goes through a 32-bit long on Windows, and the RNG
  -- state and masks reach 2^32 - 1. Every value written is an integer.
  if type(v) == "number" then return string.format("%.0f", v) end
  if type(v) == "string" then return string.format("%q", v) end
  local parts = {}
  if is_array(v) then
    for i = 1, #v do parts[i] = to_json(v[i]) end
    return "[" .. table.concat(parts, ",") .. "]"
  end
  local names = {}
  for k in pairs(v) do names[#names + 1] = k end
  table.sort(names)
  for i, k in ipairs(names) do parts[i] = string.format("%q:%s", k, to_json(v[k])) end
  return "{" .. table.concat(parts, ",") .. "}"
end

------------------------------------------------------------------------- battle

local function battle_mon(ctx, battler)
  return ctx + CTX.battleMons + battler * MON.size
end

local function stages(mon)
  local out = {}
  for i = 0, 7 do out[i + 1] = s8(mon + MON.statBoosts + i) end
  return out
end

local function live_mon(mon)
  local moves, pp = {}, {}
  for i = 0, 3 do
    moves[i + 1] = u16(mon + MON.moves + 2 * i)
    pp[i + 1] = u8(mon + MON.ppCur + i)
  end
  return {
    species = u16(mon + MON.species), level = u8(mon + MON.level),
    hp = s32(mon + MON.curHP), max_hp = u32(mon + MON.maxHP),
    status = u32(mon + MON.status), moves = moves, pp = pp, item = u16(mon + MON.heldItem),
  }
end

local function party_mon(addr)
  local pid = u32(addr + PKM.personality)
  local flags = u16(addr + PKM.flags)
  local box = decode(addr + PKM.blocks, 64, u16(addr + PKM.checksum), bit.band(flags, 2) ~= 0)
  local party = decode(addr + PKM.party, 50, pid, bit.band(flags, 1) ~= 0)
  local order = SHUFFLE[math.floor((pid % 0x40000) / 0x2000) % 24]
  local a, b = order[1] * 16, order[2] * 16 -- halfword index of blocks A and B
  return {
    species = box[a], item = box[a + 1],
    moves = { box[b], box[b + 1], box[b + 2], box[b + 3] },
    pp = { lo(box[b + 4]), hi(box[b + 4]), lo(box[b + 5]), hi(box[b + 5]) },
    status = party[0] + party[1] * 65536, level = lo(party[2]),
    hp = party[3], max_hp = party[4],
  }
end

local function side(sys, ctx, battler)
  local party_addr = u32(sys + SYS.parties + 4 * battler)
  local active = u8(ctx + CTX.selectedPartySlot + battler)
  local mon = battle_mon(ctx, battler)
  local members = {}
  for slot = 0, u32(party_addr + PARTY.currentCount) - 1 do
    if slot == active then
      members[slot + 1] = live_mon(mon)
    else
      members[slot + 1] = party_mon(party_addr + PARTY.pokemon + slot * PARTY.stride)
    end
  end
  return { active = active, stages = stages(mon), party = members }
end

local function turn_record(sys, ctx, command)
  return {
    kind = "turn", turn = s32(ctx + CTX.totalTurns), command = command,
    rng = u32(sys + SYS.seedRandNext), field = u32(ctx + CTX.fieldConditionsMask),
    sides = { side(sys, ctx, 0), side(sys, ctx, 1) },
  }
end

local function hit_record(ctx, damage, defender)
  local m0, m1 = battle_mon(ctx, 0), battle_mon(ctx, 1)
  return {
    kind = "hit", turn = s32(ctx + CTX.totalTurns),
    attacker = s32(ctx + CTX.attacker), defender = defender or s32(ctx + CTX.defender),
    move = u32(ctx + CTX.moveCur), damage = damage,
    crit = s32(ctx + CTX.criticalMul) > 1 and 1 or 0,
    active = { u8(ctx + CTX.selectedPartySlot), u8(ctx + CTX.selectedPartySlot + 1) },
    hp = { s32(m0 + MON.curHP), s32(m1 + MON.curHP) },
    status = { u32(m0 + MON.status), u32(m1 + MON.status) },
    stages = { stages(m0), stages(m1) },
  }
end

local function move_end_record(ctx)
  return {
    kind = "move_end", turn = s32(ctx + CTX.totalTurns),
    attacker = s32(ctx + CTX.attacker), move = u32(ctx + CTX.moveCur),
    flags = u32(ctx + CTX.moveStatusFlags),
  }
end

local function in_ram(a) return a >= RAM_START and a < RAM_END end

-- A BattleMon of the player's lead, then the BattleSystem that points at its context.
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

------------------------------------------------------------------------- main

local out = assert(io.open(TRACE_PATH, "w"))
local function emit(record)
  out:write(to_json(record) .. "\n")
  out:flush()
end

local sys = BATTLE_SYSTEM
local frame, prev_cmd, last_turn, done = 0, -1, nil, false

while true do
  frame = frame + 1
  if not sys and frame % LOCATE_EVERY == 0 then
    sys = locate()
    if sys then
      print(string.format("trace: located BattleSystem 0x%08X, BattleContext 0x%08X",
        sys, u32(sys + SYS.battleCtx)))
    end
  end
  if sys and not done then
    local ctx = u32(sys + SYS.battleCtx)
    if in_ram(ctx) then
      local cmd = u32(ctx + CTX.command)
      if cmd ~= prev_cmd then
        if cmd == CMD.INIT_COMMAND_SELECTION or cmd == CMD.COMMAND_SELECTION_INPUT then
          -- INPUT as well, in case the script was located after INIT this turn.
          local turn = s32(ctx + CTX.totalTurns)
          if turn ~= last_turn then
            emit(turn_record(sys, ctx, cmd))
            last_turn = turn
            print("trace: turn " .. turn)
          end
        elseif cmd == CMD.UPDATE_HP then
          local damage = -s32(ctx + CTX.damage)
          if damage ~= 0 then emit(hit_record(ctx, damage)) end
        elseif cmd == CMD.EXEC_SCRIPT then
          -- A confusion self-hit never reaches UPDATE_HP: the status check sets
          -- defender = attacker, hpCalcTemp = minus the damage, and runs
          -- subscript_hurt_self_in_confusion before LOOP_FAINTED
          -- (battle_controller_player.c:2628-2638).
          local attacker = s32(ctx + CTX.attacker)
          if u32(ctx + CTX.commandNext) == CMD.LOOP_FAINTED
              and s32(ctx + CTX.defender) == attacker then
            local damage = -s32(ctx + CTX.hpCalcTemp)
            if damage > 0 then emit(hit_record(ctx, damage, attacker)) end
          end
        elseif cmd == CMD.MOVE_END then
          emit(move_end_record(ctx))
        elseif cmd == CMD.RESULT then
          emit(turn_record(sys, ctx, cmd))
          done = true
          out:close()
          print("trace: battle over, wrote " .. TRACE_PATH)
        end
        prev_cmd = cmd
      end
    end
  end
  emu.frameadvance()
end

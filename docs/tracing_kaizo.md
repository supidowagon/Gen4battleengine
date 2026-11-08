# Recording a Kaizo battle: player guide

This page is for the person playing. Part 1 is a one-time setup. Part 2 is the **address
check** (about 10 minutes): one battle with a small script running, which tells us that
the recorder reads the right parts of the game's memory. Part 3 is the real recording of
the Roark fight, done later (K3d). You do not need to read any code.

What you need:

- DeSmuME: `G:\Farry\Downloads\DeSmuME-VS2022-x64-Release.exe` (it reports itself as
  DeSmuME 0.9.14, 64-bit)
- The project folder: `G:\Farry\Claude Workspace\Pokemon Platinum Game Engine`
  (called "the project folder" below)
- Your pre-Roark Kaizo save: `G:\Farry\Downloads\Platinum Kaizo Base.dsv`

**Your real save is never used directly.** Everything below runs on copies, so nothing can
touch your actual run.

## Part 1: set up a practice copy (about 3 minutes, once)

0. **Lua support.** This DeSmuME runs scripts only when a 64-bit `lua51.dll` (Lua 5.1)
   sits in the same folder as `DeSmuME-VS2022-x64-Release.exe`. Without it, the script
   window reports `lua51.dll was not found`. There is no `lua51.dll` in
   `G:\Farry\Downloads` yet, so ask Claude to find a trustworthy 64-bit copy for this build
   before you start, and put it next to the program.
1. Make a new folder: `G:\Farry\Kaizo Probe`.
2. Find the Kaizo ROM. Open the project folder, then the `.cache` folder, then `rom`. The
   ROM is the file whose name starts with `kaizo-` and ends in `.nds`. Today it is
   `kaizo-b787e743b5fa78e1.nds`. The 16 letters and digits change only if the Kaizo patch
   changes. Ignore the `.nds.sha1` file next to it. This is the patched Kaizo ROM, built
   from your own vanilla ROM and the Kaizo patch, and checked against the patch's
   expected checksum. If the `rom` folder or the file is missing, ask Claude to "build the
   Kaizo ROM cache".
3. **Copy** that `.nds` file (Ctrl+C, Ctrl+V; do not drag it out) into
   `G:\Farry\Kaizo Probe`, and rename the copy to `Kaizo Probe.nds`.
4. **Copy** `G:\Farry\Downloads\Platinum Kaizo Base.dsv` into the **Battery folder**
   `G:\Farry\Downloads\Battery` (the folder next to `DeSmuME-VS2022-x64-Release.exe`; see
   below) and rename the copy to `Kaizo Probe.dsv`. The name before the dot must be the
   ROM's name. Do this with DeSmuME **closed**, and when DeSmuME has already made a blank
   `Kaizo Probe.dsv` there (it does so the first time it opens the ROM), copy over that
   blank one.

Why this folder: DeSmuME stores a game's save (its "battery" or "backup memory") as a
`.dsv` file named after the ROM. This build keeps those files in a `Battery` folder next to
the DeSmuME program (`G:\Farry\Downloads\Battery`), **not** next to the ROM (set under
**Config > Path Settings**, "Battery"). The first time it opens a new ROM it creates a blank
`.dsv` there, which is why the copy in step 4 has to replace that blank file while DeSmuME
is closed. Step 7 below checks that the save was picked up, and gives a way in if it was
not. In the 2026-11-08 run the copy used for the Roark recording was
`G:\Farry\Downloads\Battery\kaizo Probe.dsv`.

## Part 2: the address check (about 10 minutes)

### Start the game from the copy

5. Start DeSmuME. Choose **File > Open ROM** and open `G:\Farry\Kaizo Probe\Kaizo Probe.nds`.
6. At the title screen, press Start (Enter by default), then A (X by default).
7. You should see **CONTINUE** with your trainer name, 0 badges and your play time.
   - If you see only NEW GAME, DeSmuME did not pick up the save. Choose
     **File > Import Backup Memory**, select the copy in the Battery folder
     (`G:\Farry\Downloads\Battery\Kaizo Probe.dsv`, never the original
     `Platinum Kaizo Base.dsv`), and confirm. Then choose **File > Reset** and go
     to step 6 again.
8. Continue the game. Open the Pokemon menu and check that your **first** Pokemon is
   **Sandshrew, level 16**. The script looks for exactly that Pokemon when a battle
   starts.
   - If your first Pokemon or its level is different, open
     `tools\probe_battle.lua` (in the project folder) in Notepad. Near the top, change the
     numbers in `LOCATE_SPECIES = LOCATE_SPECIES or 27` (the species number, 27 for
     Sandshrew) and `LOCATE_LEVEL = LOCATE_LEVEL or 16` (the level). Save the file.
9. Make a save state, so you can replay from here: **File > Save State > 1**. (To go back
   later: **File > Load State > 1**.) Anything that writes a save here, an in-game save
   or a loaded save state, writes only to the copy `Kaizo Probe.dsv` in the Battery folder.

### Run the probe script

**Status of the address check (2026-11-08).** It was run with `tools\probe_battle.lua` and
`tools\trace_battle.lua` in two Lua windows at once, on a Kaizo Roark fight (lead Prinplup,
species 394, level 16, set through small launcher files). The probe wrote only its header
line and logged nothing else, so for now **the recording itself is the address check**:
Claude reads `traces\kaizo_roark_01.jsonl` and compares the fields (`docs/tracing.md`,
"Verifying the addresses", where the result is). The steps below are kept for a repeat of
the probe. To find out why it logged nothing, run it **alone, in one Lua window**, and send
what the Lua console prints.

10. Choose **Tools > Lua Scripting > New Lua Script Window**. In the small window, click
    **Browse**, open `tools\probe_battle.lua` in the project folder, then click **Run**.
    - The window's console should print `probe: writing to ...\traces\probe.txt`. That is
      where the results go: `G:\Farry\Claude Workspace\Pokemon Platinum Game Engine\traces\probe.txt`.
    - If it says `lua51.dll was not found`, see step 0.
11. Start **any battle**: walk into tall grass, or talk to a trainer. A trainer with two or
    more Pokemon is best, because the check then also sees the opponent switch in its
    second Pokemon. A wild battle is also fine.
12. Within a couple of seconds of the battle starting, the console prints
    `probe: located BattleSystem 0x..., BattleContext 0x...`, and then `probe: turn 0` when
    the FIGHT menu appears.
    - If nothing is printed by the time the FIGHT menu appears, the script did not find
      your lead (step 8). Click **Stop**, fix the two numbers, **File > Load State > 1**,
      and run the script again.
13. Fight for **at least 3 turns**, using a damaging move each turn (in this save
    Sandshrew knows Metal Claw and Poison Sting; avoid Dig, which takes two turns). The
    console prints `probe: turn 1`, `probe: turn 2`, and so on.
    If a sandstorm or other weather happens, remember on which turn.
14. Finish the battle by winning or running away. The console prints
    `probe: battle over, wrote ...`. Click **Stop** in the Lua window. (If the battle
    ends some other way, just click Stop: the file is saved after every line.)

### What to send back

15. Tell Claude "the probe is done". Claude reads `traces\probe.txt` in the project
    folder directly. If you are working elsewhere, paste the file's contents into the
    chat instead. Also mention:
    - which battle it was (wild Pokemon, or the trainer's name),
    - whether any weather happened, and on which turn,
    - anything that looked odd.

You do not need to read the file yourself. The lines the script can judge by itself
(`totalTurns`, `seedRandNext`, `selectedPartySlot`, `damage`) end in **OK** or **CHECK**.
The weather line is judged from what you report, and Claude compares the party lines with
your save. For reference, each turn the file has:

- `totalTurns`: should count 0, 1, 2, ... one per turn.
- `fieldConditionsMask`: should be nonzero during a sandstorm (or other weather), and 0
  otherwise.
- `seedRandNext`: the battle's random number. It should change every turn.
- `selectedPartySlot[0]` and `[1]`: which party slot is out on each side (0 is the first
  Pokemon). The line checks that the party slot holds the Pokemon actually in battle.
- `damage`: after each hit, the damage number must equal the HP the target lost, unless
  the hit knocked it out.
- `party[0]` to `party[5]`: your party as the script decodes it (species number, level,
  HP and moves).

## Part 3: recording the Roark fight (K3d, later)

Do this only after Claude has confirmed the address check. It uses the same practice copy
(Part 1).

1. Load the game from the copy (steps 5 to 9) and walk to Roark. Standing in front of
   him, **save in the game** (menu > SAVE). This writes your party as it is now into the
   copy `G:\Farry\Downloads\Battery\Kaizo Probe.dsv`, which is what the conversion reads.
   Then make a save state: **File > Save State > 2**.
   - From that in-game save until the fight, nothing about the party may change: no
     other battle, no level-up, no new or forgotten move, no held-item change, no party
     reordering. If anything changes, save in the game again and make the save state
     again.
   - Do not save in the game after the Roark fight.
2. Open `tools\trace_battle.lua` in Notepad and set, near the top:
   - `TRACE_PATH = TRACE_PATH or "G:/Farry/Claude Workspace/Pokemon Platinum Game Engine/traces/roark_kaizo_01.jsonl"`
     (with forward slashes, as written; the `traces` folder was made by the probe),
   - `LOCATE_SPECIES = LOCATE_SPECIES or 27` and `LOCATE_LEVEL = LOCATE_LEVEL or 16`
     (your lead, as in step 8).
3. **Tools > Lua Scripting > New Lua Script Window**, Browse to `tools\trace_battle.lua`,
   Run. Talk to Roark. The console prints `trace: located ...`, then `trace: turn N` each
   turn.
4. Rules for a usable recording:
   - Lead with your first Pokemon, and fight until Roark's last Pokemon faints.
   - **None of your Pokemon may faint.** The recording stops being useful at the first
     faint.
   - Use only moves and switches, never an item from your bag.
   - Fight at least 3 turns.
   If a rule breaks, click Stop, **File > Load State > 2**, and start again from step 3.
5. When the console prints `trace: battle over`, click Stop, close DeSmuME (so the save
   file is fully written) and tell Claude "the Roark recording is done". Claude converts
   it with the party read straight from the copy's in-game save (step 1), so you do not
   need to type in your Pokemon:

   ```
   uv run python -m tools.trace_to_state --game kaizo --save "G:\Farry\Downloads\Battery\Kaizo Probe.dsv" --trace traces\roark_kaizo_01.jsonl --out tests\golden\roark_kaizo_01.json
   ```

   `--save` must point at the `.dsv` that DeSmuME actually wrote, in the Battery folder (a
   copy sitting next to the ROM is not the one it uses). This only reads the save. Your original `Platinum Kaizo Base.dsv` in Downloads is not
   used here: it would miss any change to your party on the way to Roark. The recording and the save stay out of the project's history
   (`traces/` and `*.dsv` are never committed).

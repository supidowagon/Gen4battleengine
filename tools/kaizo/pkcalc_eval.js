// Evaluate one PKCalc data file and write one of its globals as JSON.
//
// Usage: node pkcalc_eval.js FILE GLOBAL OUT [TIMEOUT_MS]
//
// Node's `vm` is NOT a security boundary. The protection rests on the fixed list of
// data files (tools/extract_pkcalc.py PKCALC_FILES), the zip's SHA-256 being recorded
// and reviewed, and this hardened context, in that order: a null-prototype global, no
// string code generation and no WebAssembly, no `require`/`process`/timers, and a time
// limit. Nothing but the file's own source and its declared globals reaches the context.
// Run it from the repository with paths as arguments, never from the extraction folder.
"use strict";

const fs = require("fs");
const util = require("util");
const vm = require("vm");

function fail(message) {
    process.stderr.write(message + "\n");
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length < 3 || args.length > 4) {
    fail("usage: node pkcalc_eval.js FILE GLOBAL OUT [TIMEOUT_MS]");
}
const [file, globalName, out] = args;
const timeout = args.length === 4 ? Number(args[3]) : 10000;
if (!/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(globalName)) {
    fail(`GLOBAL is not an identifier: ${globalName}`);
}
if (!Number.isInteger(timeout) || timeout <= 0) {
    fail(`TIMEOUT_MS is not a positive integer: ${args[3]}`);
}

// Top-level `const`/`let` would not become properties of the context's global object.
const code = fs.readFileSync(file, "utf8").replace(/^(const|let)(\s)/gm, "var$2");

// microtaskMode "afterEvaluate": promise jobs a file queues drain inside each
// runInContext call, under its time limit, instead of on the host after it returns.
const ctx = vm.createContext(Object.create(null), {
    codeGeneration: { strings: false, wasm: false },
    microtaskMode: "afterEvaluate",
});

// What the data file threw, read without running any of its code: no coercion, no
// getters, no proxy traps; only an own data property `message` holding a string.
function describe(e) {
    const fallback = "data file threw a non-Error value";
    if (e === null || (typeof e !== "object" && typeof e !== "function")) return fallback;
    if (util.types.isProxy(e)) return fallback;
    const d = Object.getOwnPropertyDescriptor(e, "message");
    return d && "value" in d && typeof d.value === "string" ? d.value : fallback;
}

// displayErrors: false, because with it node decorates a thrown value by reading its
// `stack` (a getter or proxy trap) after the time limit has ended.
const run = (source, filename) =>
    vm.runInContext(source, ctx, { filename, timeout, displayErrors: false });
let json;
try {
    // V8 gives every context a console (a no-op without an inspector); remove it anyway.
    run("delete globalThis.console;", "init");
    run(code, file);
    // Serialise inside the context too, so the data's getters and toJSON run under the
    // time limit; the result is checked to be a primitive string before the host uses it.
    json = run(`JSON.stringify(${globalName})`, "serialise");
} catch (e) {
    fail(`${file}: ${describe(e)}`);
}
if (typeof json !== "string") {
    fail(`${file}: ${globalName} is not JSON-serialisable`);
}
fs.writeFileSync(out, json, "utf8");
// Exit now: no host task the data file might have scheduled gets a turn.
process.exit(0);

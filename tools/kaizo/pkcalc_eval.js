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

const ctx = vm.createContext(Object.create(null), {
    codeGeneration: { strings: false, wasm: false },
});
let json;
try {
    // V8 gives every context a console (a no-op without an inspector); remove it anyway.
    vm.runInContext("delete globalThis.console;", ctx);
    vm.runInContext(code, ctx, { filename: file, timeout });
    // Serialise inside the context too, so a getter or toJSON runs under the time limit.
    json = vm.runInContext(`JSON.stringify(${globalName})`, ctx, { timeout });
} catch (e) {
    fail(`${file}: ${e && e.name ? e.name : "Error"}: ${e && e.message ? e.message : e}`);
}
if (typeof json !== "string") {
    fail(`${file}: ${globalName} is not JSON-serialisable`);
}
fs.writeFileSync(out, json, "utf8");

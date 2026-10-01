// Run the process oracle under WASI; the codec itself remains no_std + alloc.
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';

const [modulePath, ...args] = process.argv.slice(2);
if (!modulePath) {
  throw new Error('Usage: node tools/wasm/run.mjs ORACLE.wasm [arguments...]');
}
const wasi = new WASI({
  version: 'preview1',
  args: [modulePath, ...args],
  env: Object.fromEntries(Object.entries(process.env).filter(([name]) => name.startsWith('OPUS_ORACLE_'))),
  preopens: { '/': '/' },
  returnOnExit: true,
});
const module = await WebAssembly.compile(await readFile(modulePath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exitCode = wasi.start(instance);

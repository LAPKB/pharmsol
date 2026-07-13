const encoder = new TextEncoder();
const decoder = new TextDecoder();
let compilerPromise;

function decodeBase64(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function allocUtf8(exports, value) {
  const bytes = encoder.encode(value);
  if (bytes.length === 0) {
    return { ptr: 0, len: 0 };
  }
  const ptr = Number(exports.alloc(bytes.length));
  new Uint8Array(exports.memory.buffer, ptr, bytes.length).set(bytes);
  return {
    ptr,
    len: bytes.length,
  };
}

function readJsonResult(exports) {
  const ptr = Number(exports.result_ptr());
  const len = Number(exports.result_len());
  const bytes = new Uint8Array(exports.memory.buffer, ptr, len);
  return JSON.parse(decoder.decode(bytes));
}

async function loadCompiler() {
  if (!compilerPromise) {
    compilerPromise = (async () => {
      const response = await fetch(
        new URL("./generated/browser_compile_bridge.wasm", import.meta.url),
      );
      const bytes = await response.arrayBuffer();
      const { instance } = await WebAssembly.instantiate(bytes, {});
      const exports = instance.exports;

      return {
        compile(source, modelName = "") {
          const sourceBuffer = allocUtf8(exports, source);
          const modelNameBuffer = allocUtf8(exports, modelName);
          try {
            exports.compile_model(
              sourceBuffer.ptr,
              sourceBuffer.len,
              modelNameBuffer.ptr,
              modelNameBuffer.len,
            );
            const result = readJsonResult(exports);
            if (result.ok) {
              result.wasmBytes = decodeBase64(result.wasmBase64);
              delete result.wasmBase64;
            }
            return result;
          } finally {
            if (sourceBuffer.len > 0) {
              exports.dealloc(sourceBuffer.ptr, sourceBuffer.len);
            }
            if (modelNameBuffer.len > 0) {
              exports.dealloc(modelNameBuffer.ptr, modelNameBuffer.len);
            }
          }
        },
      };
    })();
  }
  return compilerPromise;
}

self.onmessage = async (event) => {
  const { type, requestId, source, modelName } = event.data;
  if (type !== "compile") {
    return;
  }

  try {
    const compiler = await loadCompiler();
    self.postMessage({
      requestId,
      result: compiler.compile(source, modelName ?? ""),
    });
  } catch (error) {
    self.postMessage({
      requestId,
      error:
        error instanceof Error
          ? `${error.message}\n${error.stack ?? ""}`
          : String(error),
    });
  }
};

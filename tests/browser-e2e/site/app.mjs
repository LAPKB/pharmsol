const precompiledInputs = Object.freeze({
  states: [100],
  params: [5, 40],
});

const compileFlowSource = `
model = example_ode
kind = ode

params = ke, v
states = central
outputs = cp

infusion(iv) -> central

dx(central) = -ke * central

out(cp) = central / v ~ continuous()
`;

const compileFlowInputs = Object.freeze({
  states: [100],
  params: [1.2, 40],
  routes: [0],
});

const invalidCompileSource = `
model = broken
kind = ode

states = central
outputs = cp

infusion(oral) -> central

dx(central) = rate(orla)

out(cp) = central ~ continuous()
`;

const compileWorker = new Worker(
  new URL("./compile-worker.mjs", import.meta.url),
  {
    type: "module",
  },
);
const pendingCompiles = new Map();
let nextRequestId = 1;

compileWorker.onmessage = (event) => {
  const { requestId, result, error } = event.data;
  const pending = pendingCompiles.get(requestId);
  if (!pending) {
    return;
  }
  pendingCompiles.delete(requestId);
  if (error) {
    pending.reject(new Error(error));
    return;
  }
  pending.resolve(result);
};

compileWorker.onerror = (event) => {
  const error = new Error(event.message || "browser compile worker failed");
  for (const pending of pendingCompiles.values()) {
    pending.reject(error);
  }
  pendingCompiles.clear();
};

function compileInWorker(source, modelName) {
  return new Promise((resolve, reject) => {
    const requestId = nextRequestId;
    nextRequestId += 1;
    pendingCompiles.set(requestId, { resolve, reject });
    compileWorker.postMessage({
      type: "compile",
      requestId,
      source,
      modelName,
    });
  });
}

async function importLoaderFromSource(loaderSource) {
  const url = URL.createObjectURL(
    new Blob([loaderSource], { type: "text/javascript" }),
  );
  try {
    return await import(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function toPlainArray(values) {
  return Array.from(values);
}

async function runPrecompiledFlow() {
  const loader = await import(
    new URL("./generated/precompiled/direct.mjs", import.meta.url)
  );
  const model = await loader.loadPharmsolDslWasmModel(
    new URL("./generated/precompiled/direct.wasm", import.meta.url),
  );
  const session = model.createSession();
  try {
    const outputs = toPlainArray(
      session.evaluateOutputsView(precompiledInputs),
    );
    const cp = model.evaluateOutput("cp", precompiledInputs);
    return {
      modelName: model.info.name,
      outputs,
      cp,
    };
  } finally {
    session.free();
  }
}

async function runCompileFlow() {
  const warmed = await compileInWorker(compileFlowSource, "example_ode");
  if (!warmed.ok) {
    return warmed;
  }

  const compiled = await compileInWorker(compileFlowSource, "example_ode");
  if (!compiled.ok) {
    return compiled;
  }

  const loader = await importLoaderFromSource(compiled.loaderSource);
  const model = await loader.instantiatePharmsolDslWasmBytes(
    compiled.wasmBytes,
  );
  const session = model.createSession();
  try {
    const outputs = toPlainArray(
      session.evaluateOutputsView(compileFlowInputs),
    );
    const cp = session.evaluateOutput("cp", compileFlowInputs);
    return {
      ok: true,
      modelName: model.info.name,
      outputs,
      cp,
      warmCacheEntries: warmed.cacheEntries,
      cacheEntries: compiled.cacheEntries,
      metadata: compiled.metadata,
    };
  } finally {
    session.free();
  }
}

async function compileInvalidFlow() {
  return compileInWorker(invalidCompileSource, "broken");
}

window.pharmsolBrowserE2E = Object.freeze({
  runPrecompiledFlow,
  runCompileFlow,
  compileInvalidFlow,
});

document.querySelector("#status").textContent = "ready";

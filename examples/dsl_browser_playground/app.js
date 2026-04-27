const palette = ["#ff6b5f", "#7ec8ff", "#f2c66d", "#b4a6ff", "#78d4b1"];
const chartTheme = {
  empty: "#6f7785",
  grid: "rgba(255,255,255,0.08)",
  axis: "rgba(255,255,255,0.22)",
  label: "#9ba3b3",
};

const state = {
  compiledWasmBytes: null,
  compileCacheEntries: 0,
  model: null,
  parameters: [],
  covariates: [],
  subject: null,
  scientificNotes: [],
  simulateTimer: null,
};

const nodes = {
  compileButton: document.querySelector("#compile-button"),
  simulateButton: document.querySelector("#simulate-button"),
  statusLine: document.querySelector("#status-line"),
  modelName: document.querySelector("#model-name"),
  modelSource: document.querySelector("#model-source"),
  modelBadges: document.querySelector("#model-badges"),
  modelSummary: document.querySelector("#model-summary"),
  parameterControls: document.querySelector("#parameter-controls"),
  covariateControls: document.querySelector("#covariate-controls"),
  subjectId: document.querySelector("#subject-id"),
  bolusList: document.querySelector("#bolus-list"),
  infusionList: document.querySelector("#infusion-list"),
  observationList: document.querySelector("#observation-list"),
  addBolus: document.querySelector("#add-bolus"),
  addInfusion: document.querySelector("#add-infusion"),
  addObservation: document.querySelector("#add-observation"),
  diagnosticBox: document.querySelector("#diagnostic-box"),
  diagnosticOutput: document.querySelector("#diagnostic-output"),
  predictionSummary: document.querySelector("#prediction-summary"),
  predictionLegend: document.querySelector("#prediction-legend"),
  predictionChart: document.querySelector("#prediction-chart"),
  predictionRows: document.querySelector("#prediction-rows"),
  scientificNotes: document.querySelector("#scientific-notes"),
};

const compileWorker = new Worker("/compile-worker.js", { type: "module" });
const pendingCompiles = new Map();
let nextCompileRequestId = 1;

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

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return response.json();
}

function compileInWorker(source, modelName) {
  return new Promise((resolve, reject) => {
    const requestId = nextCompileRequestId;
    nextCompileRequestId += 1;
    pendingCompiles.set(requestId, { resolve, reject });
    compileWorker.postMessage({
      type: "compile",
      requestId,
      source,
      modelName,
    });
  });
}

function setStatus(message) {
  nodes.statusLine.textContent = message;
}

function scheduleSimulation() {
  if (!state.compiledWasmBytes || !state.subject) {
    return;
  }
  clearTimeout(state.simulateTimer);
  state.simulateTimer = window.setTimeout(() => {
    runSimulation().catch((error) => {
      setStatus(error.message);
    });
  }, 140);
}

function clearCompiledState() {
  state.compiledWasmBytes = null;
  state.compileCacheEntries = 0;
  state.model = null;
  state.parameters = [];
  state.covariates = [];
  state.subject = null;
  state.scientificNotes = [];
}

function resetPredictions() {
  nodes.predictionSummary.textContent = "Awaiting simulation.";
  renderScientificNotes([]);
  renderPredictionRows([]);
  renderPredictionChart([]);
}

function subjectCollection(kind) {
  if (kind === "bolus") {
    return "boluses";
  }
  if (kind === "infusion") {
    return "infusions";
  }
  return "observations";
}

function normalizeModelInfo(model) {
  return {
    name: model?.name ?? "unnamed_model",
    kind: String(model?.kind ?? "unknown").toLowerCase(),
    parameters: [...(model?.parameters ?? [])],
    covariates: (model?.covariates ?? []).map((covariate) => covariate.name),
    routes: (model?.routes ?? []).map((route) => route.name),
    outputs: (model?.outputs ?? []).map((output) => output.name),
    particles: model?.particles ?? null,
  };
}

function isBimodalKeExampleModel(model) {
  return (
    model.name === "bimodal_ke" &&
    model.parameters.length === 2 &&
    model.parameters.includes("ke") &&
    model.parameters.includes("v") &&
    model.routes.length === 1 &&
    model.routes[0]?.toLowerCase() === "iv" &&
    model.outputs.length === 1 &&
    model.outputs[0]?.toLowerCase() === "cp"
  );
}

function parameterScale(name) {
  const lower = name.toLowerCase();
  if (
    lower === "tlag" ||
    lower === "lag" ||
    lower === "f" ||
    lower === "fa" ||
    lower === "f_oral" ||
    lower.startsWith("f")
  ) {
    return "linear";
  }
  return "log";
}

function parameterControl(name) {
  const lower = name.toLowerCase();
  let value = 1;
  let min = 0.1;
  let max = 10;
  let step = 0.05;

  switch (lower) {
    case "ka":
      value = 1.2;
      min = 0.05;
      max = 4;
      step = 0.05;
      break;
    case "cl":
    case "cl_i":
      value = 5;
      min = 0.5;
      max = 20;
      step = 0.1;
      break;
    case "v":
    case "vc":
    case "vol":
      value = 40;
      min = 5;
      max = 200;
      step = 0.5;
      break;
    case "ke":
    case "ke0":
      value = 0.25;
      min = 0.01;
      max = 2;
      step = 0.01;
      break;
    case "kcp":
    case "kpc":
    case "q":
      value = 0.12;
      min = 0.01;
      max = 1;
      step = 0.01;
      break;
    case "tlag":
    case "lag":
      value = 0.25;
      min = 0;
      max = 4;
      step = 0.05;
      break;
    case "f":
    case "fa":
    case "f_oral":
      value = 0.8;
      min = 0;
      max = 1;
      step = 0.01;
      break;
    case "ske":
    case "sigma":
    case "omega":
      value = 0.05;
      min = 0.001;
      max = 0.5;
      step = 0.002;
      break;
    default:
      if (lower.startsWith("f")) {
        value = 0.8;
        min = 0;
        max = 1;
        step = 0.01;
      }
      break;
  }

  return {
    name,
    label: labelize(name),
    value,
    min,
    max,
    step,
    scale: parameterScale(name),
  };
}

function covariateControl(name) {
  const lower = name.toLowerCase();
  let value = 1;
  if (lower.includes("wt") || lower.includes("weight")) {
    value = 70;
  } else if (lower.includes("age")) {
    value = 50;
  } else if (lower.includes("ht") || lower.includes("height")) {
    value = 170;
  }

  return {
    name,
    label: labelize(name),
    value,
  };
}

function labelize(name) {
  const displayName = name.replaceAll("_", " ");
  return displayName.charAt(0).toUpperCase() + displayName.slice(1);
}

function preferredName(names, candidates) {
  for (const candidate of candidates) {
    const match = names.find(
      (name) => name.toLowerCase() === candidate.toLowerCase(),
    );
    if (match) {
      return match;
    }
  }
  return null;
}

function defaultSubjectDraft(model) {
  const output =
    preferredName(model.outputs, ["cp"]) ?? model.outputs[0] ?? "output_0";
  if (isBimodalKeExampleModel(model)) {
    const infusionRoute =
      preferredName(model.routes, ["iv"]) ?? model.routes[0] ?? "";
    return {
      id: "bimodal_ke",
      boluses: [],
      infusions: infusionRoute
        ? [{ time: 0, amount: 500, route: infusionRoute, duration: 0.5 }]
        : [],
      observations: [0.5, 1, 2, 3, 4, 6, 8].map((time) => ({
        time,
        output,
      })),
    };
  }

  const bolusRoute =
    preferredName(model.routes, ["oral", "po", "ev"]) ?? model.routes[0] ?? "";
  const infusionRoute =
    preferredName(model.routes, ["iv", "venous", "central"]) ??
    model.routes[1] ??
    model.routes[0] ??
    "";

  return {
    id: `${model.name}_subject`,
    boluses: bolusRoute ? [{ time: 0, amount: 120, route: bolusRoute }] : [],
    infusions: infusionRoute
      ? [{ time: 6, amount: 60, route: infusionRoute, duration: 2 }]
      : [],
    observations: [0.5, 1, 2, 4, 8].map((time) => ({
      time,
      output,
    })),
  };
}

function reconcileNamedSelection(names, current, fallback) {
  const preserved = current ? preferredName(names, [current]) : null;
  if (preserved) {
    return preserved;
  }
  if (fallback) {
    const preferredFallback = preferredName(names, [fallback]);
    if (preferredFallback) {
      return preferredFallback;
    }
  }
  return names[0] ?? "";
}

function reconcileSubjectDraft(model, subject) {
  const seeded = defaultSubjectDraft(model);
  if (!subject) {
    return seeded;
  }

  const bolusFallback =
    seeded.boluses[0]?.route ?? seeded.infusions[0]?.route ?? "";
  const infusionFallback =
    seeded.infusions[0]?.route ?? seeded.boluses[0]?.route ?? "";
  const observationFallback = seeded.observations[0]?.output ?? "";

  return {
    id: subject.id || seeded.id,
    boluses: subject.boluses
      .map((bolus) => {
        const route = reconcileNamedSelection(
          model.routes,
          bolus.route,
          bolusFallback,
        );
        return route ? { ...bolus, route } : null;
      })
      .filter(Boolean),
    infusions: subject.infusions
      .map((infusion) => {
        const route = reconcileNamedSelection(
          model.routes,
          infusion.route,
          infusionFallback,
        );
        return route ? { ...infusion, route } : null;
      })
      .filter(Boolean),
    observations: subject.observations
      .map((observation) => {
        const output = reconcileNamedSelection(
          model.outputs,
          observation.output,
          observationFallback,
        );
        return output ? { ...observation, output } : null;
      })
      .filter(Boolean),
  };
}

function showDiagnostics(result) {
  const diagnosticText =
    result.diagnosticReport?.rendered ||
    (result.diagnosticReport
      ? JSON.stringify(result.diagnosticReport, null, 2)
      : null) ||
    result.debug ||
    result.message ||
    "Unknown compiler error.";
  nodes.diagnosticOutput.textContent = diagnosticText;
  nodes.diagnosticBox.hidden = false;
}

function hideDiagnostics() {
  nodes.diagnosticBox.hidden = true;
  nodes.diagnosticOutput.textContent = "";
}

function renderScientificNotes(notes) {
  if (!nodes.scientificNotes) {
    return;
  }
  nodes.scientificNotes.innerHTML = "";
}

function optionMarkup(values, selected) {
  if (!values.length) {
    return '<option value="">Unavailable</option>';
  }

  return values
    .map((value) => {
      const isSelected = value === selected ? "selected" : "";
      return `<option value="${escapeHtml(value)}" ${isSelected}>${escapeHtml(value)}</option>`;
    })
    .join("");
}

function renderModelSummary() {
  if (!state.model) {
    nodes.modelBadges.innerHTML = "";
    nodes.modelSummary.textContent = "Paste a model.";
    return;
  }

  const badges = [
    `${state.model.kind}`,
    `${state.model.parameters.length} parameter${state.model.parameters.length === 1 ? "" : "s"}`,
    `${state.model.routes.length} route${state.model.routes.length === 1 ? "" : "s"}`,
    `${state.model.outputs.length} output${state.model.outputs.length === 1 ? "" : "s"}`,
  ];
  if (state.model.particles) {
    badges.push(`${state.model.particles} particles`);
  }

  nodes.modelBadges.innerHTML = badges
    .map((badge) => `<span class="badge">${escapeHtml(badge)}</span>`)
    .join("");
  nodes.modelSummary.textContent = "Compiled.";
}

function sliderMin(parameter) {
  return parameter.scale === "log" ? Math.log10(parameter.min) : parameter.min;
}

function sliderMax(parameter) {
  return parameter.scale === "log" ? Math.log10(parameter.max) : parameter.max;
}

function sliderStep(parameter) {
  const span = sliderMax(parameter) - sliderMin(parameter);
  return Math.max(span / 480, parameter.scale === "log" ? 0.002 : 0.001);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function encodeSliderValue(parameter, actualValue) {
  const clamped = clamp(actualValue, parameter.min, parameter.max);
  return parameter.scale === "log" ? Math.log10(clamped) : clamped;
}

function decodeSliderValue(parameter, sliderValue) {
  const numeric = Number(sliderValue);
  const actual = parameter.scale === "log" ? 10 ** numeric : numeric;
  return clamp(actual, parameter.min, parameter.max);
}

function parameterRangeNode(index) {
  return nodes.parameterControls.querySelector(
    `[data-parameter-range="${index}"]`,
  );
}

function parameterNumberNode(index) {
  return nodes.parameterControls.querySelector(
    `[data-parameter-number="${index}"]`,
  );
}

function parameterOutputNode(index) {
  return nodes.parameterControls.querySelector(
    `[data-parameter-output="${index}"]`,
  );
}

function formatEditableNumber(value) {
  if (!Number.isFinite(value)) {
    return "";
  }
  return String(Number.parseFloat(value.toPrecision(6)));
}

function syncParameterDom(index) {
  const parameter = state.parameters[index];
  if (!parameter) {
    return;
  }

  const range = parameterRangeNode(index);
  const number = parameterNumberNode(index);
  const output = parameterOutputNode(index);

  if (range) {
    range.value = String(encodeSliderValue(parameter, parameter.value));
  }
  if (number) {
    number.value = formatEditableNumber(parameter.value);
  }
  if (output) {
    output.textContent = formatNumber(parameter.value);
  }
}

function setParameterValue(index, value) {
  const parameter = state.parameters[index];
  if (!parameter) {
    return;
  }

  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return;
  }

  parameter.value = clamp(numeric, parameter.min, parameter.max);
  syncParameterDom(index);
  scheduleSimulation();
}

function renderParameterControls() {
  if (!state.parameters.length) {
    nodes.parameterControls.innerHTML =
      '<p class="summary-copy">No parameters.</p>';
    return;
  }

  nodes.parameterControls.innerHTML = state.parameters
    .map(
      (parameter, index) => `
      <article class="slider-card">
        <div class="slider-head">
          <div>
            <strong>${escapeHtml(parameter.label)}</strong>
            <div class="slider-meta">
              <span class="control-label">${escapeHtml(parameter.name)}</span>
              <span class="control-label slider-scale">${parameter.scale} slider</span>
            </div>
          </div>
          <output data-parameter-output="${index}">${formatNumber(parameter.value)}</output>
        </div>
        <div class="slider-body">
          <input
            type="range"
            min="${sliderMin(parameter)}"
            max="${sliderMax(parameter)}"
            step="${sliderStep(parameter)}"
            value="${encodeSliderValue(parameter, parameter.value)}"
            data-parameter-range="${index}"
          />
          <input
            type="number"
            min="${parameter.min}"
            max="${parameter.max}"
            step="${parameter.step}"
            value="${formatEditableNumber(parameter.value)}"
            data-parameter-number="${index}"
          />
        </div>
      </article>
    `,
    )
    .join("");
}

function renderCovariates() {
  if (!state.covariates.length) {
    nodes.covariateControls.innerHTML =
      '<p class="summary-copy">No covariates.</p>';
    return;
  }

  nodes.covariateControls.innerHTML = state.covariates
    .map(
      (covariate, index) => `
        <label class="field">
          <span>${escapeHtml(covariate.label)}</span>
          <input type="number" step="0.1" value="${formatEditableNumber(covariate.value)}" data-covariate-index="${index}" />
        </label>
      `,
    )
    .join("");
}

function renderSubjectEditors() {
  if (!state.subject) {
    nodes.subjectId.value = "";
    nodes.bolusList.innerHTML = "";
    nodes.infusionList.innerHTML = "";
    nodes.observationList.innerHTML = "";
    return;
  }

  nodes.subjectId.value = state.subject.id;
  nodes.bolusList.innerHTML = state.subject.boluses.length
    ? state.subject.boluses
        .map(
          (bolus, index) => `
            <article class="event-card">
              <div class="event-row">
                <strong>Bolus ${index + 1}</strong>
                <button class="remove-button" type="button" data-remove="bolus" data-index="${index}">Remove</button>
              </div>
              <div class="event-grid three">
                <label class="field">
                  <span>Time</span>
                  <input type="number" step="0.1" value="${formatEditableNumber(bolus.time)}" data-kind="bolus" data-index="${index}" data-field="time" />
                </label>
                <label class="field">
                  <span>Amount</span>
                  <input type="number" step="0.1" value="${formatEditableNumber(bolus.amount)}" data-kind="bolus" data-index="${index}" data-field="amount" />
                </label>
                <label class="field">
                  <span>Route</span>
                  <select data-kind="bolus" data-index="${index}" data-field="route">${optionMarkup(state.model?.routes ?? [], bolus.route)}</select>
                </label>
              </div>
            </article>
          `,
        )
        .join("")
    : '<p class="summary-copy">No bolus.</p>';

  nodes.infusionList.innerHTML = state.subject.infusions.length
    ? state.subject.infusions
        .map(
          (infusion, index) => `
            <article class="event-card">
              <div class="event-row">
                <strong>Infusion ${index + 1}</strong>
                <button class="remove-button" type="button" data-remove="infusion" data-index="${index}">Remove</button>
              </div>
              <div class="event-grid">
                <label class="field">
                  <span>Time</span>
                  <input type="number" step="0.1" value="${formatEditableNumber(infusion.time)}" data-kind="infusion" data-index="${index}" data-field="time" />
                </label>
                <label class="field">
                  <span>Amount</span>
                  <input type="number" step="0.1" value="${formatEditableNumber(infusion.amount)}" data-kind="infusion" data-index="${index}" data-field="amount" />
                </label>
                <label class="field">
                  <span>Duration</span>
                  <input type="number" step="0.1" value="${formatEditableNumber(infusion.duration)}" data-kind="infusion" data-index="${index}" data-field="duration" />
                </label>
                <label class="field">
                  <span>Route</span>
                  <select data-kind="infusion" data-index="${index}" data-field="route">${optionMarkup(state.model?.routes ?? [], infusion.route)}</select>
                </label>
              </div>
            </article>
          `,
        )
        .join("")
    : '<p class="summary-copy">No infusions.</p>';

  nodes.observationList.innerHTML = state.subject.observations.length
    ? state.subject.observations
        .map(
          (observation, index) => `
            <article class="event-card">
              <div class="event-row">
                <strong>Observation ${index + 1}</strong>
                <button class="remove-button" type="button" data-remove="observation" data-index="${index}">Remove</button>
              </div>
              <div class="event-grid three">
                <label class="field">
                  <span>Time</span>
                  <input type="number" step="0.1" value="${formatEditableNumber(observation.time)}" data-kind="observation" data-index="${index}" data-field="time" />
                </label>
                <label class="field">
                  <span>Output</span>
                  <select data-kind="observation" data-index="${index}" data-field="output">${optionMarkup(state.model?.outputs ?? [], observation.output)}</select>
                </label>
                <div class="field">
                  <span>Type</span>
                  <input type="text" value="Missing observation" disabled />
                </div>
              </div>
            </article>
          `,
        )
        .join("")
    : '<p class="summary-copy">No observations.</p>';
}

function renderPredictionRows(predictions) {
  if (!predictions.length) {
    nodes.predictionRows.innerHTML = '<tr><td colspan="3">No data.</td></tr>';
    return;
  }

  nodes.predictionRows.innerHTML = predictions
    .map(
      (prediction) => `
        <tr>
          <td>${formatNumber(prediction.time)}</td>
          <td>${escapeHtml(prediction.output)}</td>
          <td>${formatNumber(prediction.value)}</td>
        </tr>
      `,
    )
    .join("");
}

function niceStep(rawStep) {
  if (!Number.isFinite(rawStep) || rawStep <= 0) {
    return 1;
  }

  const exponent = Math.floor(Math.log10(rawStep));
  const fraction = rawStep / 10 ** exponent;
  let niceFraction = 10;
  if (fraction <= 1) {
    niceFraction = 1;
  } else if (fraction <= 2) {
    niceFraction = 2;
  } else if (fraction <= 5) {
    niceFraction = 5;
  }
  return niceFraction * 10 ** exponent;
}

function expandDomain(
  minValue,
  maxValue,
  { includeZero = false, floorAtZero = false } = {},
) {
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return [0, 1];
  }

  let min = minValue;
  let max = maxValue;
  if (includeZero) {
    min = Math.min(min, 0);
    max = Math.max(max, 0);
  }

  if (min === max) {
    const delta = min === 0 ? 1 : Math.abs(min) * 0.18;
    min -= delta;
    max += delta;
  } else {
    const padding = (max - min) * 0.08;
    min -= padding;
    max += padding;
  }

  if (floorAtZero && minValue >= 0) {
    min = 0;
  }

  return [min, max];
}

function createTicks(minValue, maxValue, count = 5) {
  const step = niceStep((maxValue - minValue) / Math.max(count - 1, 1));
  const start = Math.ceil(minValue / step) * step;
  const ticks = [];

  for (let value = start; value <= maxValue + step * 0.5; value += step) {
    ticks.push(Number.parseFloat(value.toPrecision(12)));
  }

  if (!ticks.length) {
    ticks.push(minValue, maxValue);
  }

  return ticks;
}

function formatAxisNumber(value) {
  const absolute = Math.abs(value);
  if (absolute === 0) {
    return "0";
  }
  if (absolute >= 1000 || absolute < 0.01) {
    return value.toExponential(1).replace("e+", "e");
  }
  return formatNumber(value);
}

function renderPredictionChart(predictions) {
  if (!predictions.length) {
    nodes.predictionChart.innerHTML = `<text x="24" y="40" fill="${chartTheme.empty}">No run.</text>`;
    nodes.predictionLegend.innerHTML = "";
    return;
  }

  const groups = new Map();
  for (const prediction of predictions) {
    const bucket = groups.get(prediction.output) ?? [];
    bucket.push(prediction);
    groups.set(prediction.output, bucket);
  }

  const allTimes = predictions.map((prediction) => prediction.time);
  const allValues = predictions.map((prediction) => prediction.value);
  const [minX, maxX] = expandDomain(
    Math.min(...allTimes),
    Math.max(...allTimes),
    {
      includeZero: true,
      floorAtZero: true,
    },
  );
  const [minY, maxY] = expandDomain(
    Math.min(...allValues),
    Math.max(...allValues),
    {
      includeZero: true,
      floorAtZero: Math.min(...allValues) >= 0,
    },
  );
  const xTicks = createTicks(minX, maxX, 6);
  const yTicks = createTicks(minY, maxY, 6);
  const chartWidth = 760;
  const chartHeight = 280;
  const padding = { left: 70, right: 26, top: 22, bottom: 44 };
  const plotWidth = chartWidth - padding.left - padding.right;
  const plotHeight = chartHeight - padding.top - padding.bottom;

  const xScale = (value) =>
    padding.left + ((value - minX) / (maxX - minX)) * plotWidth;
  const yScale = (value) =>
    chartHeight -
    padding.bottom -
    ((value - minY) / (maxY - minY)) * plotHeight;

  const xGrid = xTicks
    .map((tick) => {
      const x = xScale(tick);
      return `
        <line x1="${x}" y1="${padding.top}" x2="${x}" y2="${chartHeight - padding.bottom}" stroke="${chartTheme.grid}" stroke-dasharray="3 6" />
        <text x="${x}" y="${chartHeight - padding.bottom + 20}" fill="${chartTheme.label}" font-size="12" text-anchor="middle">${escapeHtml(formatAxisNumber(tick))}</text>
      `;
    })
    .join("");

  const yGrid = yTicks
    .map((tick) => {
      const y = yScale(tick);
      return `
        <line x1="${padding.left}" y1="${y}" x2="${chartWidth - padding.right}" y2="${y}" stroke="${chartTheme.grid}" stroke-dasharray="3 6" />
        <text x="${padding.left - 10}" y="${y + 4}" fill="${chartTheme.label}" font-size="12" text-anchor="end">${escapeHtml(formatAxisNumber(tick))}</text>
      `;
    })
    .join("");

  const seriesMarkup = Array.from(groups.entries())
    .map(([output, rows], index) => {
      const color = palette[index % palette.length];
      const sorted = [...rows].sort((left, right) => left.time - right.time);
      const polyline = sorted
        .map((row) => `${xScale(row.time)},${yScale(row.value)}`)
        .join(" ");
      const dots = sorted
        .map(
          (row) =>
            `<circle cx="${xScale(row.time)}" cy="${yScale(row.value)}" r="3.5" fill="${color}" />`,
        )
        .join("");
      return `
        <g clip-path="url(#prediction-clip)">
          <polyline fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${polyline}" />
          ${dots}
        </g>
      `;
    })
    .join("");

  nodes.predictionChart.innerHTML = `
    <defs>
      <clipPath id="prediction-clip">
        <rect x="${padding.left}" y="${padding.top}" width="${plotWidth}" height="${plotHeight}" />
      </clipPath>
    </defs>
    <rect x="0" y="0" width="${chartWidth}" height="${chartHeight}" fill="transparent"></rect>
    ${xGrid}
    ${yGrid}
    <line x1="${padding.left}" y1="${chartHeight - padding.bottom}" x2="${chartWidth - padding.right}" y2="${chartHeight - padding.bottom}" stroke="${chartTheme.axis}" />
    <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${chartHeight - padding.bottom}" stroke="${chartTheme.axis}" />
    ${seriesMarkup}
    <text x="${padding.left}" y="${chartHeight - 10}" fill="${chartTheme.label}" font-size="12">Time</text>
    <text x="${chartWidth - 18}" y="${padding.top + 12}" fill="${chartTheme.label}" font-size="12" text-anchor="end">Prediction</text>
  `;

  if (groups.size <= 1) {
    nodes.predictionLegend.innerHTML = "";
  } else {
    nodes.predictionLegend.innerHTML = Array.from(groups.keys())
      .map(
        (output, index) => `
          <span class="legend-chip">
            <span class="legend-swatch" style="background:${palette[index % palette.length]}"></span>
            ${escapeHtml(output)}
          </span>
        `,
      )
      .join("");
  }
}

function updatePredictions(result) {
  state.scientificNotes = result.scientificNotes || [];
  const predictionCount = result.predictions?.length ?? 0;
  nodes.predictionSummary.textContent =
    result.predictionKind === "particles"
      ? `${predictionCount} mean predictions`
      : `${predictionCount} predictions`;
  renderScientificNotes(state.scientificNotes);
  renderPredictionRows(result.predictions || []);
  renderPredictionChart(result.predictions || []);
}

function collectParameterValues() {
  return Object.fromEntries(
    state.parameters.map((parameter) => [
      parameter.name,
      Number(parameter.value),
    ]),
  );
}

function collectCovariateValues() {
  return Object.fromEntries(
    state.covariates.map((covariate) => [
      covariate.name,
      Number(covariate.value),
    ]),
  );
}

async function compileCurrentModel() {
  setStatus("Compiling…");
  const previousSubject = state.subject;
  const result = await compileInWorker(
    nodes.modelSource.value,
    nodes.modelName.value.trim(),
  );

  if (!result.ok) {
    clearCompiledState();
    showDiagnostics(result);
    renderModelSummary();
    renderParameterControls();
    renderCovariates();
    renderSubjectEditors();
    resetPredictions();
    setStatus(result.message || "Compile failed.");
    return;
  }

  hideDiagnostics();
  state.compiledWasmBytes = result.wasmBytes;
  state.compileCacheEntries = result.cacheEntries ?? 0;
  state.model = normalizeModelInfo(result.metadata?.model);
  state.parameters = state.model.parameters.map(parameterControl);
  state.covariates = state.model.covariates.map(covariateControl);
  state.subject = reconcileSubjectDraft(state.model, previousSubject);
  renderModelSummary();
  renderParameterControls();
  renderCovariates();
  renderSubjectEditors();
  setStatus("Compiled.");
  await runSimulation();
}

async function runSimulation() {
  if (!state.compiledWasmBytes || !state.subject) {
    return;
  }

  setStatus("Simulating…");
  state.subject.id = nodes.subjectId.value.trim() || state.subject.id;

  const result = await requestJson("/api/simulate", {
    method: "POST",
    body: JSON.stringify({
      wasmBytes: Array.from(state.compiledWasmBytes),
      parameterValues: collectParameterValues(),
      covariateValues: collectCovariateValues(),
      subject: state.subject,
    }),
  });

  if (!result.ok) {
    setStatus(result.message || "Simulation failed.");
    return;
  }

  updatePredictions(result);
  setStatus("Ready.");
}

function addEvent(kind) {
  if (!state.model || !state.subject) {
    return;
  }

  const preferredRoute = state.model.routes[0] || "";
  const preferredOutput = state.model.outputs[0] || "";
  if (kind === "bolus") {
    state.subject.boluses.push({ time: 0, amount: 100, route: preferredRoute });
  }
  if (kind === "infusion") {
    state.subject.infusions.push({
      time: 0,
      amount: 50,
      route: preferredRoute,
      duration: 1,
    });
  }
  if (kind === "observation") {
    state.subject.observations.push({ time: 1, output: preferredOutput });
  }
  renderSubjectEditors();
  scheduleSimulation();
}

function removeEvent(kind, index) {
  if (!state.subject) {
    return;
  }
  const key = subjectCollection(kind);
  state.subject[key].splice(index, 1);
  renderSubjectEditors();
  scheduleSimulation();
}

function updateEvent(kind, index, field, value) {
  if (!state.subject) {
    return;
  }
  const key = subjectCollection(kind);
  if (field === "route" || field === "output") {
    state.subject[key][index][field] = value;
  } else {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return;
    }
    state.subject[key][index][field] = numeric;
  }
  scheduleSimulation();
}

function bindEvents() {
  nodes.compileButton.addEventListener("click", () => {
    compileCurrentModel().catch((error) => {
      setStatus(error.message);
    });
  });

  nodes.simulateButton.addEventListener("click", () => {
    runSimulation().catch((error) => {
      setStatus(error.message);
    });
  });

  nodes.addBolus.addEventListener("click", () => addEvent("bolus"));
  nodes.addInfusion.addEventListener("click", () => addEvent("infusion"));
  nodes.addObservation.addEventListener("click", () => addEvent("observation"));

  nodes.parameterControls.addEventListener("input", (event) => {
    const target = event.target;
    const rangeIndex = target.dataset.parameterRange;
    const numberIndex = target.dataset.parameterNumber;
    const index = Number(rangeIndex ?? numberIndex);
    if (Number.isNaN(index)) {
      return;
    }

    if (rangeIndex !== undefined) {
      setParameterValue(
        index,
        decodeSliderValue(state.parameters[index], target.value),
      );
      return;
    }

    setParameterValue(index, target.value);
  });

  nodes.parameterControls.addEventListener("change", (event) => {
    const target = event.target;
    const index = Number(target.dataset.parameterNumber);
    if (Number.isNaN(index)) {
      return;
    }
    syncParameterDom(index);
  });

  nodes.covariateControls.addEventListener("input", (event) => {
    const target = event.target;
    const index = Number(target.dataset.covariateIndex);
    if (Number.isNaN(index)) {
      return;
    }

    const numeric = Number(target.value);
    if (!Number.isFinite(numeric)) {
      return;
    }

    state.covariates[index].value = numeric;
    scheduleSimulation();
  });

  for (const container of [
    nodes.bolusList,
    nodes.infusionList,
    nodes.observationList,
  ]) {
    container.addEventListener("click", (event) => {
      const target = event.target;
      if (!target.dataset.remove) {
        return;
      }
      removeEvent(target.dataset.remove, Number(target.dataset.index));
    });

    container.addEventListener("input", (event) => {
      const target = event.target;
      if (!target.dataset.kind) {
        return;
      }
      updateEvent(
        target.dataset.kind,
        Number(target.dataset.index),
        target.dataset.field,
        target.value,
      );
    });

    container.addEventListener("change", (event) => {
      const target = event.target;
      if (!target.dataset.kind) {
        return;
      }
      updateEvent(
        target.dataset.kind,
        Number(target.dataset.index),
        target.dataset.field,
        target.value,
      );
    });
  }

  nodes.subjectId.addEventListener("input", () => {
    if (!state.subject) {
      return;
    }
    state.subject.id = nodes.subjectId.value;
    scheduleSimulation();
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatNumber(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "nan";
  }
  return numeric
    .toFixed(Math.abs(numeric) >= 100 ? 2 : 3)
    .replace(/\.0+$/, "")
    .replace(/(\.\d*[1-9])0+$/, "$1");
}

async function boot() {
  bindEvents();
  const defaults = await requestJson("/api/defaults");
  nodes.modelName.value = defaults.defaultModelName;
  nodes.modelSource.value = defaults.defaultModelSource;
  await compileCurrentModel();
}

boot().catch((error) => {
  setStatus(error.message);
});

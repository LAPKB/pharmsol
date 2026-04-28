import assert from "node:assert/strict";
import fs from "node:fs";
import { execFileSync, spawn } from "node:child_process";
import { once } from "node:events";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..", "..");
const siteDir = path.join(scriptDir, "site");
const generatedDir = path.join(siteDir, "generated");
const precompiledDir = path.join(generatedDir, "precompiled");
const bridgeTargetDir = path.join(
  repoRoot,
  "target",
  "browser-e2e-compile-bridge",
);
const bridgeWasmPath = path.join(
  bridgeTargetDir,
  "wasm32-unknown-unknown",
  "release",
  "browser_compile_bridge.wasm",
);
const port = Number(process.env.PHARMSOL_BROWSER_E2E_PORT ?? 4173);
const baseUrl = `http://127.0.0.1:${port}`;

function run(command, args, options = {}) {
  execFileSync(command, args, {
    cwd: repoRoot,
    stdio: "inherit",
    ...options,
  });
}

function prepareFixtures() {
  fs.mkdirSync(precompiledDir, { recursive: true });

  run("cargo", [
    "build",
    "--manifest-path",
    "browser-compile-bridge/Cargo.toml",
    "--target",
    "wasm32-unknown-unknown",
    "--release",
    "--target-dir",
    "target/browser-e2e-compile-bridge",
  ]);

  fs.copyFileSync(
    bridgeWasmPath,
    path.join(generatedDir, "browser_compile_bridge.wasm"),
  );

  run(
    "cargo",
    [
      "test",
      "--lib",
      "direct_browser_smoke_bundle_is_emitted_when_requested",
      "--features",
      "dsl-jit dsl-wasm",
    ],
    {
      env: {
        ...process.env,
        PHARMSOL_DSL_BROWSER_SMOKE_DIR: precompiledDir,
      },
    },
  );
}

async function waitForServer(server) {
  let stderr = "";
  server.stderr.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      const response = await fetch(baseUrl);
      if (response.ok) {
        return;
      }
    } catch {
      if (server.exitCode !== null) {
        break;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }

  throw new Error(
    `browser E2E server did not start at ${baseUrl}${stderr ? `\n${stderr}` : ""}`,
  );
}

async function main() {
  prepareFixtures();

  const server = spawn(
    "python3",
    ["-m", "http.server", String(port), "-d", siteDir],
    {
      cwd: repoRoot,
      stdio: ["ignore", "ignore", "pipe"],
    },
  );

  try {
    await waitForServer(server);

    const browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    const consoleErrors = [];
    page.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(message.text());
      }
    });

    await page.goto(baseUrl);
    await page.waitForFunction(() => Boolean(window.pharmsolBrowserE2E));

    const precompiled = await page.evaluate(() =>
      window.pharmsolBrowserE2E.runPrecompiledFlow(),
    );
    assert.equal(precompiled.modelName, "direct_w03_minimal");
    assert.equal(precompiled.outputs.length, 1);
    assert.ok(
      Math.abs(precompiled.cp - 122.5) < 1e-12,
      JSON.stringify(precompiled),
    );
    assert.ok(Math.abs(precompiled.cp - precompiled.outputs[0]) < 1e-12);

    const compiled = await page.evaluate(() =>
      window.pharmsolBrowserE2E.runCompileFlow(),
    );
    assert.equal(compiled.ok, true, JSON.stringify(compiled));
    assert.equal(compiled.modelName, "example_ode");
    assert.equal(compiled.metadata.model.name, "example_ode");
    assert.equal(compiled.outputs.length, 1);
    assert.equal(compiled.warmCacheEntries, 1, JSON.stringify(compiled));
    assert.equal(compiled.cacheEntries, 1, JSON.stringify(compiled));
    assert.ok(Math.abs(compiled.cp - 2.5) < 1e-12, JSON.stringify(compiled));
    assert.ok(Math.abs(compiled.cp - compiled.outputs[0]) < 1e-12);

    const invalid = await page.evaluate(() =>
      window.pharmsolBrowserE2E.compileInvalidFlow(),
    );
    assert.equal(invalid.ok, false, JSON.stringify(invalid));
    assert.ok(
      invalid.message.includes("unknown route `orla`"),
      JSON.stringify(invalid),
    );
    assert.equal(invalid.diagnosticReport.source.name, "inline.dsl");
    assert.equal(invalid.diagnosticReport.diagnostics[0].phase, "semantic");
    assert.equal(invalid.diagnosticReport.diagnostics[0].code, "DSL2000");
    assert.equal(invalid.cacheEntries, 1, JSON.stringify(invalid));
    assert.ok(
      invalid.diagnosticReport.diagnostics[0].suggestions.some((suggestion) =>
        suggestion.message.includes("did you mean `oral`?"),
      ),
      JSON.stringify(invalid),
    );

    assert.deepEqual(consoleErrors, []);

    await browser.close();
    console.log("browser E2E passed");
  } finally {
    server.kill("SIGTERM");
    await once(server, "exit").catch(() => {});
  }
}

main().catch((error) => {
  console.error(error.stack ?? String(error));
  process.exit(1);
});

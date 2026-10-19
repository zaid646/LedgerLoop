import { chromium } from "playwright";
import AxeBuilder from "@axe-core/playwright";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const url = process.env.A11Y_URL ?? "http://localhost:8501";
const label = process.env.A11Y_LABEL ?? "before";
const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(here, "..", process.env.A11Y_OUT ?? "evidence/a11y");
await mkdir(outDir, { recursive: true });

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
const page = await context.newPage();
await page.goto(url, { waitUntil: "domcontentloaded" });
await page.locator("h1").first().waitFor({ timeout: 45_000 });
await page.waitForTimeout(3500);

const results = await new AxeBuilder({ page }).analyze();
const summary = {
  url,
  label,
  checkedAt: new Date().toISOString(),
  violations: results.violations.map((v) => ({
    id: v.id,
    impact: v.impact,
    help: v.help,
    description: v.description,
    tags: v.tags,
    nodes: v.nodes.map((n) => ({
      target: n.target,
      html: n.html,
      failureSummary: n.failureSummary,
    })),
  })),
  passes: results.passes.length,
  incomplete: results.incomplete.length,
  inapplicable: results.inapplicable.length,
  counts: {
    serious: results.violations.filter((v) => v.impact === "serious").length,
    critical: results.violations.filter((v) => v.impact === "critical").length,
    moderate: results.violations.filter((v) => v.impact === "moderate").length,
    minor: results.violations.filter((v) => v.impact === "minor").length,
  },
};

await writeFile(
  path.join(outDir, `${label}.json`),
  JSON.stringify(summary, null, 2) + "\n",
);
await page.screenshot({
  path: path.join(outDir, `${label}.png`),
  fullPage: true,
});

console.log(
  JSON.stringify(
    {
      label,
      violations: summary.violations.length,
      counts: summary.counts,
      ids: summary.violations.map((v) => `${v.id} (${v.impact})`),
      passes: summary.passes,
    },
    null,
    2,
  ),
);
await browser.close();

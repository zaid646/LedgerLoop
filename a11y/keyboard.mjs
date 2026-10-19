import { chromium } from "playwright";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const url = process.env.A11Y_URL ?? "http://localhost:8501";
const here = path.dirname(fileURLToPath(import.meta.url));
const out = path.resolve(here, "..", "evidence/a11y/keyboard.json");

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
const page = await context.newPage();

async function load() {
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.locator("h1").first().waitFor({ timeout: 45_000 });
  await page.waitForTimeout(3000);
}

await load();
const headings = await page
  .locator("h1,h2,h3,h4,h5,h6")
  .evaluateAll((els) =>
    els.map((el) => ({ level: el.tagName, text: (el.innerText || "").trim().slice(0, 80) })),
  );

await page.keyboard.press("Tab");
const firstStop = await page.evaluate(() => {
  const el = document.activeElement;
  const style = getComputedStyle(el);
  return {
    tag: el?.tagName,
    text: (el?.innerText || el?.textContent || "").trim(),
    href: el?.getAttribute?.("href"),
    outline: style.outline,
    left: style.left,
  };
});
await page.keyboard.press("Enter");
await page.waitForTimeout(400);
const skipTarget = await page.evaluate(() => ({
  hash: location.hash,
  activeId: document.activeElement?.id,
}));

await load();
const search = page.getByRole("textbox", { name: "Search expenses" });
const searchCount = await search.count();
await search.focus();
const searchFocus = await search.evaluate((el) => {
  const style = getComputedStyle(el);
  return { outline: style.outline, ariaLabel: el.getAttribute("aria-label") };
});

await load();
const tabStops = [];
for (let index = 0; index < 10; index += 1) {
  await page.keyboard.press("Tab");
  tabStops.push(
    await page.evaluate(() => {
      const el = document.activeElement;
      return {
        tag: el?.tagName,
        name:
          el?.getAttribute?.("aria-label") ||
          (el?.innerText || el?.textContent || "").trim().slice(0, 60),
      };
    }),
  );
}

const result = {
  url,
  checkedAt: new Date().toISOString(),
  headings,
  skipLink: { firstStop, skipTarget },
  search: { accessibleNameCount: searchCount, focus: searchFocus },
  tabStops,
};
await writeFile(out, JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result, null, 2));
await browser.close();

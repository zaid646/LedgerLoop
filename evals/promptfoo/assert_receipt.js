// PromptFoo assertion: fuzzy receipt-field equality.
// Compares the model's JSON against gold vars (merchant / total / date).
function norm(s) {
  return (s === null || s === undefined ? '' : String(s))
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function ymd(s) {
  const m = (s || '').match(/(\d{1,4})[\/\-.](\d{1,2})[\/\-.](\d{1,4})/);
  if (!m) return null;
  const a = +m[1], b = +m[2], c = +m[3];
  let y, mo, d;
  if (m[1].length === 4) { y = a; mo = b; d = c; }
  else { d = a; mo = b; y = c < 100 ? c + 2000 : c; }
  return `${y}-${String(mo).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
}

module.exports = (output, context) => {
  const v = (context && context.vars) || {};
  let data;
  try {
    data = typeof output === 'string' ? JSON.parse(output) : output;
  } catch (e) {
    return { pass: false, score: 0, reason: 'output is not valid JSON' };
  }

  const results = {};

  if (v.merchant) {
    const p = norm(data.merchant), g = norm(v.merchant);
    results.merchant = !!p && (p === g || p.includes(g) || g.includes(p));
  }
  if (v.total) {
    const p = parseFloat(data.total_amount), g = parseFloat(v.total);
    results.total = !isNaN(p) && !isNaN(g) && Math.abs(p - g) <= 0.01;
  }
  if (v.date) {
    results.date = ymd(data.transaction_date) !== null && ymd(data.transaction_date) === ymd(v.date);
  }

  const checks = Object.values(results);
  const passed = checks.length === 0 ? true : checks.every(Boolean);
  const score = checks.length ? checks.filter(Boolean).length / checks.length : 1;
  return { pass: passed, score, reason: JSON.stringify(results) };
};

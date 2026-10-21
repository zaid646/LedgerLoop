import assert from 'node:assert/strict';
import test from 'node:test';
import {
  decisionStatus,
  diffFields,
  requiresCorrection,
  shouldQueue,
  validateDecision,
} from '../src/review.js';

test('routes requiring humans are queued', () => {
  assert.equal(shouldQueue('human_review'), true);
  assert.equal(shouldQueue('reject_escalate'), true);
  assert.equal(shouldQueue('auto_approve'), false);
  assert.equal(shouldQueue(undefined), false);
});

test('decision status mapping is fail-safe', () => {
  assert.equal(decisionStatus('approve'), 'approved');
  assert.equal(decisionStatus('correct'), 'corrected');
  assert.equal(decisionStatus('reject'), 'rejected');
  assert.equal(requiresCorrection('correct'), true);
  assert.equal(requiresCorrection('approve'), false);
});

test('diffFields records original and corrected values per changed field', () => {
  const diffs = diffFields(
    { merchant: 'ACME', total_amount: 12.5, transaction_date: '2026-10-01' },
    { merchant: 'ACME STORE', total_amount: 12.5, transaction_date: '2026-10-02' },
  );
  assert.deepEqual(
    diffs.map((d) => d.field).sort(),
    ['merchant', 'transaction_date'],
  );
  const merchant = diffs.find((d) => d.field === 'merchant');
  assert.deepEqual(merchant, {
    field: 'merchant',
    original: 'ACME',
    corrected: 'ACME STORE',
  });
});

test('diffFields returns nothing for an approval as-is', () => {
  assert.deepEqual(diffFields({ merchant: 'ACME' }, { merchant: 'ACME' }), []);
});

test('validateDecision enforces corrections', () => {
  assert.equal(validateDecision('correct', undefined), 'corrected values are required when decision=correct');
  assert.equal(validateDecision('correct', {}), 'corrected values are required when decision=correct');
  assert.equal(validateDecision('correct', { merchant: 'ACME' }), null);
  assert.equal(validateDecision('approve', undefined), null);
  assert.equal(
    validateDecision('approve', { merchant: 'ACME' }),
    'corrected values are only accepted with decision=correct',
  );
  assert.equal(validateDecision('reject', undefined), null);
});

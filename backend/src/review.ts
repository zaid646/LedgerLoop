export type ReviewDecision = 'approve' | 'correct' | 'reject';
export type ReviewStatus = 'pending' | 'approved' | 'corrected' | 'rejected';

export interface FieldCorrection {
  field: string;
  original: unknown;
  corrected: unknown;
}

export function decisionStatus(decision: ReviewDecision): ReviewStatus {
  if (decision === 'approve') return 'approved';
  if (decision === 'correct') return 'corrected';
  return 'rejected';
}

export function shouldQueue(route: string | undefined): boolean {
  return route === 'human_review' || route === 'reject_escalate';
}

export function requiresCorrection(decision: ReviewDecision): boolean {
  return decision === 'correct';
}

export function diffFields(
  original: Record<string, unknown>,
  corrected: Record<string, unknown>,
): FieldCorrection[] {
  const fields = new Set([...Object.keys(original), ...Object.keys(corrected)]);
  const diffs: FieldCorrection[] = [];
  for (const field of fields) {
    const before = original[field];
    const after = corrected[field];
    if (JSON.stringify(before) !== JSON.stringify(after)) {
      diffs.push({ field, original: before, corrected: after });
    }
  }
  return diffs;
}

export function validateDecision(
  decision: ReviewDecision,
  corrected: Record<string, unknown> | undefined,
): string | null {
  if (requiresCorrection(decision) && (!corrected || Object.keys(corrected).length === 0)) {
    return 'corrected values are required when decision=correct';
  }
  if (decision === 'approve' && corrected && Object.keys(corrected).length > 0) {
    return 'corrected values are only accepted with decision=correct';
  }
  return null;
}

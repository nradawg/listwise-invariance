/**
 * Every failure in this module is a refusal, not a fallback.
 *
 * A reranker that guesses when its inputs are ambiguous produces a ranking
 * that looks fine and is silently a function of something the caller never
 * intended (the array order, the pass budget, a tokenizer quirk). The whole
 * point of this package is that the output is a function of the item set and
 * the judge, so anything that would make it a function of something else has
 * to stop the call rather than be papered over.
 */

export type ListwiseErrorCode =
  /** An option value is missing, out of range, or contradicts another option. */
  | 'badConfig'
  /** The item list is malformed: duplicate ids, empty ids, mixed scoring. */
  | 'badItems'
  /** The window schedule cannot move some item to where it needs to go. */
  | 'strandedSchedule'
  /** The judge returned something that is not a permutation of its window. */
  | 'badJudgeOutput'
  /** The pass sequence is periodic and the caller asked to be told. */
  | 'oscillation'
  /** maxPasses ran out before the pass sequence could be shown to be stable. */
  | 'notConverged';

export class ListwiseError extends Error {
  constructor(
    message: string,
    readonly code: ListwiseErrorCode,
  ) {
    super(message);
    this.name = 'ListwiseError';
  }
}

/**
 * Render a position list for an error message.
 *
 * Naming only a count ("4 positions are stranded") forces the reader to redo
 * the analysis by hand to find out which ones, so the actual positions are
 * always named and the list is capped instead of summarised.
 */
export function listPositions(positions: readonly number[], cap = 8): string {
  if (positions.length === 0) return 'none';
  const shown = positions.slice(0, cap).join(', ');
  const rest = positions.length - cap;
  return rest > 0 ? `${shown} (and ${rest} more)` : shown;
}

/** Render an id for an error message with the empty string made visible. */
export function showId(id: unknown): string {
  if (typeof id !== 'string') return `${typeof id} ${String(id)}`;
  return id === '' ? '"" (the empty string)' : `"${id}"`;
}

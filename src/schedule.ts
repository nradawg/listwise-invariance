/**
 * The window schedule, and what it can and cannot do.
 *
 * Listwise reranking with a sliding window is usually described as a sort. It
 * is not. It is ONE PASS of a bubble sort with a wide comparator: the window
 * slides from the bottom of the list to the top, the judge reorders whatever
 * is inside it, and the result is written back in place. That single pass is
 * enough to float the single best item to the top only when consecutive
 * windows overlap, and it is enough for nothing else.
 *
 * Two failures follow from the schedule alone, with no reference to the
 * documents or to the judge:
 *
 *   GAPS. The obvious loop is
 *
 *       for (let start = n - w; start >= 0; start -= stride) ...
 *
 *   and it silently stops short whenever (n - w) is not a multiple of stride.
 *   With n = 10, w = 3, stride = 2 the starts are 7, 5, 3, 1 and position 0 is
 *   never inside any window. The item the first stage happened to put first is
 *   now first forever, and nothing else can ever reach the top, on any number
 *   of passes.
 *
 *   DISJOINT WINDOWS. When stride is greater than or equal to the window size
 *   the windows tile the list without sharing a position. An item can then
 *   only ever be reordered against the other items in its own window. The best
 *   document in the corpus, sitting at position 99 out of 100 with a window of
 *   4 and a stride of 4, can reach position 96 and never position 0. Running
 *   the pass again repeats the same schedule and changes nothing.
 *
 * Both are properties of (n, windowSize, stride) and are decidable before a
 * single judge call is made, which is what this file does. The reachability
 * question it answers is: assuming an adversarial judge that may place any
 * item anywhere within a window it is shown, which positions can an item
 * starting at position p occupy after one pass, and after unboundedly many?
 */

import { ListwiseError, listPositions } from './errors.js';

export interface WindowRange {
  /** Inclusive. */
  readonly start: number;
  /** Exclusive. */
  readonly end: number;
}

export interface ScheduleAnalysis {
  readonly n: number;
  readonly topK: number;
  /** In processing order: first window processed is first in the array. */
  readonly windows: readonly WindowRange[];
  /**
   * Maximal position ranges connected by window overlap. An item can move
   * anywhere inside its own component given enough passes, and can never
   * leave it, so a schedule with more than one component partitions the list
   * into groups that never compete with each other.
   */
  readonly components: readonly WindowRange[];
  /** Positions no window ever touches. */
  readonly uncovered: readonly number[];
  /** Positions at or below topK that can never reach the top topK. */
  readonly strandedFromTopK: readonly number[];
  /** Positions inside the top topK that can never be pushed out of it. */
  readonly frozenInTopK: readonly number[];
  /**
   * Passes needed before the worst-placed item outside the cutoff is even
   * able to enter the top topK. Running fewer passes than this makes the
   * cutoff a function of the first-stage order rather than of the judge.
   */
  readonly passesToTopK: number;
  /** The position that takes passesToTopK passes to reach the cutoff. */
  readonly slowestEntrant: number | null;
  /** Passes needed before an item seeded inside the cutoff can leave it. */
  readonly passesToLeaveTopK: number;
  /** The position that takes passesToLeaveTopK passes to leave the cutoff. */
  readonly slowestLeaver: number | null;
  /** Passes after which no further position becomes reachable for anyone. */
  readonly mixingPasses: number;
}

/**
 * Build the bottom-up sliding schedule, clamping the final window to start at
 * position 0.
 *
 * The clamp is the whole difference between this and the four-line loop. It
 * costs one extra, mostly overlapping, window and it is the only thing that
 * guarantees the top of the list is inside the merge at all.
 */
export function buildSlidingWindows(n: number, windowSize: number, stride: number): WindowRange[] {
  if (n <= 0) return [];
  if (windowSize >= n) return [{ start: 0, end: n }];
  const windows: WindowRange[] = [];
  let start = n - windowSize;
  for (;;) {
    windows.push({ start, end: start + windowSize });
    if (start === 0) break;
    start = Math.max(0, start - stride);
  }
  return windows;
}

/** Merge windows into overlap-connected components. */
function componentsOf(windows: readonly WindowRange[]): WindowRange[] {
  const sorted = [...windows].sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: WindowRange[] = [];
  for (const w of sorted) {
    const last = merged[merged.length - 1];
    // Strict inequality on purpose. [0,3) and [3,6) are adjacent but share no
    // position, so no item can pass between them and they are two components,
    // not one. Merging on `w.start <= last.end` would report exactly the
    // stride-equals-window case as safe.
    if (last !== undefined && w.start < last.end) {
      if (w.end > last.end) merged[merged.length - 1] = { start: last.start, end: w.end };
    } else {
      merged.push({ start: w.start, end: w.end });
    }
  }
  return merged;
}

interface Spread {
  lo: number;
  hi: number;
}

/**
 * Reachability, and why it stays an interval.
 *
 * From a position p inside window i, one pass can leave the item anywhere in
 * window i, and then anywhere in any later window that overlaps what has been
 * reached so far. Every set produced this way is a union of intervals that all
 * contain a point of the previous interval, so it is itself an interval. That
 * collapses the reachable set to two numbers and makes the whole analysis
 * linear instead of a set closure over n positions.
 */
function spreadsOf(windows: readonly WindowRange[]): Spread[] {
  const spread: Spread[] = windows.map((w) => ({ lo: w.start, hi: w.end }));
  for (let i = 0; i < windows.length; i++) {
    const s = spread[i]!;
    for (let j = i + 1; j < windows.length; j++) {
      const w = windows[j]!;
      if (w.start < s.hi && s.lo < w.end) {
        if (w.start < s.lo) s.lo = w.start;
        if (w.end > s.hi) s.hi = w.end;
      }
    }
  }
  return spread;
}

interface Simulation {
  /** Passes after which the reachable interval stops growing. */
  readonly passes: number;
  readonly lo: number;
  readonly hi: number;
  /** First pass at which the reachable interval touches [0, topK). */
  readonly reachesTopKAt: number;
  /** First pass at which the reachable interval touches [topK, n). */
  readonly leavesTopKAt: number;
}

export function analyzeSchedule(
  n: number,
  windows: readonly WindowRange[],
  topK: number,
): ScheduleAnalysis {
  const components = componentsOf(windows);

  // firstWindowOf[p] is the index of the earliest window in PROCESSING order
  // that contains p. Later windows in the same pass are what carry an item
  // further, so where a pass can take an item depends on where it first enters
  // the merge, not on every window that happens to contain it.
  const firstWindowOf = new Array<number>(n).fill(-1);
  for (let i = 0; i < windows.length; i++) {
    const w = windows[i]!;
    for (let p = Math.max(0, w.start); p < Math.min(n, w.end); p++) {
      if (firstWindowOf[p] === -1) firstWindowOf[p] = i;
    }
  }

  const uncovered: number[] = [];
  for (let p = 0; p < n; p++) if (firstWindowOf[p] === -1) uncovered.push(p);

  const spread = spreadsOf(windows);
  const loAt = new Array<number>(n);
  const hiAt = new Array<number>(n);
  for (let p = 0; p < n; p++) {
    const i = firstWindowOf[p]!;
    if (i === -1) {
      // An untouched position is a fixed point of the merge in both
      // directions: the item there never moves and nothing can arrive.
      loAt[p] = p;
      hiAt[p] = p + 1;
    } else {
      const s = spread[i]!;
      loAt[p] = s.lo;
      hiAt[p] = s.hi;
    }
  }

  const simulate = (windowIndex: number): Simulation => {
    const s = spread[windowIndex]!;
    let a = s.lo;
    let b = s.hi;
    let passes = 1;
    let minLo = Number.POSITIVE_INFINITY;
    let maxHi = Number.NEGATIVE_INFINITY;
    // Each position is folded into the running bounds exactly once, so the
    // whole simulation is linear in n no matter how many passes it takes.
    const fold = (from: number, to: number): void => {
      for (let q = from; q < to; q++) {
        const l = loAt[q]!;
        const h = hiAt[q]!;
        if (l < minLo) minLo = l;
        if (h > maxHi) maxHi = h;
      }
    };
    fold(a, b);
    let reachesTopKAt = a < topK ? 1 : Number.POSITIVE_INFINITY;
    let leavesTopKAt = b > topK ? 1 : Number.POSITIVE_INFINITY;
    for (;;) {
      const nextA = Math.min(a, minLo);
      const nextB = Math.max(b, maxHi);
      if (nextA === a && nextB === b) break;
      fold(nextA, a);
      fold(b, nextB);
      a = nextA;
      b = nextB;
      passes++;
      if (!Number.isFinite(reachesTopKAt) && a < topK) reachesTopKAt = passes;
      if (!Number.isFinite(leavesTopKAt) && b > topK) leavesTopKAt = passes;
    }
    return { passes, lo: a, hi: b, reachesTopKAt, leavesTopKAt };
  };

  const sims = windows.map((_, i) => simulate(i));

  const strandedFromTopK: number[] = [];
  const frozenInTopK: number[] = [];
  let passesToTopK = 0;
  let slowestEntrant: number | null = null;
  let passesToLeaveTopK = 0;
  let slowestLeaver: number | null = null;
  let mixingPasses = 0;

  for (let p = 0; p < n; p++) {
    const i = firstWindowOf[p]!;
    const sim = i === -1 ? null : sims[i]!;
    if (sim !== null && sim.passes > mixingPasses) mixingPasses = sim.passes;

    if (p >= topK) {
      const lo = sim === null ? p : sim.lo;
      if (lo >= topK) strandedFromTopK.push(p);
      else if (sim !== null && sim.reachesTopKAt > passesToTopK) {
        passesToTopK = sim.reachesTopKAt;
        slowestEntrant = p;
      }
    } else if (topK < n) {
      // Only meaningful when there is something outside the cutoff to be
      // pushed into. With topK === n every position is inside it by
      // definition and "frozen in the top k" would flag the entire list.
      const hi = sim === null ? p + 1 : sim.hi;
      if (hi <= topK) frozenInTopK.push(p);
      else if (sim !== null && sim.leavesTopKAt > passesToLeaveTopK) {
        passesToLeaveTopK = sim.leavesTopKAt;
        slowestLeaver = p;
      }
    }
  }

  return {
    n,
    topK,
    windows: [...windows],
    components,
    uncovered,
    strandedFromTopK,
    frozenInTopK,
    passesToTopK,
    slowestEntrant,
    passesToLeaveTopK,
    slowestLeaver,
    mixingPasses,
  };
}

function describeComponents(components: readonly WindowRange[], cap = 4): string {
  const shown = components
    .slice(0, cap)
    .map((c) => `${c.start} to ${c.end - 1}`)
    .join('; ');
  const rest = components.length - cap;
  return rest > 0 ? `${shown} (and ${rest} more)` : shown;
}

/**
 * Refuse a schedule that cannot produce the ranking the caller asked for.
 *
 * This runs before any judge call, because the failure is not a quality
 * problem that better judgements could fix. It is arithmetic.
 *
 * The requirement checked here is that every position can reach every other
 * one given enough passes, which is the same as saying the windows form a
 * single overlap-connected block covering the whole list. Anything weaker
 * leaves some pair of items whose relative order the judge can never be asked
 * about, and that pair is then ordered by whatever the input array happened to
 * say. `strandedFromTopK` and `frozenInTopK` on the analysis describe the same
 * failure narrowed to the cutoff the caller cares about, for callers who want
 * to inspect a partial schedule rather than be stopped by one.
 */
export function assertScheduleUsable(analysis: ScheduleAnalysis): void {
  const { n, uncovered, components } = analysis;

  if (uncovered.length > 0) {
    throw new ListwiseError(
      `The window schedule never covers position ${listPositions(uncovered)} of ${n}. ` +
        `An item at a position no window touches is never shown to the judge, so it keeps ` +
        `whatever rank the first stage gave it however many passes you run, and no other item ` +
        `can move into that position either. This is a property of the schedule alone and not ` +
        `of the documents. Build the schedule with buildSlidingWindows, which clamps the last ` +
        `window to start at 0, or supply windows that tile positions 0 through ${n - 1} with ` +
        `overlapping ranges.`,
      'strandedSchedule',
    );
  }

  const single = components.length === 1 && components[0]!.start === 0 && components[0]!.end === n;
  if (n > 0 && !single) {
    throw new ListwiseError(
      `The schedule splits the list into ${components.length} groups of positions that never ` +
        `share a window: ${describeComponents(components)}. An item only moves within the ` +
        `windows it is shown in, so it can only travel through a chain of windows that share ` +
        `positions, and no such chain crosses these boundaries. An item in one group can ` +
        `therefore never overtake an item in another however many passes you run, and their ` +
        `relative order in the result is whatever the input array happened to say. Use a ` +
        `stride strictly smaller than the window size so consecutive windows overlap, or ` +
        `widen the window.`,
      'strandedSchedule',
    );
  }
}

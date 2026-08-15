/**
 * The driver: seed, pass loop, stability detection.
 *
 * Three separate things have to hold before a listwise reranking is a function
 * of the item set and the judge, rather than of incidental facts about the
 * call.
 *
 *   1. THE SEED. The merge starts from the order the caller passed. Two
 *      callers holding the same 100 documents in different array orders then
 *      get different rankings out of the same judge. This module throws the
 *      array order away and seeds from a canonical order derived from the
 *      items themselves, so the array order cannot reach the output at all.
 *
 *   2. THE SCHEDULE. Covered in schedule.ts. Refused before any judge call.
 *
 *   3. THE PASS COUNT. Iterating the pass function on a finite set of
 *      permutations either reaches a fixed point or enters a cycle. The naive
 *      loop runs a fixed number of passes and returns whatever state it is
 *      standing on, so if the judge is intransitive (and an LLM judge always
 *      is) the answer depends on the pass budget: three passes and four passes
 *      return different rankings, and neither is more correct. This module
 *      detects the fixed point or the cycle and returns something that is the
 *      same for every budget large enough to find it, or refuses.
 *
 * The judge cache is load bearing for (3), not just a cost saving. Cycle
 * detection is only meaningful if the pass function is a function: if the same
 * window can produce two different verdicts, the sequence of states never
 * repeats and stability can never be established. Caching on the exact
 * sequence of ids shown makes the pass deterministic within a call, and if
 * caching is turned off a nondeterministic judge surfaces as a notConverged
 * refusal instead of a quietly arbitrary answer.
 */

import { ListwiseError, showId } from './errors.js';
import { PreferenceLedger, aggregateByMeanRank, type Triad } from './preference.js';
import {
  analyzeSchedule,
  assertScheduleUsable,
  buildSlidingWindows,
  type ScheduleAnalysis,
  type WindowRange,
} from './schedule.js';

export interface RankItem {
  /** Stable, unique, and not derived from array position. */
  readonly id: string;
  /** What the judge is shown. */
  readonly text: string;
  /**
   * First-stage score, higher is better. Optional, but all or nothing across
   * the list. Used only to seed the canonical order.
   */
  readonly priorScore?: number;
  /** Ignored by this module and handed back untouched on the result. */
  readonly payload?: unknown;
}

/**
 * Ranks one window, best first, by returning its ids.
 *
 * Deliberately given nothing but the items: no pass index, no window index, no
 * position offsets. A judge that could see those could vary its verdict with
 * them, and the pass function would stop being a function of the list, which
 * would make every stability claim in this module false.
 */
export type Judge = (window: readonly RankItem[]) => readonly string[];

export interface WindowEvent {
  readonly passIndex: number;
  readonly windowIndex: number;
  readonly range: WindowRange;
  readonly shown: readonly string[];
  readonly ranked: readonly string[];
}

export type WindowObserver = (event: WindowEvent) => void;

export type Convergence =
  | { readonly kind: 'fixedPoint'; readonly passes: number }
  | {
      readonly kind: 'cycle';
      /** Index of the state the cycle closes back onto, 0 being the seed. */
      readonly enteredAtPass: number;
      readonly length: number;
      readonly states: readonly (readonly string[])[];
    }
  | { readonly kind: 'unstable'; readonly passes: number };

export interface RerankOptions {
  readonly judge: Judge;
  /** Items shown to the judge at once. Default 20. */
  readonly windowSize?: number;
  /** Positions the window advances between calls. Default 10. */
  readonly stride?: number;
  /** An explicit schedule. Mutually exclusive with windowSize and stride. */
  readonly windows?: readonly WindowRange[];
  /**
   * The prefix of the ranking the caller actually intends to consume. Used to
   * decide whether the schedule and the pass budget are adequate. Default is
   * the whole list.
   */
  readonly topK?: number;
  /** Default 4. */
  readonly maxPasses?: number;
  /** Default 'aggregate'. See the note on Convergence above. */
  readonly onOscillation?: 'aggregate' | 'error';
  /** Default 'error'. */
  readonly onNotConverged?: 'aggregate' | 'error';
  /** Default true. Turning it off makes judge nondeterminism observable. */
  readonly cacheJudge?: boolean;
  /** Cap on reported intransitive triads. Default 32. */
  readonly maxTriads?: number;
  readonly onWindow?: WindowObserver;
}

export interface RerankResult {
  /** Best first. Contains every input item exactly once. */
  readonly ranking: readonly RankItem[];
  readonly convergence: Convergence;
  readonly schedule: ScheduleAnalysis;
  /** The canonical seed order, before any judging. */
  readonly seedOrder: readonly string[];
  readonly judgeCalls: number;
  readonly cacheHits: number;
  readonly comparisons: number;
  /** Pairs the judge ordered both ways in different windows. */
  readonly contradictedPairs: readonly (readonly [string, string])[];
  readonly intransitiveTriads: readonly Triad[];
  readonly triadSearchTruncated: boolean;
}

/**
 * Ranking states are compared by joining their ids into a single string, so
 * the joiner has to be a character that cannot occur inside an id. NUL is
 * rejected in ids for exactly this reason, which makes the join injective:
 * two different rankings can never produce the same key and be mistaken for
 * a repeat.
 */
const STATE_SEPARATOR = '\u0000';

function validateItems(items: readonly RankItem[]): void {
  const seen = new Map<string, number>();
  let scored = 0;
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item === null || typeof item !== 'object') {
      throw new ListwiseError(
        `Item at index ${i} is ${String(item)} rather than an object with an id and a text. ` +
          `Every entry has to carry the id the judge will return and the text the judge will ` +
          `read, so there is nothing sensible to rank here.`,
        'badItems',
      );
    }
    if (typeof item.id !== 'string' || item.id === '') {
      throw new ListwiseError(
        `Item at index ${i} has the id ${showId(item.id)}. Ids are how a judge verdict is ` +
          `matched back to items, and an empty or non-string id cannot be matched, so the ` +
          `verdict would be applied to the wrong document. Give every item a non-empty string ` +
          `id that is stable across calls.`,
        'badItems',
      );
    }
    if (item.id.includes(STATE_SEPARATOR)) {
      throw new ListwiseError(
        `Item at index ${i} has an id containing a NUL character. Ranking states are compared ` +
          `by joining ids with NUL, so an id containing one could make two different rankings ` +
          `compare equal and a cycle be reported where none exists. Remove the NUL from the id.`,
        'badItems',
      );
    }
    if (typeof item.text !== 'string') {
      throw new ListwiseError(
        `Item ${showId(item.id)} has a text of type ${typeof item.text} rather than a string. ` +
          `The judge is shown this value, so a non-string would be stringified differently by ` +
          `different judges and the ranking would depend on which one you used.`,
        'badItems',
      );
    }
    const first = seen.get(item.id);
    if (first !== undefined) {
      throw new ListwiseError(
        `Items at index ${first} and index ${i} share the id ${showId(item.id)}. A judge verdict ` +
          `names ids, so a duplicate id makes the verdict ambiguous about which of the two ` +
          `items it ranked and one of them would be silently dropped from the result. ` +
          `Deduplicate before ranking, or give the copies distinct ids.`,
        'badItems',
      );
    }
    seen.set(item.id, i);
    if (item.priorScore !== undefined) {
      if (typeof item.priorScore !== 'number' || !Number.isFinite(item.priorScore)) {
        throw new ListwiseError(
          `Item ${showId(item.id)} has a priorScore of ${String(item.priorScore)}. The seed order ` +
            `is produced by comparing scores, and a NaN or infinite score compares false in both ` +
            `directions, which would leave the seed depending on the order you passed the array ` +
            `in. Pass a finite number or omit priorScore entirely.`,
          'badItems',
        );
      }
      scored++;
    }
  }
  if (scored !== 0 && scored !== items.length) {
    throw new ListwiseError(
      `${scored} of ${items.length} items carry a priorScore and the rest do not. There is no ` +
        `defensible place to put an unscored item relative to a scored one, so the seed order ` +
        `would come down to the array order for exactly the items whose position is least ` +
        `certain. Score all of them or none of them.`,
      'badItems',
    );
  }
}

/**
 * The canonical seed order: score descending, then id ascending.
 *
 * `id` comparison uses the plain relational operators rather than
 * localeCompare, because localeCompare is locale sensitive and the same call
 * on two machines would then seed differently, which is the whole failure this
 * function exists to prevent. Ids are unique by the time this runs, so the
 * comparator is a total order and the result does not rely on sort stability.
 */
export function canonicalOrder(items: readonly RankItem[]): RankItem[] {
  return [...items].sort((x, y) => {
    const xs = x.priorScore;
    const ys = y.priorScore;
    if (xs !== undefined && ys !== undefined && xs !== ys) return ys - xs;
    return x.id < y.id ? -1 : x.id > y.id ? 1 : 0;
  });
}

function applyVerdict(
  shown: readonly RankItem[],
  returned: unknown,
  range: WindowRange,
): RankItem[] {
  const where = `window covering positions ${range.start} to ${range.end - 1}`;
  if (!Array.isArray(returned)) {
    throw new ListwiseError(
      `The judge returned ${typeof returned} for the ${where} instead of an array of ids. A ` +
        `listwise judge has to answer with the ids of the items it was shown, in ranked order, ` +
        `best first.`,
      'badJudgeOutput',
    );
  }
  if (returned.length !== shown.length) {
    throw new ListwiseError(
      `The judge returned ${returned.length} ids for the ${where}, which held ${shown.length} ` +
        `items. The merge writes the returned order straight back into those positions, so a ` +
        `short answer would drop items out of the ranking and a long one would duplicate them. ` +
        `Return a permutation of exactly the ids you were shown.`,
      'badJudgeOutput',
    );
  }
  const byId = new Map<string, RankItem>();
  for (const item of shown) byId.set(item.id, item);
  const used = new Set<string>();
  const ordered: RankItem[] = [];
  for (let i = 0; i < returned.length; i++) {
    const id: unknown = returned[i];
    if (typeof id !== 'string') {
      throw new ListwiseError(
        `The judge returned ${typeof id} at position ${i} of its answer for the ${where}. Every ` +
          `entry has to be one of the id strings the judge was shown, because that is the only ` +
          `way to know which document it meant.`,
        'badJudgeOutput',
      );
    }
    const item = byId.get(id);
    if (item === undefined) {
      throw new ListwiseError(
        `The judge returned the id ${showId(id)} for the ${where}, which was not among the ids ` +
          `it was shown. This usually means the model invented an id or echoed one from an ` +
          `earlier window, and applying it would put a document into positions it was never ` +
          `judged against. Constrain the judge to the ids in its prompt.`,
        'badJudgeOutput',
      );
    }
    if (used.has(id)) {
      throw new ListwiseError(
        `The judge returned the id ${showId(id)} twice for the ${where}. A ranking places each ` +
          `item once, and writing a duplicate back would overwrite whichever item the judge ` +
          `left out. Return each id exactly once.`,
        'badJudgeOutput',
      );
    }
    used.add(id);
    ordered.push(item);
  }
  return ordered;
}

/**
 * One bubble-sort pass. Exported because it is the thing this package is about
 * and because a caller comparing against the naive behaviour needs to be able
 * to run it without the guards.
 *
 * Windows holding fewer than two items are skipped rather than sent to the
 * judge: there is no ordering question to ask, and a call whose answer is
 * forced would still cost a request and could still return a malformed one.
 */
export function runSinglePass(
  ranking: readonly RankItem[],
  windows: readonly WindowRange[],
  judge: Judge,
  observer?: (windowIndex: number, range: WindowRange, shown: RankItem[], ranked: RankItem[]) => void,
): RankItem[] {
  const out = [...ranking];
  for (let wi = 0; wi < windows.length; wi++) {
    const range = windows[wi]!;
    const shown = out.slice(range.start, range.end);
    if (shown.length < 2) continue;
    const ordered = applyVerdict(shown, judge(shown), range);
    for (let k = 0; k < ordered.length; k++) out[range.start + k] = ordered[k]!;
    observer?.(wi, range, shown, ordered);
  }
  return out;
}

interface ResolvedConfig {
  readonly windows: readonly WindowRange[];
  readonly topK: number;
  readonly maxPasses: number;
  readonly onOscillation: 'aggregate' | 'error';
  readonly onNotConverged: 'aggregate' | 'error';
  readonly cacheJudge: boolean;
  readonly maxTriads: number;
}

function requireInteger(value: number, name: string, min: number, max: number): void {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new ListwiseError(
      `${name} is ${String(value)} but has to be a whole number between ${min} and ${max}. ` +
        `The schedule is built from these numbers before anything is judged, so a fractional ` +
        `or out of range value would produce window boundaries that do not line up with item ` +
        `positions.`,
      'badConfig',
    );
  }
}

function resolveConfig(n: number, options: RerankOptions): ResolvedConfig {
  if (typeof options.judge !== 'function') {
    throw new ListwiseError(
      `No judge was supplied. There is deliberately no default: the ranking this module ` +
        `produces is a function of the item set and the judge, and inventing a judge would ` +
        `make it a function of a heuristic the caller never chose.`,
      'badConfig',
    );
  }

  const hasExplicit = options.windows !== undefined;
  if (hasExplicit && (options.windowSize !== undefined || options.stride !== undefined)) {
    throw new ListwiseError(
      `Both an explicit windows array and windowSize or stride were supplied. These describe ` +
        `two different schedules and there is no way to tell which one you meant, so no window ` +
        `list is built. Pass windows on its own, or windowSize and stride on their own.`,
      'badConfig',
    );
  }

  const topK = options.topK ?? n;
  requireInteger(topK, 'topK', n === 0 ? 0 : 1, n);

  const maxPasses = options.maxPasses ?? 4;
  requireInteger(maxPasses, 'maxPasses', 1, 1_000_000);

  const maxTriads = options.maxTriads ?? 32;
  requireInteger(maxTriads, 'maxTriads', 0, 1_000_000);

  let windows: readonly WindowRange[];
  if (hasExplicit) {
    const supplied = options.windows!;
    if (!Array.isArray(supplied)) {
      throw new ListwiseError(
        `windows is ${typeof supplied} rather than an array of {start, end} ranges.`,
        'badConfig',
      );
    }
    for (let i = 0; i < supplied.length; i++) {
      const w = supplied[i]!;
      if (
        !Number.isInteger(w.start) ||
        !Number.isInteger(w.end) ||
        w.start < 0 ||
        w.end > n ||
        w.end <= w.start
      ) {
        throw new ListwiseError(
          `windows[${i}] is {start: ${String(w.start)}, end: ${String(w.end)}}, which is not a ` +
            `non-empty range of whole positions inside a list of ${n} items. A window that runs ` +
            `off the end of the list would show the judge undefined entries, and an empty or ` +
            `reversed one would silently do nothing.`,
          'badConfig',
        );
      }
    }
    windows = supplied.map((w) => ({ start: w.start, end: w.end }));
  } else {
    const windowSize = options.windowSize ?? 20;
    const stride = options.stride ?? 10;
    // A window larger than the list is allowed and means one window over
    // everything, which is the correct degenerate case and is also what the
    // default of 20 does on a short candidate list.
    requireInteger(windowSize, 'windowSize', 2, 1_000_000);
    requireInteger(stride, 'stride', 1, 1_000_000);
    if (stride >= windowSize && windowSize < n) {
      throw new ListwiseError(
        `stride is ${stride} and windowSize is ${windowSize}, so consecutive windows share no ` +
          `position. The merge can only move an item within a window it is shown in, so with ` +
          `disjoint windows an item is confined to its own block of ${windowSize} positions ` +
          `permanently: the best document in the list, if it starts near the bottom, can never ` +
          `reach the top however many passes you run. Use a stride strictly smaller than ` +
          `windowSize, for example ${Math.max(1, Math.floor(windowSize / 2))}.`,
        'badConfig',
      );
    }
    windows = buildSlidingWindows(n, windowSize, stride);
  }

  return {
    windows,
    topK,
    maxPasses,
    onOscillation: options.onOscillation ?? 'aggregate',
    onNotConverged: options.onNotConverged ?? 'error',
    cacheJudge: options.cacheJudge ?? true,
    maxTriads,
  };
}

function assertPassBudget(analysis: ScheduleAnalysis, maxPasses: number): void {
  if (analysis.topK >= analysis.n) return;
  if (maxPasses < analysis.passesToTopK) {
    throw new ListwiseError(
      `maxPasses is ${maxPasses} but this schedule needs ${analysis.passesToTopK} passes before ` +
        `the item at position ${String(analysis.slowestEntrant)} can so much as enter the top ` +
        `${analysis.topK}. A pass carries an item only as far as the chain of overlapping ` +
        `windows reaches from where it entered the merge, so with a smaller budget the cutoff ` +
        `is decided by the first-stage order for everything further down, and raising maxPasses ` +
        `later would change the answer. Raise maxPasses to at least ${analysis.passesToTopK}, ` +
        `or widen the window so the chain is shorter.`,
      'badConfig',
    );
  }
  if (maxPasses < analysis.passesToLeaveTopK) {
    throw new ListwiseError(
      `maxPasses is ${maxPasses} but this schedule needs ${analysis.passesToLeaveTopK} passes ` +
        `before the item at position ${String(analysis.slowestLeaver)}, which starts inside the ` +
        `top ${analysis.topK}, can be pushed out of it. With a smaller budget anything the ` +
        `first stage placed near the head of the list survives into the result whatever the ` +
        `judge says about it. Raise maxPasses to at least ${analysis.passesToLeaveTopK}, or ` +
        `widen the window.`,
      'badConfig',
    );
  }
}

export function rerank(items: readonly RankItem[], options: RerankOptions): RerankResult {
  if (!Array.isArray(items)) {
    throw new ListwiseError(
      `The item list is ${typeof items} rather than an array. Pass the candidates to rerank as ` +
        `an array of {id, text} objects.`,
      'badItems',
    );
  }
  validateItems(items);

  const n = items.length;
  const config = resolveConfig(n, options);
  const analysis = analyzeSchedule(n, config.windows, config.topK);

  const seed = canonicalOrder(items);
  const seedOrder = seed.map((item) => item.id);
  const canonicalIndex = new Map<string, number>(seedOrder.map((id, i) => [id, i]));
  const ledger = new PreferenceLedger();

  // A list of zero or one item has exactly one ordering, so there is no
  // schedule to satisfy and no judgement that could change the answer. This
  // returns before assertScheduleUsable so that the degenerate case is not
  // rejected for having a window that cannot mix anything.
  if (n < 2) {
    return {
      ranking: seed,
      convergence: { kind: 'fixedPoint', passes: 0 },
      schedule: analysis,
      seedOrder,
      judgeCalls: 0,
      cacheHits: 0,
      comparisons: 0,
      contradictedPairs: [],
      intransitiveTriads: [],
      triadSearchTruncated: false,
    };
  }

  assertScheduleUsable(analysis);
  assertPassBudget(analysis, config.maxPasses);

  const cache = new Map<string, readonly string[]>();
  let judgeCalls = 0;
  let cacheHits = 0;
  const judge: Judge = (window) => {
    if (!config.cacheJudge) {
      judgeCalls++;
      return options.judge(window);
    }
    const key = window.map((item) => item.id).join(STATE_SEPARATOR);
    const hit = cache.get(key);
    if (hit !== undefined) {
      cacheHits++;
      return hit;
    }
    judgeCalls++;
    const answer = options.judge(window);
    // Stored before validation deliberately: applyVerdict throws on a bad
    // answer, so nothing downstream ever sees the cached copy, and storing it
    // here keeps the cache key and the answer written in one place.
    cache.set(key, answer);
    return answer;
  };

  let passIndex = 0;
  const observe = (
    windowIndex: number,
    range: WindowRange,
    shown: RankItem[],
    ranked: RankItem[],
  ): void => {
    const rankedIds = ranked.map((item) => item.id);
    ledger.record(rankedIds);
    options.onWindow?.({
      passIndex,
      windowIndex,
      range,
      shown: shown.map((item) => item.id),
      ranked: rankedIds,
    });
  };

  const states: string[][] = [seedOrder];
  const seenAt = new Map<string, number>([[seedOrder.join(STATE_SEPARATOR), 0]]);
  let current = seed;
  let convergence: Convergence | null = null;
  let cycleStates: readonly (readonly string[])[] = [];

  for (let pass = 1; pass <= config.maxPasses; pass++) {
    passIndex = pass;
    const next = runSinglePass(current, config.windows, judge, observe);
    const ids = next.map((item) => item.id);
    const key = ids.join(STATE_SEPARATOR);
    const previous = seenAt.get(key);
    current = next;
    if (previous !== undefined) {
      // A fixed point is just a cycle of length one under this bookkeeping,
      // which keeps the two cases from drifting apart.
      cycleStates = states.slice(previous);
      convergence =
        cycleStates.length === 1
          ? { kind: 'fixedPoint', passes: pass }
          : {
              kind: 'cycle',
              enteredAtPass: previous,
              length: cycleStates.length,
              states: cycleStates,
            };
      break;
    }
    seenAt.set(key, states.length);
    states.push(ids);
  }

  let finalOrder: readonly string[];
  if (convergence === null) {
    if (config.onNotConverged === 'error') {
      throw new ListwiseError(
        `The ranking had not repeated a state after ${config.maxPasses} passes, so it is not ` +
          `known whether it settles or oscillates and the ranking after ${config.maxPasses} ` +
          `passes is simply the one this budget happened to stop on: a larger budget would ` +
          `return something else. This is normal when the judge contradicts itself. Either ` +
          `raise maxPasses until a state repeats, or set onNotConverged to 'aggregate' to ` +
          `average the ${config.maxPasses + 1} states seen so far, which is at least a stated ` +
          `policy rather than an accident of the budget.` +
          (config.cacheJudge
            ? ''
            : ` cacheJudge is off, so a judge that answers the same window differently on ` +
              `different calls would also produce this: with the cache on, the pass function ` +
              `is deterministic and a repeat is guaranteed eventually.`),
        'notConverged',
      );
    }
    convergence = { kind: 'unstable', passes: config.maxPasses };
    finalOrder = aggregateByMeanRank(states, canonicalIndex);
  } else if (convergence.kind === 'cycle') {
    if (config.onOscillation === 'error') {
      throw new ListwiseError(
        `The ranking oscillates with a period of ${convergence.length} passes rather than ` +
          `settling, which means the judge's pairwise verdicts cannot all be satisfied by any ` +
          `single ordering. There is no ranking here that more passes would find: passes ` +
          `${convergence.enteredAtPass} onward walk the same ${convergence.length} states ` +
          `forever, so the answer you get is decided by your pass budget. Set onOscillation to ` +
          `'aggregate' to average the states in the cycle, which gives the same result for ` +
          `every budget large enough to detect it, or use a judge whose comparisons are ` +
          `consistent.`,
        'oscillation',
      );
    }
    finalOrder = aggregateByMeanRank(cycleStates, canonicalIndex);
  } else {
    finalOrder = states[states.length - 1]!;
  }

  const byId = new Map<string, RankItem>(items.map((item) => [item.id, item]));
  const ranking = finalOrder.map((id) => byId.get(id)!);
  const triadSearch = ledger.intransitiveTriads(config.maxTriads);

  return {
    ranking,
    convergence,
    schedule: analysis,
    seedOrder,
    judgeCalls,
    cacheHits,
    comparisons: ledger.comparisons,
    contradictedPairs: ledger.contradictedPairs(),
    intransitiveTriads: triadSearch.triads,
    triadSearchTruncated: triadSearch.truncated,
  };
}

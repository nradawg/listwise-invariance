/**
 * listwise-invariance: listwise LLM reranking whose output does not depend on
 * the order the candidates arrived in, on how many passes you paid for, or on
 * a schedule that quietly cannot move some items at all.
 *
 * The sliding-window merge that every listwise reranker uses is one pass of a
 * bubble sort, not a sort. Three consequences, all of which this package
 * treats as correctness problems rather than quality problems:
 *
 *   The schedule can strand items. With a stride at least as large as the
 *   window, or with a loop that stops before it reaches position 0, there are
 *   positions from which no item can ever reach the top. That is decidable
 *   from (n, windowSize, stride) alone, before a single judge call, and it is
 *   refused here rather than discovered as a mysteriously bad ranking.
 *
 *   LLM judgements are intransitive. Extra passes therefore oscillate rather
 *   than converge, so more compute returns a different answer, not a better
 *   one. This package detects the cycle and returns a result that is the same
 *   for every pass budget large enough to find it.
 *
 *   The merge starts from the input order, so the same documents in a
 *   different array order rank differently. The input order is discarded and
 *   replaced with a canonical order derived from the items.
 *
 * None of the three is visible to a test that runs one pass over a short list,
 * which is why they survive in production code.
 */

export { ListwiseError, type ListwiseErrorCode } from './errors.js';

export {
  analyzeSchedule,
  assertScheduleUsable,
  buildSlidingWindows,
  type ScheduleAnalysis,
  type WindowRange,
} from './schedule.js';

export {
  PreferenceLedger,
  aggregateByMeanRank,
  type Triad,
  type TriadSearch,
} from './preference.js';

export {
  canonicalOrder,
  rerank,
  runSinglePass,
  type Convergence,
  type Judge,
  type RankItem,
  type RerankOptions,
  type RerankResult,
  type WindowEvent,
  type WindowObserver,
} from './rerank.js';

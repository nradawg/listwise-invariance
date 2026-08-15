import { describe, expect, it } from 'vitest';
import {
  ListwiseError,
  buildSlidingWindows,
  canonicalOrder,
  rerank,
  runSinglePass,
  type Judge,
  type RankItem,
} from '../src/index.js';
import {
  ids,
  idsIn,
  oracleJudge,
  plainList,
  relevanceOf,
  restlessJudge,
  scoredList,
  shuffled,
} from './helpers.js';

const CYCLE: readonly [string, string, string] = ['ka', 'kb', 'kc'];
const FILLER = new Map<string, number>([
  ['ka', 5],
  ['kb', 5],
  ['kc', 5],
  ['za', 1],
  ['zb', 0],
]);

function restlessFixture(): { list: RankItem[]; judge: Judge } {
  return {
    list: plainList(['ka', 'kb', 'kc', 'za', 'zb']),
    judge: restlessJudge(CYCLE, FILLER),
  };
}

describe('the input order does not reach the output', () => {
  const list = ids(24);
  const relevance = relevanceOf(list);
  const judge = oracleJudge(relevance);
  const truth = [...list].reverse();

  it('gives the same ranking for fifty different array orders', () => {
    const first = rerank(plainList(list), { judge, maxPasses: 12 });
    for (let seed = 1; seed <= 50; seed++) {
      const scrambled = rerank(plainList(shuffled(list, seed)), { judge, maxPasses: 12 });
      expect(idsIn(scrambled.ranking), `seed ${seed}`).toEqual(idsIn(first.ranking));
    }
  });

  it('gives the same seed order for every array order', () => {
    const first = rerank(plainList(list), { judge, maxPasses: 12 });
    for (let seed = 1; seed <= 10; seed++) {
      const scrambled = rerank(plainList(shuffled(list, seed)), { judge, maxPasses: 12 });
      expect(scrambled.seedOrder).toEqual(first.seedOrder);
    }
  });

  it('reproduces the ground truth order from a transitive judge', () => {
    const result = rerank(plainList(list), { judge, maxPasses: 12 });
    expect(idsIn(result.ranking)).toEqual(truth);
    expect(result.convergence.kind).toBe('fixedPoint');
  });

  it('lifts the best document from the very bottom of the seed to rank 0', () => {
    const result = rerank(plainList(list), { judge, maxPasses: 12 });
    expect(result.seedOrder[result.seedOrder.length - 1]).toBe('d23');
    expect(result.ranking[0]!.id).toBe('d23');
  });

  it('returns every input item exactly once', () => {
    const result = rerank(plainList(shuffled(list, 7)), { judge, maxPasses: 12 });
    expect(new Set(idsIn(result.ranking)).size).toBe(list.length);
    expect(idsIn(result.ranking).slice().sort()).toEqual([...list].sort());
  });
});

describe('canonicalOrder', () => {
  it('sorts by priorScore descending', () => {
    const list = scoredList(['b', 'a', 'c']);
    expect(idsIn(canonicalOrder(list))).toEqual(['b', 'a', 'c']);
  });

  it('breaks priorScore ties by id rather than by array position', () => {
    const forward: RankItem[] = [
      { id: 'zz', text: 'z', priorScore: 1 },
      { id: 'aa', text: 'a', priorScore: 1 },
    ];
    const reversed = [...forward].reverse();
    expect(idsIn(canonicalOrder(forward))).toEqual(['aa', 'zz']);
    expect(idsIn(canonicalOrder(reversed))).toEqual(['aa', 'zz']);
  });

  it('falls back to id order when nothing is scored', () => {
    expect(idsIn(canonicalOrder(plainList(['m', 'a', 'z'])))).toEqual(['a', 'm', 'z']);
  });
});

describe('a schedule that cannot move an item is refused, not run', () => {
  const list = ids(12);
  const relevance = relevanceOf(list);
  const judge = oracleJudge(relevance);
  const disjoint = [
    { start: 8, end: 12 },
    { start: 4, end: 8 },
    { start: 0, end: 4 },
  ];

  it('leaves the best document stuck near the bottom under the naive merge', () => {
    // A perfect judge, fifty passes, and the best document still never gets
    // past position 8. The failure is the schedule, not the judgements.
    let ranking: readonly RankItem[] = scoredList(list);
    for (let pass = 0; pass < 50; pass++) {
      ranking = runSinglePass(ranking, disjoint, judge);
    }
    expect(ranking[0]!.id).not.toBe('d11');
    expect(idsIn(ranking).slice(8)).toContain('d11');
    expect(ranking[8]!.id).toBe('d11');
  });

  it('refuses the same windows before making a single judge call', () => {
    let calls = 0;
    const counting: Judge = (window) => {
      calls++;
      return judge(window);
    };
    expect(() => rerank(scoredList(list), { judge: counting, windows: disjoint })).toThrow(
      ListwiseError,
    );
    expect(calls).toBe(0);
  });

  it('refuses a stride that is not smaller than the window size', () => {
    try {
      rerank(scoredList(list), { judge, windowSize: 4, stride: 4 });
      throw new Error('expected a refusal');
    } catch (error) {
      expect((error as ListwiseError).code).toBe('badConfig');
      expect((error as ListwiseError).message).toContain('share no position');
    }
  });

  it('surfaces the best document once the windows overlap', () => {
    const result = rerank(scoredList(list), { judge, windowSize: 4, stride: 2, maxPasses: 12 });
    expect(result.ranking[0]!.id).toBe('d11');
  });
});

describe('oscillation is detected instead of being averaged over by accident', () => {
  it('reports a cycle rather than a fixed point', () => {
    const { list, judge } = restlessFixture();
    const result = rerank(list, { judge, maxPasses: 10 });
    expect(result.convergence.kind).toBe('cycle');
  });

  it('reports the period of the cycle and the states in it', () => {
    const { list, judge } = restlessFixture();
    const result = rerank(list, { judge, maxPasses: 10 });
    if (result.convergence.kind !== 'cycle') throw new Error('expected a cycle');
    expect(result.convergence.length).toBe(3);
    expect(result.convergence.states.length).toBe(3);
    expect(new Set(result.convergence.states.map((s) => s.join(','))).size).toBe(3);
  });

  it('returns the same ranking for every pass budget large enough to see it', () => {
    const { list, judge } = restlessFixture();
    const budgets = [3, 4, 7, 20, 51];
    const answers = budgets.map((maxPasses) => idsIn(rerank(list, { judge, maxPasses }).ranking));
    for (const answer of answers) expect(answer).toEqual(answers[0]);
  });

  it('is exactly what the raw pass loop fails to do', () => {
    // Three consecutive pass budgets, three different answers, none of them
    // better than the others. This is what a fixed maxPasses returns.
    const { list, judge } = restlessFixture();
    const windows = buildSlidingWindows(5, 20, 10);
    let current: readonly RankItem[] = canonicalOrder(list);
    const byBudget = new Map<number, string>();
    for (let pass = 1; pass <= 6; pass++) {
      current = runSinglePass(current, windows, judge);
      byBudget.set(pass, idsIn(current).join(','));
    }
    expect(byBudget.get(4)).not.toBe(byBudget.get(5));
    expect(byBudget.get(5)).not.toBe(byBudget.get(6));
    expect(byBudget.get(4)).not.toBe(byBudget.get(6));
  });

  it('breaks the tie among the cycle members in canonical order', () => {
    // Every member of this cycle holds every rank once, so mean rank ties and
    // the canonical order decides. Any other rule would depend on which state
    // the loop stopped on.
    const { list, judge } = restlessFixture();
    const result = rerank(list, { judge, maxPasses: 10 });
    expect(idsIn(result.ranking)).toEqual(['ka', 'kb', 'kc', 'za', 'zb']);
  });

  it('throws with the period named when the caller asks to be told', () => {
    const { list, judge } = restlessFixture();
    try {
      rerank(list, { judge, maxPasses: 10, onOscillation: 'error' });
      throw new Error('expected a refusal');
    } catch (error) {
      expect((error as ListwiseError).code).toBe('oscillation');
      expect((error as ListwiseError).message).toContain('period of 3 passes');
    }
  });

  it('records the pairs the judge ordered both ways', () => {
    const { list, judge } = restlessFixture();
    const result = rerank(list, { judge, maxPasses: 10 });
    expect(result.contradictedPairs.length).toBeGreaterThan(0);
  });

  it('records no contradictions for a transitive judge', () => {
    const list = ids(16);
    const result = rerank(plainList(list), {
      judge: oracleJudge(relevanceOf(list)),
      maxPasses: 12,
    });
    expect(result.contradictedPairs).toEqual([]);
    expect(result.intransitiveTriads).toEqual([]);
  });
});

describe('two input orders, one answer', () => {
  it('the raw pass loop keeps them apart forever', () => {
    const judge = restlessJudge(CYCLE, FILLER);
    const windows = buildSlidingWindows(5, 20, 10);
    let a: readonly RankItem[] = plainList(['ka', 'kb', 'kc', 'za', 'zb']);
    let b: readonly RankItem[] = plainList(['kb', 'ka', 'kc', 'za', 'zb']);
    for (let pass = 0; pass < 2; pass++) {
      a = runSinglePass(a, windows, judge);
      b = runSinglePass(b, windows, judge);
    }
    expect(idsIn(a)).not.toEqual(idsIn(b));
  });

  it('rerank collapses them onto the same ranking', () => {
    const judge = restlessJudge(CYCLE, FILLER);
    const a = rerank(plainList(['ka', 'kb', 'kc', 'za', 'zb']), { judge, maxPasses: 10 });
    const b = rerank(plainList(['kb', 'ka', 'kc', 'za', 'zb']), { judge, maxPasses: 10 });
    expect(idsIn(a.ranking)).toEqual(idsIn(b.ranking));
  });
});

describe('the pass budget has to be able to reach the cutoff', () => {
  const list = ids(100);
  const judge = oracleJudge(relevanceOf(list));

  it('refuses a budget too small to push a seeded item out of the top half', () => {
    try {
      rerank(plainList(list), { judge, topK: 50, maxPasses: 4 });
      throw new Error('expected a refusal');
    } catch (error) {
      expect((error as ListwiseError).code).toBe('badConfig');
      expect((error as ListwiseError).message).toContain('needs 5 passes');
      expect((error as ListwiseError).message).toContain('position 0');
    }
  });

  it('accepts the budget the analysis asked for', () => {
    expect(() =>
      rerank(plainList(list), { judge, topK: 50, maxPasses: 20 }),
    ).not.toThrow();
  });

  it('does not object when the caller consumes the whole ranking', () => {
    expect(() => rerank(plainList(list), { judge, maxPasses: 20 })).not.toThrow();
  });
});

describe('a judge that is not a function of its window', () => {
  /** Rotates by a call counter, so the same window answers differently. */
  function drifting(): Judge {
    let calls = 0;
    return (window) => {
      calls++;
      const order = window.map((item) => item.id);
      const shift = calls % order.length;
      return [...order.slice(shift), ...order.slice(0, shift)];
    };
  }

  it('is reported as not converged rather than answered arbitrarily', () => {
    const list = plainList(ids(7));
    try {
      rerank(list, { judge: drifting(), maxPasses: 3, cacheJudge: false });
      throw new Error('expected a refusal');
    } catch (error) {
      expect((error as ListwiseError).code).toBe('notConverged');
      expect((error as ListwiseError).message).toContain('cacheJudge is off');
    }
  });

  it('can be aggregated instead, when the caller says so explicitly', () => {
    const list = plainList(ids(7));
    const result = rerank(list, {
      judge: drifting(),
      maxPasses: 3,
      cacheJudge: false,
      onNotConverged: 'aggregate',
    });
    expect(result.convergence.kind).toBe('unstable');
    expect(idsIn(result.ranking).slice().sort()).toEqual(ids(7).slice().sort());
  });
});

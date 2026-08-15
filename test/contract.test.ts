import { describe, expect, it } from 'vitest';
import {
  ListwiseError,
  PreferenceLedger,
  aggregateByMeanRank,
  rerank,
  type Judge,
  type RankItem,
  type WindowEvent,
} from '../src/index.js';
import { ids, idsIn, oracleJudge, plainList, relevanceOf } from './helpers.js';

const SIX = plainList(ids(6));
const GOOD: Judge = (window) => window.map((item) => item.id);

function refusal(run: () => unknown): ListwiseError {
  try {
    run();
  } catch (error) {
    if (error instanceof ListwiseError) return error;
    throw error;
  }
  throw new Error('expected a refusal but the call returned');
}

describe('the judge has to return a permutation of its window', () => {
  it('rejects an answer that is too short', () => {
    const error = refusal(() =>
      rerank(SIX, { judge: (w) => w.slice(1).map((item) => item.id) }),
    );
    expect(error.code).toBe('badJudgeOutput');
    expect(error.message).toContain('returned 5 ids');
    expect(error.message).toContain('positions 0 to 5');
  });

  it('rejects an answer that is too long', () => {
    const error = refusal(() =>
      rerank(SIX, { judge: (w) => [...w.map((item) => item.id), 'd00'] }),
    );
    expect(error.code).toBe('badJudgeOutput');
    expect(error.message).toContain('returned 7 ids');
  });

  it('rejects a duplicated id', () => {
    const error = refusal(() =>
      rerank(SIX, {
        judge: (w) => {
          const list = w.map((item) => item.id);
          list[1] = list[0]!;
          return list;
        },
      }),
    );
    expect(error.code).toBe('badJudgeOutput');
    expect(error.message).toContain('twice');
  });

  it('rejects an id that was never shown', () => {
    const error = refusal(() =>
      rerank(SIX, {
        judge: (w) => {
          const list = w.map((item) => item.id);
          list[0] = 'invented';
          return list;
        },
      }),
    );
    expect(error.code).toBe('badJudgeOutput');
    expect(error.message).toContain('"invented"');
  });

  it('rejects a non-array answer', () => {
    const error = refusal(() => rerank(SIX, { judge: (() => 'd00') as unknown as Judge }));
    expect(error.code).toBe('badJudgeOutput');
    expect(error.message).toContain('instead of an array of ids');
  });

  it('rejects a non-string entry', () => {
    const error = refusal(() =>
      rerank(SIX, {
        judge: (w) => {
          const list: unknown[] = w.map((item) => item.id);
          list[2] = 2;
          return list as string[];
        },
      }),
    );
    expect(error.code).toBe('badJudgeOutput');
    expect(error.message).toContain('at position 2');
  });
});

describe('the item list has to be unambiguous', () => {
  it('rejects duplicate ids and names both indexes', () => {
    const list = plainList(['a', 'b', 'a', 'c']);
    const error = refusal(() => rerank(list, { judge: GOOD }));
    expect(error.code).toBe('badItems');
    expect(error.message).toContain('index 0 and index 2');
  });

  it('rejects an empty id', () => {
    const error = refusal(() => rerank(plainList(['a', '', 'c']), { judge: GOOD }));
    expect(error.code).toBe('badItems');
    expect(error.message).toContain('the empty string');
  });

  it('rejects a non-string id', () => {
    const list = [{ id: 7, text: 'x' }, ...plainList(['a'])] as unknown as RankItem[];
    const error = refusal(() => rerank(list, { judge: GOOD }));
    expect(error.code).toBe('badItems');
  });

  it('rejects an id containing the state separator', () => {
    const list = plainList(['a', `b${String.fromCharCode(0)}c`]);
    const error = refusal(() => rerank(list, { judge: GOOD }));
    expect(error.code).toBe('badItems');
    expect(error.message).toContain('NUL');
  });

  it('rejects a non-string text', () => {
    const list = [{ id: 'a', text: 5 }] as unknown as RankItem[];
    const error = refusal(() => rerank(list, { judge: GOOD }));
    expect(error.code).toBe('badItems');
    expect(error.message).toContain('rather than a string');
  });

  it('rejects a NaN priorScore', () => {
    const list: RankItem[] = [
      { id: 'a', text: 'a', priorScore: Number.NaN },
      { id: 'b', text: 'b', priorScore: 1 },
    ];
    const error = refusal(() => rerank(list, { judge: GOOD }));
    expect(error.code).toBe('badItems');
    expect(error.message).toContain('NaN');
  });

  it('rejects an infinite priorScore', () => {
    const list: RankItem[] = [
      { id: 'a', text: 'a', priorScore: Number.POSITIVE_INFINITY },
      { id: 'b', text: 'b', priorScore: 1 },
    ];
    expect(refusal(() => rerank(list, { judge: GOOD })).code).toBe('badItems');
  });

  it('rejects a list where only some items are scored', () => {
    const list: RankItem[] = [
      { id: 'a', text: 'a', priorScore: 3 },
      { id: 'b', text: 'b' },
      { id: 'c', text: 'c', priorScore: 1 },
    ];
    const error = refusal(() => rerank(list, { judge: GOOD }));
    expect(error.code).toBe('badItems');
    expect(error.message).toContain('2 of 3 items');
  });

  it('rejects an entry that is not an object', () => {
    const list = [null, { id: 'a', text: 'a' }] as unknown as RankItem[];
    expect(refusal(() => rerank(list, { judge: GOOD })).code).toBe('badItems');
  });

  it('rejects a non-array item list', () => {
    const error = refusal(() => rerank('nope' as unknown as RankItem[], { judge: GOOD }));
    expect(error.code).toBe('badItems');
  });
});

describe('the configuration has to describe one schedule', () => {
  it('requires a judge', () => {
    const error = refusal(() => rerank(SIX, {} as unknown as { judge: Judge }));
    expect(error.code).toBe('badConfig');
    expect(error.message).toContain('no default');
  });

  it('rejects a window of one item', () => {
    const error = refusal(() => rerank(SIX, { judge: GOOD, windowSize: 1, stride: 1 }));
    expect(error.code).toBe('badConfig');
    expect(error.message).toContain('windowSize');
  });

  it('rejects a stride of zero', () => {
    const error = refusal(() => rerank(SIX, { judge: GOOD, windowSize: 4, stride: 0 }));
    expect(error.code).toBe('badConfig');
    expect(error.message).toContain('stride');
  });

  it('rejects a fractional window size', () => {
    expect(refusal(() => rerank(SIX, { judge: GOOD, windowSize: 3.5, stride: 1 })).code).toBe(
      'badConfig',
    );
  });

  it('rejects a topK of zero on a non-empty list', () => {
    expect(refusal(() => rerank(SIX, { judge: GOOD, topK: 0 })).code).toBe('badConfig');
  });

  it('rejects a topK larger than the list', () => {
    expect(refusal(() => rerank(SIX, { judge: GOOD, topK: 7 })).code).toBe('badConfig');
  });

  it('rejects a maxPasses of zero', () => {
    expect(refusal(() => rerank(SIX, { judge: GOOD, maxPasses: 0 })).code).toBe('badConfig');
  });

  it('rejects an explicit schedule alongside windowSize', () => {
    const error = refusal(() =>
      rerank(SIX, { judge: GOOD, windows: [{ start: 0, end: 6 }], windowSize: 4 }),
    );
    expect(error.code).toBe('badConfig');
    expect(error.message).toContain('two different schedules');
  });

  it('rejects a window that runs off the end of the list', () => {
    const error = refusal(() => rerank(SIX, { judge: GOOD, windows: [{ start: 0, end: 9 }] }));
    expect(error.code).toBe('badConfig');
    expect(error.message).toContain('windows[0]');
  });

  it('rejects a reversed window', () => {
    expect(
      refusal(() => rerank(SIX, { judge: GOOD, windows: [{ start: 4, end: 2 }] })).code,
    ).toBe('badConfig');
  });
});

describe('caching, observation and degenerate lists', () => {
  const list = ids(24);
  const judge = oracleJudge(relevanceOf(list));

  it('reuses answers for windows it has already seen', () => {
    const result = rerank(plainList(list), { judge, maxPasses: 12 });
    expect(result.cacheHits).toBeGreaterThan(0);
  });

  it('asks the judge more often once the cache is turned off', () => {
    const cached = rerank(plainList(list), { judge, maxPasses: 12 });
    const uncached = rerank(plainList(list), { judge, maxPasses: 12, cacheJudge: false });
    expect(uncached.judgeCalls).toBeGreaterThan(cached.judgeCalls);
    expect(uncached.cacheHits).toBe(0);
  });

  it('reaches the same ranking either way for a judge that is a real function', () => {
    const cached = rerank(plainList(list), { judge, maxPasses: 12 });
    const uncached = rerank(plainList(list), { judge, maxPasses: 12, cacheJudge: false });
    expect(idsIn(uncached.ranking)).toEqual(idsIn(cached.ranking));
  });

  it('reports every window to the observer', () => {
    const events: WindowEvent[] = [];
    const result = rerank(plainList(list), {
      judge,
      maxPasses: 12,
      onWindow: (event) => events.push(event),
    });
    expect(events.length).toBeGreaterThan(0);
    expect(events.every((e) => e.shown.length === e.ranked.length)).toBe(true);
    expect(events[0]!.passIndex).toBe(1);
    expect(events[0]!.range).toEqual(result.schedule.windows[0]);
  });

  it('handles an empty list without calling the judge', () => {
    let calls = 0;
    const result = rerank([], {
      judge: (w) => {
        calls++;
        return GOOD(w);
      },
    });
    expect(result.ranking).toEqual([]);
    expect(calls).toBe(0);
    expect(result.convergence).toEqual({ kind: 'fixedPoint', passes: 0 });
  });

  it('handles a single item without calling the judge', () => {
    let calls = 0;
    const result = rerank(plainList(['only']), {
      judge: (w) => {
        calls++;
        return GOOD(w);
      },
    });
    expect(idsIn(result.ranking)).toEqual(['only']);
    expect(calls).toBe(0);
  });

  it('spends a second pass proving the first one settled', () => {
    // Two items, two calls: one to order them and one to establish that the
    // order is a fixed point. Stopping after the first would be assuming what
    // this module exists to check.
    const relevance = new Map([
      ['a', 1],
      ['b', 2],
    ]);
    const result = rerank(plainList(['a', 'b']), { judge: oracleJudge(relevance) });
    expect(idsIn(result.ranking)).toEqual(['b', 'a']);
    expect(result.judgeCalls).toBe(2);
    expect(result.convergence).toEqual({ kind: 'fixedPoint', passes: 2 });
  });

  it('hands the original item objects back', () => {
    const items: RankItem[] = [
      { id: 'a', text: 'a', payload: { row: 1 } },
      { id: 'b', text: 'b', payload: { row: 2 } },
    ];
    const result = rerank(items, { judge: GOOD });
    expect(result.ranking.map((item) => item.payload)).toEqual([{ row: 1 }, { row: 2 }]);
  });
});

describe('PreferenceLedger', () => {
  it('records every pair a verdict implies, not just adjacent ones', () => {
    const ledger = new PreferenceLedger();
    ledger.record(['a', 'b', 'c']);
    expect(ledger.wins('a', 'c')).toBe(1);
    expect(ledger.comparisons).toBe(3);
  });

  it('counts repeated verdicts', () => {
    const ledger = new PreferenceLedger();
    ledger.record(['a', 'b']);
    ledger.record(['a', 'b']);
    expect(ledger.wins('a', 'b')).toBe(2);
    expect(ledger.wins('b', 'a')).toBe(0);
  });

  it('treats an even split as no preference in either direction', () => {
    const ledger = new PreferenceLedger();
    ledger.record(['a', 'b']);
    ledger.record(['b', 'a']);
    expect(ledger.prefers('a', 'b')).toBe(false);
    expect(ledger.prefers('b', 'a')).toBe(false);
  });

  it('takes the majority when the judge disagrees with itself', () => {
    const ledger = new PreferenceLedger();
    ledger.record(['a', 'b']);
    ledger.record(['a', 'b']);
    ledger.record(['b', 'a']);
    expect(ledger.prefers('a', 'b')).toBe(true);
    expect(ledger.contradictedPairs()).toEqual([['a', 'b']]);
  });

  it('finds a rock-paper-scissors triad', () => {
    const ledger = new PreferenceLedger();
    ledger.record(['a', 'b']);
    ledger.record(['b', 'c']);
    ledger.record(['c', 'a']);
    const found = ledger.intransitiveTriads();
    expect(found.triads).toEqual([['a', 'b', 'c']]);
    expect(found.truncated).toBe(false);
  });

  it('reports each cycle once rather than once per rotation', () => {
    const ledger = new PreferenceLedger();
    ledger.record(['a', 'b']);
    ledger.record(['b', 'c']);
    ledger.record(['c', 'a']);
    expect(ledger.intransitiveTriads().triads.length).toBe(1);
  });

  it('finds nothing in a consistent tournament', () => {
    const ledger = new PreferenceLedger();
    ledger.record(['a', 'b', 'c', 'd']);
    ledger.record(['a', 'c', 'd']);
    expect(ledger.intransitiveTriads().triads).toEqual([]);
  });

  it('marks the search truncated when the cap is hit', () => {
    const ledger = new PreferenceLedger();
    const nodes = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5'];
    for (let i = 0; i < nodes.length; i++) {
      ledger.record([nodes[i]!, nodes[(i + 1) % nodes.length]!]);
      ledger.record([nodes[i]!, nodes[(i + 2) % nodes.length]!]);
    }
    const found = ledger.intransitiveTriads(2);
    expect(found.triads.length).toBe(2);
    expect(found.truncated).toBe(true);
  });

  it('marks the search truncated when the work budget runs out', () => {
    const ledger = new PreferenceLedger();
    ledger.record(['a', 'b']);
    ledger.record(['b', 'c']);
    ledger.record(['c', 'a']);
    expect(ledger.intransitiveTriads(32, 1).truncated).toBe(true);
  });
});

describe('aggregateByMeanRank', () => {
  it('orders by mean rank across the states', () => {
    const canonical = new Map([
      ['a', 0],
      ['b', 1],
      ['c', 2],
    ]);
    const order = aggregateByMeanRank(
      [
        ['a', 'b', 'c'],
        ['a', 'c', 'b'],
      ],
      canonical,
    );
    expect(order).toEqual(['a', 'b', 'c']);
  });

  it('breaks a tie with the canonical order rather than the state order', () => {
    const canonical = new Map([
      ['b', 0],
      ['a', 1],
      ['c', 2],
    ]);
    const order = aggregateByMeanRank(
      [
        ['a', 'b', 'c'],
        ['b', 'a', 'c'],
      ],
      canonical,
    );
    expect(order).toEqual(['b', 'a', 'c']);
  });

  it('is unchanged by the order the states are supplied in', () => {
    const canonical = new Map([
      ['a', 0],
      ['b', 1],
      ['c', 2],
    ]);
    const states: string[][] = [
      ['c', 'a', 'b'],
      ['a', 'b', 'c'],
      ['b', 'c', 'a'],
    ];
    const forward = aggregateByMeanRank(states, canonical);
    const backward = aggregateByMeanRank([...states].reverse(), canonical);
    expect(backward).toEqual(forward);
  });
});

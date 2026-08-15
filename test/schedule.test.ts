import { describe, expect, it } from 'vitest';
import {
  ListwiseError,
  analyzeSchedule,
  assertScheduleUsable,
  buildSlidingWindows,
  type WindowRange,
} from '../src/index.js';

/** The loop everyone writes, which stops before it reaches position 0. */
function naiveWindows(n: number, windowSize: number, stride: number): WindowRange[] {
  const out: WindowRange[] = [];
  for (let start = n - windowSize; start >= 0; start -= stride) {
    out.push({ start, end: start + windowSize });
  }
  return out;
}

function covered(n: number, windows: readonly WindowRange[]): boolean[] {
  const hit = new Array<boolean>(n).fill(false);
  for (const w of windows) for (let p = w.start; p < w.end; p++) hit[p] = true;
  return hit;
}

describe('buildSlidingWindows', () => {
  it('covers every position for a wide range of parameters', () => {
    for (let n = 2; n <= 40; n++) {
      for (let w = 2; w <= 12; w++) {
        for (let s = 1; s < w; s++) {
          const windows = buildSlidingWindows(n, w, s);
          expect(covered(n, windows).every(Boolean), `n=${n} w=${w} s=${s}`).toBe(true);
        }
      }
    }
  });

  it('always clamps the last window to start at position 0', () => {
    for (let n = 2; n <= 40; n++) {
      for (let s = 1; s < 6; s++) {
        const windows = buildSlidingWindows(n, 6, s);
        expect(windows[windows.length - 1]!.start).toBe(0);
      }
    }
  });

  it('returns one window when it is at least as wide as the list', () => {
    expect(buildSlidingWindows(5, 20, 10)).toEqual([{ start: 0, end: 5 }]);
    expect(buildSlidingWindows(5, 5, 2)).toEqual([{ start: 0, end: 5 }]);
  });

  it('returns nothing for an empty list', () => {
    expect(buildSlidingWindows(0, 20, 10)).toEqual([]);
  });

  it('processes bottom up, so the first window is the tail of the list', () => {
    const windows = buildSlidingWindows(30, 10, 5);
    expect(windows[0]).toEqual({ start: 20, end: 30 });
    expect(windows[windows.length - 1]).toEqual({ start: 0, end: 10 });
  });

  it('makes consecutive windows overlap whenever the stride is smaller', () => {
    const windows = buildSlidingWindows(37, 8, 3);
    for (let i = 1; i < windows.length; i++) {
      expect(windows[i]!.start).toBeLessThan(windows[i - 1]!.end);
      expect(windows[i - 1]!.start).toBeLessThan(windows[i]!.end);
    }
  });
});

describe('the gap the naive loop leaves', () => {
  it('never covers position 0 when (n - windowSize) is not a multiple of stride', () => {
    const windows = naiveWindows(10, 3, 2);
    expect(windows.map((w) => w.start)).toEqual([7, 5, 3, 1]);
    expect(covered(10, windows)[0]).toBe(false);
  });

  it('reports the gap as an uncovered position', () => {
    const analysis = analyzeSchedule(10, naiveWindows(10, 3, 2), 10);
    expect(analysis.uncovered).toEqual([0]);
  });

  it('refuses the schedule and names the position', () => {
    const analysis = analyzeSchedule(10, naiveWindows(10, 3, 2), 10);
    try {
      assertScheduleUsable(analysis);
      throw new Error('expected a refusal');
    } catch (error) {
      expect(error).toBeInstanceOf(ListwiseError);
      expect((error as ListwiseError).code).toBe('strandedSchedule');
      expect((error as ListwiseError).message).toContain('never covers position 0');
    }
  });

  it('accepts the same parameters once the last window is clamped', () => {
    const analysis = analyzeSchedule(10, buildSlidingWindows(10, 3, 2), 10);
    expect(analysis.uncovered).toEqual([]);
    expect(() => assertScheduleUsable(analysis)).not.toThrow();
  });

  it('leaves the item at position 0 permanently first under the naive schedule', () => {
    // Reachability, not simulation: nothing can enter position 0, so whatever
    // the first stage put there is the answer no matter what the judge says.
    const analysis = analyzeSchedule(10, naiveWindows(10, 3, 2), 1);
    expect(analysis.strandedFromTopK).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});

describe('disjoint windows strand items permanently', () => {
  const windows = [
    { start: 8, end: 12 },
    { start: 4, end: 8 },
    { start: 0, end: 4 },
  ];

  it('splits the list into one component per window', () => {
    const analysis = analyzeSchedule(12, windows, 12);
    expect(analysis.components).toEqual([
      { start: 0, end: 4 },
      { start: 4, end: 8 },
      { start: 8, end: 12 },
    ]);
  });

  it('treats adjacent windows as disconnected because they share no position', () => {
    const analysis = analyzeSchedule(
      6,
      [
        { start: 3, end: 6 },
        { start: 0, end: 3 },
      ],
      6,
    );
    expect(analysis.components.length).toBe(2);
  });

  it('treats windows overlapping by a single position as connected', () => {
    const analysis = analyzeSchedule(
      6,
      [
        { start: 3, end: 6 },
        { start: 0, end: 4 },
      ],
      6,
    );
    expect(analysis.components).toEqual([{ start: 0, end: 6 }]);
  });

  it('is refused even though every position is covered', () => {
    const analysis = analyzeSchedule(12, windows, 12);
    expect(analysis.uncovered).toEqual([]);
    expect(() => assertScheduleUsable(analysis)).toThrow(ListwiseError);
  });

  it('names the disconnected groups in the refusal', () => {
    const analysis = analyzeSchedule(12, windows, 12);
    try {
      assertScheduleUsable(analysis);
      throw new Error('expected a refusal');
    } catch (error) {
      expect((error as ListwiseError).message).toContain('3 groups');
      expect((error as ListwiseError).message).toContain('0 to 3');
    }
  });

  it('reports the bottom block as unable to reach the top eight positions', () => {
    const analysis = analyzeSchedule(12, windows, 8);
    expect(analysis.strandedFromTopK).toEqual([8, 9, 10, 11]);
  });

  it('reports the top two blocks as unable to leave the top eight positions', () => {
    const analysis = analyzeSchedule(12, windows, 8);
    expect(analysis.frozenInTopK).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  it('does not improve with more passes, because the closure is the window', () => {
    // mixingPasses is the point at which nothing further becomes reachable.
    // One pass already exhausts a disjoint schedule.
    const analysis = analyzeSchedule(12, windows, 12);
    expect(analysis.mixingPasses).toBe(1);
  });
});

describe('pass budget analysis on a healthy schedule', () => {
  const n = 100;
  const windows = buildSlidingWindows(n, 20, 10);

  it('forms a single component over the whole list', () => {
    const analysis = analyzeSchedule(n, windows, n);
    expect(analysis.components).toEqual([{ start: 0, end: n }]);
    expect(() => assertScheduleUsable(analysis)).not.toThrow();
  });

  it('lets an item climb from the bottom to the top in a single pass', () => {
    const analysis = analyzeSchedule(n, windows, 10);
    expect(analysis.passesToTopK).toBe(1);
  });

  it('is far slower downward than upward', () => {
    // The asymmetry is structural. A pass ends at the top of the list, so an
    // item that enters the merge low is carried all the way up by the chain of
    // later windows, while an item at the top only moves as far as the last
    // window reaches and has to wait for the next pass to go further.
    const analysis = analyzeSchedule(n, windows, 50);
    expect(analysis.passesToLeaveTopK).toBe(5);
    expect(analysis.slowestLeaver).toBe(0);
  });

  it('reports how many passes make every position reachable', () => {
    const analysis = analyzeSchedule(n, windows, n);
    expect(analysis.mixingPasses).toBe(9);
  });

  it('needs one pass when a single window covers everything', () => {
    const analysis = analyzeSchedule(6, buildSlidingWindows(6, 20, 10), 6);
    expect(analysis.mixingPasses).toBe(1);
    expect(analysis.components).toEqual([{ start: 0, end: 6 }]);
  });

  it('reports no cutoff work when the caller consumes the whole ranking', () => {
    const analysis = analyzeSchedule(n, windows, n);
    expect(analysis.passesToTopK).toBe(0);
    expect(analysis.passesToLeaveTopK).toBe(0);
    expect(analysis.frozenInTopK).toEqual([]);
  });
});

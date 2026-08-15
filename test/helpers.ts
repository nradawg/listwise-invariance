import type { Judge, RankItem } from '../src/index.js';

export function ids(n: number, prefix = 'd'): string[] {
  return Array.from({ length: n }, (_, i) => `${prefix}${String(i).padStart(2, '0')}`);
}

/** No priorScore, so the canonical seed is the ids in ascending order. */
export function plainList(list: readonly string[]): RankItem[] {
  return list.map((id) => ({ id, text: `text of ${id}` }));
}

/** priorScore descending in the given order, so the seed is that order. */
export function scoredList(list: readonly string[]): RankItem[] {
  return list.map((id, i) => ({ id, text: `text of ${id}`, priorScore: list.length - i }));
}

export function relevanceOf(list: readonly string[]): Map<string, number> {
  // Later in the array means more relevant, so the truly best document starts
  // at the bottom of the seed order.
  return new Map(list.map((id, i) => [id, i]));
}

/** A perfectly transitive judge. Any correct merge must reproduce its order. */
export function oracleJudge(relevance: ReadonlyMap<string, number>): Judge {
  return (window) =>
    [...window]
      .sort((a, b) => {
        const diff = (relevance.get(b.id) ?? 0) - (relevance.get(a.id) ?? 0);
        return diff !== 0 ? diff : a.id < b.id ? -1 : 1;
      })
      .map((item) => item.id);
}

/**
 * A judge that always finds something it prefers to whatever is currently on
 * top, and promotes it.
 *
 * Given a rock-paper-scissors triple this never settles: the promoted item is
 * itself beaten by another member of the triple, so the head rotates forever.
 * It is a pure function of the window it is shown, which is what makes the
 * resulting oscillation a property of the judge rather than of randomness.
 */
export function restlessJudge(
  cycle: readonly [string, string, string],
  fallback: ReadonlyMap<string, number>,
): Judge {
  const beats = (a: string, b: string): boolean => {
    const ia = cycle.indexOf(a);
    const ib = cycle.indexOf(b);
    if (ia >= 0 && ib >= 0) return cycle[(ia + 1) % 3] === b;
    return (fallback.get(a) ?? 0) > (fallback.get(b) ?? 0);
  };
  return (window) => {
    const order = [...window];
    const head = order[0]!;
    const challenger = order.findIndex((item, i) => i > 0 && beats(item.id, head.id));
    if (challenger === -1) return order.map((item) => item.id);
    const moved = order.splice(challenger, 1)[0]!;
    order.unshift(moved);
    return order.map((item) => item.id);
  };
}

/** Deterministic shuffle so a failing case can be reproduced from its seed. */
export function shuffled<T>(input: readonly T[], seed: number): T[] {
  const out = [...input];
  let state = (seed * 2654435761) % 4294967296;
  const next = (): number => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    const a = out[i]!;
    out[i] = out[j]!;
    out[j] = a;
  }
  return out;
}

export function idsIn(items: readonly RankItem[]): string[] {
  return items.map((item) => item.id);
}

# listwise-invariance

Listwise LLM reranking whose output does not depend on the order the candidates arrived in, on how many passes you paid for, or on a window schedule that quietly cannot move some of the items at all.

```ts
import { rerank } from 'listwise-invariance';

const result = rerank(candidates, {
  judge: askTheModel,   // (window: RankItem[]) => ids, best first
  windowSize: 20,
  stride: 10,
  topK: 10,             // the prefix you actually intend to consume
  maxPasses: 4,
});

result.ranking;              // best first, every input item exactly once
result.convergence;          // { kind: 'fixedPoint' } | { kind: 'cycle', length: 3, states }
result.schedule.mixingPasses;    // 9 for 100 candidates at windowSize 20, stride 10
result.contradictedPairs;    // pairs the judge ordered both ways
result.intransitiveTriads;   // a above b above c above a
```

Zero runtime dependencies. The judge is injected, so nothing here talks to a model and the tests are exact.

## The sliding window merge is one bubble-sort pass, not a sort

Every listwise reranker does the same thing: slide a window of 20 up a list of 100, ask the model to order what is inside it, write the answer back, repeat. That is one pass of a bubble sort with a wide comparator. It floats the single best item to the top, and it does nothing else that a sort does.

Three failures follow, and none of them is visible to a test that runs one pass over a list of eight documents.

### The loop stops before it reaches position 0

This is the loop that gets written:

```ts
for (let start = n - windowSize; start >= 0; start -= stride) { ... }
```

With `n = 10`, `windowSize = 3`, `stride = 2` the starts are 7, 5, 3, 1, and then the loop ends. Position 0 is never inside a window. The document the first stage happened to rank first is now ranked first forever, no other document can ever reach the top, and running more passes repeats the same schedule. Nothing about this depends on the documents or on the model.

`buildSlidingWindows` clamps the final start to 0. `analyzeSchedule` reports `uncovered: [0]` for the loop above, and `assertScheduleUsable` refuses it by name before any judge call is made.

### A stride the size of the window strands items permanently

With `windowSize = 4` and `stride = 4` the windows tile the list without sharing a position: `[8,12)`, `[4,8)`, `[0,4)`. An item can only be reordered against the items in its own block, so it can never leave that block. Give the merge a perfect oracle judge, put the best document at position 11 of 12, and run fifty passes: it ends at position 8 and stays there. There is a test that does exactly that.

The check is a reachability question that has nothing to do with content. From a position `p` inside window `i`, one pass can leave the item anywhere in window `i`, then anywhere in any later window overlapping what has been reached, and so on. Every set produced that way is an interval, which collapses the whole analysis to merging the windows into overlap-connected components. Two intervals count as connected only when they share a position, so `[0,3)` and `[3,6)` are two components rather than one, which is precisely the stride-equals-window case.

A schedule whose components are anything other than a single block covering the whole list is refused, because a pair of items in different components is ordered by whatever the input array said and the judge is never asked about it.

### Climbing is fast, sinking is slow, and the pass budget decides the cutoff

For 100 candidates at `windowSize: 20, stride: 10`, an item at the very bottom can reach position 0 in a single pass: it enters the merge in the first window and the chain of later windows carries it all the way up. The reverse is not true. An item the first stage put at position 0 only reaches position 19 in the first pass, position 29 in the second, and needs nine passes before the bottom of the list is even reachable. `analyzeSchedule` reports that as `mixingPasses: 9`.

So a pass budget is a correctness parameter, not a quality knob. Asking for `topK: 50` with `maxPasses: 4` is refused:

```
maxPasses is 4 but this schedule needs 5 passes before the item at position 0,
which starts inside the top 50, can be pushed out of it. With a smaller budget
anything the first stage placed near the head of the list survives into the
result whatever the judge says about it.
```

`topK` changes nothing about what is computed. It is the prefix you say you intend to consume, and it is what the adequacy checks are measured against.

### Intransitive judgements oscillate, so more compute returns a different answer

LLM judges contradict themselves. Given a rock-paper-scissors triple, a judge that always promotes something it prefers to whatever is currently on top never settles: the promoted item is itself beaten by another member of the triple, and the head rotates forever. The package ships a deterministic judge with exactly that property in its tests, and the raw pass loop gives three different rankings after four, five and six passes. None of them is better than the others. A reranker that runs a fixed number of passes and returns the state it stopped on is returning a function of its budget.

Iterating a deterministic pass function over a finite set of permutations either reaches a fixed point or enters a cycle, and both are detectable. Each pass state is hashed and recorded, so:

* a repeat of the immediately previous state is a fixed point, returned as is;
* a repeat of an earlier state is a cycle, and the ranking returned is the mean rank of the items across the members of that cycle, with ties broken by the canonical order. That is a function of the cycle, so every pass budget large enough to detect it returns the same ranking. `onOscillation: 'error'` refuses instead, naming the period;
* running out of `maxPasses` without a repeat proves nothing, so it throws by default. `onNotConverged: 'aggregate'` averages what was seen, which is at least a stated policy rather than an accident of the budget.

The judge cache is load bearing here, not just a cost saving. Cycle detection is only meaningful if the pass function is a function. Answers are cached on the exact sequence of ids shown, which makes the pass deterministic within a call; with the cache off, a judge that answers the same window differently on different calls surfaces as a `notConverged` refusal rather than as a quietly arbitrary ranking. For the same reason the judge signature receives nothing but its window: no pass index, no window index, no offsets. A judge that could see those could vary its verdict with them and every stability claim above would be false.

### The input order is thrown away

The merge starts from the array the caller passed, so the same 100 documents in a different array order rank differently. The seed here is a canonical order derived from the items: `priorScore` descending, then id ascending, with plain relational comparison rather than `localeCompare`, which is locale sensitive and would seed differently on different machines. Ids are unique by the time this runs, so the comparator is a total order and the result never leans on sort stability.

Anything that would let the array order back in is refused rather than resolved:

* duplicate ids, because a verdict names ids and one of the two copies would be silently dropped;
* a `NaN` or infinite `priorScore`, because it compares false in both directions and the seed would fall back to array position for exactly the item whose place is least certain;
* a list where some items carry a `priorScore` and some do not, because there is no defensible place to put an unscored item relative to a scored one.

Every window verdict is also recorded as the pairs it implies, giving two direct measurements of the judge rather than of the ranking: `contradictedPairs`, where the same two documents were ordered both ways in different windows, and `intransitiveTriads`, the three-cycles in the resulting tournament. A strict preference requires a majority across every time the pair was seen, because treating an even split as a preference would let the tie-break rule invent cycles the judge never claimed.

## Known limitations

**Aggregating a cycle is a policy, not a discovery.** When the judge is intransitive there is no ranking that satisfies all of its verdicts. Mean rank over the cycle is stable and defensible, and it is still one choice among several. The cycle members are on the result so you can apply a different rule.

**Stability is stability of this pass function, not of the model.** The cache makes repeated windows return the repeated answer, which is what makes a fixed point detectable. A model at temperature above zero would give a different ranking on a fresh call with a cleared cache, and nothing here can detect that from inside a single call.

**Position bias inside a window is not corrected.** The output is invariant to the order the caller supplied, because that order is discarded. It is not invariant to the order items appear in within a window, which the schedule determines. Correcting that needs the judge queried on permuted windows, which multiplies the number of model calls and is not done here.

**The reachability analysis assumes an adversarial judge.** It asks where an item *could* go, so it will pass a schedule that is technically connected through a long thin chain of overlaps that a real judge would never walk. `mixingPasses` is the number to look at in that case.

**Schedule analysis is quadratic in the number of windows.** Fine for the tens of windows a rerank actually uses, and not intended for schedules with thousands.

**`topK` does not reduce work.** The whole list is reranked and the whole ranking is returned; `topK` only decides how strict the adequacy checks are. Pass fewer candidates if you want fewer judge calls.

## Test

```bash
npm install
npm test   # 98 tests: stranded schedules, oscillation, order invariance, judge contract
```

The adversarial cases are tested against the naive behaviour directly: the unclamped loop, the disjoint schedule with a perfect judge and fifty passes, and three consecutive pass budgets returning three different rankings.

## License

MIT

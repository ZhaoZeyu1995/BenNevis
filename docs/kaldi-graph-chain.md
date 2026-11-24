# What the Kaldi graph chain actually does

Read this before changing anything in `BenNevis/decode/graph.py`.

`egs/*/asr/run/prepare_graph.sh` builds the decoding graph with six OpenFST
and Kaldi binaries in two pipelines. Goal 2 replaces them, and a faithful
replacement needs to know what each one *means*, not just that it was in the
chain. Everything below was read from
`~/git/kaldi/src/{fstbin,fstext,decoder}` at OpenFST 1.8.4, which is the
version `tools/Makefile` pins.

## The chain, as shipped

```bash
# LG, kept with its disambiguation symbols
fsttablecompose L_disambig.fst G.fst \
  | fstdeterminizestar --use-log=true \
  | fstminimizeencoded \
  | fstarcsort --sort_type=ilabel > tmp/LG.fst

# TLG, which is what decode-faster reads
fsttablecompose T.fst tmp/LG.fst \
  | fstdeterminizestar --use-log=true \
  | fstrmsymbols disambig.int \
  | fstrmepslocal \
  | fstminimizeencoded \
  | fstarcsort --sort_type=ilabel > TLG.fst
```

The `mmictc-1` topology skips the second `fstdeterminizestar`, because that
topology is deliberately non-deterministic.

## Stage by stage

**`fsttablecompose`** is ordinary composition in the tropical semiring. Its
own usage says it is "more efficient for certain cases -- in particular,
where one of the FSTs (the left one, if `--match-side=left`) has large
out-degree". So it is an *optimisation*, not a different operation:
`k2.compose` is semantically the same thing. Measured on mini-librispeech,
composing L with G takes 1.9s in k2, so nothing here needs replacing.

**`fstdeterminizestar --use-log=true`** is two departures from a plain
determinize, and both matter.

The *star* is epsilon removal. `determinize-star.h` says it: "DeterminizeStar
implements determinization with epsilon removal, which we distinguish with a
star." The algorithm is subset construction where each element is an input
state plus a residual output string and weight, with input epsilons detected
and "treated specially" -- the epsilon closure is folded into the subset.
`k2.determinize` does not do this; its docstring says the input "is expected
to be epsilon-free" and that "epsilon will be treated as a normal symbol".

`--use-log=true` determinizes **in the log semiring**.
`DeterminizeStarInLog` in `fstext-utils-inl.h` casts the FST to `LogArc`,
determinizes, and casts back. In the log semiring `Plus` is log-add, so when
determinization merges paths carrying the same input string their
probabilities are **summed**; in the tropical semiring the cheapest is kept
instead. k2 has no equivalent: `weight_pushing_type` chooses how weights are
pushed, not which semiring the merge happens in.

`delta` defaults to `kDelta`, 1/1024, and decides when two weights count as
equal.

**`fstminimizeencoded`** is four operations, and `MinimizeEncoded` in
`fstext/fstext-utils.h` is short enough to quote in full:

```cpp
ArcMap(fst, QuantizeMapper<Arc>(delta));
EncodeMapper<Arc> encoder(kEncodeLabels | kEncodeWeights, ENCODE);
Encode(fst, &encoder);
internal::AcceptorMinimize(fst);
Decode(fst, encoder);
```

The quantisation is the first line and nothing else does it, so **this is
where CLAUDE.md's 1/1024 grid comes from**. The rule is
`TropicalWeight::Quantize` in OpenFST's `float-weight.h`:

```cpp
return TropicalWeightTpl<T>(std::floor(Value() / delta + 0.5F) * delta);
```

Round half up onto a grid of `delta`, in float32, with infinity left alone.
That is checkable rather than merely observed: `-log(0.1)` is 2.30258509,
and `floor(2.30258509 * 1024 + 0.5) / 1024` is `2358/1024` = **2.30273438**,
exactly the weight yesno's `LG.fst` carries where its `G.fst` carries
2.30258512.

The encoding folds each (label, weight) pair into a single label, which
turns a weighted transducer into an **unweighted acceptor** -- the only
thing `AcceptorMinimize` accepts -- and `Decode` unfolds it afterwards.

**Minimisation does not require a deterministic input, and this was worth
checking rather than assuming.** The natural reading is that
`fstdeterminizestar` and `fstminimizeencoded` are an inseparable pair,
because textbook minimisation is defined on a DFA. OpenFST's
`AcceptorMinimize` says otherwise:

```cpp
static constexpr auto revuz_props = kAcyclic | kIDeterministic;
if (fst->Properties(revuz_props, true) == revuz_props) {
  // Acyclic minimization (Revuz).
} else {
  // Either the FST has cycles, or it's generated from non-deterministic
  // input (which the Revuz algorithm can't handle), so use the cyclic
  // minimization algorithm of Hopcroft.
}
```

and `Minimize`'s own comment is explicit that it handles "deterministic
weighted automata and transducers, and also non-deterministic ones **if
they use an idempotent semiring**". The tropical semiring is idempotent --
`Plus` is `min`, and `min(a, a) = a`. So minimising the undeterminised
tropical LG that `compose_lg` now produces is a legitimate operation, not
a stage that became unavailable when determinization was dropped. The two
are **separable**, which is the opposite of what the shell pipeline
suggests.

Also deliberate: `fstminimizeencoded` exists *because* it does not push
weights. OpenFST's public `Minimize` calls
`Push(fst, REWEIGHT_TO_INITIAL, delta)` on a weighted acceptor first;
Kaldi's comment says "it will not push the weights, or the symbols. This
is better for our recipes, as we avoid ever pushing the weights." A
replacement should inherit that choice rather than reach for the
convenient library call.

One caveat Kaldi states and a replacement must not lose: it "will only
minimize optimally if your graphs are such that the symbols are as far to
the left as they can go, and the weights in combinable paths are the
same". So this is not guaranteed-minimal output; it is a cheap merge that
happens to work well on the graphs these recipes build.

**`fstrmsymbols disambig.int`** maps the listed labels to `<eps>`, and the
defaults are worth confirming rather than assuming: `fstrmsymbols.cc` sets
`remove_arcs = false` and `apply_to_output = false`, and its usage says
"With no options, replaces a subset of symbols with epsilon, wherever they
appear on the **input** side". So it is a relabelling, not a deletion --
`--remove-arcs=true` would send those arcs to a dead state and `Connect`
them away instead, which is a different graph. Note *which* list each
pipeline uses: `disambig.int` for the token-level graph and
`phones/disambig.int` for the phone-level one.

**Our equivalent is a threshold, not a list**, and the difference is an
assumption. `compose_lg` does `labels[labels >= first_disambig] = 0` with
`first_disambig = min(id of every symbol starting with "#")`. That matches
`fstrmsymbols` only while the `#` symbols occupy a **contiguous suffix** of
`phones.txt`. Measured, they do: yesno has them at 8..9 of 10 symbols and
mini-librispeech at 347..363 of 364. But if an ordinary phone ever sorted
above them, the threshold would map that phone to epsilon -- and the graph
would still compose, the decoder would still run and the transcriptions
would be quietly wrong. `build_graph` now refuses such a directory
instead.

**`fstrmepslocal`** is **not** full epsilon removal, and it is not only
epsilon removal either. Its usage: "Removes some (but not all) epsilons in
an algorithm that will always reduce the number of arcs+states." That
guarantee is the point -- full epsilon removal can grow a graph without
bound, and this is the conservative version chosen to avoid that.
`remove-eps-local.h` adds that the algorithm "is not optimal but is
reasonably clever", so the output is not canonical and two correct
implementations need not agree.

The part a replacement would miss: **it also combines pairs of
input-epsilon and output-epsilon arcs into one.** That is a transducer
operation with no counterpart in `k2.remove_epsilon`, which removes arcs
whose *input* is epsilon. An arc pair `(x:eps)(eps:y)` becomes `(x:y)`,
which is how this stage can shrink a graph that has few input epsilons to
remove.

**Called with no flags it is `RemoveEpsLocalSpecial`, not
`RemoveEpsLocal`.** `fstrmepslocal.cc` defaults `use_log = false` and
`stochastic_in_log = true`, and that combination selects the special
variant: equivalence in the **tropical** semiring, stochasticity preserved
in the **log** sense where possible. The header explains why this cannot
be had by casting -- "in that case we would no longer be able to guarantee
equivalence in the original semiring (this arises from what happens when we
combine identical arcs)" -- and notes that where the two cannot both hold,
equivalence wins.

The file's own worked example shows what the semiring choice costs. Given

```
0 1 0 0
0 0
1 0
```

a two-state FST with an epsilon arc and both states final, `fstrmepslocal`
gives a single state `0`, while `fstrmepslocal --use-log=true` gives
`0  -0.693147182` -- which is `-log(2)`, the two identical paths log-added.
Equivalence in one semiring is a different answer from equivalence in the
other, on a three-line FST.

**`fstarcsort --sort_type=ilabel`** sorts arcs by input label. Note this does
**not** make a k2 FSA `ARC_SORTED`; CLAUDE.md records that measurement.

## Where `BenNevis.decode.graph` currently diverges

| | Kaldi | `compose_lg` / `build_decoding_graph` |
| --- | --- | --- |
| compose L, G | `fsttablecompose`, tropical | `k2.compose` — equivalent |
| determinize LG | log semiring, with epsilon removal | **not done**, measured: see below |
| minimise LG | `fstminimizeencoded`, quantises to 1/1024 | *not done* — but **available**, see above: Hopcroft needs no determinism |
| disambiguation symbols | kept in `tmp/LG.fst`, **through** the T composition | stripped from LG **before** composing T |
| determinize T∘LG | again, log semiring | *not done* |
| epsilon removal | `RemoveEpsLocalSpecial`, never grows the graph, **also merges `(x:eps)(eps:y)` arc pairs** | `k2.remove_epsilon`, removes all input epsilons, merges no pairs, can grow it |
| minimise TLG | `fstminimizeencoded` | *not done* |

Three of these change the weights or the size of the result, and one
forecloses the faithful chain: **stripping the disambiguation symbols before
composing T removes exactly the symbols that make T∘LG determinisable.** They
exist for that purpose, which is why Kaldi carries them through the
composition and drops them after. Any replacement that wants Kaldi's second
determinization has to keep them until then.

## Measurements, mini-librispeech, this machine

```
L_disambig    1,176,708 states   1,589,729 arcs    4.2s to build
G               249,864 states   1,806,563 arcs   11.7s to build
LG            2,628,241 states   5,447,106 arcs    1.9s to compose + connect
```

For comparison, from the nightly's own log, Kaldi's whole first pipeline --
compose, determinize-star in the log semiring, minimise, arcsort -- takes
**19 seconds**, and the second takes 36. `k2.determinize` on that same 5.4M
arc LG takes **3358 seconds** -- 56 minutes, peaking at 2.1GiB -- and
returns 18,732,970 arcs, 3.4 times what it was given. It grinds rather than
explodes, so the CI job would have finished with another 40 minutes.

**So `compose_lg` does not determinize, as of the measurement above.**
Stripping the disambiguation symbols and removing the epsilons instead costs
1.5 seconds and gives 6,111,314 arcs -- 12% above the raw composition, and
three times smaller than the determinised graph. The whole of `compose_lg`
then runs in 2.9s, against Kaldi's 19 for its first pipeline.
`intersect_dense_pruned` does not require a deterministic graph.

What that gives up is not a tolerance. `--use-log=true` merges paths sharing
an input string by **adding their probabilities**, so Kaldi's arc weights can
be lower than any single path's; without determinization the cheapest path
stands alone. One-best Viterbi picks the same path either way, which is all
the decoder asks, but a lattice or any sum over paths would differ. Kaldi's
own LG is **substantially smaller** than our undeterminised composition,
and the counts are now measured rather than estimated. Run 45501684352
recorded them:

| | Kaldi states | Kaldi arcs | ours states | ours arcs | ratio |
| --- | --- | --- | --- | --- | --- |
| `L_disambig` | 1,176,707 | 1,589,728 | 1,176,708 | 1,589,729 | **exact** |
| `LG` | 1,383,213 | 4,367,485 | 2,160,124 | 6,111,314 | **1.40x** |
| `TLG` | 3,469,109 | 12,711,953 | 4,646,729 | 43,263,880 | **3.40x** |

`L_disambig` agrees to the arc, which is a check on the counting itself:
1,589,728 arcs plus one final line is the 1,589,729 lines recorded for the
Python build.

**The estimate this replaces was 33% too high, and it changed a verdict.**
`LG.fst.txt` is 138,155,979 bytes, which divided by a guessed line length
gave "about 5.8M arcs" and the conclusion that Kaldi arrives "within a few
percent" of our composition. It is 4,367,485. We are **40% larger on LG**,
not a few percent, and minimisation there buys about **29%** rather than
the 5% claimed below. This is exactly why `shapes` was added; quote the
table, not a byte count divided by anything.

L_disambig has **2** input epsilons in 1,589,729 arcs, from the optional
silence start arcs, and G's input side carries words plus `#0` with no
epsilons. So the missing epsilon removal is *not* what makes k2 slow here --
a hypothesis worth discarding explicitly, since the naming of
`determinizestar` invites it.

## Is minimisation worth reimplementing?

Not yet, and the reason is arithmetic rather than difficulty.

What it would cost: k2 has no `minimize`, so this is Hopcroft over an
encoded acceptor, written in Python with Numba to stand a chance --
O(e log v), about 6.1M arcs by 21, so some 10^8 partition operations, plus
the (label, weight) encoding and the float32 quantisation above to stay
faithful, with a correctness obligation on every graph the project builds.

What it would buy, now that the counts are measured: our `LG` is 6,111,314
arcs against Kaldi's 4,367,485, so **29%** -- and our `TLG` is 43,263,880
against 12,711,953, which is **3.4 times larger**. The earlier verdict here
was that this bought "about 5%" and so did not pay for a Hopcroft
implementation. That rested on an estimate that was 33% high, and **the
case is now materially stronger**, especially for `TLG`, which is the graph
the decoder actually reads and loads into memory.

It is still not obviously worth a Hopcroft implementation, because
`intersect_dense_pruned` needs the graph to be neither deterministic nor
minimal, so this is a cost question rather than a correctness one. But it
is a real cost, and three things now bear on it:

- **Whether the graphs are functionally equivalent at all.** The nightly
  gate answers that, and until it has run cleanly the size difference is
  not even known to be *only* a size difference.
- **Where the 3.4x comes from.** `LG` is 1.40x and `TLG` is 3.40x, so the
  topology composition is where most of it appears -- not the lexicon and
  grammar. That points at the second pipeline's epsilon-pair merging and
  minimisation rather than at determinization.
- **What it costs at decoding time**, in memory and in speed, which nothing
  has measured.

## Reimplementing `DeterminizeStar`

`determinize-star-inl.h` is 1,134 lines. The shape:

- an `Element` is `(input state, weight, residual output string)`;
- an output state is a canonicalised **subset** of elements, looked up in a
  hash so equal subsets share a state;
- `StringRepository` interns output label sequences as integers, with fast
  paths for the empty and single-label cases, plus `RemovePrefix` for
  factoring out a common prefix;
- a FIFO worklist drives it: pop a subset, take its epsilon closure, emit
  final weights, group outgoing arcs by input label, and push new subsets;
- `max_states` and `allow_partial` exist because this construction can blow
  up, and `delta` decides weight equality.

A Numba port is possible but it is not small: the hot structures are a hash
keyed on variable-length subsets and an arena of variable-length strings,
neither of which fits Numba's typed containers, so it would need hand-rolled
open addressing over flat integer arrays. Against Kaldi's 19 seconds, a
careful implementation within 3-10x would be fine and a pure-Python one at
50-200x would buy nothing. **Measure the alternatives in k2 before writing
it.**

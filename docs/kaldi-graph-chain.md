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

**`fstminimizeencoded`** minimises "after encoding [similar to fstminimize,
but **no weight-pushing**]". Encoding folds each (label, weight) pair into a
single label so the transducer becomes an acceptor that can be minimised as
an ordinary DFA, then decodes. Its `--delta` is documented as the "delta
likelihood used for quantization of weights", and **this is where the 1/1024
grid in CLAUDE.md comes from** -- yesno's `G.fst` carries 2.30258512 for
-log(0.1) while its `LG.fst` carries 2.30273438, which is 2358/1024.

**`fstrmsymbols disambig.int`** maps the listed labels to `<eps>`. Note
*which* list each pipeline uses: `disambig.int` for the token-level graph and
`phones/disambig.int` for the phone-level one.

**`fstrmepslocal`** is **not** full epsilon removal. Its usage: "Removes some
(but not all) epsilons in an algorithm that will always reduce the number of
arcs+states." That guarantee is the point -- full epsilon removal can grow a
graph without bound, and this is the conservative version chosen to avoid
that. Called with no flags, so it preserves tropical equivalence and
stochasticity in the log semiring.

**`fstarcsort --sort_type=ilabel`** sorts arcs by input label. Note this does
**not** make a k2 FSA `ARC_SORTED`; CLAUDE.md records that measurement.

## Where `BenNevis.decode.graph` currently diverges

| | Kaldi | `compose_lg` / `build_decoding_graph` |
| --- | --- | --- |
| compose L, G | `fsttablecompose`, tropical | `k2.compose` — equivalent |
| determinize LG | log semiring, with epsilon removal | **not done**, measured: see below |
| minimise LG | `fstminimizeencoded`, quantises to 1/1024 | *not done* |
| disambiguation symbols | kept in `tmp/LG.fst`, **through** the T composition | stripped from LG **before** composing T |
| determinize T∘LG | again, log semiring | *not done* |
| epsilon removal | `fstrmepslocal`, never grows the graph | `k2.remove_epsilon`, removes all, can grow it |
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
own LG is about the size of the undeterminised composition --
`LG.fst.txt` is 138,155,979 bytes, on the order of 5.8M lines -- which says
the `fstminimizeencoded` we skip is doing real work, and that k2's
determinization without it inflates where Kaldi's compacts.

L_disambig has **2** input epsilons in 1,589,729 arcs, from the optional
silence start arcs, and G's input side carries words plus `#0` with no
epsilons. So the missing epsilon removal is *not* what makes k2 slow here --
a hypothesis worth discarding explicitly, since the naming of
`determinizestar` invites it.

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

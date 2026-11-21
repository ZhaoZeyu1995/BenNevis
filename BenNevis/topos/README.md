# Topologies

A topology is the token-level state machine a phone is expanded into: how many
tokens it contributes, in what order they may occur, and how a frame may be
repeated or skipped. It determines both the FSTs used for training and decoding
and the size of the neural network's output layer, so the two must agree.

Each topology lives in one YAML file in [`specs/`](specs/). There is no Python
to write to add one.

```
python -m BenNevis.topos list                 # what is available
python -m BenNevis.topos describe 3state-skip # what the machine looks like
python -m BenNevis.topos spec ctc             # the file itself
```

## What a spec contains

```yaml
name: 2state
summary: |
  0 -> (a_0:a) -> (a_1:<eps>) -> 0

tokens_per_phone: ["{phone}_0", "{phone}_1"]

variants:
  lang:                                      # builds T.fst
    stride: 2
    arcs:
      - [hub, s0, "{phone}_0", "{phone}"]
      - [s0, s1, "{phone}_1", "<eps>"]
      - [s1, hub, "<eps>", "<eps>"]
  k2:                                        # builds k2/T.fst
    stride: 2
    disambig: skip
    arcs: ...
```

`hub` is state 0, which every phone leaves from and returns to and which
carries the `<blk>:<eps>` self-loop. `s0`, `s1`, ... are the states of the
phone being expanded; each phone is assigned a contiguous block of `stride`
state ids, so `s0` means a different state for each phone. `{phone}` in a
label is the phone being expanded, and in `inter_phone_arcs` `{dst_phone}` is
the phone owning the destination block.

`tokens_per_phone` is the inventory: it is used both to build `tokens.txt` and
to check every arc's input label, so a spec whose arcs use a token the
inventory does not list is rejected at load time rather than producing a
network that cannot express its own topology.

## The two variants

Every topology defines two FSTs, which is why there used to be two scripts per
topology:

| Variant | File | Input symbols |
| --- | --- | --- |
| `lang` | `T.fst` | `tokens_disambig.txt`: includes `<eps>` and `#N` |
| `k2` | `k2/T.fst` | `k2/tokens.txt`: exactly the network's outputs |

The `lang` variant may use `<eps>` input labels, which keeps it deterministic
and lets it compose with Kaldi's `L_disambig.fst`. The `k2` variant may not:
its input alphabet *is* the network's output layer, and a network cannot emit
`<eps>`. Removing the epsilons with `fstrmepsilon` instead would produce a far
larger FST, so the `k2` variants give up determinism rather than size. This is
checked when a spec is loaded.

## In a language directory

`prepare_topo.sh` copies the spec into the language directory it prepares, as
`k2/topo.yaml`, and `Lang.load_topo` expands *that* into the training FSA
directly:

```
spec + k2/tokens.txt + k2/phones.txt  ->  k2.Fsa
```

So training needs neither OpenFST nor a shell, and a language directory keeps
describing the topology it was actually built with even if the bundled spec of
the same name later changes. `k2/T.fst` is still compiled, because the
decoding graph is built with Kaldi and because it is a useful thing to inspect
with the OpenFST tools, but nothing in training reads it. A directory prepared
before `k2/topo.yaml` existed still loads, through a `fstprint` fallback that
warns.

## Adding a topology

1. Write `specs/<name>.yaml`.
2. Run `python -m BenNevis.topos fst <name> path/to/phones.txt` and read it.
3. Add a golden file: `python tests/regenerate_topo_golden.py` writes one for
   every topology, so commit only the new files and leave the rest unchanged.
4. The structural tests in `tests/topos/test_build.py` pick the new topology up
   automatically; `prepare_topo.sh` needs no change at all.

## History

This replaces fifteen `get_token_fst_*.py` scripts, which each printed OpenFST
text directly, shared their logic by copy and paste, and had to be registered
in `prepare_topo.sh` by hand along with an `awk` pipeline to derive the token
inventory. The specs reproduce every one of those scripts' output byte for
byte; `tests/golden/topos/` holds the recorded output and is what pins that.

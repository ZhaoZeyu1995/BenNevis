"""
Tests for the k2 decoding graph.

Tier 2: these need k2, and nothing else -- no Kaldi, no OpenFST. The
lexicon and the grammar are built in Python by `BenNevis.lm`, so the whole
decoding graph can be constructed from a dictionary and an ARPA model.

The comparison with Kaldi's own `LG.fst` is bounded rather than exact, and
the bound is Kaldi's own quantisation: `fstminimizeencoded` rounds every
weight to a multiple of OpenFST's `kDelta = 1/1024`, so yesno's `G.fst`
carries 2.30258512 and its `LG.fst` carries 2358/1024 = 2.30273438. The
k2 graph keeps full precision, which is better and different.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

k2 = pytest.importorskip("k2")

from BenNevis.decode.graph import (  # noqa: E402
    QUANTISATION_DELTA,
    build_decoding_graph,
    compose_lg,
    load_openfst_text,
)
from BenNevis.lm import build_fst_text, read_arpa  # noqa: E402
from BenNevis.lm.lexicon import build_disambig_fst_text, read_lexicon  # noqa: E402
from BenNevis.utils.fingerprint import shortest_distance  # noqa: E402

pytestmark = pytest.mark.k2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE = REPO_ROOT / "tests" / "golden" / "reference" / "yesno"
YESNO = REPO_ROOT / "egs" / "yesno" / "asr" / "input"

#: YES and NO as yesno's lexicon pronounces them.
PRONUNCIATIONS = {"YES": ("Y", "E", "S"), "NO": ("N", "O")}


def symbol_table(path):
    table = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            table[fields[0]] = int(fields[1])
    return table


def as_dicts(fsa):
    """A k2 FSA as (arcs by source, final weights), like `fingerprint`'s."""
    arcs, finals = {}, {}
    values = fsa.arcs.values().cpu()
    scores = fsa.scores.detach().cpu()
    for index in range(values.shape[0]):
        source = int(values[index][0])
        destination = int(values[index][1])
        label = int(values[index][2])
        # k2 stores scores, which are negated weights.
        weight = -float(scores[index])
        if label == -1:
            finals[source] = weight
        else:
            arcs.setdefault(source, []).append((destination, label, weight))
    return arcs, finals


def read_openfst(path):
    """Kaldi's own FST, in the same shape, with its start state."""
    arcs, finals, start = {}, {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 4:
            source = int(fields[0])
            if start is None:
                start = source
            arcs.setdefault(source, []).append(
                (
                    int(fields[1]),
                    int(fields[2]),
                    float(fields[4]) if len(fields) > 4 else 0.0,
                )
            )
        elif fields:
            finals[int(fields[0])] = float(fields[1]) if len(fields) > 1 else 0.0
    return arcs, finals, start


@pytest.fixture(scope="module")
def tables():
    return (
        symbol_table(REFERENCE / "lang" / "words.txt"),
        symbol_table(REFERENCE / "lang" / "phones.txt"),
    )


@pytest.fixture(scope="module")
def drawn(tables):
    """Word sequences the lexicon pronounces, with their phone sequences."""
    from BenNevis.utils.fingerprint import (
        position_dependent_table,
        sample_lexicon_sequences,
    )

    _words, phones = tables
    return sample_lexicon_sequences(
        read_lexicon(YESNO / "lexicon.txt"),
        count=24,
        position_dependent_phones=position_dependent_table(phones.keys()),
    )


@pytest.fixture(scope="module")
def kaldi_fingerprint(tables, drawn):
    """What Kaldi's own LG gives each drawn sequence."""
    from BenNevis.utils.fingerprint import fingerprint_phone_sequences

    _words, phones = tables
    arcs, finals, start = read_openfst(REFERENCE / "graph" / "ctc" / "LG.fst.txt")
    return fingerprint_phone_sequences(arcs, finals, phones, start, drawn)


@pytest.fixture(scope="module")
def built(tables):
    """LG for yesno, from a dictionary and an ARPA model and nothing else."""
    words, phones = tables
    lexicon = load_openfst_text(
        build_disambig_fst_text(
            read_lexicon(YESNO / "lexicon.txt"),
            words,
            phones,
            sil_prob=0.0,
        )
    )
    grammar = load_openfst_text(build_fst_text(read_arpa(YESNO / "task.arpabo"), words))
    first = min(value for key, value in phones.items() if key.startswith("#"))
    return compose_lg(lexicon, grammar, first), first


class TestLoadOpenfstText:
    def test_a_four_field_arc_gets_its_missing_weight(self):
        """
        `fstprint` and the builders after it omit the semiring One, and k2
        wants five fields. Filling that in here is what replaces
        `Lang.load_lexicon`'s `fstprint | awk` pipeline -- one of the
        project's remaining POSIX-only sites.
        """
        fsa = load_openfst_text("0\t1\t1\t1\n1\n", acceptor=False)
        assert fsa.num_arcs >= 1

    def test_a_five_field_arc_keeps_its_weight(self):
        fsa = load_openfst_text("0\t1\t1\t1\t2.5\n1\n", acceptor=False)
        arcs, _finals = as_dicts(fsa)
        assert arcs[0][0][2] == pytest.approx(2.5)

    def test_the_result_is_arc_sorted(self):
        """
        `k2.intersect_dense` documents that its graphs must be, and
        `GraphLoss.get_den_graph` not doing it is a known violation.
        """
        fsa = load_openfst_text("0\t1\t2\t2\n0\t1\t1\t1\n1\n")
        assert fsa.properties & k2.fsa_properties.ARC_SORTED


class TestComposition:
    def test_the_disambiguation_symbols_are_gone(self, built):
        """
        They exist only to make the composition determinisable. Leaving one
        in would have the decoder wait for a `#1` no acoustic model emits.
        """
        graph, first = built
        assert int((graph.labels >= first).sum()) == 0

    def test_only_lexicon_phones_remain(self, built, tables):
        """
        `<SIL>` and `<UNK>` are in the lexicon but not in yesno's grammar,
        so composition should have removed them along with their phones.
        """
        graph, _first = built
        _words, phones = tables
        labels = set(graph.labels.tolist()) - {-1, 0}
        expected = {phones[p] for pron in PRONUNCIATIONS.values() for p in pron}
        assert labels == expected

    def test_it_is_not_empty(self, built):
        graph, _first = built
        assert graph.num_arcs > 0
        assert graph.shape[0] > 1


class TestFunctionalEquivalenceWithKaldi:
    """
    The same phone sequences must cost the same, within Kaldi's own
    quantisation.
    """

    @pytest.mark.parametrize(
        "sentence",
        [
            ["YES"],
            ["NO"],
            ["YES", "NO"],
            ["NO", "YES"],
            ["YES", "YES", "NO"],
            ["NO"] * 5,
            ["YES", "NO", "NO", "YES"],
        ],
    )
    def test_the_weights_agree_within_quantisation(self, built, tables, sentence):
        graph, _first = built
        _words, phones = tables
        sequence = [
            phones[phone] for word in sentence for phone in PRONUNCIATIONS[word]
        ]

        kaldi_arcs, kaldi_finals, kaldi_start = read_openfst(
            REFERENCE / "graph" / "ctc" / "LG.fst.txt"
        )
        theirs = shortest_distance(kaldi_arcs, kaldi_finals, sequence, kaldi_start)
        arcs, finals = as_dicts(graph)
        ours = shortest_distance(arcs, finals, sequence, 0)

        assert theirs is not None, f"Kaldi rejects {sentence}"
        assert ours is not None, f"this graph rejects {sentence}"
        # One word costs one arc plus the final, so the bound counts both.
        bound = (len(sentence) + 1) * QUANTISATION_DELTA
        assert abs(theirs - ours) <= bound, (
            f"{' '.join(sentence)}: Kaldi {theirs:.6f}, this {ours:.6f}, "
            f"gap {abs(theirs - ours):.6f} exceeds {bound:.6f}"
        )

    def test_the_gap_is_kaldis_quantisation_and_not_ours(self):
        """
        Names the cause rather than just tolerating the symptom.
        `fstminimizeencoded` rounds to a multiple of 1/1024, so yesno's
        `G.fst` weight becomes 2358/1024 in `LG.fst`.
        """
        import struct

        grammar = read_openfst(REFERENCE / "graph" / "ctc" / "G.fst.txt")
        composed = read_openfst(REFERENCE / "graph" / "ctc" / "LG.fst.txt")
        g_weight = grammar[0][0][0][2]
        lg_weight = composed[0][0][0][2]
        assert g_weight != lg_weight
        quantised = round(g_weight / QUANTISATION_DELTA) * QUANTISATION_DELTA
        as_float32 = struct.unpack("f", struct.pack("f", quantised))[0]
        assert f"{as_float32:.9g}" == f"{lg_weight:.9g}"

    def test_the_delta_is_openfsts(self):
        assert QUANTISATION_DELTA == 1.0 / 1024


class TestWithTheTopology:
    def test_the_graph_takes_token_labels(self, tables):
        """
        The topology is what makes this a *token* graph: it maps the
        network's output labels to phones, so composing it on the left
        gives an FSA whose input alphabet is the output layer.
        """
        from BenNevis.topos import build as topo_build
        from BenNevis.topos import spec as topo_spec

        words, phones = tables

        lexicon = load_openfst_text(
            build_disambig_fst_text(
                read_lexicon(YESNO / "lexicon.txt"),
                words,
                phones,
                sil_prob=0.0,
            )
        )
        grammar = load_openfst_text(
            build_fst_text(read_arpa(YESNO / "task.arpabo"), words)
        )
        first = min(v for key, v in phones.items() if key.startswith("#"))

        # The topology is the one input that never came from Kaldi: it is
        # expanded from the YAML spec, which is already Python.
        topology_spec = topo_spec.load("ctc")
        phone_list = topo_build.read_phone_table(REFERENCE / "lang" / "phones.txt")
        tokens = {
            token: index
            for index, token in enumerate(
                topo_build.build_tokens(topology_spec, phone_list)
            )
        }
        text = topo_build.build_openfst_int_text(
            topology_spec,
            "k2",
            phone_list,
            isymbols=tokens,
            osymbols=phones,
        )
        topology = load_openfst_text(text)
        assert topology.num_arcs > 0

        graph = build_decoding_graph(topology, lexicon, grammar, first)
        assert graph.num_arcs > 0
        # Its input alphabet is the network's output layer, not phones.
        assert int(graph.labels.max()) < len(tokens)


class TestItDoesNotDeterminize:
    """
    Skipping `k2.determinize` is a measured decision, pinned here.

    On mini-librispeech's 206,510-word lexicon and its pruned 3-gram,
    `k2.determinize` took **3358 seconds** and turned 5,447,106 arcs into
    18,732,970, against the 19 seconds Kaldi spends determinising *in the
    log semiring* and minimising as well. Skipping it costs 1.5s and
    leaves 6,111,314 arcs, and the whole of `compose_lg` then runs in
    2.9s.

    Reinstating it would not break anything visibly -- the graph would
    still be correct and the tests would still pass. It would just cost
    an hour per language directory, which in CI reads as a hang. So the
    absence is asserted rather than left to a comment.
    """

    def test_determinize_is_never_called(self, tables, monkeypatch):
        """
        Fails loudly if `k2.determinize` comes back.

        Patching k2 rather than inspecting the source, so a call added
        through any path is caught.
        """
        words, phones = tables

        def refuse(*args, **kwargs):
            raise AssertionError(
                "compose_lg called k2.determinize: 3358s and 3.4x the arcs "
                "on mini-librispeech, against Kaldi's 19s for more work. "
                "See docs/kaldi-graph-chain.md before reinstating it."
            )

        monkeypatch.setattr(k2, "determinize", refuse)
        lexicon = load_openfst_text(
            build_disambig_fst_text(
                read_lexicon(YESNO / "lexicon.txt"), words, phones, sil_prob=0.0
            )
        )
        grammar = load_openfst_text(
            build_fst_text(read_arpa(YESNO / "task.arpabo"), words)
        )
        first = min(value for key, value in phones.items() if key.startswith("#"))
        compose_lg(lexicon, grammar, first)

    def test_the_graph_is_epsilon_free(self, built):
        """
        The invariant that makes skipping it safe.

        An undeterminised graph is fine for `intersect_dense_pruned`, but
        an input epsilon is not: the acoustic model has no label for it,
        so one surviving here would be a symbol the decoder can never
        consume. `remove_epsilon` is what guarantees this, and it is the
        step that replaces determinization rather than accompanying it.
        """
        graph, _first = built
        assert int((graph.labels == 0).sum()) == 0


class TestEquivalenceOnLexiconDrawnSequences:
    """
    The gate that generalises past hardcoded sentences.

    `TestFunctionalEquivalenceWithKaldi` asks about seven sentences whose
    pronunciations are written into this file as `PRONUNCIATIONS`. That
    works for a four-word lexicon and is useless for mini-librispeech's
    206,510 entries, which is the one corpus where a graph replacement
    matters -- and whose `LG.fst.txt` is 138,155,979 bytes, so the capture
    keeps its digest and not its bytes and there is no committed FST to
    compare against at all.

    `sample_lexicon_sequences` is the machinery for that: word sequences
    drawn reproducibly from the words the *lexicon* pronounces, with the
    phone sequence `L` accepts for each, suffixed by the project's own
    `position_dependent` so the expansion cannot drift from the lexicon
    FST it probes. Sampling from `words.txt` instead would be no use,
    because a phone-input FST rejects almost every sequence drawn that way
    and a fingerprint of rejections compares nothing.

    These tests exercise that machinery here, where Kaldi's `LG.fst.txt`
    is 20KB and committed, so it is checked against a real Kaldi graph
    before mini-librispeech has to rely on it. Note that a sequence the
    *lexicon* permits can still be rejected by `G`: yesno's grammar is
    restrictive, so about a third of the draws survive. That is expected,
    and it is why the fingerprint records only what was accepted.
    """

    def test_kaldis_graph_accepts_some_of_them(self, kaldi_fingerprint, drawn):
        """
        Guards the whole class against being vacuous.

        If the sampler produced sequences Kaldi rejected outright, every
        comparison below would pass by having nothing to compare -- the
        same shape as the `G.fst` gate that built its path wrongly and
        skipped unconditionally.
        """
        assert drawn, "the sampler drew nothing"
        assert kaldi_fingerprint, (
            "Kaldi's LG accepted none of the lexicon-drawn sequences, so "
            "the comparison would be empty"
        )

    def test_our_graph_agrees_within_quantisation(
        self, built, tables, drawn, kaldi_fingerprint
    ):
        """
        Every sequence Kaldi scores, we must score the same, within the
        1/1024 grid `fstminimizeencoded` puts Kaldi's weights on.
        """
        graph, _first = built
        _words, phones = tables
        arcs, finals = as_dicts(graph)

        compared = 0
        for words, phone_sequence in drawn:
            key = " ".join(words)
            if key not in kaldi_fingerprint:
                continue
            sequence = [phones[phone] for phone in phone_sequence]
            ours = shortest_distance(arcs, finals, sequence, 0)
            assert ours is not None, f"this graph rejects {key}"
            theirs = kaldi_fingerprint[key]
            bound = (len(words) + 1) * QUANTISATION_DELTA
            assert abs(theirs - ours) <= bound, (
                f"{key}: Kaldi {theirs:.6f}, this {ours:.6f}, gap "
                f"{abs(theirs - ours):.6f} exceeds {bound:.6f}"
            )
            compared += 1
        assert compared, "nothing was compared"

    def test_the_sequences_are_reproducible(self, drawn):
        """
        Two implementations must be asked the same questions, so the draw
        is seeded. A fingerprint recorded from a different draw compares
        nothing, and would look like agreement on an empty set.
        """
        from BenNevis.utils.fingerprint import (
            position_dependent_table,
            sample_lexicon_sequences,
        )

        again = sample_lexicon_sequences(
            read_lexicon(YESNO / "lexicon.txt"),
            count=24,
            position_dependent_phones=position_dependent_table({"Y": 1}.keys()),
        )
        assert again == drawn

    def test_the_phones_come_from_the_lexicon(self, drawn, tables):
        """
        Every phone drawn must be in the table the FST is written against,
        or the sequence could never be scored and the draw is the bug.
        """
        _words, phones = tables
        for words, phone_sequence in drawn:
            unknown = [p for p in phone_sequence if p not in phones]
            assert not unknown, f"{' '.join(words)} wants {unknown}"


@pytest.fixture(scope="module")
def our_tlg(tables, built):
    """
    `TLG` for yesno, from a dictionary, an ARPA model and a topology.

    The same three parts Kaldi's `prepare_graph.sh` composes, by different
    algorithms -- which is the whole question this file exists to settle.
    """
    from BenNevis.topos import build as topo_build
    from BenNevis.topos import spec as topo_spec

    words, phones = tables
    first = min(value for key, value in phones.items() if key.startswith("#"))
    lexicon = load_openfst_text(
        build_disambig_fst_text(
            read_lexicon(YESNO / "lexicon.txt"), words, phones, sil_prob=0.0
        )
    )
    grammar = load_openfst_text(build_fst_text(read_arpa(YESNO / "task.arpabo"), words))
    # The topology's k2 variant: tokens in, phones out, with neither the
    # disambiguation symbols nor `<eps>`, because `compose_lg` has already
    # stripped the former and a network cannot emit the latter.
    loaded = topo_spec.load("ctc")
    out_phones = {
        key: value for key, value in phones.items() if not key.startswith("#")
    }
    phone_list = [key for key, _ in sorted(out_phones.items(), key=lambda kv: kv[1])]
    tokens = {
        token: index
        for index, token in enumerate(
            topo_build.build_tokens(
                loaded, phone_list, with_eps=False, with_disambig=False
            )
        )
    }
    topology = load_openfst_text(
        topo_build.build_openfst_int_text(
            loaded, "k2", phone_list, isymbols=tokens, osymbols=out_phones
        )
    )
    return build_decoding_graph(topology, lexicon, grammar, first)


@pytest.fixture(scope="module")
def probes(our_tlg):
    """Both TLGs, inverted and epsilon-free, ready to search by words."""
    from BenNevis.decode.graph import prepare_output_probe

    kaldi = load_openfst_text(
        (REFERENCE / "graph" / "ctc" / "TLG.fst.txt").read_text(),
        acceptor=False,
    )
    return prepare_output_probe(kaldi), prepare_output_probe(our_tlg)


class TestTlgEquivalenceByOutput:
    """
    `TLG` compared as the function a decoder uses.

    This is the graph `decode-faster` actually reads, and the question
    that matters is what it costs to recognise a word sequence. Under a
    CTC topology that is a **minimum over every token realisation** of
    those words -- blanks anywhere, labels repeated -- so the search runs
    on the *output* side. Probing one hand-picked token sequence would
    compare an alignment no decoder need take, and two graphs could agree
    on it while differing on the paths that matter.

    Probing the output side also sidesteps the token tables entirely:
    `prepare_topo.sh` writes `tokens.txt` with `<eps>` for Kaldi and
    `k2/tokens.txt` without it for us, so every token id differs by one,
    whereas `words.txt` is the same table on both sides.

    On mini-librispeech our `TLG` is 43,263,880 arcs against Kaldi's
    (estimated) 14M, so this gate is where that difference will have to
    be explained -- but the arithmetic cannot run there yet, because the
    ARPA model is downloaded rather than committed. yesno is where the
    machinery gets checked against a real Kaldi graph first.
    """

    @pytest.mark.parametrize(
        "sentence",
        [
            ["YES"],
            ["NO"],
            ["YES", "NO"],
            ["NO", "YES"],
            ["YES", "YES", "NO"],
            ["NO", "NO", "NO"],
            ["YES", "YES", "YES", "YES"],
        ],
    )
    def test_the_weights_agree_within_quantisation(self, probes, tables, sentence):
        from BenNevis.decode.graph import weight_of_word_sequence

        kaldi_probe, our_probe = probes
        words, _phones = tables
        ids = [words[word] for word in sentence]

        theirs = weight_of_word_sequence(kaldi_probe, ids)
        ours = weight_of_word_sequence(our_probe, ids)
        assert theirs is not None, f"Kaldi's TLG rejects {sentence}"
        assert ours is not None, f"our TLG rejects {sentence}"

        bound = (len(sentence) + 1) * QUANTISATION_DELTA
        assert abs(theirs - ours) <= bound, (
            f"{' '.join(sentence)}: Kaldi {theirs:.6f}, ours {ours:.6f}, "
            f"gap {abs(theirs - ours):.6f} exceeds {bound:.6f}"
        )

    def test_a_sequence_the_grammar_rejects_returns_none(self, probes, tables):
        """
        None rather than infinity, and rather than a crash: a rejection is
        a fact about the lexicon and grammar, so a caller records it as
        absent. `<SIL>` is in yesno's lexicon but not its grammar.
        """
        from BenNevis.decode.graph import weight_of_word_sequence

        kaldi_probe, our_probe = probes
        words, _phones = tables
        if "<SIL>" not in words:
            pytest.skip("no out-of-grammar word to try")
        ids = [words["<SIL>"]]
        assert weight_of_word_sequence(kaldi_probe, ids) is None
        assert weight_of_word_sequence(our_probe, ids) is None

    def test_the_probe_is_epsilon_free(self, probes):
        """
        The precondition that makes the lattice acyclic, and so makes
        `k2.shortest_path` applicable at all. Most of a token graph's arcs
        emit no word -- 20 of 29 in yesno's TLG -- and as input labels
        those form cycles, including blank self-loops.
        """
        for probe in probes:
            assert int((probe.labels == 0).sum()) == 0

    def test_longer_sequences_cost_more(self, probes, tables):
        """
        A sanity check on the probe itself: if it returned a constant, or
        the weight of the empty path, every comparison above would agree
        while measuring nothing.
        """
        from BenNevis.decode.graph import weight_of_word_sequence

        _kaldi, our_probe = probes
        words, _phones = tables
        one = weight_of_word_sequence(our_probe, [words["YES"]])
        four = weight_of_word_sequence(our_probe, [words["YES"]] * 4)
        assert one is not None and four is not None
        assert four > one * 2


class TestWeightOfLabelSequence:
    """
    The probe that survives Kaldi's own LG, which the composition did not.

    Three routes were tried on the real graph and the first two fail.
    `k2.intersect` with `treat_epsilons_specially=False` treats label 0 as
    an ordinary symbol, so every path carrying an input epsilon is
    rejected -- on mini-librispeech that was *every* path, and the gate
    reported `compared LG on 0 sequences`. With the flag True the
    epsilons match correctly, but an n-gram back-off arc is an input
    epsilon and those form cycles, so `k2.shortest_path` fails with
    `dest_state > state_idx01 (17 vs. 17)`, an error that says nothing
    about epsilons. Removing them first is the third route and the
    dearest: `k2.remove_epsilon` on the 43M-arc TLG does not fit a 16GiB
    runner.

    So this is a Dijkstra over (state, position) pairs, and the property
    it rests on is that the weights are non-negative -- every arc of a
    graph built from `-log` probabilities costs nothing or more, so a
    cycle can only make a path dearer. These tests put an epsilon
    self-loop on the path deliberately, because that is the shape no
    composition-and-shortest-path can answer.
    """

    @staticmethod
    def cyclic():
        """A B, reachable only through a state with an epsilon self-loop."""
        import k2

        return k2.Fsa.from_str(
            "0 1 1 0 0.0\n"
            "1 1 0 0 -0.25\n"  # an epsilon self-loop, cost 0.25
            "1 2 0 0 -0.5\n"  # an epsilon arc onward, cost 0.5
            "2 3 2 0 0.0\n"
            "3 4 -1 0 0.0\n"
            "4\n",
            acceptor=False,
        )

    def test_it_reads_a_sequence_across_an_epsilon_cycle(self):
        from BenNevis.decode.graph import weight_of_label_sequence

        # 0.5 for the epsilon arc onward, and the self-loop not taken:
        # going round it is never cheaper, which is the whole argument.
        assert weight_of_label_sequence(self.cyclic(), [1, 2]) == pytest.approx(0.5)

    def test_the_same_graph_defeats_shortest_path(self):
        """
        The premise, pinned. If a k2 release ever made `shortest_path`
        tolerate this, the Dijkstra would be redundant rather than
        necessary -- and that should be a failing test, not a thing
        nobody notices.
        """
        import k2

        lattice = k2.connect(
            k2.intersect(
                k2.arc_sort(self.cyclic()),
                k2.arc_sort(k2.linear_fsa([1, 2])),
                treat_epsilons_specially=True,
            )
        )
        with pytest.raises(RuntimeError):
            k2.shortest_path(k2.create_fsa_vec([lattice]), use_double_scores=True)

    def test_a_sequence_it_cannot_read_is_absent(self):
        from BenNevis.decode.graph import weight_of_label_sequence

        # Both halves matter: a prefix must not be accepted, because the
        # final arc is only taken once the sequence is exhausted, and the
        # order must matter or the probe is comparing a bag of labels.
        assert weight_of_label_sequence(self.cyclic(), [1]) is None
        assert weight_of_label_sequence(self.cyclic(), [2, 1]) is None

    def test_a_prefix_is_not_accepted_where_a_final_arc_is_reachable(self):
        """
        The hole the other rejection test could not see.

        `test_a_sequence_it_cannot_read_is_absent` asserts a prefix is
        rejected, but on that graph the super-final arc is only reachable
        after the whole sequence, so dropping the `position == length`
        check entirely still passed it -- a mutation that survived.
        Here the graph is final *both* after A and after A B, so reading
        A B must cost what A B costs and not what stopping early would.
        """
        import k2

        from BenNevis.decode.graph import weight_of_label_sequence

        graph = k2.Fsa.from_str(
            "0 1 1 0 0.0\n"
            "1 3 -1 0 0.0\n"  # final after A alone
            "1 2 2 0 -0.5\n"
            "2 3 -1 0 0.0\n"  # final after A B
            "3\n",
            acceptor=False,
        )
        assert weight_of_label_sequence(graph, [1]) == pytest.approx(0.0)
        assert weight_of_label_sequence(graph, [1, 2]) == pytest.approx(0.5)

    def test_the_epsilon_label_is_the_callers_choice(self):
        """
        Label 0 is `<blk>` and not `<eps>` on a k2 topology's input side,
        so a probe that hardcoded 0 would skip blanks on one graph and
        epsilons on another while looking identical.
        """
        import k2

        from BenNevis.decode.graph import weight_of_label_sequence

        # A, then label 2, then C. Whether the middle arc is skipped
        # decides which sequence the graph reads, and nothing else about
        # the graph changes.
        graph = k2.Fsa.from_str(
            "0 1 1 0 0.0\n" "1 2 2 0 -0.5\n" "2 3 3 0 0.0\n" "3 4 -1 0 0.0\n" "4\n",
            acceptor=False,
        )
        assert weight_of_label_sequence(graph, [1, 2, 3]) == pytest.approx(0.5)
        assert weight_of_label_sequence(graph, [1, 3]) is None
        # Calling 2 the epsilon reverses both answers.
        assert weight_of_label_sequence(graph, [1, 3], epsilon=2) == pytest.approx(0.5)
        assert weight_of_label_sequence(graph, [1, 2, 3], epsilon=2) is None

    def test_it_finds_the_cheapest_of_two_paths(self):
        import k2

        from BenNevis.decode.graph import weight_of_label_sequence

        graph = k2.Fsa.from_str(
            "0 1 1 0 -2.0\n"
            "0 1 1 0 -0.75\n"  # the same label, cheaper
            "1 2 -1 0 0.0\n"
            "2\n",
            acceptor=False,
        )
        assert weight_of_label_sequence(graph, [1]) == pytest.approx(0.75)


@pytest.fixture(scope="module")
def inverted(our_tlg):
    """
    Both TLGs inverted and nothing else -- the route the gate takes.

    `prepare_output_probe` follows the inversion with
    `k2.remove_epsilon`, and that is the step mini-librispeech's
    43,263,880-arc TLG does not survive on a 15.6GiB runner. Inverting
    alone is cheap on any graph; what pays for it is a probe that
    tolerates the epsilons instead of removing them.
    """
    kaldi = load_openfst_text(
        (REFERENCE / "graph" / "ctc" / "TLG.fst.txt").read_text(),
        acceptor=False,
    )
    return k2.invert(kaldi), k2.invert(our_tlg)


class TestTlgEquivalenceBySearch:
    """
    The TLG half of the nightly gate, on the one corpus where both
    graphs are committed.

    It asks the same question as `TestTlgEquivalenceByOutput` above by a
    different route: invert so the words are the input labels, then
    search with `weight_of_label_sequence` rather than remove the
    epsilons and compose. The two routes are compared against each
    other here as well as against Kaldi, because yesno is where that
    comparison is affordable and mini-librispeech is where only one of
    them is.
    """

    def test_the_inverted_graphs_still_have_input_epsilons(self, inverted):
        """
        The premise, and without it the agreement below would prove
        nothing: a probe that tolerates epsilons can only be shown to do
        so on a graph that has them. Most of a token graph's arcs emit
        no word -- 20 of 29 in yesno's TLG -- and inverted those are
        input epsilons, including the blank self-loops that make the
        graph cyclic. This is the state `prepare_output_probe` exists to
        leave behind and this route does not need to.
        """
        for graph in inverted:
            assert int((graph.labels == 0).sum()) > 0

    @pytest.mark.parametrize(
        "sentence",
        [
            ["YES"],
            ["NO"],
            ["YES", "NO"],
            ["NO", "YES"],
            ["YES", "YES", "NO"],
            ["NO", "NO", "NO"],
            ["YES", "YES", "YES", "YES"],
        ],
    )
    def test_the_weights_agree_within_quantisation(self, inverted, tables, sentence):
        from BenNevis.decode.graph import weight_of_label_sequence

        their_graph, our_graph = inverted
        words, _phones = tables
        ids = [words[word] for word in sentence]

        theirs = weight_of_label_sequence(their_graph, ids)
        ours = weight_of_label_sequence(our_graph, ids)
        assert theirs is not None, f"Kaldi's TLG rejects {sentence}"
        assert ours is not None, f"our TLG rejects {sentence}"

        bound = (len(sentence) + 1) * QUANTISATION_DELTA
        assert abs(theirs - ours) <= bound, (
            f"{' '.join(sentence)}: Kaldi {theirs:.6f}, ours {ours:.6f}, "
            f"gap {abs(theirs - ours):.6f} exceeds {bound:.6f}"
        )

    @pytest.mark.parametrize("sentence", [["YES"], ["YES", "NO"], ["NO", "NO", "NO"]])
    def test_it_finds_what_the_composition_route_finds(
        self, inverted, probes, tables, sentence
    ):
        """
        The two routes must give the *same* weight, not merely weights
        that agree to Kaldi's quantisation -- they are searching one
        graph for one minimum, so any difference is a defect in one of
        them rather than a tolerance.

        This is what licences the gate using the cheap route on a graph
        where the expensive one cannot be run for comparison.
        """
        from BenNevis.decode.graph import (
            weight_of_label_sequence,
            weight_of_word_sequence,
        )

        words, _phones = tables
        ids = [words[word] for word in sentence]
        for searched, composed in zip(inverted, probes):
            by_search = weight_of_label_sequence(searched, ids)
            by_composition = weight_of_word_sequence(composed, ids)
            assert by_search is not None and by_composition is not None
            assert by_search == pytest.approx(by_composition, abs=1e-5)

    def test_a_sequence_the_grammar_rejects_returns_none(self, inverted, tables):
        """
        `<SIL>` is in yesno's lexicon and not in its grammar. The route
        must report that as absent rather than as a weight, and rather
        than looping on the epsilon cycles it is tolerating.
        """
        from BenNevis.decode.graph import weight_of_label_sequence

        words, _phones = tables
        if "<SIL>" not in words:
            pytest.skip("no out-of-grammar word to try")
        ids = [words["<SIL>"]]
        for graph in inverted:
            assert weight_of_label_sequence(graph, ids) is None

    def test_longer_sequences_cost_more(self, inverted, tables):
        """
        A probe returning a constant, or the weight of the empty path,
        would pass every comparison above while measuring nothing.
        """
        from BenNevis.decode.graph import weight_of_label_sequence

        _their_graph, our_graph = inverted
        words, _phones = tables
        one = weight_of_label_sequence(our_graph, [words["YES"]])
        four = weight_of_label_sequence(our_graph, [words["YES"]] * 4)
        assert one is not None and four is not None
        assert four > one * 2


class TestTheProbeIsPackedOncePerGraph:
    """
    What the `LabelProbe` class is for, and it is a measured saving
    rather than a tidiness.

    The arrays are a property of the graph, so a probe rebuilt per
    question repacks a graph that has not changed. On run 45705231089
    that was 48 repacks of 43M and 12.9M arcs for 24 sequences; back
    when the pack was `tolist()` it was the whole of the LG half's
    125.7s, which fell to 6.2s when the pack became `frombytes`. Neither
    number is visible from a small graph, which is why this is pinned by
    counting rather than by timing.
    """

    def test_one_probe_packs_once_however_many_questions(self, inverted, tables):
        from BenNevis.decode import graph as graph_module

        packs = []
        original = graph_module._arc_arrays

        def counted(graph):
            packs.append(graph)
            return original(graph)

        graph_module._arc_arrays = counted
        try:
            _their_graph, our_graph = inverted
            words, _phones = tables
            probe = graph_module.LabelProbe(our_graph)
            for sentence in (["YES"], ["NO"], ["YES", "NO"], ["NO", "YES"]):
                assert probe.weight([words[word] for word in sentence]) is not None
        finally:
            graph_module._arc_arrays = original
        assert len(packs) == 1, f"packed {len(packs)} times for four questions"

    def test_the_one_off_function_still_answers_the_same(self, inverted, tables):
        """
        `weight_of_label_sequence` is the single-question form and must
        not drift from the class it wraps -- every test above this file's
        TLG classes still calls it.
        """
        from BenNevis.decode.graph import LabelProbe, weight_of_label_sequence

        _their_graph, our_graph = inverted
        words, _phones = tables
        ids = [words["YES"], words["NO"]]
        probe = LabelProbe(our_graph)
        assert weight_of_label_sequence(our_graph, ids) == probe.weight(ids)

    def test_the_epsilon_belongs_to_the_probe(self, inverted, tables):
        """
        It is a property of the graph's alphabet rather than of the
        question, so it is given once. Taking it per question would
        invite two questions of one graph being asked under different
        alphabets, which is how a token-side probe comes to skip blanks
        instead of epsilons.
        """
        from BenNevis.decode.graph import LabelProbe

        _their_graph, our_graph = inverted
        words, _phones = tables
        ids = [words["YES"]]
        # `<eps>` is 0 on the word side; calling the blank's id epsilon
        # instead must change the answer rather than be ignored.
        assert LabelProbe(our_graph, epsilon=0).weight(ids) is not None
        assert LabelProbe(our_graph, epsilon=words["YES"]).weight(ids) != (
            LabelProbe(our_graph, epsilon=0).weight(ids)
        )

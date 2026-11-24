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

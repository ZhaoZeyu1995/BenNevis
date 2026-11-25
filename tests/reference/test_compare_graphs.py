"""
Tests for the nightly's graph equivalence gate.

The gate itself runs inside a recipe, because mini-librispeech's `LG.fst`
and `TLG.fst` are 138MB and 340MB and kept in the capture as digests, and
our side needs a 3-gram model that is downloaded rather than committed.
So the *script* cannot run here -- but its comparison functions can, and
they are what decides pass from fail.

They are imported rather than reimplemented. A comparison rewritten in
the test would pass while the shipped one failed, which is the mistake
`tests/recipes/test_decode_faster.py` exists to avoid.

Checked in both directions, because the dangerous failure is a gate that
cannot fail: it must agree on real graphs, and it must report a mismatch
when one is introduced. A third test covers the case that has already
shipped once elsewhere in this project -- agreement on an empty set --
by asserting the functions report *how many* sequences they compared.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.k2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE = REPO_ROOT / "tests" / "golden" / "reference" / "yesno"
YESNO = REPO_ROOT / "egs" / "yesno" / "asr" / "input"

if str(REPO_ROOT / "tests" / "reference") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))


@pytest.fixture(scope="module")
def pieces():
    """Both LGs, the tables, and the lexicon-drawn sequences."""
    from compare_graphs import read_table

    from BenNevis.decode.graph import compose_lg, load_openfst_text
    from BenNevis.lm import build_fst_text, read_arpa
    from BenNevis.lm.lexicon import build_disambig_fst_text, read_lexicon
    from BenNevis.utils.fingerprint import sample_lexicon_sequences

    words = read_table(REFERENCE / "lang" / "words.txt")
    phones = read_table(REFERENCE / "lang" / "phones.txt")
    first = min(value for key, value in phones.items() if key.startswith("#"))
    lexicon = load_openfst_text(
        build_disambig_fst_text(
            read_lexicon(YESNO / "lexicon.txt"), words, phones, sil_prob=0.0
        )
    )
    grammar = load_openfst_text(build_fst_text(read_arpa(YESNO / "task.arpabo"), words))
    ours = compose_lg(lexicon, grammar, first)
    theirs = load_openfst_text(
        (REFERENCE / "graph" / "ctc" / "LG.fst.txt").read_text(), acceptor=False
    )
    drawn = sample_lexicon_sequences(read_lexicon(YESNO / "lexicon.txt"), count=24)
    return ours, theirs, words, phones, drawn


class TestReadTable:
    def test_it_reads_a_symbol_table(self, tmp_path):
        from compare_graphs import read_table

        path = tmp_path / "words.txt"
        path.write_text("<eps> 0\nYES 1\nNO 2\n")
        assert read_table(path) == {"<eps>": 0, "YES": 1, "NO": 2}

    def test_a_malformed_line_is_skipped(self, tmp_path):
        from compare_graphs import read_table

        path = tmp_path / "words.txt"
        path.write_text("<eps> 0\njunk\nYES 1 extra\nNO 2\n")
        assert read_table(path) == {"<eps>": 0, "NO": 2}


class TestCompareLg:
    def test_the_real_graphs_agree(self, pieces):
        from compare_graphs import compare_lg

        from BenNevis.decode.graph import QUANTISATION_DELTA

        ours, theirs, _words, phones, drawn = pieces
        compared, failures = compare_lg(ours, theirs, phones, drawn, QUANTISATION_DELTA)
        assert failures == []
        assert compared > 0, "the gate compared nothing, so it proved nothing"

    def test_a_perturbed_graph_is_caught(self, pieces):
        """
        The direction that matters. A gate that cannot fail is worse than
        no gate, because it reads as evidence.
        """
        from compare_graphs import compare_lg

        from BenNevis.decode.graph import QUANTISATION_DELTA

        ours, theirs, _words, phones, drawn = pieces
        perturbed = ours.clone()
        perturbed.scores = perturbed.scores - 0.05
        compared, failures = compare_lg(
            perturbed, theirs, phones, drawn, QUANTISATION_DELTA
        )
        assert compared > 0
        assert failures, "a 0.05 weight shift went unreported"
        assert "exceeds" in failures[0]

    def test_nothing_drawn_compares_nothing(self, pieces):
        """
        So the caller can refuse to pass on an empty comparison, which is
        how the `G.fst` gate came to skip unconditionally.
        """
        from compare_graphs import compare_lg

        from BenNevis.decode.graph import QUANTISATION_DELTA

        ours, theirs, _words, phones, _drawn = pieces
        compared, failures = compare_lg(ours, theirs, phones, [], QUANTISATION_DELTA)
        assert compared == 0
        assert failures == []

    def test_a_sequence_with_an_unknown_phone_is_skipped(self, pieces):
        from compare_graphs import compare_lg

        from BenNevis.decode.graph import QUANTISATION_DELTA

        ours, theirs, _words, phones, _drawn = pieces
        compared, failures = compare_lg(
            ours,
            theirs,
            phones,
            [(("X",), ("NOT_A_PHONE",))],
            QUANTISATION_DELTA,
        )
        assert compared == 0
        assert failures == []


@pytest.fixture(scope="module")
def tlgs(pieces):
    """Both TLGs, ours built the way stage 10 builds it."""
    from BenNevis.decode.graph import build_decoding_graph, load_openfst_text
    from BenNevis.lm import build_fst_text, read_arpa
    from BenNevis.lm.lexicon import build_disambig_fst_text, read_lexicon
    from BenNevis.topos import build as topo_build
    from BenNevis.topos import spec as topo_spec

    _ours_lg, _theirs_lg, words, phones, drawn = pieces
    first = min(value for key, value in phones.items() if key.startswith("#"))
    lexicon = load_openfst_text(
        build_disambig_fst_text(
            read_lexicon(YESNO / "lexicon.txt"), words, phones, sil_prob=0.0
        )
    )
    grammar = load_openfst_text(build_fst_text(read_arpa(YESNO / "task.arpabo"), words))
    # The k2 variant: tokens in, phones out, with neither the
    # disambiguation symbols nor `<eps>`.
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
    ours = build_decoding_graph(topology, lexicon, grammar, first)
    theirs = load_openfst_text(
        (REFERENCE / "graph" / "ctc" / "TLG.fst.txt").read_text(),
        acceptor=False,
    )
    return ours, theirs, words, drawn


class TestInputEpsilons:
    """
    The failure that made the gate compare nothing on mini-librispeech.

    Kaldi's LG keeps input epsilons -- `fstrmepslocal` removes only the
    ones it can remove without growing the graph -- while ours is
    epsilon-free, because `compose_lg` ends in `remove_epsilon`. Under
    `treat_epsilons_specially=False` label 0 is an ordinary symbol, so a
    single epsilon on a path makes a linear probe reject it, Kaldi
    rejects every sequence, and the run reports `compared LG on 0
    sequences`.

    yesno could not show this: its LG has no input epsilons on the
    surviving paths, so both settings agree there and the gate looked
    verified on the one corpus where both graphs are committed. These
    tests use a hand-built pair instead, which is the only way to put an
    epsilon where it matters.
    """

    @staticmethod
    def _pair():
        import k2

        # Both score the phone sequence A B at zero. `theirs` reaches it
        # through an epsilon arc, as Kaldi's does; `ours` does not.
        theirs = k2.arc_sort(
            k2.Fsa.from_str(
                "0 1 1 0 0.0\n1 2 0 0 0.0\n2 3 2 0 0.0\n3 4 -1 0 0.0\n4\n",
                acceptor=False,
            )
        )
        ours = k2.arc_sort(
            k2.Fsa.from_str(
                "0 1 1 0 0.0\n1 2 2 0 0.0\n2 3 -1 0 0.0\n3\n",
                acceptor=False,
            )
        )
        return ours, theirs

    def test_an_epsilon_on_kaldis_side_is_not_a_rejection(self):
        from compare_graphs import compare_lg

        from BenNevis.decode.graph import QUANTISATION_DELTA

        ours, theirs = self._pair()
        phones = {"<eps>": 0, "A": 1, "B": 2}
        drawn = [(["word"], ["A", "B"])]
        compared, failures = compare_lg(ours, theirs, phones, drawn, QUANTISATION_DELTA)
        assert failures == []
        assert compared == 1, (
            "an input epsilon on Kaldi's side made the probe reject the "
            "sequence, which is the mini-librispeech failure"
        )

    def test_a_real_disagreement_is_still_caught_across_an_epsilon(self):
        """
        The other direction, so the fix cannot be a blanket pass: the
        epsilon must be stepped over without the weights behind it being
        ignored.
        """
        import k2
        from compare_graphs import compare_lg

        from BenNevis.decode.graph import QUANTISATION_DELTA

        ours, theirs = self._pair()
        theirs.scores[1] += 0.25
        compared, failures = compare_lg(
            k2.arc_sort(ours),
            k2.arc_sort(theirs),
            {"A": 1, "B": 2},
            [(["w"], ["A", "B"])],
            QUANTISATION_DELTA,
        )
        assert compared == 1
        assert len(failures) == 1 and "0.25" in failures[0]


class TestCompareTlg:
    def test_the_real_graphs_agree(self, tlgs):
        from compare_graphs import compare_tlg

        from BenNevis.decode.graph import QUANTISATION_DELTA

        ours, theirs, words, drawn = tlgs
        compared, failures = compare_tlg(ours, theirs, words, drawn, QUANTISATION_DELTA)
        assert failures == []
        assert compared > 0, "the gate compared nothing, so it proved nothing"

    def test_a_perturbed_graph_is_caught(self, tlgs):
        from compare_graphs import compare_tlg

        from BenNevis.decode.graph import QUANTISATION_DELTA

        ours, theirs, words, drawn = tlgs
        perturbed = ours.clone()
        perturbed.scores = perturbed.scores - 0.05
        compared, failures = compare_tlg(
            perturbed, theirs, words, drawn, QUANTISATION_DELTA
        )
        assert compared > 0
        assert failures, "a 0.05 weight shift went unreported"

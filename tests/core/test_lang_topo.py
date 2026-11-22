"""
Tests for building the training topology FSA inside `Lang`.

`Lang.load_topo` used to read the compiled `k2/T.fst` back through
`fstprint | awk`, which put OpenFST and a POSIX shell on the training path.
It now expands `k2/topo.yaml`, the specification that `prepare_topo.sh` copies
into the language directory.

Two things are checked, and they are the two that could go wrong:

1. The FSA is the same machine the old route produced. The old route is
   simulated here -- substitute the symbol tables, then order the arcs by
   output label the way `fstarcsort --sort_type=olabel` did -- and both FSAs
   are put through the real loss.
2. Nothing shells out. `os.popen` is replaced with something that fails, so a
   test cannot pass by quietly falling back.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

torch = pytest.importorskip("torch")
k2 = pytest.importorskip("k2")

from BenNevis.core.lang import Lang  # noqa: E402
from BenNevis.core.losses import GraphLoss  # noqa: E402
from BenNevis.topos import build, spec  # noqa: E402

pytestmark = pytest.mark.k2

WORD_IDS = [[1, 2, 3], [2, 2], [3, 1, 2, 1]]

# Spelled out rather than imported from conftest, which is not a package.
# These must agree with the tables the `lang_dir` fixture writes; the first
# test below fails loudly if they drift.
PHONE_IDS = {"<eps>": 0, "a": 1, "b": 2, "c": 3}
TOKEN_IDS = {"<blk>": 0, "a": 1, "b": 2, "c": 3}
NUM_TOKENS = 4


@pytest.fixture
def no_shell(monkeypatch):
    """Make any shell-out fail loudly, so a fallback cannot pass silently."""

    def forbidden(*args, **kwargs):
        raise AssertionError(f"shelled out: {args!r}")

    monkeypatch.setattr("os.popen", forbidden)
    return forbidden


def old_route_topo(phones_path):
    """
    The FSA the `fstcompile | fstarcsort | fstprint | awk` route produced.

    The compilation and the printing cancel out, so what is left of the round
    trip is the symbol substitution and the sort by output label.
    """
    text = build.build_openfst_text(
        spec.load("ctc"), "k2", build.read_phone_table(phones_path)
    )
    arcs, finals = [], []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 1:
            finals.append(int(fields[0]))
            continue
        src, dst, ilabel, olabel = fields
        arcs.append((int(src), int(dst), TOKEN_IDS[ilabel], PHONE_IDS[olabel]))

    arcs.sort(key=lambda arc: (arc[0], arc[3]))  # fstarcsort --sort_type=olabel
    lines = [f"{src} {dst} {i} {o} 0.0" for src, dst, i, o in arcs]
    lines.extend(str(state) for state in finals)
    return k2.Fsa.from_openfst("\n".join(lines) + "\n", acceptor=False)


class TestBuildsFromTheSpec:
    """`load_topo=True` works with no OpenFST and no shell."""

    def test_tables_match_the_fixture(self, lang_dir):
        """Guards the constants above against the fixture changing."""
        assert build.read_symbol_table(lang_dir / "k2" / "tokens.txt") == TOKEN_IDS
        assert build.read_symbol_table(lang_dir / "k2" / "phones.txt") == PHONE_IDS

    def test_builds_without_shelling_out(self, lang_dir, no_shell):
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=True)
        assert isinstance(obj.topo, k2.Fsa)
        assert obj.topo.num_arcs > 0

    def test_does_not_need_the_compiled_fst(self, lang_dir, no_shell):
        """There is no k2/T.fst in the fixture at all."""
        assert not (lang_dir / "k2" / "T.fst").exists()
        Lang(str(lang_dir), load_topo=True, load_lexicon=False)

    def test_labels_are_the_network_outputs(self, lang_dir, no_shell):
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=False)
        labels = set(obj.topo.labels.tolist()) - {-1}
        assert labels <= set(range(NUM_TOKENS))
        assert 0 in labels  # the blank self-loop

    def test_output_labels_are_phones(self, lang_dir, no_shell):
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=False)
        aux = set(obj.topo.aux_labels.tolist()) - {-1}
        assert aux <= set(PHONE_IDS.values())

    def test_token_ids_come_from_the_table_on_disc(self, lang_dir, no_shell):
        """
        `k2/tokens.txt` fixes the network's output indices, so a table with
        different ids must move the FSA's input labels with it.
        """
        plain = Lang(str(lang_dir), load_topo=True, load_lexicon=False)
        (lang_dir / "k2" / "tokens.txt").write_text(
            "<blk> 0\nc 1\nb 2\na 3\n"  # a and c swapped
        )
        swapped = Lang(str(lang_dir), load_topo=True, load_lexicon=False)

        remap = {0: 0, 1: 3, 2: 2, 3: 1, -1: -1}
        assert [remap[label] for label in plain.topo.labels.tolist()] == (
            swapped.topo.labels.tolist()
        )

    def test_missing_token_is_reported_against_the_table(self, lang_dir, no_shell):
        (lang_dir / "k2" / "tokens.txt").write_text("<blk> 0\na 1\nb 2\n")
        with pytest.raises(KeyError, match="input symbol 'c'"):
            Lang(str(lang_dir), load_topo=True, load_lexicon=False)


class TestEquivalenceWithTheOldRoute:
    """The spec-built FSA must behave exactly as the compiled one did."""

    def test_same_arc_and_state_count(self, lang_dir, no_shell):
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=False)
        old = old_route_topo(lang_dir / "k2" / "phones.txt")
        assert obj.topo.num_arcs == old.num_arcs
        assert obj.topo.shape[0] == old.shape[0]

    def test_same_arc_set(self, lang_dir, no_shell):
        """
        The same arcs, differing only in order within a state -- which is all
        `fstarcsort` ever changed, and which `k2.compose` re-sorts anyway.
        """
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=False)
        old = old_route_topo(lang_dir / "k2" / "phones.txt")

        def arc_set(fsa):
            arcs = fsa.arcs.values()[:, :3].tolist()
            aux = fsa.aux_labels.tolist()
            return sorted(tuple(a) + (b,) for a, b in zip(arcs, aux))

        assert arc_set(obj.topo) == arc_set(old)

    def test_same_loss(self, lang_dir, no_shell):
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=True)
        torch.manual_seed(0)
        log_probs = torch.randn(3, 20, NUM_TOKENS).log_softmax(-1)
        lens = torch.tensor([20, 18, 15], dtype=torch.int32)
        target_lens = torch.tensor([3, 2, 4], dtype=torch.int32)

        from_spec = GraphLoss(obj, use_den=False, reduction="sum")(
            log_probs, lens, WORD_IDS, target_lens
        ).item()
        obj.topo = old_route_topo(lang_dir / "k2" / "phones.txt")
        from_old = GraphLoss(obj, use_den=False, reduction="sum")(
            log_probs, lens, WORD_IDS, target_lens
        ).item()

        assert from_spec == pytest.approx(from_old, rel=1e-9, abs=1e-9)

    def test_same_denominator(self, lang_dir, no_shell):
        """The denominator uses the topology on its own, with no lexicon."""
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=True)
        torch.manual_seed(1)
        log_probs = torch.randn(2, 16, NUM_TOKENS).log_softmax(-1)
        lens = torch.tensor([16, 14], dtype=torch.int32)

        def den_scores(topo):
            obj.topo = topo
            loss = GraphLoss(obj, use_den=True)
            graph = loss.get_den_graph(2, torch.device("cpu"))
            dense = loss.get_dense_fsa(log_probs, lens)
            lattice = k2.intersect_dense(graph, dense, 10.0)
            return lattice.get_tot_scores(
                log_semiring=True, use_double_scores=True
            ).tolist()

        from_spec = den_scores(
            Lang(str(lang_dir), load_topo=True, load_lexicon=False).topo
        )
        from_old = den_scores(old_route_topo(lang_dir / "k2" / "phones.txt"))
        assert from_spec == pytest.approx(from_old, rel=1e-9, abs=1e-9)

    def test_matches_k2_ctc_topo(self, lang_dir, no_shell):
        """And the whole thing still agrees with k2's own CTC topology."""
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=True)
        torch.manual_seed(2)
        log_probs = torch.randn(3, 20, NUM_TOKENS).log_softmax(-1)
        lens = torch.tensor([20, 18, 15], dtype=torch.int32)
        target_lens = torch.tensor([3, 2, 4], dtype=torch.int32)

        from_spec = GraphLoss(obj, use_den=False, reduction="sum")(
            log_probs, lens, WORD_IDS, target_lens
        ).item()
        obj.topo = k2.ctc_topo(NUM_TOKENS - 1)
        from_builtin = GraphLoss(obj, use_den=False, reduction="sum")(
            log_probs, lens, WORD_IDS, target_lens
        ).item()

        assert from_spec == pytest.approx(from_builtin, rel=1e-5, abs=1e-5)


class TestFallbackToTheCompiledFst:
    """A language directory prepared before the spec was copied in still loads."""

    def test_warns_and_uses_fstprint(self, lang_dir, monkeypatch):
        (lang_dir / "k2" / "topo.yaml").unlink()
        (lang_dir / "k2" / "T.fst").write_bytes(b"not really an fst")

        text = build.build_openfst_int_text(
            spec.load("ctc"),
            "k2",
            build.read_phone_table(lang_dir / "k2" / "phones.txt"),
            isymbols=TOKEN_IDS,
            osymbols=PHONE_IDS,
        )

        class FakePipe:
            def read(self):
                return text

        calls = []

        def fake_popen(cmd):
            calls.append(cmd)
            return FakePipe()

        monkeypatch.setattr("os.popen", fake_popen)

        with pytest.warns(UserWarning, match="fstprint"):
            obj = Lang(str(lang_dir), load_topo=True, load_lexicon=False)

        assert len(calls) == 1
        assert "fstprint" in calls[0]
        assert obj.topo.num_arcs > 0

    def test_neither_present_names_both(self, lang_dir, no_shell):
        (lang_dir / "k2" / "topo.yaml").unlink()
        with pytest.raises(AssertionError, match="topo.yaml"):
            Lang(str(lang_dir), load_topo=True, load_lexicon=False)

    def test_spec_wins_when_both_present(self, lang_dir, no_shell):
        """With a spec there, nothing reads the compiled FST."""
        (lang_dir / "k2" / "T.fst").write_bytes(b"not really an fst")
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=False)
        assert obj.topo.num_arcs > 0


class TestTheDenominatorGraphIsArcSorted:
    """
    `k2.intersect_dense` documents that its graphs must be arc-sorted, and
    `get_den_graph` was handing it the topology unsorted.

    `prepare_topo.sh` runs `fstarcsort --sort_type=olabel`, which sorts by
    *output* label and does not make a k2 FSA `ARC_SORTED` -- it measured
    False for all eight topologies, before and after the spec rewrite. The
    numerator path is unaffected because `k2.compose` arc-sorts its first
    argument itself, so this was the only site violating the precondition.

    It had not been biting because the CPU scores come out the same either
    way, which the second test here pins so the baseline cannot move. On
    CUDA an unsorted graph is undefined rather than merely slow, and that
    remains unchecked -- these tests run on the CPU.
    """

    def test_the_topology_itself_is_not_arc_sorted(self, lang_dir, no_shell):
        """
        The premise. If a topology ever came out sorted on its own this
        test would start failing, which is the right way to find out --
        the fix would then be unnecessary rather than silently redundant.
        """
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=False)
        assert not (obj.topo.properties & k2.fsa_properties.ARC_SORTED)

    def test_the_denominator_graph_is(self, lang_dir, no_shell):
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=True)
        loss = GraphLoss(obj, use_den=True)
        graph = loss.get_den_graph(2, torch.device("cpu"))
        assert graph.properties & k2.fsa_properties.ARC_SORTED

    def test_the_scores_are_unchanged_by_sorting(self, lang_dir, no_shell):
        """
        The test that protects the WER ledger: sorting must not move the
        denominator, or every recorded error rate would need re-measuring.
        Computed both ways on the same batch and compared exactly, not to a
        tolerance -- a difference here would be a real difference.
        """
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=True)
        torch.manual_seed(7)
        log_probs = torch.randn(3, 18, NUM_TOKENS).log_softmax(-1)
        lens = torch.tensor([18, 16, 12], dtype=torch.int32)
        loss = GraphLoss(obj, use_den=True)
        dense = loss.get_dense_fsa(log_probs, lens)

        def scores(graph):
            lattice = k2.intersect_dense(graph, dense, 10.0)
            return lattice.get_tot_scores(
                log_semiring=True, use_double_scores=True
            ).tolist()

        sorted_graph = loss.get_den_graph(3, torch.device("cpu"))
        unsorted = k2.create_fsa_vec([obj.topo for _ in range(3)])
        assert not (unsorted.properties & k2.fsa_properties.ARC_SORTED)

        assert scores(sorted_graph) == scores(unsorted)

    def test_the_graph_still_carries_no_gradient(self, lang_dir, no_shell):
        """
        The denominator is a constant. A graph that required grad would
        have the loss backpropagate into the topology, which is not a
        parameter.
        """
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=True)
        loss = GraphLoss(obj, use_den=True)
        assert loss.get_den_graph(2, torch.device("cpu")).requires_grad is False

    def test_one_graph_per_utterance(self, lang_dir, no_shell):
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=True)
        loss = GraphLoss(obj, use_den=True)
        for size in (1, 2, 5):
            assert loss.get_den_graph(size, torch.device("cpu")).shape[0] == size

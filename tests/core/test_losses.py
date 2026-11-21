"""
Tests for GraphLoss, the k2 graph loss.

The central test is equivalence with `torch.nn.functional.ctc_loss`: under a
CTC topology and a one-to-one lexicon, the graph loss computes exactly the CTC
loss, so torch's own implementation is an independent oracle for both the
forward value and the gradients. That pins the numerator computation, the
dense FSA construction, the supervision segments and both reductions at once.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("k2")

from BenNevis.core.losses import GraphLoss  # noqa: E402

pytestmark = pytest.mark.k2

# Word sequences and their matching phone sequences under the fixture lexicon.
# [2, 2] is the interesting one: a repeated label, which CTC can only emit with
# a blank in between.
WORD_IDS = [[1, 2, 3], [2, 2], [3, 1, 2, 1]]


def make_inputs(seed=0, batch=3, frames=20, tokens=4):
    """Normalised log probabilities and their lengths."""
    torch.manual_seed(seed)
    log_probs = torch.randn(batch, frames, tokens).log_softmax(-1)
    lens = torch.tensor([frames, frames - 2, frames - 5][:batch], dtype=torch.int32)
    return log_probs, lens


def torch_ctc(log_probs, lens, targets, target_lens, reduction):
    """The same loss via torch, which expects (T, B, C) and a flat target."""
    flat = torch.tensor([p for t in targets for p in t], dtype=torch.int32)
    return torch.nn.functional.ctc_loss(
        log_probs.transpose(0, 1),
        flat,
        lens,
        target_lens,
        blank=0,
        reduction=reduction,
    )


@pytest.fixture
def batch(lang):
    """A batch plus the phone targets implied by its word ids."""
    log_probs, lens = make_inputs()
    targets = [lang.wids2pids([w])[0] for w in WORD_IDS]
    target_lens = torch.tensor([len(t) for t in targets], dtype=torch.int32)
    return log_probs, lens, targets, target_lens


class TestAgainstTorchCTC:
    """Under a CTC topology the graph loss must equal torch's CTC loss."""

    @pytest.mark.parametrize("reduction", ["sum", "mean"])
    def test_forward_matches(self, lang, batch, reduction):
        log_probs, lens, targets, target_lens = batch

        got = GraphLoss(lang, use_den=False, reduction=reduction)(
            log_probs, lens, WORD_IDS, target_lens
        )
        want = torch_ctc(log_probs, lens, targets, target_lens, reduction)

        assert got.item() == pytest.approx(want.item(), rel=1e-5, abs=1e-5)

    def test_per_utterance_losses_match(self, lang, batch):
        """reduction='none' must give the same per-utterance values."""
        log_probs, lens, targets, target_lens = batch

        got = GraphLoss(lang, use_den=False, reduction="none")(
            log_probs, lens, WORD_IDS, target_lens
        )
        want = torch_ctc(log_probs, lens, targets, target_lens, "none")

        assert got.shape == (len(WORD_IDS),)
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_gradients_match_with_respect_to_the_logits(self, lang):
        """
        Agreement on the backward pass, which is what training depends on.

        The comparison must be made with respect to the logits, with the
        log_softmax inside the graph, because the two libraries define their
        gradients differently with respect to log probabilities:

        - k2 returns the true d(loss)/d(log_probs), which is -gamma and sums
          to -1 over each frame.
        - torch's ctc_loss backward returns (p - gamma), the gradient with
          respect to the pre-softmax logits, which sums to 0 over each frame.
          Verified against a finite difference of torch's own forward pass:
          the finite difference sums to -1, torch's analytic gradient to 0.

        Both are then invariant under the log_softmax Jacobian, because
        (I - p 1^T)^T applied to either gives (p - gamma). So the two agree
        exactly where it matters and differ only if compared one step too
        early. Comparing raw log-prob gradients would fail for a reason that
        has nothing to do with correctness.
        """
        torch.manual_seed(0)
        logits = torch.randn(3, 20, 4)
        lens = torch.tensor([20, 18, 15], dtype=torch.int32)
        targets = [lang.wids2pids([w])[0] for w in WORD_IDS]
        target_lens = torch.tensor([len(t) for t in targets], dtype=torch.int32)

        a = logits.clone().requires_grad_(True)
        GraphLoss(lang, use_den=False, reduction="sum")(
            a.log_softmax(-1), lens, WORD_IDS, target_lens
        ).backward()

        b = logits.clone().requires_grad_(True)
        torch_ctc(b.log_softmax(-1), lens, targets, target_lens, "sum").backward()

        assert a.grad is not None and b.grad is not None
        assert torch.allclose(a.grad, b.grad, rtol=1e-4, atol=1e-4)

    def test_log_prob_gradient_is_the_occupancy(self, lang, batch):
        """
        k2's gradient with respect to log probabilities is -gamma, so it sums
        to -1 per frame: every frame is occupied by exactly one state. Pinning
        this records which of the two conventions BenNevis actually receives.
        """
        log_probs, lens, _, target_lens = batch
        x = log_probs.clone().requires_grad_(True)

        GraphLoss(lang, use_den=False, reduction="sum")(
            x, lens, WORD_IDS, target_lens
        ).backward()

        for i, length in enumerate(lens.tolist()):
            sums = x.grad[i, :length].sum(-1)
            assert torch.allclose(sums, torch.full_like(sums, -1.0), atol=1e-4)

    @pytest.mark.parametrize("seed", [0, 1, 2, 7])
    def test_holds_across_inputs(self, lang, seed):
        """Not a fluke of one random draw."""
        log_probs, lens = make_inputs(seed=seed)
        targets = [lang.wids2pids([w])[0] for w in WORD_IDS]
        target_lens = torch.tensor([len(t) for t in targets], dtype=torch.int32)

        got = GraphLoss(lang, use_den=False, reduction="sum")(
            log_probs, lens, WORD_IDS, target_lens
        )
        want = torch_ctc(log_probs, lens, targets, target_lens, "sum")

        assert got.item() == pytest.approx(want.item(), rel=1e-5, abs=1e-5)


class TestDenominator:
    """Behaviour of the denominator graph."""

    def test_denominator_of_normalised_outputs_is_about_zero(self, lang, batch):
        """
        The CTC topology accepts any token sequence, so with log-softmaxed
        inputs the total probability over the denominator is 1 and its loss is
        0. The full loss should therefore still match CTC.
        """
        log_probs, lens, targets, target_lens = batch

        with_den = GraphLoss(lang, use_den=True, reduction="sum")(
            log_probs, lens, WORD_IDS, target_lens
        )
        want = torch_ctc(log_probs, lens, targets, target_lens, "sum")

        assert with_den.item() == pytest.approx(want.item(), abs=1e-3)

    def test_den_grad_without_den_changes_gradients_not_the_value(self, lang, batch):
        """
        use_den=False with use_den_grad=True subtracts the denominator from the
        reported loss under no_grad, so the value shifts but the gradient is
        the numerator's alone.
        """
        log_probs, lens, _, target_lens = batch

        a = log_probs.clone().requires_grad_(True)
        GraphLoss(lang, use_den=False, use_den_grad=False, reduction="sum")(
            a, lens, WORD_IDS, target_lens
        ).backward()

        b = log_probs.clone().requires_grad_(True)
        GraphLoss(lang, use_den=False, use_den_grad=True, reduction="sum")(
            b, lens, WORD_IDS, target_lens
        ).backward()

        assert torch.allclose(a.grad, b.grad, rtol=1e-5, atol=1e-6)


class TestConfiguration:
    """Guard rails on how GraphLoss is constructed and called."""

    def test_rejects_an_unknown_reduction(self, lang):
        with pytest.raises(AssertionError):
            GraphLoss(lang, reduction="average")

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_accepts_the_documented_reductions(self, lang, reduction):
        assert GraphLoss(lang, reduction=reduction).reduction == reduction

    def test_numerator_graph_does_not_require_grad(self, lang):
        """A training graph carrying gradients would be a serious bug."""
        graph = GraphLoss(lang).get_num_graph(WORD_IDS, torch.device("cpu"))
        assert graph.requires_grad is False

    def test_denominator_graph_has_one_copy_per_utterance(self, lang):
        graph = GraphLoss(lang).get_den_graph(5, torch.device("cpu"))
        assert graph.shape[0] == 5
        assert graph.requires_grad is False

    def test_dense_fsa_respects_the_given_lengths(self, lang, batch):
        """
        Padding must be excluded: the dense FSA has one state per frame plus a
        final state for each utterance.
        """
        log_probs, lens, _, _ = batch

        dense = GraphLoss(lang).get_dense_fsa(log_probs, lens)

        assert dense.dim0() == log_probs.shape[0]
        assert dense.duration.tolist() == lens.tolist()


class TestTopologyEquivalence:
    """
    The project's own CTC generator against k2's built-in ctc_topo.

    This is the oracle the declarative topology rewrite will have to satisfy:
    any new implementation must produce a topology that behaves identically.
    """

    def test_generated_ctc_topo_has_the_same_size(self, generated_ctc_topo, lang):
        assert generated_ctc_topo.num_arcs == lang.topo.num_arcs

    def test_generated_ctc_topo_gives_the_same_loss(
        self, lang, generated_ctc_topo, batch
    ):
        log_probs, lens, _, target_lens = batch

        from_builtin = GraphLoss(lang, use_den=False, reduction="sum")(
            log_probs, lens, WORD_IDS, target_lens
        ).item()
        lang.topo = generated_ctc_topo
        from_generator = GraphLoss(lang, use_den=False, reduction="sum")(
            log_probs, lens, WORD_IDS, target_lens
        ).item()

        assert from_generator == pytest.approx(from_builtin, rel=1e-6, abs=1e-6)

    def test_generated_ctc_topo_also_matches_torch_ctc(
        self, lang, generated_ctc_topo, batch
    ):
        """The generator, not just k2's builtin, implements real CTC."""
        log_probs, lens, targets, target_lens = batch
        lang.topo = generated_ctc_topo

        got = GraphLoss(lang, use_den=False, reduction="sum")(
            log_probs, lens, WORD_IDS, target_lens
        )
        want = torch_ctc(log_probs, lens, targets, target_lens, "sum")

        assert got.item() == pytest.approx(want.item(), rel=1e-5, abs=1e-5)

"""
The code in this file implements loss functions for the model training.
The GraphLoss class is used to compute the graph loss in k2, which is inspired by icefall's implementation.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""
import k2
import torch
from typing import Optional, Tuple, Dict, Any, List
from BenNevis.core.lang import Lang


class GraphLoss(torch.nn.Module):
    """
    Graph Loss computation in k2.

    Arguments
    ---------
    lang : Lang
        The Lang object used for graph compilation.
    use_den : bool
        Whether to consider the denominator graph in the loss computation, by default True.
    use_den_grad : bool
        Whether to use the denominator graph in the gradient computation, by default False.
        Note that this argument is only effective when `use_den` is False. That said, if `use_den` is True,
        the denominator graph will always be used in the gradient computation.
        On the contrary, if `use_den` is False and `use_den_grad` is True, the denominator graph will be used in the
        loss computation, but not in the gradient computation.
    output_beam : float
        Beam for k2.intersect_dense, by default 10.0.
        You may refer to the official k2 documentation for more information.
    reduction : str
        Specifies the reduction to apply to the output:
        "none" | "mean" | "sum".
        * "none": no reduction will be applied.
        * "mean": the output losses will be divided by the target lengths and then the mean over the batch is taken.
        * "sum": sum the output losses over batches.

    """
    def __init__(self,
                 lang: Lang,
                 use_den: bool = True,
                 use_den_grad: bool = False,
                 output_beam: float = 10.0,
                 reduction: str = "mean",
                 ):
        super(GraphLoss, self).__init__()
        assert reduction in ("none", "mean", "sum"), \
            f"reduction must be 'none', 'mean' or 'sum', but got {reduction}"
        self.lang = lang
        self.use_den = use_den
        self.use_den_grad = use_den_grad
        self.output_beam = output_beam
        self.reduction = reduction

    def forward(self,
                log_probs: torch.Tensor,
                log_probs_lens: torch.Tensor,
                word_ids: List[List[int]],
                target_lengths: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """
        Compute the Graph loss given the log probabilities of the output symbols.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the output symbols, with shape `(batch_size, max_seq_len, num_symbols)`.
        log_probs_lens : torch.Tensor
            The length of the log probabilities, with shape `(batch_size,)`.
        word_ids : List[List[int]]
            The word IDs of the target sequences.
        target_lengths : Optional[torch.Tensor]
            The length of the target (phone-level) sequences. If `reduction` is `mean`, this argument is required.

        Returns
        -------
        loss : torch.Tensor
            The graph loss.
            If `reduction` is `none`, the shape is `(batch_size,)`.
            If `reduction` is `mean` or `sum`, the shape is `()`.
        """
        device = log_probs.device
        batch_size = log_probs.size(0)
        num_graph = self.get_num_graph(word_ids, device)
        dense_fsa = self.get_dense_fsa(log_probs, log_probs_lens)
        num = self.compute(num_graph, dense_fsa, target_lengths)
        if not self.use_den:
            if self.use_den_grad:
                with torch.no_grad():
                    den_graph = self.get_den_graph(batch_size, device)
                    den = self.compute(den_graph, dense_fsa, target_lengths)
                return num - den
            else:
                return num
        else:
            den_graph = self.get_den_graph(batch_size, device)
            den = self.compute(den_graph, dense_fsa, target_lengths)
            return num - den

    def compute(self,
                graph: k2.Fsa,
                dense_fsa_vec: k2.DenseFsaVec,
                target_lengths: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """
        Compute the Graph loss given a decoding graph and a dense fsa vector.

        Arguments
        ---------
        graph : k2.Fsa
            The graph used for training, which is usually derived from Lang.compile_training_graph().
            For more information, please refer to BenNevis.core.lang.Lang.compile_training_graph.
        dense_fsa_vec : k2.DenseFsaVec
            The dense fsa that denotes the output posterior probabilities.
        target_lengths : Optional[torch.Tensor]
            The length of the target (phone-level) sequences. If `reduction` is `mean`, this argument is required.

        Returns
        -------
        loss : torch.Tensor
            The graph loss of the numerator or the denominator.
            If `reduction` is `none`, the shape is `(batch_size,)`.
            If `reduction` is `mean` or `sum`, the shape is `()`.
        """

        lattice = k2.intersect_dense(graph, dense_fsa_vec,
                                     self.output_beam)

        tot_scores = lattice.get_tot_scores(log_semiring=True, use_double_scores=False)
        loss = -1 * tot_scores
        loss = loss.to(torch.float32)

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:
            assert self.reduction == "mean"
            loss /= target_lengths
            return loss.mean()

    def get_num_graph(self,
                      word_ids: List[List[int]],
                      device: torch.device,
                      ) -> k2.Fsa:
        """
        Get the numerator graph for the loss computation.

        Arguments
        ---------
        word_ids: List[List[int]]
            The word IDs of the target sequences.
        device: torch.device
            The device to put the training graph on.
            Usually it is the same device as neural network outputs.

        Returns
        -------
        decoding_graph: k2.Fsa
            The numerator graph for the loss computation.
        """
        decoding_graph = self.lang.compile_training_graph(
            word_ids, device
        )

        assert decoding_graph.requires_grad is False
        return decoding_graph

    def get_dense_fsa(self,
                      log_probs: torch.Tensor,
                      log_probs_lens: torch.Tensor,
                      ) -> k2.DenseFsaVec:
        """
        Get the dense fsa vector for the loss computation.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the output symbols, with shape `(batch_size, max_seq_len, num_symbols)`.
        log_probs_lens : torch.Tensor
            The length of the log probabilities, with shape `(batch_size,)`.

        Returns
        -------
        dense_fsa_vec: k2.DenseFsaVec
            The dense fsa vector for the loss computation.
        """

        batch_size = log_probs.shape[0]
        supervision_segments = torch.tensor(
            [[i, 0, log_probs_lens[i]] for i in range(batch_size)],
            device="cpu",
            dtype=torch.int32,
        )
        dense_fsa_vec = k2.DenseFsaVec(
            log_probs=log_probs,
            supervision_segments=supervision_segments,
        )
        return dense_fsa_vec

    def get_den_graph(self,
                      batch_size: int,
                      device: torch.device,
                      ) -> k2.Fsa:
        """
        Get the denominator graph for the loss computation.

        Arguments
        ---------
        batch_size: int
            The size of the batch.
        device: torch.device
            The device to put the training graph on.
            Usually it is the same device as neural network outputs.

        Returns
        -------
        den_decoding_graph: k2.Fsa
            The denominator graph for the loss computation.
        """
        den_decoding_graph = k2.create_fsa_vec(
            [self.lang.topo.to(device) for _ in range(batch_size)]
        )

        assert den_decoding_graph.requires_grad is False
        return den_decoding_graph

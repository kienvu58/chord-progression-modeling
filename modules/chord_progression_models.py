import torch
import numpy as np

from overrides import overrides
from typing import Iterator, Dict, List, Tuple, Union

from allennlp.models import Model
from allennlp.modules.text_field_embedders import (
    TextFieldEmbedder,
    BasicTextFieldEmbedder,
)
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics.average import Average
from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError

TM_ONE_HOT = "one_hot"
TM_FIXED = "fixed_temperature"
TM_DECREASED = "decreased_temperature"
TM_NO = "no_temperature"


class Cpm(Model):
    """
    The ``Cpm`` applies a "contextualizing"
    ``Seq2SeqEncoder`` to uncontextualized embeddings, using a ``torch.nn.functional.kl_div``
    module to compute the language modeling loss.
    If bidirectional is True,  the language model is trained to predict the next and
    previous tokens for each token in the input. In this case, the contextualizer must
    be bidirectional. If bidirectional is False, the language model is trained to only
    predict the next token for each token in the input; the contextualizer should also
    be unidirectional.
    If your language model is bidirectional, it is IMPORTANT that your bidirectional
    ``Seq2SeqEncoder`` contextualizer does not do any "peeking ahead". That is, for its
    forward direction it should only consider embeddings at previous timesteps, and for
    its backward direction only embeddings at subsequent timesteps. Similarly, if your
    language model is unidirectional, the unidirectional contextualizer should only
    consider embeddings at previous timesteps. If this condition is not met, your
    language model is cheating.
    Parameters
    ----------
    vocab: ``Vocabulary``
    text_field_embedder: ``TextFieldEmbedder``
        Used to embed the indexed tokens we get in ``forward``.
    contextualizer: ``Seq2SeqEncoder``
        Used to "contextualize" the embeddings. As described above,
        this encoder must not cheat by peeking ahead.
    dropout: ``float``, optional (default: None)
        If specified, dropout is applied to the contextualized embeddings before computation of
        the softmax. The contextualized embeddings themselves are returned without dropout.
    bidirectional: ``bool``, optional (default: False)
        Train a bidirectional language model, where the contextualizer
        is used to predict the next and previous token for each input token.
        This must match the bidirectionality of the contextualizer.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        contextualizer: Seq2SeqEncoder,
        hparams: Dict,
    ) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder

        self.contextualizer = contextualizer
        self.bidirectional = contextualizer.is_bidirectional()

        if self.bidirectional:
            self.forward_dim = contextualizer.get_output_dim() // 2
        else:
            self.forward_dim = contextualizer.get_output_dim()

        dropout = hparams["dropout"]
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

        self.hidden2chord = torch.nn.Sequential(
            torch.nn.Linear(self.forward_dim, hparams["fc_hidden_dim"]),
            torch.nn.ReLU(True),
            torch.nn.Linear(hparams["fc_hidden_dim"], vocab.get_vocab_size()),
        )
        self.perplexity = Perplexity()
        self.accuracy = CategoricalAccuracy()
        self.real_loss = Average()

        self.similarity_matrix = hparams["similarity_matrix"]
        self.training_mode = hparams["training_mode"]

        self.T_initial = hparams["T_initial"]
        self.T = self.T_initial
        self.decay_rate = hparams["decay_rate"]

        self.batches_per_epoch = hparams["batches_per_epoch"]
        self.epoch = 0
        self.batch_counter = 0

    def num_layers(self) -> int:
        """
        Returns the depth of this LM. That is, how many layers the contextualizer has plus one for
        the non-contextual layer.
        """
        if hasattr(self.contextualizer, "num_layers"):
            return self.contextualizer.num_layers + 1
        else:
            raise NotImplementedError(
                f"Contextualizer of type {type(self.contextualizer)} "
                + "does not report how many layers it has."
            )

    def loss_helper(
        self, direction_embeddings: torch.Tensor, direction_targets: torch.Tensor
    ):
        mask = direction_targets > 0
        # we need to subtract 1 to undo the padding id since the softmax
        # does not include a padding dimension

        # shape (batch_size * timesteps, )
        non_masked_targets = direction_targets.masked_select(mask)

        # shape (batch_size * timesteps, embedding_dim)
        non_masked_embeddings = direction_embeddings.masked_select(
            mask.unsqueeze(-1)
        ).view(-1, self.forward_dim)
        # note: need to return average loss across forward and backward
        # directions, but total sum loss across all batches.
        # Assuming batches include full sentences, forward and backward
        # directions have the same number of samples, so sum up loss
        # here then divide by 2 just below
        probs = torch.nn.functional.log_softmax(
            self.hidden2chord(non_masked_embeddings), dim=-1
        )

        real_loss = torch.nn.functional.nll_loss(
            probs, non_masked_targets, reduction="sum"
        )
        # transform targets into probability distributions using Embedding
        # then compute loss using torch.nn.functional.kl_div
        if self.training:
            if self.training_mode == TM_ONE_HOT:
                train_loss = real_loss
            elif self.training_mode == TM_NO:
                target_distributions = self.similarity_matrix(non_masked_targets)
                train_loss = torch.nn.functional.kl_div(
                    probs, target_distributions, reduction="sum"
                )
            elif self.training_mode == TM_FIXED or self.training_mode == TM_DECREASED:
                target_distributions = self.similarity_matrix(non_masked_targets)
                target_distributions = torch.nn.functional.softmax(
                    target_distributions / self.T, dim=1
                )
                train_loss = torch.nn.functional.kl_div(
                    probs, target_distributions, reduction="sum"
                )
            else:
                raise ValueError("Unknown training mode: {}".format(self.training_mode))
        else:
            train_loss = real_loss
        return train_loss, real_loss

    @overrides
    def forward(
        self,
        input_tokens: Dict[str, torch.LongTensor],
        forward_output_tokens: Dict[str, torch.LongTensor],
        backward_output_tokens: Dict[str, torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the averaged forward (and backward, if language model is bidirectional)
        LM loss from the batch.
        Returns
        -------
        Dict with keys:
        ``'loss'``: ``torch.Tensor``
            forward negative log likelihood, or the average of forward/backward
            if language model is bidirectional
        ``'forward_loss'``: ``torch.Tensor``
            forward direction negative log likelihood
        ``'backward_loss'``: ``torch.Tensor`` or ``None``
            backward direction negative log likelihood. If language model is not
            bidirectional, this is ``None``.
        ``'contextual_embeddings'``: ``Union[torch.Tensor, List[torch.Tensor]]``
            (batch_size, timesteps, embed_dim) tensor of top layer contextual representations or
            list of all layers. No dropout applied.
        ``'noncontextual_token_embeddings'``: ``torch.Tensor``
            (batch_size, timesteps, token_embed_dim) tensor of bottom layer noncontextual
            representations
        ``'mask'``: ``torch.Tensor``
            (batch_size, timesteps) mask for the embeddings
        """
        self.batch_counter += 1
        if self.batch_counter % self.batches_per_epoch == 0:
            self.epoch += 1
            if self.training_mode == TM_DECREASED:
                self.T *= 1 / (1 + self.decay_rate * self.epoch)
                if self.T < 1e-20:
                    self.T = 1e-20

        mask = get_text_field_mask(input_tokens)

        # shape (batch_size, timesteps, embedding_size)
        embeddings = self.text_field_embedder(input_tokens)

        contextual_embeddings = self.contextualizer(embeddings, mask)
        contextual_embeddings_with_dropout = self.dropout(contextual_embeddings)

        if self.bidirectional:
            forward_embeddings, backward_embeddings = contextual_embeddings_with_dropout.chunk(
                2, -1
            )
            backward_logits = self.hidden2chord(backward_embeddings)
        else:
            forward_embeddings = contextual_embeddings_with_dropout
            backward_logits = None
        forward_logits = self.hidden2chord(forward_embeddings)

        forward_targets = forward_output_tokens.get("tokens")
        if self.bidirectional:
            backward_targets = backward_output_tokens.get("tokens")

        # compute loss
        forward_loss, forward_real_loss = self.loss_helper(
            forward_embeddings, forward_targets
        )
        if self.bidirectional:
            backward_loss, backward_real_loss = self.loss_helper(
                backward_embeddings, backward_targets
            )
        else:
            backward_loss, backward_real_loss = None, None

        return_dict = {}

        num_targets = torch.sum((forward_targets > 0).long())
        if num_targets > 0:
            if self.bidirectional:
                average_loss = (
                    0.5 * (forward_loss + backward_loss) / num_targets.float()
                )
                average_real_loss = (
                    0.5 * (forward_real_loss + backward_real_loss) / num_targets.float()
                )
            else:
                average_loss = forward_loss / num_targets.float()
                average_real_loss = forward_real_loss / num_targets.float()
        else:
            average_loss = torch.tensor(0.0).to(forward_targets.device)
            average_real_loss = torch.tensor(0.0).to(forward_targets.device)

        self.perplexity(average_real_loss)
        self.accuracy(forward_logits, forward_targets, mask)
        self.real_loss(average_real_loss)

        return_dict.update({"loss": average_loss})

        return_dict.update(
            {
                # Note: These embeddings do not have dropout applied.
                "contextual_embeddings": contextual_embeddings,
                "noncontextual_token_embeddings": embeddings,
                "forward_logits": forward_logits,
                "backward_logits": backward_logits,
                "mask": mask,
            }
        )

        return return_dict

    def get_metrics(self, reset: bool = False):
        return {
            "perplexity": self.perplexity.get_metric(reset=reset),
            "accuracy": self.accuracy.get_metric(reset=reset),
            "real_loss": float(self.real_loss.get_metric(reset=reset)),
        }


@Metric.register("perplexity")
class Perplexity(Average):
    """
    Perplexity is a common metric used for evaluating how well a language model
    predicts a sample.
    Notes
    -----
    Assumes negative log likelihood loss of each batch (base e). Provides the
    average perplexity of the batches.
    """

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        """
        Returns
        -------
        The accumulated perplexity.
        """
        average_loss = super().get_metric(reset)
        if average_loss == 0:
            return 0.0

        # Exponentiate the loss to compute perplexity
        return float(torch.exp(average_loss))

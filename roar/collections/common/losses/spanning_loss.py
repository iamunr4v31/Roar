from torch import nn

from roar.core.classes import Loss, typecheck
from roar.core.neural_types import ChannelType, LogitsType, LossType, NeuralType

__all__ = ["SpanningLoss"]


class SpanningLoss(Loss):
    """
    implements start and end loss of a span e.g. for Question Answering.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "logits": NeuralType(("B", "T", "D"), LogitsType()),
            "start_positions": NeuralType(tuple("B"), ChannelType()),
            "end_positions": NeuralType(tuple("B"), ChannelType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {
            "loss": NeuralType(elements_type=LossType()),
            "start_logits": NeuralType(("B", "T"), LogitsType()),
            "end_logits": NeuralType(("B", "T"), LogitsType()),
        }

    def __init__(
        self,
    ):
        super().__init__()

    @typecheck()
    def forward(self, logits, start_positions, end_positions):
        """
        Args:
            logits: Output of question answering head, which is a token classfier.
            start_positions: Ground truth start positions of the answer w.r.t.
                input sequence. If question is unanswerable, this will be
                pointing to start token, e.g. [CLS], of the input sequence.
            end_positions: Ground truth end positions of the answer w.r.t.
                input sequence. If question is unanswerable, this will be
                pointing to start token, e.g. [CLS], of the input sequence.
        """
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss, start_logits, end_logits

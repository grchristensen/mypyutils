import torch
from torch import Tensor
from torch.nn import Module, functional as fn


class EncoderAdditiveAlignment(Module):
    """Computes the alignment scores for a set of encoder states."""

    def __init__(self, key_layer, energy_layer):
        """
        :param key_layer: The layer to apply to the key (encoder states). Must be compatible with the :param
        energy_layer
        :param energy_layer: The layer to apply after the :param key_layer
        """
        super().__init__()
        self.key_layer = key_layer
        self.energy_layer = energy_layer

    def forward(self, x: Tensor, seq_lens: Tensor = None) -> Tensor:
        """
        Apply the forward pass for the model.

        :param x: The key (encoder states)
        :param seq_lens: If given, values that fall out of each sequence length will be masked
        :return: The alignment scores for this key
        """
        # This module is a trainable 2-layer MLP
        key = torch.tanh(self.key_layer(x))
        energy = self.energy_layer(key).squeeze(2)

        if seq_lens is not None:
            # If seq_lens are given, we must mask values that are outside the sequence boundary
            max_length = max(seq_lens)

            # Shapes: (seq_len, 1) >= (1, batch_size)
            mask = torch.arange(max_length)[:, None] >= seq_lens[None, :]

            # Masking with -inf will cause softmax to output 0. for that value
            energy = energy.masked_fill(mask, value=float('-inf'))

        # Softmax so that a weighted average can be taken with these scores
        alignment_scores = fn.softmax(energy, dim=0)
        return alignment_scores

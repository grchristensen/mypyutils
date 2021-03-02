import torch
from torch import Tensor
from torch.nn import Module, functional as fn, Linear


class AdditiveAttention(Module):
    """Computes a context vector based on the given key."""

    def __init__(self, key_size, query_size=None, bidirectional=False):
        """
        :param key_size: The size of the encoder hidden states
        :param query_size: NOT SUPPORTED
        :param bidirectional: If True, key_size will be doubled to account for bidirectional encoders
        """
        # TODO: Look into whether super can be initialized with different classes...
        super().__init__()
        if bidirectional:
            self.key_size = 2 * key_size
        else:
            self.key_size = key_size

        # TODO: Support decoder attention
        if query_size is not None:
            raise
        query_size = 0

        layer_size = max(key_size, query_size)
        key_layer = Linear(self.key_size, layer_size)
        energy_layer = Linear(layer_size, 1)
        alignment = EncoderAdditiveAlignment(key_layer=key_layer, energy_layer=energy_layer)
        self._impl = EncoderAdditiveAttention(alignment=alignment)

    def forward(self, *args, **kwargs):
        """Apply a forward pass"""
        return self._impl(args, kwargs)


class EncoderAdditiveAttention(Module):
    """Computes a context vector based on the weighted average of the given encoder states."""

    def __init__(self, alignment):
        """
        :param alignment: The module used to compute the alignment scores
        """
        super().__init__()
        self.alignment = alignment

    def forward(self, x, seq_lens=None):
        """
        Apply the forward pass to :param x.

        :param x: The key (encoder states) to combine
        :param seq_lens: If provided, anything past each seq_len will not be included in the weighted average
        :return: The combined key, a weighted average of the encoder states
        """
        alignment_scores = self.alignment(x, seq_lens)

        weighted_encoder_states = alignment_scores[..., None] * x
        weighted_average = torch.sum(weighted_encoder_states, dim=0)

        return weighted_average


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
        Apply the forward pass to :param x.

        :param x: The key (encoder states)
        :param seq_lens: If given, values that fall out of each sequence length will be masked
        :return: The alignment scores for this key
        """
        # This module is a trainable 2-layer MLP
        key = torch.tanh(self.key_layer(x))
        energy = self.energy_layer(key).squeeze(2)

        if seq_lens is not None:
            energy = mask_energy(energy, seq_lens)

        # Softmax so that a weighted average can be taken with these scores
        alignment_scores = fn.softmax(energy, dim=0)
        return alignment_scores


def mask_energy(energy, seq_lens):
    max_length = max(seq_lens)

    # Shapes: (seq_len, 1) >= (1, batch_size)
    mask = torch.arange(max_length)[:, None] >= seq_lens[None, :]

    # Masking with -inf will cause softmax to output 0. for that value
    energy = energy.masked_fill(mask, value=float('-inf'))

    return energy


class DecoderAdditiveAlignment(Module):
    def __init__(self, query_layer, key_layer, energy_layer):
        super().__init__()
        self.query_layer = query_layer
        self.key_layer = key_layer
        self.energy_layer = energy_layer

    def forward(self, query, key, seq_lens=None):
        query = self.query_layer(query)
        key = self.key_layer(key)

        energy = self.energy_layer(torch.tanh(query + key)).squeeze(2)

        if seq_lens is not None:
            energy = mask_energy(energy, seq_lens)

        alignment_scores = fn.softmax(energy, dim=0)
        return alignment_scores

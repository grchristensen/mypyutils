import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class Encoder(Module):
    """Encodes a sequence into one vector."""

    def __init__(self, rnn, attn=None):
        """
        :param rnn: The RNN module to use for encoding.
        :param attn: The attention module to use after encoding.
        """
        super().__init__()

        if attn is None:
            self._impl = NoAttnEncoder(rnn)
        else:
            self._impl = AttnEncoder(rnn, attn)

    def forward(self, x):
        """
        Pass :param x through the network.
        :param x: The input for the network, should be of shape (seq_len, batch_size, num_features). If attention is
        used, this should not be a packed sequence.
        :return: The output of the network.
        """
        return self._impl(x)


class NoAttnEncoder(Module):
    """Encodes a sequence into a vector, without using attention."""

    def __init__(self, rnn):
        """
        :param rnn: The RNN module to use in encoding.
        """
        super().__init__()
        self.rnn = rnn

    def forward(self, x: Tensor) -> Tensor:
        """
        Pass :param x through the network.
        :param x: The input to the network, should be of shape (seq_len, batch_size, num_features).
        :return: The output of the network.
        """
        encoding, _ = self.rnn(x)

        # The way that we retrieve the last encoding depends on whether the sequence is packed.
        if isinstance(encoding, PackedSequence):
            encoding, seq_lens = pad_packed_sequence(encoding)

            # If the sequence is packed, then each encoding is at the position of the last item in the sequence
            # (seq_len - 1).
            encoding = encoding[seq_lens - 1, torch.arange(encoding.shape[1])]
        else:
            # If the sequence is not packed, then we know each encoding is simply located at the last position in the
            # sequence.
            encoding = encoding[-1]

        return encoding


class AttnEncoder(Module):
    """Encodes a sequence into a vector, with attention."""

    def __init__(self, rnn, attn):
        """
        :param rnn: The RNN module to use in encoding.
        :param attn: The attention module to use after encoding.
        """
        super().__init__()
        self.rnn = rnn
        self.attn = attn

    def forward(self, x):
        """
        Pass :param x through the network.
        :param x: The input to the network, should be of shape (seq_len, batch_size, num_features).
        :return: The output of the network.
        """
        seq_outputs, _ = self.rnn(x)

        if isinstance(seq_outputs, PackedSequence):
            seq_outputs, seq_lens = pad_packed_sequence(seq_outputs)

            encoding = self.attn(seq_outputs, seq_lens)
        else:
            encoding = self.attn(seq_outputs)

        return encoding

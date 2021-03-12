import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


class Encoder(Module):
    """Encodes a sequence into one vector."""

    def __init__(self, rnn, attn=None, whole_sequence=False):
        """
        :param rnn: The RNN module to use for encoding.
        :param attn: The attention module to use after encoding.
        :param whole_sequence: If True, this encoder will return the whole sequence
        """
        super().__init__()

        if attn is None:
            self._impl = NoAttnEncoder(rnn, whole_sequence)
        else:
            self._impl = AttnEncoder(rnn, attn)

    def forward(self, x):
        """
        Pass :param x through the network.
        :param x: The input for the network, should be of shape (seq_len, batch_size,
        num_features). If attention is used, this should not be a packed sequence.
        :return: The output of the network.
        """
        return self._impl(x)


class NoAttnEncoder(Module):
    """Encodes a sequence into a vector, without using attention."""

    def __init__(self, rnn, whole_sequence=False):
        """
        :param rnn: The RNN module to use in encoding.
        :param whole_sequence: If True, the encoder returns the entire annotated
        sequence.
        """
        super().__init__()
        self.rnn = rnn
        self.whole_sequence = whole_sequence

    def forward(self, x: Tensor) -> Tensor:
        """
        Pass :param x through the network.
        :param x: The input to the network, should be of shape (seq_len, batch_size,
        num_features).
        :return: The output of the network.
        """
        encoding, _ = self.rnn(x)

        if self.whole_sequence:
            return encoding

        # The way that we retrieve the last encoding depends on whether the sequence is
        # packed.
        if isinstance(encoding, PackedSequence):
            encoding, seq_lens = pad_packed_sequence(encoding)

            # If the sequence is packed, then each encoding is at the position of the
            # last item in the sequence (seq_len - 1).
            encoding = encoding[seq_lens - 1, torch.arange(encoding.shape[1])]
        else:
            # If the sequence is not packed, then we know each encoding is simply
            # located at the last position in the sequence.
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
        :param x: The input to the network, should be of shape (seq_len, batch_size,
        num_features).
        :return: The output of the network.
        """
        seq_outputs, _ = self.rnn(x)

        if isinstance(seq_outputs, PackedSequence):
            seq_outputs, seq_lens = pad_packed_sequence(seq_outputs)

            encoding = self.attn(seq_outputs, seq_lens)
        else:
            encoding = self.attn(seq_outputs)

        return encoding


# TODO: Check if RNNs can accept non-flattened inputs, (and check if it still works with
# pad/pack)
# TODO: Check whether computing a value and then padding it with pack_padded_sequence
# will still have it affect grad
class ChainedEncoder(Module):
    """Encodes nested sequences using multiple encoders."""

    def __init__(self, *args):
        super().__init__()
        self.encoders = args

    def forward(self, x, seq_lens_list):
        """
        Encode the nested sequences

        :param x The first level of the sequences to encode
        :param seq_lens_list The seq_lens for each sequence after the first. If an
        integer, no packing occurs, whereas tensors cause packing
        """
        encodings = self.encoders[0](x).unsqueeze(0)

        for index, seq_lens in enumerate(seq_lens_list):
            # We base the new shape off of the old one
            new_shape = list(encodings.shape)

            if isinstance(seq_lens, Tensor):
                # Since seq_lens is a sequence, we assume that each sequence is the
                # maximum length, and pad sequences that don't fit this description.
                seq_len = max(seq_lens)
            else:
                # seq_lens is just an integer, no packing required
                seq_len = seq_lens

            new_shape[0] = int(new_shape[1] / seq_len)
            new_shape[1] = seq_len

            encodings = torch.reshape(encodings, new_shape)
            encodings = torch.transpose(encodings, 0, 1)

            if isinstance(seq_lens, Tensor):
                encodings = pack_padded_sequence(
                    encodings, seq_lens, enforce_sorted=False
                )

            encodings = self.encoders[index + 1](encodings)
            if isinstance(encodings, PackedSequence):
                return encodings

            encodings = encodings.unsqueeze(0)

        return encodings.squeeze(0)

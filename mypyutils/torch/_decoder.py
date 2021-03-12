import torch
from torch.nn import Module, functional as fn


class Decoder(Module):
    """Decodes a vector into a sequence"""

    def __init__(self, rnn, max_length, attn=None, softmax=False, sos_value=0.0):
        """
        :param rnn: The rnn module to use for decoding
        :param max_length: The sequence length of the output
        :param attn: The attention module to use for decoding
        :param softmax: If True, a softmax function will be applied after the rnn output
        :param sos_value: The value to create the start of string vector with
        """
        super().__init__()

        if attn is None:
            self._impl = NoAttnDecoder(
                rnn, max_length, softmax=softmax, sos_value=sos_value
            )
        else:
            self._impl = AttnDecoder(
                rnn, attn, max_length, softmax=softmax, sos_value=sos_value
            )

    def forward(self, *args, **kwargs):
        """Decode the vector"""
        return self._impl(args, kwargs)

    def teacher_force(self, *args, **kwargs):
        """Decode the vector, but outputs are forced"""
        return self._impl(args, kwargs)


# TODO: Pretty much all of these functions break DRY, comments would need to be synced
# across them
class AttnDecoder(Module):
    # TODO: Make sure you use SOS token properly
    def __init__(self, rnn, attn, max_length, softmax=False, sos_value=0.0):
        super().__init__()
        self.rnn = rnn
        self.attn = attn
        self.max_length = max_length
        self.softmax = softmax
        self.sos_value = sos_value

    def forward(self, query, key, seq_lens=None):
        batch_size, _ = query.shape

        output = torch.zeros(self.max_length, batch_size, self.rnn.hidden_size)
        decoder_input = torch.full(
            [1, batch_size, self.rnn.hidden_size], fill_value=self.sos_value
        )

        for index in range(self.max_length):
            context_vector = self.attn(query, key, seq_lens=seq_lens)

            decoder_input, query = self.rnn(decoder_input, context_vector)

            if self.softmax:
                decoder_input = fn.softmax(decoder_input, dim=-1)

            output[index] = decoder_input[0]

        return output

    def teacher_force(self, query, key, forced, seq_lens=None):
        batch_size, _ = query.shape

        output = torch.zeros(self.max_length, batch_size, self.rnn.hidden_size)
        decoder_input = torch.full(
            [1, batch_size, self.rnn.hidden_size], fill_value=self.sos_value
        )

        for index in range(self.max_length):
            context_vector = self.attn(query, key, seq_lens=seq_lens)

            decoder_output, query = self.rnn(decoder_input, context_vector)

            decoder_input = forced[index][None, ...]

            output[index] = decoder_output[0]

        return output


class NoAttnDecoder(Module):
    def __init__(self, rnn, max_length, softmax=False, sos_value=0.0):
        super().__init__()
        self.rnn = rnn
        self.max_length = max_length
        self.softmax = softmax
        self.sos_value = sos_value

    def forward(self, context_vector):
        batch_size, _ = context_vector.shape

        output = torch.zeros(self.max_length, batch_size, self.rnn.hidden_size)
        decoder_input = torch.full(
            [1, batch_size, self.rnn.hidden_size], fill_value=self.sos_value
        )

        for index in range(self.max_length):
            decoder_input, context_vector = self.rnn(decoder_input, context_vector)

            if self.softmax:
                decoder_input = fn.softmax(decoder_input, dim=-1)

            output[index] = decoder_input[0]

        return output

    def teacher_force(self, context_vector, forced):
        batch_size, _ = context_vector.shape

        output = torch.zeros(self.max_length, batch_size, self.rnn.hidden_size)
        decoder_input = torch.full(
            [1, batch_size, self.rnn.hidden_size], fill_value=self.sos_value
        )

        for index in range(self.max_length):
            decoder_output, context_vector = self.rnn(decoder_input, context_vector)
            # To be fed into the rnn later decoder_input needs to at least have a
            # seq_len of 1
            decoder_input = forced[index][None, ...]

            output[index] = decoder_output[0]

        return output

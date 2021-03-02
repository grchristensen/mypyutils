from unittest.mock import MagicMock

import torch
from torch.nn import functional as fn

from mypyutils.torch import EncoderAdditiveAttention, DecoderAdditiveAttention


class IdentityRNN(MagicMock):
    def __init__(self):
        def mock_forward(x):
            return x, 2

        super().__init__(side_effect=mock_forward)
        self.forward = mock_forward


class AddHiddenRNN(MagicMock):
    def __init__(self, hidden_size):
        def mock_forward(x, h0=None):
            seq_len, batch_size, _ = x.shape
            if h0 is None:
                h0 = torch.zeros(batch_size, hidden_size)[None, ...]

            acc = torch.arange(seq_len)[:, None, None] + 1

            output = h0 + acc
            return output, output[-1]

        super().__init__(side_effect=mock_forward)
        self.hidden_size = hidden_size
        self.forward = mock_forward


class AddHiddenAndInputRNN(MagicMock):
    def __init__(self, hidden_size):
        def mock_forward(x, h0=None):
            seq_len, batch_size, _ = x.shape
            if h0 is None:
                h0 = torch.zeros(batch_size, hidden_size)[None, ...]

            acc = torch.arange(seq_len)[:, None, None] + 1

            output = x + h0 + acc
            return output, output[-1]

        super().__init__(side_effect=mock_forward)
        self.hidden_size = hidden_size
        self.forward = mock_forward


class IdentityLinear(MagicMock):
    def __init__(self, output_size):
        def mock_forward(x):
            # Just return the same input but only up to output_size length.
            return x[..., :output_size]

        super().__init__(side_effect=mock_forward)
        self.forward = mock_forward


class HalfToFirstAlignment(MagicMock):
    def __init__(self):
        def mock_forward(x, seq_lens=None):
            max_seq_len, batch_size = x.shape[0], x.shape[1]
            alignment_scores = torch.zeros(max_seq_len, batch_size)

            # If seq_lens is available we must treat each seq_len separately to avoid giving weight to padding
            if seq_lens is not None:
                for batch_index, seq_len in enumerate(seq_lens):
                    # If there is only one item in the sequence, that item gets the full weight
                    if seq_len == 1:
                        alignment_scores[0, batch_index] = 1.
                        # If max_seq_len is 1, then its not safe to index after 0
                        if max_seq_len != 1:
                            alignment_scores[1:, batch_index] = 0.
                    else:
                        # The first weight is 0.5 and the rest (that are valid) are even
                        alignment_scores[0, batch_index] = 0.5
                        alignment_scores[1:seq_len, batch_index] = 0.5 / (seq_len - 1)
            else:
                # If there is only one item in the sequence, that item gets the full weight
                if max_seq_len != 1:
                    # If there is more than 1 item in the sequence, the first item gets half weight and the rest are
                    # even
                    alignment_scores[0, :] = 0.5
                    alignment_scores[1:, :] = 0.5 / (max_seq_len - 1)
                else:
                    alignment_scores[0, :] = 1.

            return alignment_scores

        super().__init__(side_effect=mock_forward)
        self.forward = mock_forward


class HalfToFirstAttention(EncoderAdditiveAttention):
    def __init__(self):
        super().__init__(HalfToFirstAlignment())


class MeanAttention(MagicMock):
    def __init__(self):
        def mock_forward(x, seq_lens=None):
            if seq_lens is not None:
                max_length = max(seq_lens)

                # Shapes: (seq_len, 1) >= (1, batch_size)
                mask = torch.arange(max_length)[:, None] >= seq_lens[None, :]

                x = x.masked_fill(mask.unsqueeze(2), value=0.)  # noqa: Unresolved Attribute 'unsqueeze'

                return torch.sum(x, dim=0) / seq_lens.unsqueeze(1)

            return torch.mean(x, dim=0)

        super().__init__(side_effect=mock_forward)
        self.forward = mock_forward


class SumToFirstAlignment(MagicMock):
    def __init__(self):
        def mock_forward(query, key, seq_lens=None):
            seq_len, batch_size, _ = key.shape
            query_sum = torch.sum(query, dim=1)
            energy = torch.zeros(seq_len, batch_size)

            energy[0] = query_sum

            if seq_lens is not None:
                max_length = max(seq_lens)

                # Shapes: (seq_len, 1) >= (1, batch_size)
                mask = torch.arange(max_length)[:, None] >= seq_lens[None, :]

                energy = energy.masked_fill(mask.unsqueeze(2), value=float('-inf'))

            return fn.softmax(energy, dim=0)

        super().__init__(side_effect=mock_forward)
        self.forward = mock_forward


class SumToFirstAttention(DecoderAdditiveAttention):
    def __init__(self):
        super().__init__(SumToFirstAlignment())

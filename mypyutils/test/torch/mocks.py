from unittest.mock import MagicMock

import torch


class IdentityLinear(MagicMock):
    def __init__(self, output_size):
        def mock_forward(x):
            # Just return the same input but only up to output_size length.
            return x[..., :output_size]

        super().__init__(side_effect=mock_forward)
        self.forward = mock_forward


class FirstToHalfAlignment(MagicMock):
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

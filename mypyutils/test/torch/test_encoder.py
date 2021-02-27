import pytest  # noqa
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from mypyutils.test.torch.mocks import IdentityRNN, MeanAttention, HalfToFirstAttention
from mypyutils.torch import NoAttnEncoder, AttnEncoder


class TestNoAttnEncoder:
    normal_tensors = [
        (torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 2, 3]
        ]),
         torch.tensor([1, 2, 3])),
        (torch.tensor([
            [0., 0., 0., 0.],
            [1., 1., 1., 1.],
            [5., 4., 3., 2.]
        ]),
         torch.tensor([5., 4., 3., 2.])),
        (torch.repeat_interleave(torch.tensor([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 2, 3, 4]
        ]).unsqueeze(2), 3, dim=2),
         torch.tensor([
             [1, 1, 1],
             [2, 2, 2],
             [3, 3, 3],
             [4, 4, 4]
         ]))
    ]

    @pytest.mark.parametrize('x, expected_output', normal_tensors)
    def test_encoder_should_retrieve_last_seq_item(self, x, expected_output):
        encoder = NoAttnEncoder(IdentityRNN())

        output = encoder(x)

        assert torch.equal(output, expected_output)

    packed_tensors = [
        (pack_padded_sequence(torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 3],
            [0, 2, 0]
        ]), torch.tensor([2, 4, 3]), enforce_sorted=False),
         torch.tensor([1, 2, 3])),
        (pack_padded_sequence(torch.tensor([
            [1., 4., 1., 1.],
            [1., 0., 3., 1.],
            [5., 0., 0., 2.]
        ]), torch.tensor([3, 1, 2, 3]), enforce_sorted=False),
         torch.tensor([5., 4., 3., 2.])),
        (pack_padded_sequence(torch.repeat_interleave(torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 3, 4],
            [1, 2, 0, 0]
        ]).unsqueeze(2), 2, dim=2), torch.tensor([3, 3, 2, 2]), enforce_sorted=False),
         torch.tensor([
             [1, 1],
             [2, 2],
             [3, 3],
             [4, 4]
         ]))
    ]

    @pytest.mark.parametrize('x, expected_output', packed_tensors)
    def test_encoder_should_retrieve_last_packed_seq_item(self, x, expected_output):
        encoder = NoAttnEncoder(IdentityRNN())

        output = encoder(x)

        assert torch.equal(output, expected_output)


class TestAttnEncoder:
    test_data = [
        (torch.tensor([
            [2, -2, 3.],
            [4., 2., 6.]
        ]),
         torch.tensor([3., 0., 4.5])),
        (torch.repeat_interleave(torch.tensor([
            [1., 6., -9.],
            [3., 8., 9.]
        ]).unsqueeze(2), 4, dim=2),
         torch.tensor([
             [2., 2., 2., 2.],
             [7., 7., 7., 7.],
             [0., 0., 0., 0.]
         ]))
    ]

    @pytest.mark.parametrize('x, expected_output', test_data)
    def test_encoder_uses_attn(self, x, expected_output):
        encoder = AttnEncoder(IdentityRNN(), MeanAttention())

        output = encoder(x)

        assert torch.equal(output, expected_output)

    packed_tensors = [
        pack_padded_sequence(torch.tensor([
            [
                [-1., -1., -1., -1., -1., -1., -1.],
                [+1., +1., +1., +1., +1., +1., +1.],
            ],
            [
                [-2., -2., -2., -2., -2., -2., -2.],
                [+2., +2., +2., +2., +2., +2., +2.],
            ],
            [
                [-4., -4., -4., -4., -4., -4., -4.],
                [+0., +0., +0., +0., +0., +0., +0.],
            ]
        ]), torch.tensor([3, 2]), enforce_sorted=False)
    ]

    expected_outputs = [
        torch.tensor([
            [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
            [+1.5, +1.5, +1.5, +1.5, +1.5, +1.5, +1.5],
        ])
    ]

    @pytest.mark.parametrize('x, expected_output', zip(packed_tensors, expected_outputs))
    def test_encoder_accepts_packed_sequences(self, x, expected_output):
        encoder = AttnEncoder(IdentityRNN(), HalfToFirstAttention())

        output = encoder(x)

        assert torch.equal(output, expected_output)

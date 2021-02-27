import pytest  # noqa
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tests.torch.mocks import IdentityRNN, MeanAttention, HalfToFirstAttention
from mypyutils.torch import NoAttnEncoder, AttnEncoder, ChainedEncoder


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


class TestChainedEncoder:
    sentences_list = [
        torch.tensor([  # 4 sentences per paragraph, each sentence is 5 words long. Total 2 paragraphs
            [1.00, 5.00, 10.0, -1.0, 5.00, 101., -5.0, -11.],
            [2.00, 4.00, 9.00, -2.0, 10.0, 102., -4.0, -12.],
            [3.00, 3.00, 8.00, -3.0, 15.0, 103., -3.0, -13.],
            [4.00, 2.00, 7.00, -4.0, 20.0, 104., -2.0, -14.],
            [5.00, 1.00, 6.00, -5.0, 25.0, 105., -1.0, -15.]
        ]).unsqueeze(2),
        torch.tensor([  # 4 sentences per paragraph, each sentence is 5 words long. Total 2 paragraphs
            [1.00, 5.00, 0.00, 0.00, 5.00, 101., -5.0, -11.],
            [2.00, 4.00, 0.00, 0.00, 10.0, 102., -4.0, -12.],
            [3.00, 3.00, 0.00, 0.00, 15.0, 103., -3.0, -13.],
            [4.00, 2.00, 0.00, 0.00, 20.0, 104., -2.0, -14.],
            [5.00, 1.00, 0.00, 0.00, 25.0, 105., -1.0, -15.]
        ]).unsqueeze(2)
    ]

    paragraph_lens_list = [
        4,
        torch.tensor([2, 4])
    ]

    expected_outputs = [
        torch.tensor([1.75, 28.5]).unsqueeze(1),
        torch.tensor([3.00, 28.5]).unsqueeze(1)
    ]

    @pytest.mark.parametrize('x, paragraph_lens, expected', zip(sentences_list, paragraph_lens_list, expected_outputs))
    def test_two_encoders(self, x, paragraph_lens, expected):
        encoder1 = NoAttnEncoder(IdentityRNN())
        encoder2 = AttnEncoder(IdentityRNN(), MeanAttention())
        chained_encoder = ChainedEncoder(encoder1, encoder2)

        output = chained_encoder(x, [paragraph_lens])

        assert torch.equal(output, expected)

    sentences_list_2 = [
        torch.transpose(torch.tensor([  # 4 sentences per paragraph, 3 paragraphs per page
            [1.00, 2.00, 3.00, 4.00, 5.00],
            [5.00, 4.00, 3.00, 2.00, 1.00],
            [-5.0, -4.0, -3.0, -2.0, -1.0],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [101., 102., 103., 104., 105.],
            [-11., -12., -13., -14., -15.],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [10.0, 9.00, 8.00, 7.00, 6.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [101., 102., 103., 104., 105.],
            [-11., -12., -13., -14., -15.],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [1.00, 2.00, 3.00, 4.00, 5.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [101., 102., 103., 104., 105.],
            [-11., -12., -13., -14., -15.],
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [0.00, 0.00, 0.00, 0.00, 0.00],
        ]), 0, 1).unsqueeze(2)
    ]

    seq_lens_list = [
        [torch.tensor([3, 4, 1, 2, 1, 3]), torch.tensor([2, 3])]
    ]

    expected_outputs_2 = [
        torch.tensor([12.20833300, 30.83333325]).unsqueeze(1)
    ]

    @pytest.mark.parametrize('x, seq_lens, expected', zip(sentences_list_2, seq_lens_list, expected_outputs_2))
    def test_three_encoders(self, x, seq_lens, expected):
        encoder1 = NoAttnEncoder(IdentityRNN())
        encoder2 = AttnEncoder(IdentityRNN(), MeanAttention())
        encoder3 = AttnEncoder(IdentityRNN(), HalfToFirstAttention())
        chained_encoder = ChainedEncoder(encoder1, encoder2, encoder3)

        output = chained_encoder(x, seq_lens)

        assert torch.allclose(output, expected)

    expected_outputs_3 = [
        torch.tensor([
            [1.666666, 45.00000],
            [22.75000, 5.000000],
            [0.000000, 28.33333]
        ]).unsqueeze(2)
    ]

    @pytest.mark.parametrize('x, seq_lens, expected', zip(sentences_list_2, seq_lens_list, expected_outputs_3))
    def test_whole_sequence(self, x, seq_lens, expected):
        encoder1 = NoAttnEncoder(IdentityRNN())
        encoder2 = AttnEncoder(IdentityRNN(), MeanAttention())
        encoder3 = NoAttnEncoder(IdentityRNN(), whole_sequence=True)
        chained_encoder = ChainedEncoder(encoder1, encoder2, encoder3)

        output = chained_encoder(x, seq_lens)
        output, _ = pad_packed_sequence(output)

        assert torch.allclose(output, expected)

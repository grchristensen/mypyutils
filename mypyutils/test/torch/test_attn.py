import pytest  # noqa
import torch

from mypyutils.test.torch.mocks import IdentityLinear, HalfToFirstAlignment
from mypyutils.torch import EncoderAdditiveAlignment, EncoderAdditiveAttention


class TestEncoderAdditiveAttention:
    encoder_states_list = [
        torch.tensor([
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ],
            [
                [1.2, 2.2, 3.2, 4.2, 5.2],
                [1.2, 2.2, 3.2, 4.2, 5.2],
                [1.2, 2.2, 3.2, 4.2, 5.2],
                [1.2, 2.2, 3.2, 4.2, 5.2],
            ],
            [
                [1.4, 2.4, 3.4, 4.4, 5.4],
                [1.4, 2.4, 3.4, 4.4, 5.4],
                [1.4, 2.4, 3.4, 4.4, 5.4],
                [1.4, 2.4, 3.4, 4.4, 5.4],
            ]
        ])
    ]

    expected_outputs = [
        torch.tensor([
            [1.15, 2.15, 3.15, 4.15, 5.15],
            [1.15, 2.15, 3.15, 4.15, 5.15],
            [1.15, 2.15, 3.15, 4.15, 5.15],
            [1.15, 2.15, 3.15, 4.15, 5.15]
        ])
    ]

    @pytest.mark.parametrize('x, expected_output', zip(encoder_states_list, expected_outputs))
    def test_forward_pass(self, x, expected_output):
        alignment = HalfToFirstAlignment()
        attn = EncoderAdditiveAttention(alignment=alignment)

        output = attn(x)

        assert torch.equal(output, expected_output)

    padded_encoder_states_list = [
        torch.tensor([
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ],
            [
                [1.2, 2.2, 3.2, 4.2, 5.2],
                [1.2, 2.2, 3.2, 4.2, 5.2],
                [1.2, 2.2, 3.2, 4.2, 5.2],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.4, 2.4, 3.4, 4.4, 5.4],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ])
    ]

    seq_lens_list = [
        torch.tensor([2, 3, 2, 1])
    ]

    padded_expected_outputs = [
        torch.tensor([
            [1.10, 2.10, 3.10, 4.10, 5.10],
            [1.15, 2.15, 3.15, 4.15, 5.15],
            [1.10, 2.10, 3.10, 4.10, 5.10],
            [1.00, 2.00, 3.00, 4.00, 5.00]
        ])
    ]

    @pytest.mark.parametrize('x, seq_lens, expected_output',
                             zip(padded_encoder_states_list, seq_lens_list, padded_expected_outputs))
    def test_packed_forward_pass(self, x, seq_lens, expected_output):
        alignment = HalfToFirstAlignment()
        attn = EncoderAdditiveAttention(alignment=alignment)

        output = attn(x, seq_lens)

        assert torch.equal(output, expected_output)


class TestEncoderAdditiveAlignment:
    atol = 1e-4
    rtol = 1e-8

    encoder_states_list = [
        torch.tensor([
            [
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.]
            ],
            [
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]
            ],
            [
                [3., 3., 3., 3., 3.],
                [3., 3., 3., 3., 3.],
                [3., 3., 3., 3., 3.],
                [3., 3., 3., 3., 3.]
            ]
        ])
    ]

    expected_outputs = [
        torch.tensor([
            [0.2868, 0.2868, 0.2868, 0.2868],
            [0.3511, 0.3511, 0.3511, 0.3511],
            [0.3622, 0.3622, 0.3622, 0.3622]
        ])
    ]

    @pytest.mark.parametrize('x, expected_output', zip(encoder_states_list, expected_outputs))
    def test_forward_pass(self, x, expected_output):
        key_layer = IdentityLinear(3)
        energy_layer = IdentityLinear(1)
        alignment = EncoderAdditiveAlignment(key_layer=key_layer, energy_layer=energy_layer)

        output = alignment(x)

        assert torch.allclose(output, expected_output, atol=self.atol, rtol=self.rtol)

    padded_encoder_states_list = [
        torch.tensor([
            [
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.]
            ],
            [
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [0., 0., 0., 0., 0.]
            ],
            [
                [3., 3., 3., 3., 3.],
                [0., 0., 0., 0., 0.],
                [3., 3., 3., 3., 3.],
                [0., 0., 0., 0., 0.]
            ]
        ])
    ]

    seq_lens_list = [
        torch.tensor([3, 2, 3, 1])
    ]

    padded_expected_outputs = [
        torch.tensor([
            [0.2868, 0.4496, 0.2868, 1.0000],
            [0.3511, 0.5504, 0.3511, 0.0000],
            [0.3622, 0.0000, 0.3622, 0.0000]
        ])
    ]

    @pytest.mark.parametrize('x, seq_lens, expected_output',
                             zip(padded_encoder_states_list, seq_lens_list, padded_expected_outputs))
    def test_packed_forward_pass(self, x, seq_lens, expected_output):
        key_layer = IdentityLinear(3)
        energy_layer = IdentityLinear(1)
        alignment = EncoderAdditiveAlignment(key_layer=key_layer, energy_layer=energy_layer)

        output = alignment(x, seq_lens)

        assert torch.allclose(output, expected_output, atol=self.atol, rtol=self.rtol)

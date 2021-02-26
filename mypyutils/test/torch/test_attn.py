import pytest  # noqa
import torch

from mypyutils.test.torch.mocks import IdentityLinear
from mypyutils.torch import EncoderAdditiveAlignment


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
        attn = EncoderAdditiveAlignment(key_layer=key_layer, energy_layer=energy_layer)

        output = attn(x)

        assert torch.allclose(output, expected_output, atol=self.atol, rtol=self.rtol)

    packed_encoder_states_list = [
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

    packed_expected_outputs = [
        torch.tensor([
            [0.2868, 0.4496, 0.2868, 1.],
            [0.3511, 0.5504, 0.3511, 0.],
            [0.3622, 0.0000, 0.3622, 0.]
        ])
    ]

    @pytest.mark.parametrize('x, seq_lens, expected_output',
                             zip(packed_encoder_states_list, seq_lens_list, packed_expected_outputs))
    def test_packed_forward_pass(self, x, seq_lens, expected_output):
        key_layer = IdentityLinear(3)
        energy_layer = IdentityLinear(1)
        attn = EncoderAdditiveAlignment(key_layer=key_layer, energy_layer=energy_layer)

        output = attn(x, seq_lens)

        assert torch.allclose(output, expected_output, atol=self.atol, rtol=self.rtol)

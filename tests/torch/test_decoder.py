import pytest
import torch

from mypyutils.torch import NoAttnDecoder, AttnDecoder
from tests.torch.mocks import (
    AddHiddenRNN,
    SumToFirstAttention,
    AddHiddenAndInputRNN,
)


class TestNoAttnDecoder:
    context_vectors = [
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ]

    expected_outputs = [
        torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                [
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                ],
                [
                    [4.0, 4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0],
                ],
                [
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                ],
            ]
        )
    ]

    @pytest.mark.parametrize("x, expected", zip(context_vectors, expected_outputs))
    def test_forward_pass(self, x, expected):
        decoder = NoAttnDecoder(AddHiddenRNN(5), max_length=5)

        output = decoder(x)

        assert torch.equal(output, expected)

    forced_outputs = [
        torch.tensor(
            [
                [
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                    [3.0, 3.0, 3.0, 3.0, 3.0],
                ],
                [
                    [6.0, 6.0, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 6.0, 6.0, 6.0],
                ],
                [
                    [9.0, 9.0, 9.0, 9.0, 9.0],
                    [9.0, 9.0, 9.0, 9.0, 9.0],
                    [9.0, 9.0, 9.0, 9.0, 9.0],
                    [9.0, 9.0, 9.0, 9.0, 9.0],
                ],
                [
                    [12.0, 12.0, 12.0, 12.0, 12.0],
                    [12.0, 12.0, 12.0, 12.0, 12.0],
                    [12.0, 12.0, 12.0, 12.0, 12.0],
                    [12.0, 12.0, 12.0, 12.0, 12.0],
                ],
            ]
        )
    ]

    expected_forced_outputs = [
        torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                ],
                [
                    [12.0, 12.0, 12.0, 12.0, 12.0],
                    [12.0, 12.0, 12.0, 12.0, 12.0],
                    [12.0, 12.0, 12.0, 12.0, 12.0],
                    [12.0, 12.0, 12.0, 12.0, 12.0],
                ],
                [
                    [22.0, 22.0, 22.0, 22.0, 22.0],
                    [22.0, 22.0, 22.0, 22.0, 22.0],
                    [22.0, 22.0, 22.0, 22.0, 22.0],
                    [22.0, 22.0, 22.0, 22.0, 22.0],
                ],
            ]
        )
    ]

    @pytest.mark.parametrize(
        "x, forced, expected",
        zip(context_vectors, forced_outputs, expected_forced_outputs),
    )
    def test_teacher_force(self, x, forced, expected):
        decoder = NoAttnDecoder(AddHiddenAndInputRNN(5), max_length=4)

        output = decoder.teacher_force(x, forced)

        assert torch.equal(output, expected)


class TestAttnDecoder:
    atol = 1e-4
    rtol = 1e-8

    keys = [
        torch.tensor(
            [
                [
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                ],
                [
                    [-2.0, -2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0],
                    [-2.0, -2.0, -2.0, -2.0, -2.0],
                ],
                [
                    [-3.0, -3.0, -3.0, -3.0, -3.0],
                    [-3.0, -3.0, -3.0, -3.0, -3.0],
                    [-3.0, -3.0, -3.0, -3.0, -3.0],
                ],
                [
                    [-4.0, -4.0, -4.0, -4.0, -4.0],
                    [-4.0, -4.0, -4.0, -4.0, -4.0],
                    [-4.0, -4.0, -4.0, -4.0, -4.0],
                ],
            ]
        )
    ]

    first_queries = [
        torch.tensor(
            [
                [1.0, 0.2, 0.0, 0.5, 0.3],
                [5.0, 5.0, 5.0, 5.0, 5.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ]

    expected_outputs = [
        torch.tensor(
            [
                [
                    [-0.5775, -0.5775, -0.5775, -0.5775, -0.5775],
                    [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000],
                    [-1.5000, -1.5000, -1.5000, -1.5000, -1.5000],
                ],
                [
                    [-1.9635, -1.9635, -1.9635, -1.9635, -1.9635],
                    [-1.5000, -1.5000, -1.5000, -1.5000, -1.5000],
                    [-1.9996, -1.9996, -1.9996, -1.9996, -1.9996],
                ],
                [
                    [-2.0000, -2.0000, -2.0000, -2.0000, -2.0000],
                    [-1.9996, -1.9996, -1.9996, -1.9996, -1.9996],
                    [-2.0000, -2.0000, -2.0000, -2.0000, -2.0000],
                ],
            ]
        )
    ]

    @pytest.mark.parametrize(
        "key, first_query, expected", zip(keys, first_queries, expected_outputs)
    )
    def test_forward_pass(self, key, first_query, expected):
        decoder = AttnDecoder(AddHiddenRNN(5), SumToFirstAttention(), max_length=3)

        output = decoder(first_query, key)

        assert torch.allclose(output, expected, atol=self.atol, rtol=self.rtol)

    forced_outputs = [
        torch.tensor(
            [
                [
                    [-10.0, -10.0, -10.0, -10.0, -10.0],
                    [-10.0, -10.0, -10.0, -10.0, -10.0],
                    [-10.0, -10.0, -10.0, -10.0, -10.0],
                ],
                [
                    [-9.0, -9.0, -9.0, -9.0, -9.0],
                    [-9.0, -9.0, -9.0, -9.0, -9.0],
                    [-9.0, -9.0, -9.0, -9.0, -9.0],
                ],
                [
                    [-8.0, -8.0, -8.0, -8.0, -8.0],
                    [-8.0, -8.0, -8.0, -8.0, -8.0],
                    [-8.0, -8.0, -8.0, -8.0, -8.0],
                ],
            ]
        )
    ]

    expected_forced_outputs = [
        torch.tensor(
            [
                [
                    [-0.5775, -0.5775, -0.5775, -0.5775, -0.5775],
                    [-0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
                    [-1.5000, -1.5000, -1.5000, -1.5000, -1.5000],
                ],
                [
                    [-11.9635, -11.9635, -11.9635, -11.9635, -11.9635],
                    [-11.5000, -11.5000, -11.5000, -11.5000, -11.5000],
                    [-11.9996, -11.9996, -11.9996, -11.9996, -11.9996],
                ],
                [
                    [-11.000, -11.000, -11.000, -11.000, -11.000],
                    [-11.000, -11.000, -11.000, -11.000, -11.000],
                    [-11.000, -11.000, -11.000, -11.000, -11.000],
                ],
            ]
        )
    ]

    @pytest.mark.parametrize(
        "first_query, key, forced, expected",
        zip(first_queries, keys, forced_outputs, expected_forced_outputs),
    )
    def test_teacher_force(self, first_query, key, forced, expected):
        decoder = AttnDecoder(
            AddHiddenAndInputRNN(5), SumToFirstAttention(), max_length=3
        )

        output = decoder.teacher_force(first_query, key, forced)

        assert torch.allclose(output, expected, atol=self.atol, rtol=self.rtol)

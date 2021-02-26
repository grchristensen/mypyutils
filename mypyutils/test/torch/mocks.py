from unittest.mock import MagicMock


class IdentityLinear(MagicMock):
    def __init__(self, output_size):
        def mock_forward(x):
            # Just return the same input but only up to output_size length.
            return x[..., :output_size]

        super().__init__(side_effect=mock_forward)
        self.forward = mock_forward

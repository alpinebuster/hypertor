"""Tests for `util.py`."""

from typing import Union
from collections import namedtuple
import pytest
import torch

from hypertor.core.util import SimpleTransformerIO, \
    SeparateTransformerIO, JointTransformerIO

IOParams = namedtuple("IOParams", [
    "batch_size",
    "embedding_dim",
    "num_weights",
    "num_labels",
    "flows_shape",
])


@pytest.fixture(params=[
    (1, 4, 16, 4, 32),
    (8, 8, 128, 6, 64),
    (16, 16, 256, 8, 128),
])
def transformer_io_params(request) -> IOParams:
    """
    Fixture for `*TransformerIO` test parameters.

    Returns a dict with keys:
        batch_size, embedding_dim, num_weights, num_labels, flows_shape
    """
    return IOParams(*request.param)

class TestUtil:
    """Tests for util.py (PyTorch version)."""

    def _encode_samples(
        self,
        batch_size: int,
        flows_shape: int,
        io: Union[SimpleTransformerIO, SeparateTransformerIO, JointTransformerIO],
    ) -> torch.Tensor:
        """Creates empty images and labels and encodes for Transformer."""
        # [batch_size, channels, flows_shape, flows_shape]
        images: torch.Tensor = torch.zeros(
            batch_size, 1, flows_shape, flows_shape, dtype=torch.float32
        )
        labels: torch.Tensor = torch.zeros(batch_size, dtype=torch.long)
        return io.encode_samples(images, labels)

    def _check_weights(
        self,
        encoded: torch.Tensor,
        num_weights: int,
        flows_shape: int,
        io: Union[SimpleTransformerIO, SeparateTransformerIO, JointTransformerIO],
    ) -> None:
        """Decodes weights and checks their number and their shapes."""
        weights: list[torch.Tensor] = io.decode_weights(encoded)

        assert isinstance(weights, list)
        assert len(weights) == num_weights

        for weight in weights:
            assert weight.shape == (flows_shape**2,)

    def test_simple_transformer_io(
        self,
        transformer_io_params: IOParams,
    ) -> None:
        """Tests SimpleTransformerIO Transformer adapter."""
        batch_size = transformer_io_params.batch_size
        embedding_dim = transformer_io_params.embedding_dim
        num_weights = transformer_io_params.num_weights
        num_labels = transformer_io_params.num_labels
        flows_shape = transformer_io_params.flows_shape

        io = SimpleTransformerIO(
            num_labels=num_labels,
            num_weights=num_weights,
            embedding_dim=embedding_dim,
            weight_block_size=flows_shape**2,
        )
        encoded: torch.Tensor = self._encode_samples(
            batch_size, flows_shape, io
        )

        assert encoded.shape == (
            batch_size,
            embedding_dim + (flows_shape**2),
        )
        self._check_weights(
            encoded,
            num_weights=num_weights,
            flows_shape=flows_shape,
            io=io,
        )

    def test_joint_transformer_io(
        self,
        transformer_io_params: IOParams,
    ) -> None:
        """Tests JointTransformerIO Transformer adapter."""
        batch_size = transformer_io_params.batch_size
        embedding_dim = transformer_io_params.embedding_dim
        num_weights = transformer_io_params.num_weights
        num_labels = transformer_io_params.num_labels
        flows_shape = transformer_io_params.flows_shape

        io = JointTransformerIO(
            num_labels=num_labels,
            num_weights=num_weights,
            embedding_dim=embedding_dim,
            weight_block_size=flows_shape**2,
        )
        encoded: torch.Tensor = self._encode_samples(
            batch_size, flows_shape, io
        )

        expected_shape: tuple[int, int] = (
            batch_size + num_weights,
            embedding_dim + (flows_shape**2),
        )
        assert encoded.shape == expected_shape
        self._check_weights(
            encoded,
            num_weights=num_weights,
            flows_shape=flows_shape,
            io=io,
        )

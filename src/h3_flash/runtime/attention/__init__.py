"""Dense and explicitly approximate attention backends."""

from .dense_fa4 import DenseFA4MiniMaxH3Processor, install_dense_fa4

__all__ = ["DenseFA4MiniMaxH3Processor", "install_dense_fa4"]

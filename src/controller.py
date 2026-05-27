"""Neural controller — single-layer forward network for evolved agents."""

from __future__ import annotations

import numpy as np


class NeuralController:
    """Single-layer neural network for stigmergic sorting agents.

    Architecture: 6 inputs → 12 hidden (tanh) → 5 outputs (softmax)
    Total parameters: 6×12 + 12 + 12×5 + 5 = 149
    """

    INPUTS = 6
    HIDDEN = 12
    OUTPUTS = 5
    TOTAL_PARAMS = 149  # 6*12 + 12 + 12*5 + 5

    def __init__(self, weights: np.ndarray) -> None:
        """Unpack flat weight vector into matrices.

        Args:
            weights: 1-D array of length 149.

        Raises:
            ValueError: If weights length is not 149.
        """
        if weights.shape != (self.TOTAL_PARAMS,):
            raise ValueError(
                f"Expected {self.TOTAL_PARAMS} weights, got {weights.shape}"
            )

        self._weights = weights.copy()
        self._unpack()

    def _unpack(self) -> None:
        """Split flat weights into W1, b1, W2, b2."""
        i = 0
        s = self.INPUTS * self.HIDDEN  # 72
        self.W1 = self._weights[i : i + s].reshape(self.INPUTS, self.HIDDEN)
        i += s
        self.b1 = self._weights[i : i + self.HIDDEN]  # 12
        i += self.HIDDEN
        s = self.HIDDEN * self.OUTPUTS  # 60
        self.W2 = self._weights[i : i + s].reshape(self.HIDDEN, self.OUTPUTS)
        i += s
        self.b2 = self._weights[i : i + self.OUTPUTS]  # 5

    def forward(self, input_vec: np.ndarray) -> np.ndarray:
        """Forward pass: tanh hidden, softmax output.

        Args:
            input_vec: 1-D array of length 6.

        Returns:
            Probability vector of length 5.
        """
        h = self._tanh(input_vec @ self.W1 + self.b1)
        out = self._softmax(h @ self.W2 + self.b2)
        return out

    def to_list(self) -> list[float]:
        """Serialize back to flat list of 149 floats."""
        return self._weights.tolist()

    @classmethod
    def from_list(cls, weights_list: list[float]) -> "NeuralController":
        """Deserialize from list."""
        return cls(np.array(weights_list, dtype=np.float64))

    @classmethod
    def random_init(
        cls,
        rng: np.random.RandomState,
        pickup_bias: float = 0.5,
        move_bias: float = 0.0,
    ) -> "NeuralController":
        """Xavier init for W1, W2. Zero init for b1.

        b2 biases: PICKUP index gets +pickup_bias, MOVE index gets +move_bias.

        Xavier scale = sqrt(2 / (fan_in + fan_out))
        """
        w1_scale = np.sqrt(2.0 / (cls.INPUTS + cls.HIDDEN))
        w2_scale = np.sqrt(2.0 / (cls.HIDDEN + cls.OUTPUTS))

        W1 = rng.randn(cls.INPUTS, cls.HIDDEN) * w1_scale
        b1 = np.zeros(cls.HIDDEN)
        W2 = rng.randn(cls.HIDDEN, cls.OUTPUTS) * w2_scale
        b2 = np.zeros(cls.OUTPUTS)
        b2[Action.PICKUP] = pickup_bias
        b2[Action.MOVE] = move_bias

        flat = np.concatenate([W1.ravel(), b1, W2.ravel(), b2])
        return cls(flat)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x)

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        """Element-wise tanh."""
        return np.tanh(x)


# Import Action here to avoid circular imports at module level.
# We need it for random_init's pickup_bias indexing.
from .agents import Action  # noqa: E402

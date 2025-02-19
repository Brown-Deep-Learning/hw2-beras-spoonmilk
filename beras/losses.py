import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return np.mean((y_true - y_pred) ** 2, axis=-1)
 
    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs[0] , self.inputs[1]
        y_batch = y_pred.shape[0]
        return [2 * (y_pred - y_true) / y_batch]


class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        return -np.sum(y_true * np.log(y_pred + 1e-9), axis=-1)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs[0], self.inputs[1]
        return [y_pred - y_true]
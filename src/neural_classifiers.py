"""Two-hidden-layer RMSprop neural classifiers used for NNET and BNNET.

Architecture values stated by the paper are preserved. Unreported choices
(ReLU, learning rate, batch size, validation split, epoch cap and Gaussian
variational prior) are explicit constructor parameters and fidelity metadata.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src._validation import binary_labels, finite_real, numeric_matrix, positive_integer


def _split_validation(y: np.ndarray, fraction: float, rng: np.random.Generator):
    validation = []
    for label in np.unique(y):
        locations = np.flatnonzero(y == label)
        rng.shuffle(locations)
        count = min(len(locations) - 1, max(1, int(round(len(locations) * fraction))))
        if count > 0:
            validation.extend(locations[:count])
    validation = np.asarray(sorted(validation), dtype=int)
    if not len(validation):
        return np.arange(len(y)), np.array([], dtype=int)
    training = np.setdiff1d(np.arange(len(y)), validation)
    return training, validation


class RMSpropNeuralClassifier(ClassifierMixin, BaseEstimator):
    """Paper-shaped NNET or mean-field variational BNNET."""

    def __init__(self, hidden_units: int = 20, dropout: float = 0.25,
                 warmup_epochs: int = 100, patience: int = 20,
                 max_epochs: int = 500, batch_size: int = 64,
                 learning_rate: float = 0.001, rmsprop_decay: float = 0.9,
                 validation_fraction: float = 0.1, bayesian: bool = False,
                 prior_std: float = 1.0, posterior_samples: int = 100,
                 random_state: int = 42):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rmsprop_decay = rmsprop_decay
        self.validation_fraction = validation_fraction
        self.bayesian = bayesian
        self.prior_std = prior_std
        self.posterior_samples = posterior_samples
        self.random_state = random_state

    def _parameters(self):
        return self.means_ if self.bayesian else self.parameters_

    @staticmethod
    def _forward(X, parameters, dropout, rng=None):
        weights, biases = parameters[:3], parameters[3:]
        activations, preactivations, masks = [X], [], []
        value = X
        for layer in range(2):
            z = value @ weights[layer] + biases[layer]
            value = np.maximum(z, 0.0)
            mask = None
            if dropout:
                mask = (rng.random(value.shape) >= dropout) / (1.0 - dropout)
                value = value * mask
            preactivations.append(z); activations.append(value); masks.append(mask)
        logits = (value @ weights[2] + biases[2]).reshape(-1)
        return expit(np.clip(logits, -40, 40)), (activations, preactivations, masks)

    @staticmethod
    def _gradients(y, probability, cache, parameters, dropout):
        weights = parameters[:3]
        activations, preactivations, masks = cache
        delta = (probability - y)[:, None] / len(y)
        grad_w = [None] * 3; grad_b = [None] * 3
        grad_w[2], grad_b[2] = activations[2].T @ delta, delta.sum(axis=0)
        delta = delta @ weights[2].T
        for layer in (1, 0):
            if dropout:
                delta *= masks[layer]
            delta *= preactivations[layer] > 0
            grad_w[layer], grad_b[layer] = activations[layer].T @ delta, delta.sum(axis=0)
            if layer:
                delta = delta @ weights[layer].T
        return grad_w + grad_b

    def _sample_parameters(self, rng):
        eps = [rng.normal(size=value.shape) for value in self.means_]
        std = [np.logaddexp(0.0, rho) for rho in self.rhos_]
        return [mu + sigma * noise for mu, sigma, noise in zip(self.means_, std, eps)], eps, std

    def fit(self, X: Any, y: Any):
        for key in ("parameters_", "means_", "rhos_", "classes_", "n_features_in_"):
            self.__dict__.pop(key, None)
        Xv = numeric_matrix(X); yv, classes = binary_labels(y, len(Xv))
        hidden = positive_integer(self.hidden_units, "hidden_units")
        warmup = positive_integer(self.warmup_epochs, "warmup_epochs")
        patience = positive_integer(self.patience, "patience")
        epochs = positive_integer(self.max_epochs, "max_epochs")
        batch = positive_integer(self.batch_size, "batch_size")
        samples = positive_integer(self.posterior_samples, "posterior_samples")
        lr = finite_real(self.learning_rate, "learning_rate")
        decay = finite_real(self.rmsprop_decay, "rmsprop_decay")
        dropout = finite_real(self.dropout, "dropout")
        validation_fraction = finite_real(self.validation_fraction, "validation_fraction")
        prior = finite_real(self.prior_std, "prior_std")
        if not (0 < lr and 0 < decay < 1 and 0 <= dropout < 1
                and 0 < validation_fraction < .5 and prior > 0):
            raise ValueError("Neural scalar parameters are outside their valid ranges.")
        if not isinstance(self.bayesian, bool):
            raise ValueError("bayesian must be boolean.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)) \
                or not 0 <= int(self.random_state) < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32).")
        rng = np.random.default_rng(int(self.random_state))
        train, validation = _split_validation(yv, validation_fraction, rng)
        target = (yv == classes[1]).astype(float)
        shapes = [(Xv.shape[1], hidden), (hidden, hidden), (hidden, 1),
                  (hidden,), (hidden,), (1,)]
        means = [rng.normal(0, np.sqrt(2 / max(1, shape[0])), shape) if len(shape) == 2
                 else np.zeros(shape) for shape in shapes]
        if self.bayesian:
            self.means_ = means
            self.rhos_ = [np.full(shape, -5.0) for shape in shapes]
            optimised = self.means_ + self.rhos_
        else:
            self.parameters_ = means
            optimised = self.parameters_
        averages = [np.zeros_like(parameter) for parameter in optimised]
        best, best_loss, waits, completed = None, np.inf, 0, 0
        for epoch in range(epochs):
            order = rng.permutation(train)
            for start in range(0, len(order), batch):
                indices = order[start:start + batch]
                if self.bayesian:
                    parameters, noises, stds = self._sample_parameters(rng)
                else:
                    parameters, noises, stds = self.parameters_, None, None
                probability, cache = self._forward(Xv[indices], parameters, dropout, rng)
                gradients = self._gradients(target[indices], probability, cache, parameters, dropout)
                if self.bayesian:
                    scale = 1.0 / len(train)
                    mean_gradients, rho_gradients = [], []
                    for mu, rho, sigma, noise, gradient in zip(self.means_, self.rhos_, stds, noises, gradients):
                        mean_gradients.append(gradient + scale * mu / prior**2)
                        d_sigma = gradient * noise + scale * (sigma / prior**2 - 1.0 / sigma)
                        rho_gradients.append(d_sigma * expit(rho))
                    gradients = mean_gradients + rho_gradients
                for i, (parameter, gradient) in enumerate(zip(optimised, gradients)):
                    averages[i] = decay * averages[i] + (1 - decay) * gradient**2
                    parameter -= lr * gradient / (np.sqrt(averages[i]) + 1e-8)
            monitor = validation if len(validation) else train
            if self.bayesian:
                eval_rng = np.random.default_rng(int(self.random_state) + epoch + 100000)
                predictions = np.mean([self._forward(Xv[monitor], self._sample_parameters(eval_rng)[0], 0)[0]
                                       for _ in range(5)], axis=0)
            else:
                predictions = self._forward(Xv[monitor], self.parameters_, 0)[0]
            p = np.clip(predictions, 1e-12, 1 - 1e-12)
            loss = float(np.mean(-target[monitor] * np.log(p) - (1-target[monitor]) * np.log(1-p)))
            completed = epoch + 1
            if loss < best_loss - 1e-8:
                best_loss, waits = loss, 0
                best = [value.copy() for value in optimised]
            else:
                waits += 1
            if completed >= warmup and waits >= patience:
                break
        for destination, saved in zip(optimised, best):
            destination[...] = saved
        self.classes_, self.n_features_in_ = classes, Xv.shape[1]
        self.diagnostics_ = {
            "architecture": [Xv.shape[1], hidden, hidden, 1], "activation": "ReLU/sigmoid",
            "optimizer": "RMSprop", "dropout": dropout, "warmup_epochs": warmup,
            "patience": patience, "epochs_completed": completed,
            "validation_rows": int(len(validation)), "bayesian": self.bayesian,
            "posterior_samples": samples if self.bayesian else None,
            "inference": "mean-field Gaussian variational" if self.bayesian else None,
        }
        return self

    def posterior_probabilities(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classes_")
        if not self.bayesian:
            raise ValueError("Posterior draws are available only for bayesian=True.")
        Xv = numeric_matrix(X)
        if Xv.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature width differs from fitting data.")
        rng = np.random.default_rng(int(self.random_state) + 200000)
        return np.column_stack([
            self._forward(Xv, self._sample_parameters(rng)[0], 0)[0]
            for _ in range(positive_integer(self.posterior_samples, "posterior_samples"))])

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "classes_")
        Xv = numeric_matrix(X)
        if Xv.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature width differs from fitting data.")
        if self.bayesian:
            p1 = self.posterior_probabilities(Xv).mean(axis=1)
        else:
            p1 = self._forward(Xv, self.parameters_, 0)[0]
        return np.column_stack((1.0 - p1, p1))

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

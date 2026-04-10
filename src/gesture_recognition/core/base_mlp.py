"""
Base MLP Classifier
====================
Generic MLP with Adam optimizer, dropout, validation split
Reusable base class สำหรับ hand, face, body, หรืออะไรก็ได้

Subclass ต้อง implement:
- feature extraction methods (เฉพาะ domain)
- train() wrapper ที่แปลงข้อมูลเป็น X_list, y_list แล้วเรียก train_from_data()
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional


class BaseMLP:

    def __init__(self, input_size: int, hidden_sizes: List[int] = None,
                 learning_rate: float = 0.001, dropout_rate: float = 0.3):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        self.classes = []
        self.is_trained = False

        # Feature normalization params (z-score)
        self.feature_mean = None
        self.feature_std = None

        # Training stats
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        # Adam optimizer state
        self._adam_m = []  # first moment
        self._adam_v = []  # second moment
        self._adam_t = 0   # timestep

        # Local RNG (ไม่ reset global state)
        self._rng = np.random.default_rng(42)

    # ---- Normalization ----

    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Z-score normalization (mean=0, std=1)"""
        if fit:
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0) + 1e-8
        if self.feature_mean is not None and self.feature_std is not None:
            return (X - self.feature_mean) / self.feature_std
        return X

    # ---- Activation functions ----

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)

    # ---- Weight initialization ----

    def _init_weights(self, num_classes: int):
        """He initialization สำหรับ ReLU"""
        self.weights = []
        self.biases = []

        layer_sizes = [self.input_size] + self.hidden_sizes + [num_classes]

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            std = np.sqrt(2.0 / fan_in)
            w = self._rng.standard_normal((fan_in, layer_sizes[i + 1])) * std
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

        # Reset Adam state
        self._adam_m = [np.zeros_like(w) for w in self.weights]
        self._adam_v = [np.zeros_like(w) for w in self.weights]
        self._adam_m_b = [np.zeros_like(b) for b in self.biases]
        self._adam_v_b = [np.zeros_like(b) for b in self.biases]
        self._adam_t = 0

    # ---- Forward / Backward ----

    def _forward(self, X: np.ndarray, training: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Forward pass, return (activations, pre_activations, dropout_masks)"""
        activations = [X]
        pre_activations = [X]
        dropout_masks = []

        current = X
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            current = self._relu(z)

            # Inverted dropout (only during training, only on hidden layers)
            if training and self.dropout_rate > 0:
                mask = (self._rng.random(current.shape) > self.dropout_rate).astype(float)
                current = current * mask / (1.0 - self.dropout_rate)
                dropout_masks.append(mask)
            else:
                dropout_masks.append(np.ones_like(current))

            activations.append(current)

        # Output layer (no dropout)
        z = current @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        output = self._softmax(z)
        activations.append(output)

        return activations, pre_activations, dropout_masks

    def _backward(self, X: np.ndarray, y_onehot: np.ndarray,
                  activations: List[np.ndarray], pre_activations: List[np.ndarray],
                  dropout_masks: List[np.ndarray]):
        """Backpropagation with Adam optimizer"""
        m = X.shape[0]
        self._adam_t += 1

        delta = activations[-1] - y_onehot

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for i in range(len(self.weights) - 1, -1, -1):
            gw = activations[i].T @ delta / m
            gb = np.mean(delta, axis=0)

            # Adam update for weights
            self._adam_m[i] = beta1 * self._adam_m[i] + (1 - beta1) * gw
            self._adam_v[i] = beta2 * self._adam_v[i] + (1 - beta2) * (gw ** 2)
            m_hat = self._adam_m[i] / (1 - beta1 ** self._adam_t)
            v_hat = self._adam_v[i] / (1 - beta2 ** self._adam_t)
            self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            # Adam update for biases
            self._adam_m_b[i] = beta1 * self._adam_m_b[i] + (1 - beta1) * gb
            self._adam_v_b[i] = beta2 * self._adam_v_b[i] + (1 - beta2) * (gb ** 2)
            m_hat_b = self._adam_m_b[i] / (1 - beta1 ** self._adam_t)
            v_hat_b = self._adam_v_b[i] / (1 - beta2 ** self._adam_t)
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + eps)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_derivative(pre_activations[i])
                # Apply dropout mask from forward pass
                delta = delta * dropout_masks[i - 1] / (1.0 - self.dropout_rate) if self.dropout_rate > 0 else delta

    # ---- Stratified split ----

    @staticmethod
    def _stratified_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, rng=None):
        """Stratified train/val split ให้แต่ละ class มี proportion เท่ากัน"""
        if rng is None:
            rng = np.random.default_rng(42)

        classes = np.unique(y)
        train_idx, val_idx = [], []

        for c in classes:
            c_idx = np.where(y == c)[0]
            rng.shuffle(c_idx)
            n_val = max(1, int(len(c_idx) * val_ratio))
            val_idx.extend(c_idx[:n_val])
            train_idx.extend(c_idx[n_val:])

        return np.array(train_idx), np.array(val_idx)

    # ---- Generic training ----

    def train_from_data(self, X_list: List[np.ndarray], y_list: List[int],
                        class_names: List[str], epochs: int = 300,
                        batch_size: int = 32, verbose: bool = True,
                        progress_callback=None) -> Optional[Dict]:
        """
        Train MLP จาก feature arrays และ labels

        Features:
        - Adam optimizer
        - Dropout (inverted)
        - Validation split (80/20 stratified)
        - Early stopping on val_loss (patience=20)
        - Adaptive data augmentation
        """
        self.classes = class_names

        X_orig = np.array(X_list)
        y_orig = np.array(y_list)

        if len(X_orig) < 2:
            if verbose:
                print("ข้อมูลไม่เพียงพอสำหรับ training")
            return None

        # === Stratified train/val split (on original data) ===
        train_idx, val_idx = self._stratified_split(X_orig, y_orig, val_ratio=0.2, rng=self._rng)
        X_train_orig = X_orig[train_idx]
        y_train_orig = y_orig[train_idx]
        X_val_orig = X_orig[val_idx]
        y_val_orig = y_orig[val_idx]

        # === Feature normalization — fit on train only ===
        self.feature_mean = np.mean(X_train_orig, axis=0)
        self.feature_std = np.std(X_train_orig, axis=0) + 1e-8

        # === Adaptive augmentation ===
        n_samples = len(X_train_orig)
        if n_samples <= 5:
            aug_factor = 30
            noise_std = 0.03
        elif n_samples <= 20:
            aug_factor = 15
            noise_std = 0.02
        else:
            aug_factor = 5
            noise_std = 0.01

        # Augment training data only
        X_aug = [X_train_orig]
        y_aug = [y_train_orig]
        for _ in range(aug_factor):
            noise = self._rng.standard_normal(X_train_orig.shape) * (noise_std * self.feature_std)
            X_aug.append(X_train_orig + noise)
            y_aug.append(y_train_orig)

        X_train = np.vstack(X_aug)
        y_train = np.concatenate(y_aug)

        if verbose:
            print(f"  Train: {len(X_train_orig)} → {len(X_train)} (aug) | Val: {len(X_val_orig)}")

        # Normalize
        X_train = self._normalize_features(X_train, fit=False)
        X_val = self._normalize_features(X_val_orig, fit=False)

        # Shuffle training data
        perm = self._rng.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        num_classes = len(self.classes)
        self._init_weights(num_classes)

        # One-hot encode
        y_train_oh = np.zeros((len(y_train), num_classes))
        y_train_oh[np.arange(len(y_train)), y_train] = 1

        y_val_oh = np.zeros((len(y_val_orig), num_classes))
        y_val_oh[np.arange(len(y_val_orig)), y_val_orig] = 1

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        best_weights = None
        best_biases = None

        for epoch in range(epochs):
            # Shuffle training data each epoch
            perm = self._rng.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]
            y_train_oh = y_train_oh[perm]

            # Mini-batch training
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                X_batch = X_train[start:end]
                y_batch = y_train_oh[start:end]

                activations, pre_activations, dropout_masks = self._forward(X_batch, training=True)
                self._backward(X_batch, y_batch, activations, pre_activations, dropout_masks)

            # Training stats (no dropout)
            train_act, _, _ = self._forward(X_train, training=False)
            train_preds = np.argmax(train_act[-1], axis=1)
            train_acc = np.mean(train_preds == y_train) * 100
            train_loss = -np.mean(np.sum(y_train_oh * np.log(train_act[-1] + 1e-8), axis=1))

            # Validation stats
            val_act, _, _ = self._forward(X_val, training=False)
            val_preds = np.argmax(val_act[-1], axis=1)
            val_acc = np.mean(val_preds == y_val_orig) * 100
            val_loss = -np.mean(np.sum(y_val_oh * np.log(val_act[-1] + 1e-8), axis=1))

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if progress_callback:
                progress_callback(epoch, train_loss, train_acc, val_loss, val_acc)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"Train: {train_acc:.1f}% (loss={train_loss:.4f}) | "
                      f"Val: {val_acc:.1f}% (loss={val_loss:.4f})")

            # Early stopping on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch+1}: val_loss not improving for {patience} epochs")
                break

        # Restore best weights
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        self.is_trained = True

        final_val_acc = self.val_accuracies[-1] if self.val_accuracies else 0
        if verbose:
            print(f"  Training complete! Val accuracy: {final_val_acc:.1f}%")

        return {
            'final_loss': self.train_losses[-1],
            'final_accuracy': self.train_accuracies[-1],
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'final_val_accuracy': final_val_acc,
            'losses': self.train_losses,
            'accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }

    # ---- Prediction ----

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """ทำนายจาก feature vector"""
        if not self.is_trained or len(self.classes) == 0:
            return "", 0.0

        if features.ndim == 1:
            features = features.reshape(1, -1)

        features_norm = self._normalize_features(features, fit=False)
        activations, _, _ = self._forward(features_norm, training=False)
        probs = activations[-1][0]

        best_idx = np.argmax(probs)
        confidence = probs[best_idx] * 100

        if confidence < 50.0:
            return "", 0.0

        return self.classes[best_idx], confidence

    # ---- Save / Load ----

    def save(self, filepath: str):
        """บันทึก trained model"""
        data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'classes': self.classes,
            'hidden_sizes': self.hidden_sizes,
            'input_size': self.input_size,
            'is_trained': self.is_trained,
            'dropout_rate': self.dropout_rate,
            'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.tolist() if self.feature_std is not None else None,
            'train_losses': self.train_losses[-10:] if self.train_losses else [],
            'train_accuracies': self.train_accuracies[-10:] if self.train_accuracies else [],
            'val_losses': self.val_losses[-10:] if self.val_losses else [],
            'val_accuracies': self.val_accuracies[-10:] if self.val_accuracies else [],
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, filepath: str) -> bool:
        """โหลด trained model"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.weights = [np.array(w) for w in data['weights']]
            self.biases = [np.array(b) for b in data['biases']]
            self.classes = data['classes']
            self.hidden_sizes = data.get('hidden_sizes', [128, 64])
            self.input_size = data.get('input_size', self.input_size)
            self.is_trained = data.get('is_trained', True)
            self.dropout_rate = data.get('dropout_rate', 0.3)
            self.feature_mean = np.array(data['feature_mean']) if data.get('feature_mean') is not None else None
            self.feature_std = np.array(data['feature_std']) if data.get('feature_std') is not None else None
            self.train_losses = data.get('train_losses', [])
            self.train_accuracies = data.get('train_accuracies', [])
            self.val_losses = data.get('val_losses', [])
            self.val_accuracies = data.get('val_accuracies', [])
            return True
        except Exception as e:
            print(f"Error loading MLP model: {e}")
            return False

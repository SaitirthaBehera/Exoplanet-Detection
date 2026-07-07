import logging
import os
from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config import (
    WINDOW_SIZE,
    CLF_CONV_FILTERS,
    CLF_KERNEL_SIZE,
    CLF_LSTM_UNITS,
    CLF_DROPOUT,
    CLF_DENSE_UNITS,
    CLF_LEARNING_RATE,
    CLF_BATCH_SIZE,
    CLF_EPOCHS,
    CLF_PATIENCE,
    CLF_CLASS_WEIGHT,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def build_classifier(window_size: int = WINDOW_SIZE) -> tf.keras.Model:
    f1, f2 = CLF_CONV_FILTERS
    k = CLF_KERNEL_SIZE

    inp = layers.Input(shape=(window_size, 1), name="clf_input")

    x = layers.Conv1D(f1, kernel_size=k, activation="relu", padding="same",
                      name="clf_conv1")(inp)
    x = layers.BatchNormalization(name="clf_bn1")(x)
    x = layers.MaxPooling1D(pool_size=4, name="clf_pool1")(x)

    x = layers.Conv1D(f2, kernel_size=k, activation="relu", padding="same",
                      name="clf_conv2")(x)
    x = layers.BatchNormalization(name="clf_bn2")(x)
    x = layers.MaxPooling1D(pool_size=4, name="clf_pool2")(x)

    x = layers.LSTM(CLF_LSTM_UNITS, return_sequences=False, name="clf_lstm")(x)
    x = layers.Dropout(CLF_DROPOUT, name="clf_drop1")(x)

    x = layers.Dense(CLF_DENSE_UNITS, activation="relu", name="clf_dense")(x)
    x = layers.Dropout(CLF_DROPOUT, name="clf_drop2")(x)
    out = layers.Dense(1, activation="sigmoid", name="clf_output")(x)

    model = Model(inputs=inp, outputs=out, name="transit_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CLF_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    logger.info(
        "Classifier built — window_size=%d, params=%s",
        window_size, f"{model.count_params():,}",
    )

    return model


def train_classifier(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = CLF_EPOCHS,
    batch_size: int = CLF_BATCH_SIZE,
    validation_split: float = 0.2,
    class_weight: Optional[Dict[int, float]] = None,
) -> tf.keras.callbacks.History:
    if class_weight is None:
        class_weight = CLF_CLASS_WEIGHT

    if X_train.ndim == 2:
        X_train = X_train[..., np.newaxis]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CLF_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]

    logger.info(
        "Training classifier — %d samples (%.0f%% positive), epochs=%d",
        len(y_train),
        100.0 * np.mean(y_train),
        epochs,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info(
        "Training complete — best val_loss=%.4f, best val_auc=%.4f",
        min(history.history["val_loss"]),
        max(history.history.get("val_auc", [0.0])),
    )

    return history


def predict_transit(
    model: tf.keras.Model,
    curve: np.ndarray,
) -> Union[float, np.ndarray]:
    single = curve.ndim <= 1 or (curve.ndim == 2 and curve.shape[-1] == 1
                                  and curve.shape[0] != 1)

    x = curve.copy()
    if x.ndim == 1:
        x = x[np.newaxis, :, np.newaxis]
    elif x.ndim == 2:
        if x.shape[-1] == 1 and x.shape[0] != 1:
            x = x[np.newaxis, ...]
        else:
            x = x[..., np.newaxis]

    try:
        probs = model.predict(x, verbose=0).squeeze(axis=-1)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise

    if single:
        return float(probs.squeeze())
    return probs


def evaluate_classifier(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if X_test.ndim == 2:
        X_test = X_test[..., np.newaxis]

    y_prob = model.predict(X_test, verbose=0).squeeze()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_test, y_prob)),
    }

    logger.info(
        "Evaluation — acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  auc=%.3f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc_roc"],
    )

    return metrics


def save_classifier(
    model: tf.keras.Model,
    filepath: Optional[str] = None,
) -> None:
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, "classifier.keras")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    logger.info("Classifier saved → %s", filepath)


def load_classifier(
    filepath: Optional[str] = None,
) -> tf.keras.Model:
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, "classifier.keras")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No classifier model found at {filepath}")

    model = tf.keras.models.load_model(filepath)
    logger.info("Classifier loaded ← %s", filepath)
    return model

import logging
import os
from typing import Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from src.config import (
    WINDOW_SIZE,
    AE_FILTERS,
    AE_KERNEL_SIZE,
    AE_LEARNING_RATE,
    AE_BATCH_SIZE,
    AE_EPOCHS,
    AE_PATIENCE,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


def _pad_size(window_size: int) -> int:
    remainder = window_size % 4
    if remainder == 0:
        return window_size
    return window_size + (4 - remainder)


def build_autoencoder(window_size: int = WINDOW_SIZE) -> tf.keras.Model:
    padded = _pad_size(window_size)
    pad_total = padded - window_size
    pad_right = pad_total

    f1, f2, f3 = AE_FILTERS
    k = AE_KERNEL_SIZE

    inp = layers.Input(shape=(window_size, 1), name="encoder_input")

    x = inp

    if pad_total > 0:
        x = layers.ZeroPadding1D(padding=(0, pad_right), name="input_padding")(x)

    x = layers.Conv1D(f1, kernel_size=k, activation="relu", padding="same",
                      name="enc_conv1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="enc_pool1")(x)

    x = layers.Conv1D(f2, kernel_size=k, activation="relu", padding="same",
                      name="enc_conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="enc_pool2")(x)

    x = layers.Conv1D(f3, kernel_size=k, activation="relu", padding="same",
                      name="bottleneck")(x)

    x = layers.Conv1DTranspose(f3, kernel_size=k, strides=1, activation="relu",
                               padding="same", name="dec_convt1")(x)
    x = layers.UpSampling1D(size=2, name="dec_up1")(x)

    x = layers.Conv1DTranspose(f2, kernel_size=k, strides=1, activation="relu",
                               padding="same", name="dec_convt2")(x)
    x = layers.UpSampling1D(size=2, name="dec_up2")(x)

    x = layers.Conv1D(1, kernel_size=k, activation="linear", padding="same",
                      name="dec_output_conv")(x)

    if pad_total > 0:
        x = layers.Cropping1D(cropping=(0, pad_right), name="output_crop")(x)

    model = Model(inputs=inp, outputs=x, name="denoising_autoencoder")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE),
        loss="mse",
    )

    logger.info(
        "Autoencoder built — window_size=%d, padded=%d, params=%s",
        window_size, padded, f"{model.count_params():,}",
    )

    return model


def train_autoencoder(
    model: tf.keras.Model,
    noisy_data: np.ndarray,
    clean_data: np.ndarray,
    epochs: int = AE_EPOCHS,
    batch_size: int = AE_BATCH_SIZE,
    validation_split: float = 0.2,
) -> tf.keras.callbacks.History:
    if noisy_data.ndim == 2:
        noisy_data = noisy_data[..., np.newaxis]
    if clean_data.ndim == 2:
        clean_data = clean_data[..., np.newaxis]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=AE_PATIENCE,
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
        "Training autoencoder — %d samples, epochs=%d, batch_size=%d",
        len(noisy_data), epochs, batch_size,
    )

    history = model.fit(
        noisy_data,
        clean_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info(
        "Training complete — best val_loss=%.6f",
        min(history.history["val_loss"]),
    )

    return history


def denoise(model: tf.keras.Model, noisy_curves: np.ndarray) -> np.ndarray:
    original_shape = noisy_curves.shape
    single = noisy_curves.ndim == 1

    x = noisy_curves.copy()
    if single:
        x = x[np.newaxis, :, np.newaxis]
    elif x.ndim == 2:
        x = x[..., np.newaxis]

    try:
        denoised = model.predict(x, verbose=0)
    except Exception as exc:
        logger.error("Denoising failed: %s", exc)
        raise

    if single:
        denoised = denoised.squeeze()
    elif len(original_shape) == 2:
        denoised = denoised.squeeze(axis=-1)

    return denoised


def save_autoencoder(
    model: tf.keras.Model,
    filepath: Optional[str] = None,
) -> None:
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, "autoencoder.keras")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    logger.info("Autoencoder saved → %s", filepath)


def load_autoencoder(
    filepath: Optional[str] = None,
) -> tf.keras.Model:
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, "autoencoder.keras")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No autoencoder model found at {filepath}")

    model = tf.keras.models.load_model(filepath)
    logger.info("Autoencoder loaded ← %s", filepath)
    return model

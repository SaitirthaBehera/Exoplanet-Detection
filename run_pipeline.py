#!/usr/bin/env python3
"""
🔭 Cosmic Orbit — Full Pipeline Runner
========================================
Run this script to execute the complete exoplanet detection pipeline:
  1. Generate synthetic training data
  2. Train the denoising autoencoder
  3. Train the CNN-LSTM classifier
  4. Run detection on a target
  5. Extract features and validate

Usage (local):
    python run_pipeline.py

Usage (Google Colab):
    !git clone https://github.com/YOUR_USERNAME/bah-2026-exoplanet-detection.git
    %cd bah-2026-exoplanet-detection
    !pip install -r requirements.txt
    !python run_pipeline.py
"""

import os
import sys
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pipeline")

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    print("=" * 70)
    print("  🔭 COSMIC ORBIT — Exoplanet Transit Detection Pipeline")
    print("  BAH 2026 — ISRO × Hack2skill")
    print("=" * 70)
    print()

    # ================================================================
    # STEP 1: Generate Synthetic Training Data
    # ================================================================
    print("━" * 50)
    print("📊 STEP 1: Generating Synthetic Training Data")
    print("━" * 50)

    from src.preprocessing import generate_synthetic_dataset, prepare_classification_dataset
    from src.config import SYNTHETIC_N_SAMPLES, WINDOW_SIZE, NOISE_SIGMA

    noisy, clean, labels = generate_synthetic_dataset(
        n_samples=SYNTHETIC_N_SAMPLES,
        window_size=WINDOW_SIZE,
        noise_sigma=NOISE_SIGMA
    )
    print(f"  ✅ Generated {SYNTHETIC_N_SAMPLES} synthetic light curves")
    print(f"     Noisy shape:  {noisy.shape}")
    print(f"     Clean shape:  {clean.shape}")
    print(f"     Labels shape: {labels.shape}")
    print(f"     Transit ratio: {labels.mean():.1%}")
    print()

    # ================================================================
    # STEP 2: Train Denoising Autoencoder
    # ================================================================
    print("━" * 50)
    print("🧹 STEP 2: Training Denoising Autoencoder")
    print("━" * 50)

    from src.autoencoder import build_autoencoder, train_autoencoder, save_autoencoder, denoise

    ae_model = build_autoencoder(window_size=WINDOW_SIZE)
    ae_model.summary()
    print()

    ae_history = train_autoencoder(
        ae_model, noisy, clean,
        epochs=50, batch_size=64, validation_split=0.2
    )

    # Save model
    save_autoencoder(ae_model)
    final_loss = ae_history.history['val_loss'][-1]
    print(f"  ✅ Autoencoder trained — final val_loss: {final_loss:.6f}")
    print()

    # ================================================================
    # STEP 3: Train CNN-LSTM Classifier
    # ================================================================
    print("━" * 50)
    print("🧠 STEP 3: Training CNN-LSTM Classifier")
    print("━" * 50)

    from src.classifier import (
        build_classifier, train_classifier, save_classifier,
        evaluate_classifier, predict_transit
    )

    # Prepare classification data (use denoised curves as input)
    denoised = denoise(ae_model, noisy)
    X_train, X_test, y_train, y_test = prepare_classification_dataset(
        denoised, clean, labels, test_size=0.2
    )
    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    clf_model = build_classifier(window_size=WINDOW_SIZE)
    clf_model.summary()
    print()

    clf_history = train_classifier(
        clf_model, X_train, y_train,
        epochs=50, batch_size=32, validation_split=0.2
    )

    # Evaluate
    metrics = evaluate_classifier(clf_model, X_test, y_test)
    save_classifier(clf_model)

    print(f"\n  ✅ Classifier trained — Test Metrics:")
    print(f"     Accuracy:  {metrics['accuracy']:.4f}")
    print(f"     Precision: {metrics['precision']:.4f}")
    print(f"     Recall:    {metrics['recall']:.4f}")
    print(f"     F1 Score:  {metrics['f1']:.4f}")
    print(f"     AUC-ROC:   {metrics.get('auc_roc', 'N/A')}")
    print()

    # ================================================================
    # STEP 4: Run Detection on Demo Data
    # ================================================================
    print("━" * 50)
    print("🎯 STEP 4: Running Detection on Sample Data")
    print("━" * 50)

    # Try loading real data first, fall back to synthetic
    try:
        from src.data_pipeline import load_lightcurve
        time, flux = load_lightcurve("HAT-P-7b")
        print("  Using real Kepler data for HAT-P-7b")
    except FileNotFoundError:
        print("  ⚠️ Real data not found. Using synthetic demo curve.")
        time = np.linspace(0, 50, WINDOW_SIZE)
        # Create a demo curve with known transit
        flux = np.zeros(WINDOW_SIZE)
        phase = (time % 10.0) / 10.0
        in_transit = np.abs(phase - 0.5) < 0.015
        transit_phase = (phase[in_transit] - 0.5) / 0.015
        flux[in_transit] -= 0.012 * 0.5 * (1 + np.cos(np.pi * transit_phase))
        flux += np.random.normal(0, 0.002, WINDOW_SIZE)

    # Denoise
    flux_input = flux.reshape(1, -1, 1)
    denoised_demo = denoise(ae_model, flux_input).squeeze()

    # Classify
    prob = predict_transit(clf_model, denoised_demo)
    print(f"  Transit probability: {prob:.4f}")
    print(f"  Classification: {'🟢 TRANSIT DETECTED' if prob > 0.5 else '🔴 No transit'}")
    print()

    # Feature extraction
    try:
        from src.feature_extraction import extract_all_features
        features = extract_all_features(time, denoised_demo)
        if features:
            print(f"  Extracted features:")
            print(f"    Period:   {features.get('best_period', 'N/A')} days")
            print(f"    Depth:    {features.get('depth', 'N/A')}")
            print(f"    Duration: {features.get('duration_hours', 'N/A')} hours")
            print(f"    SNR:      {features.get('snr', 'N/A')}")
    except Exception as e:
        print(f"  ⚠️ Feature extraction skipped: {e}")

    print()

    # ================================================================
    # DONE
    # ================================================================
    print("=" * 70)
    print("  ✅ PIPELINE COMPLETE")
    print("  Models saved to: models/")
    print("  Launch dashboard: streamlit run app/streamlit_app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

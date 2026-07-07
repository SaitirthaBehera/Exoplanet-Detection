import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)

DATA_COLUMNS = ["time", "flux"]

TARGETS = {
    "Kepler-10b": {
        "search_name": "Kepler-10",
        "kic_id": 11904151,
        "mission": "Kepler",
        "known_period_days": 0.837495,
        "known_depth_ppm": 152,
        "known_duration_hours": 1.81,
        "description": "First confirmed rocky exoplanet — strong, short-period signal"
    },
    "Kepler-452b": {
        "search_name": "Kepler-452",
        "kic_id": 8311864,
        "mission": "Kepler",
        "known_period_days": 384.843,
        "known_depth_ppm": 84,
        "known_duration_hours": 9.67,
        "description": "Earth-like planet in habitable zone — long period, subtle dip"
    },
    "Kepler-90g": {
        "search_name": "Kepler-90",
        "kic_id": 11442793,
        "mission": "Kepler",
        "known_period_days": 210.60697,
        "known_depth_ppm": 335,
        "known_duration_hours": 7.46,
        "description": "8-planet system — multi-transit complexity"
    },
    "HAT-P-7b": {
        "search_name": "HAT-P-7",
        "kic_id": 10666592,
        "mission": "Kepler",
        "known_period_days": 2.204737,
        "known_depth_ppm": 6180,
        "known_duration_hours": 3.63,
        "description": "Hot Jupiter — deep, easy-to-detect transit (sanity check)"
    },
    "Kepler-22b": {
        "search_name": "Kepler-22",
        "kic_id": 10593626,
        "mission": "Kepler",
        "known_period_days": 289.8623,
        "known_depth_ppm": 492,
        "known_duration_hours": 7.42,
        "description": "First planet in habitable zone — moderate difficulty"
    },
}

WINDOW_SIZE = 2001
FLATTEN_WINDOW = 401
SIGMA_CLIP = 5.0
NOISE_SIGMA = 0.001

SYNTHETIC_N_SAMPLES = 5000
TRANSIT_DEPTH_RANGE = (0.0005, 0.02)
TRANSIT_DURATION_RANGE = (0.01, 0.05)
TRANSIT_PERIOD_RANGE = (0.5, 50.0)

AE_FILTERS = [64, 32, 16]
AE_KERNEL_SIZE = 7
AE_LATENT_DIM = 16
AE_LEARNING_RATE = 1e-3
AE_BATCH_SIZE = 64
AE_EPOCHS = 50
AE_PATIENCE = 10

CLF_CONV_FILTERS = [64, 128]
CLF_KERNEL_SIZE = 5
CLF_LSTM_UNITS = 64
CLF_DROPOUT = 0.3
CLF_DENSE_UNITS = 64
CLF_LEARNING_RATE = 1e-4
CLF_BATCH_SIZE = 32
CLF_EPOCHS = 50
CLF_PATIENCE = 10
CLF_CLASS_WEIGHT = {0: 1.0, 1: 5.0}

BLS_PERIOD_MIN = 0.5
BLS_PERIOD_MAX = 400.0
BLS_PERIOD_NPOINTS = 50000
BLS_DURATIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.33]

PERIOD_TOLERANCE = 0.01
DEPTH_TOLERANCE = 0.30
DURATION_TOLERANCE = 0.30

DASHBOARD_TITLE = "🔭 Cosmic Orbit — Exoplanet Transit Detector"
DASHBOARD_SUBTITLE = "AI-enabled Detection from Noisy Astronomical Light Curves"
CONFIDENCE_THRESHOLD = 0.5

COLORS = {
    "primary": "#FF6B00",
    "secondary": "#0A1628",
    "accent": "#00D4FF",
    "success": "#00E676",
    "warning": "#FFD600",
    "danger": "#FF1744",
    "bg_dark": "#0D1117",
    "bg_card": "#161B22",
    "text_primary": "#E6EDF3",
    "text_secondary": "#8B949E",
    "transit_highlight": "#FF6B00",
    "raw_curve": "#8B949E",
    "denoised_curve": "#00D4FF",
    "model_fit": "#FF6B00",
}

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

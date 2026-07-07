# Architecture Documentation

## Cosmic Orbit — AI-Enabled Exoplanet Transit Detection Pipeline
### BAH 2026 — ISRO × Hack2skill | Problem Statement #07

---

## Pipeline Overview

```mermaid
flowchart TD
    subgraph INPUT["📡 Data Input"]
        A1["NASA Exoplanet Archive"]
        A2["Kepler Mission (MAST)"]
        A3["TESS Mission (MAST)"]
    end

    subgraph PIPELINE["🔧 Data Pipeline Layer — Person B"]
        B1["lightkurve.search_lightcurve()"]
        B2["Download & Stitch Quarters"]
        B3["Remove NaN + Sigma-Clip Outliers"]
        B4["Flatten (Savitzky-Golay Filter)"]
        B5["Export .npy files"]
    end

    subgraph MODELS["🧠 Modeling Layer — Person A"]
        C1["Normalize Flux"]
        C2["Create Windows (2001 pts)"]

        subgraph STAGE1["Stage 1: Denoising"]
            D1["Conv1D Autoencoder"]
            D2["Encoder: 64→32→16 filters"]
            D3["Decoder: 16→32→64 filters"]
            D4["Output: Cleaned signal"]
        end

        subgraph STAGE2["Stage 2: Classification"]
            E1["Conv1D Feature Extraction"]
            E2["LSTM Temporal Analysis"]
            E3["Dense Classifier"]
            E4["Output: P(transit)"]
        end

        subgraph BLS["Feature Extraction"]
            F1["Box Least Squares Search"]
            F2["Period, Depth, Duration"]
            F3["SNR Computation"]
        end
    end

    subgraph VALIDATE["✅ Validation Layer — Person C"]
        G1["NASA TAP API Query"]
        G2["Known Transit Parameters"]
        G3["Parameter Comparison"]
        G4["Metrics: Accuracy, F1, AUC"]
    end

    subgraph PRESENT["📊 Presentation Layer — Person D"]
        H1["Streamlit Dashboard"]
        H2["Raw → Denoised → Detection"]
        H3["Validation Comparison Table"]
    end

    A1 & A2 & A3 --> B1
    B1 --> B2 --> B3 --> B4 --> B5
    B5 --> C1 --> C2
    C2 --> D1 --> D4
    D4 --> E1 --> E2 --> E3 --> E4
    D4 --> F1 --> F2 --> F3
    E4 & F3 --> G1
    G1 --> G2 --> G3 --> G4
    G4 --> H1 --> H2 & H3
```

---

## Data Format Specifications

### Light Curve Storage (`.npy`)

All light curves are stored as NumPy arrays with shape `(N, 2)`:

| Column | Index | Type | Description |
|--------|-------|------|-------------|
| Time | 0 | float64 | Barycentric Julian Date (BJD) |
| Flux | 1 | float64 | Normalized relative flux |

**Normalization**: `flux = (raw_flux / median(raw_flux)) - 1`
- Baseline ≈ 0.0
- Transit dips → negative values (e.g., -0.001 = 1000 ppm depth)

### Model Input Format

| Shape | Description |
|-------|-------------|
| `(batch, 2001, 1)` | Windowed light curve segments |
| `(batch, 1)` | Classification labels (0 = no transit, 1 = transit) |

---

## Model Architecture Details

### Stage 1: Denoising Autoencoder

```mermaid
flowchart LR
    subgraph ENCODER["Encoder"]
        I["Input (2001, 1)"] --> P["ZeroPad → (2004, 1)"]
        P --> C1["Conv1D(64, k=7, relu)"]
        C1 --> M1["MaxPool1D(2) → (1002, 64)"]
        M1 --> C2["Conv1D(32, k=7, relu)"]
        C2 --> M2["MaxPool1D(2) → (501, 32)"]
        M2 --> C3["Conv1D(16, k=7, relu) → (501, 16)"]
    end

    subgraph DECODER["Decoder"]
        C3 --> CT1["Conv1DT(16, k=7, relu)"]
        CT1 --> U1["UpSample(2) → (1002, 16)"]
        U1 --> CT2["Conv1DT(32, k=7, relu)"]
        CT2 --> U2["UpSample(2) → (2004, 32)"]
        U2 --> CO["Conv1D(1, k=7, linear)"]
        CO --> CR["Crop1D(0, 3) → (2001, 1)"]
    end

    style ENCODER fill:#0A1628,stroke:#00D4FF
    style DECODER fill:#0A1628,stroke:#FF6B00
```

**Key Design Decisions**:
- **Padding strategy**: Input padded from 2001 → 2004 (nearest multiple of 4) to ensure dimension compatibility through encode/decode
- **Linear activation** on output: light curve values can be negative (transit dips)
- **Kernel size 7**: captures ~3.5 hours of Kepler long-cadence data per kernel

### Stage 2: CNN-LSTM Classifier

```mermaid
flowchart LR
    I["Input (2001, 1)"] --> C1["Conv1D(64, k=5)"]
    C1 --> BN1["BatchNorm"]
    BN1 --> M1["MaxPool(4) → (500, 64)"]
    M1 --> C2["Conv1D(128, k=5)"]
    C2 --> BN2["BatchNorm"]
    BN2 --> M2["MaxPool(4) → (125, 128)"]
    M2 --> L["LSTM(64)"]
    L --> D1["Dropout(0.3)"]
    D1 --> F1["Dense(64, relu)"]
    F1 --> D2["Dropout(0.3)"]
    D2 --> OUT["Dense(1, sigmoid)"]

    style I fill:#0A1628,stroke:#00D4FF
    style OUT fill:#00E676,color:#000
```

**Key Design Decisions**:
- **Conv1D → LSTM hybrid**: CNNs extract local transit shapes; LSTM captures periodicity
- **BatchNormalization**: stabilizes training on variable stellar data
- **Class weights {0: 1.0, 1: 5.0}**: compensates for transit rarity (~1% of data)
- **Aggressive pooling (4×)**: reduces LSTM input to manageable sequence length

---

## Feature Extraction (BLS)

The Box Least Squares algorithm searches for periodic box-shaped dips:

1. **Period Grid**: 50,000 points from 0.5 to 400 days (log-uniform)
2. **Duration Grid**: [0.05, 0.10, 0.15, 0.20, 0.25, 0.33] days
3. **Best Period**: argmax of BLS power spectrum
4. **Transit Parameters**: extracted from phase-folded curve at best period

---

## API Endpoints

### NASA Exoplanet Archive (Validation)

```
GET https://exoplanetarchive.ipac.caltech.edu/TAP/sync
    ?query=SELECT pl_name,pl_orbper,pl_trandep,pl_trandur FROM ps
           WHERE pl_name='Kepler-10 b' AND default_flag=1
    &format=csv
```

### MAST (Data Download via lightkurve)

```python
lightkurve.search_lightcurve(
    target="Kepler-10",
    mission="Kepler",
    author="Kepler"
)
```

---

## Validation Methodology

1. **Select** 5 known exoplanets spanning easy → hard difficulty
2. **Download** real Kepler light curves via MAST
3. **Run** full pipeline: denoise → extract features → classify
4. **Compare** detected parameters against NASA catalog values
5. **Pass/Fail** criteria:
   - Period: within 1% of known value
   - Depth: within 30% of known value
   - Duration: within 30% of known value

---

## Synthetic Data Generation

Training data is bootstrapped from synthetic light curves:

1. **Baseline**: flat signal (flux = 0 after normalization)
2. **Transit injection**: cosine-tapered box dips with randomized:
   - Depth: 50 – 20,000 ppm
   - Duration: 1% – 5% of window width
   - Center: random position
3. **Noise injection**: Gaussian (σ = 0.001) + slow sinusoidal trends
4. **Split**: 50% transit, 50% no-transit
5. **Augmentation**: random noise level variation (0.0005 – 0.003 σ)

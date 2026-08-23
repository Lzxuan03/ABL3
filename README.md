# ABL3: Source-Anchored Causal Online EEG Calibration

This repository contains the key implementation of ABL3 and its integration with a CMCRD-style EEG--eye emotion recognizer.

To keep the public release compact, we provide one representative setting only:

- **Protocol:** Cross-Subject
- **Dataset:** (SSED、SEED_IV、)SEED-V
- **Random seed:** 2023,2024,2025,2026
- **EEG input:** 62 channels × 5 differential-entropy bands = 310 dimensions
- **Eye input:** 33 dimensions
- **Recognizer:** DGCNN-based CMCRD

This is a **key-code release**. Dataset files, checkpoints, generated predictions, internal audit files, discontinued variants, and multi-dataset experiment-management code are not included.

## ABL3

ABL3 performs causal EEG calibration before a frozen recognizer. It uses a source-training statistical prior and updates the target state only after the current prediction has been produced.

```text
source prior + past target state
            |
            v
     calibrate EEG x_t
            |
            v
      frozen recognizer
            |
            v
       prediction y_t
            |
            v
  update state with EEG x_t
```

The current target sample does not enter the persistent state before its own prediction.

ABL3 uses:

- source-training statistics only for initialization;
- causal online normalization;
- frequency-resolved variance drift over five DE bands;
- adaptive retention;
- a shared retention factor for the main calibration state;
- strict Predict-then-Update inference.

ABL3 does **not** use target labels, pseudo-labels, gradients, optimizer updates, or future target samples during target inference.

## Key files

The main files in this compact release are:

```text
README.md
requirements.txt
configs/frozen_abl3_multiseed.json

source/DGCNN.py
source/load_data.py
source/data/label_provider.py

source/cmcrd_abl3_plugin_mainprotocol_seed2023_v2/
└── scripts/mainprotocol_pipeline_v1.py

source/cmcrd_strict_same_protocol_seed2023_v1/
└── losses/canonical_unis_crd_v1.py

source/replacement_a8_abl3_shared_retention_seed2023_v1/
└── adapters/abl3_transform_v1.py

source/bandwise_drift_final_validation_v1/
└── calibration/
    ├── bandwise_drift_control_v1.py
    └── cumulative_welford_v1.py

source/causal_reliability_calibration_v1/
├── online_statistics_v1.py
└── seedv_data_v1.py

source/replacement_a8_strict_cross_multiseed_confirmation_v1/
├── common_v1.py
└── adapters/protocol_data_v1.py

source/replacement_a8_strict_cross_fulltarget_confirmation_v1/
├── common_v1.py
└── adapters/full_coverage_sampler_v1.py

source/strict_cross_dgcnn_sourceval_seed2022_v1/
└── data/build_sourceval_cross_v1.py
```

### Role of the main files

- `abl3_transform_v1.py`: final ABL3 transform and Predict-then-Update implementation.
- `bandwise_drift_control_v1.py`: frequency-resolved drift estimation and retention candidates.
- `cumulative_welford_v1.py`: retention-weighted online mean/variance state.
- `online_statistics_v1.py`: causal normalization utilities.
- `DGCNN.py`: DGCNN recognizer used by the representative experiment.
- `canonical_unis_crd_v1.py`: CMCRD distillation loss used for EEG student training.
- `mainprotocol_pipeline_v1.py`: representative Cross-Subject integration of the CMCRD recognizer and ABL3.
- `protocol_data_v1.py`: source-train/source-validation/target split logic.
- `full_coverage_sampler_v1.py`: deterministic source-training sampler.
- `seedv_data_v1.py`, `load_data.py`, and `label_provider.py`: SEED-V feature and label loading.

## Frozen ABL3 configuration

The public configuration is stored in:

```text
configs/frozen_abl3_multiseed.json
```

The main ABL3 parameters are:

```text
warmup = 64
recent_window = 64
lambda_min = 0.98
drift_theta = 1.0
drift_temperature = 0.25
eta = 0.5
retention_policy = SHARED_MEAN_OF_CANONICAL_BAND_RETENTIONS
```

These parameters are not adapted using target labels or target performance.

## Representative Cross-Subject protocol

The released example uses SEED-V and seed 2023.

For each of the 16 target subjects:

1. the target subject is held out for formal evaluation;
2. one source subject is selected as the source-validation subject by a deterministic source-only rule;
3. the remaining source subjects form the source-training set;
4. normalization statistics are fitted on source-training EEG only;
5. checkpoint selection uses source-validation classification loss only;
6. the formal target subject is not used for training, normalization fitting, or checkpoint selection.

## Paired comparison

The baseline and ABL3 branches must use the same frozen recognizer checkpoint and the same target order.

```text
Baseline:
raw EEG + Eye -> frozen recognizer -> prediction

ABL3:
raw EEG -> ABL3 + same Eye -> same frozen recognizer -> prediction
```

Only the EEG input is changed by ABL3. The eye-movement input path remains unchanged.

## SEED-V data

The SEED-V dataset is not redistributed in this repository. Please obtain it from the official source and follow its license.

The code expects the official SEED-V feature files, for example:

```text
SEED-V/
├── EEG_DE_features/
│   ├── 1_123.npz
│   ├── 2_123.npz
│   └── ...
├── Eye_movement_features/
│   ├── 1_123.npz
│   ├── 2_123.npz
│   └── ...
└── emotion_label_and_stimuli_order.xlsx
```

Use a configurable dataset root rather than a machine-specific absolute path. The recommended environment variable is:

```bash
export SEED_V_ROOT=/path/to/SEED-V
```

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

The implementation uses PyTorch, NumPy, SciPy, pandas, and scikit-learn.

## Metrics

We report:

- Accuracy

Metrics are computed per held-out target subject and then averaged across the 16 SEED-V subjects for the Cross-Subject task.

## What is not included

To keep the repository focused on the proposed method, this compact release does not include:

- SEED or SEED-IV experiment code;
- Forward Cross-Session experiment automation;
- other recognizer portability experiments;
- hyperparameter sweeps;
- exploratory or discontinued variants;
- internal audit and forensic scripts;
- pretrained checkpoints;
- raw or processed dataset files;
- generated predictions, logs, or result tables.

## Notes for public release

Before publishing, remove machine-specific absolute paths from the source files. In particular, dataset paths should be configured through `SEED_V_ROOT` or an equivalent command-line argument.

Do not upload internal source manifests containing local filesystem paths, usernames, experiment directories, or development-only metadata.

## Citation

If you use this code, please cite the corresponding paper. Citation information will be added after publication.


# Raw Dataset

This directory stores the original datasets used by the Hybrid Quantum Feature Selection Framework (HQFSF).

## Purpose

The files in this folder are the original source datasets and should remain unchanged throughout the project. All preprocessing operations are performed on copies of these datasets.

Do not modify, rename, or overwrite the original dataset files.

---

## Files

### Breast_Cancer_Wisconsin.data

Original Breast Cancer Wisconsin (Diagnostic) dataset obtained from the UCI Machine Learning Repository.

- Samples: 569
- Features: 30
- Target: Diagnosis
- Classes:
  - M – Malignant
  - B – Benign

---

### Breast_Cancer_Wisconsin.names

Dataset description provided by the UCI Machine Learning Repository.

Contains:

- Dataset information
- Feature descriptions
- Attribute definitions
- Citation information

---

## Source

- UCI Machine Learning Repository
- Dataset: Breast Cancer Wisconsin (Diagnostic)

---

## Workflow

```text
Raw Dataset
      │
      ▼
Dataset Validation
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Scaling
      │
      ▼
Train/Test Split
      │
      ▼
Processed Dataset
```

---

## Directory Structure

```text
datasets/
└── raw/
    ├── Breast_Cancer_Wisconsin.data
    ├── Breast_Cancer_Wisconsin.names
    └── README.md
```

---

## Notes

- Original datasets should remain read-only.
- Do not edit the contents of the raw dataset files.
- Any cleaning, encoding, scaling, or transformation should be performed using the preprocessing pipeline.
- Processed datasets are saved in the `datasets/processed/` directory.
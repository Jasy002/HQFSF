# External Datasets

This directory is reserved for externally downloaded datasets, archives, and supplementary resources that may be used by the HQFSF project.

Unlike the `raw/` directory, which stores the primary dataset, this folder is intended for optional datasets used for benchmarking, comparison, experimentation, or future research.

---

## Typical Contents

Examples of files that may be stored in this directory include:

- ZIP or TAR archives
- Kaggle datasets
- UCI Machine Learning Repository downloads
- Benchmark datasets
- Additional datasets for comparison studies
- Backup copies of datasets
- Compressed dataset packages

---

## Current Status

No external datasets are currently stored in this directory.

The HQFSF project currently uses the **Breast Cancer Wisconsin Diagnostic Dataset**, which is located in:

```text
datasets/raw/
```

---

## Notes

- Files in this directory are optional.
- Large datasets should be compressed whenever possible.
- Original external datasets should not be modified.
- Preprocessed datasets should be stored in `datasets/processed/`.
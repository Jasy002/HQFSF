# Datasets

This directory contains all datasets used by the HQFSF project.

## Folder Structure

- `raw/` – Original datasets (read-only).
- `processed/` – Cleaned, scaled, and split datasets.
- `metadata/` – Dataset information and feature descriptions.
- `external/` – Optional external archives or downloaded resources.

## Workflow

1. Store the original dataset in `raw/`.
2. Validate the dataset.
3. Preprocess and split the data.
4. Save processed files in `processed/`.
5. Keep metadata synchronized with the dataset.
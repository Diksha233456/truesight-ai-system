# Evaluation Protocol: Speaker-Independent Voice Analysis

## Purpose

This document outlines the standard evaluation protocol required to accurately measure acoustic deception detection and voice stress classification performance without data leakage or speaker bias.

---

## Protocol Specifications

### 1. Speaker Grouping
- All dataset metadata MUST contain a `speaker_id` column.
- Audio clips derived from the same speaker (regardless of session or segment) MUST share the exact same `speaker_id`.

### 2. GroupKFold Cross-Validation
- `n_splits = 5`
- Split strategy: `sklearn.model_selection.GroupKFold(n_splits=5)`
- Group parameter: `groups = df['speaker_id']`
- Rule: No speaker in fold $k$ training set may appear in fold $k$ test set.

### 3. Metric Reporting
For each cross-validation fold, the following metrics must be recorded and averaged:
- **Speaker-Independent Accuracy**
- **Balanced Accuracy**
- **Macro F1-Score**
- **ROC-AUC Score**
- **Confusion Matrix** (Aggregated across folds)

### 4. Comparison Metric
Models MUST report both **Random Split Accuracy** and **Speaker-Independent GroupKFold Accuracy** side-by-side to highlight speaker identity memorization vs true generalization.

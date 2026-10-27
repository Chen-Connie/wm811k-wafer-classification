# WM811K Wafer Defect Classification

An ongoing project investigating semiconductor wafer defect classification using the WM811K dataset and a PyTorch CNN baseline.

## Current Progress

- Loaded and inspected 811,457 wafer maps.
- Cleaned failure-type labels and retained 172,950 labeled samples across nine categories.
- Examined wafer-map dimensions, class distributions, and representative defect patterns.
- Identified 10,762 manufacturing lots in the labeled data.
- Implemented an approximately 80/10/10 lot-level train/validation/test split using random seed 42.
- Verified zero lot overlap between partitions.
- Created a balanced training subset containing 108 wafers per class, totaling 972 samples.
- Implemented image preprocessing, PyTorch datasets, data loaders, and a CNN baseline.
- Completed CNN training and evaluation experiments, including a separate 200-epoch run.

## Classification Categories

Eight defect categories plus the `none` category:

- none
- Center
- Donut
- Edge-Loc
- Edge-Ring
- Loc
- Near-full
- Random
- Scratch

## Experiment A: Balanced CNN Baseline

### Data Preparation

- Balanced training subset: 972 wafers, with 108 samples per class.
- Validation set: 17,249 wafers.
- Test set: 16,714 wafers.
- Validation and test sets were not resampled.
- Wafer maps were resized to 224 × 224 using nearest-neighbor interpolation.
- Pixel values were scaled from 0–2 to 0–1.

### Model and Evaluation

Implemented a nine-class CNN in PyTorch using convolutional layers,
batch normalization, max pooling, and fully connected layers.

Training used cross-entropy loss and the Adam optimizer.
Models were selected using validation macro-F1.

Evaluation included accuracy, macro-F1, per-class precision and recall,
and confusion matrices.

### 200-Epoch Experiment

The saved training history contains metrics for epochs 1–200.

The highest validation macro-F1 occurred at epoch 200:

- Validation accuracy: 78.05%
- Validation macro-F1: 0.5637

These are validation results, not test results.
A test evaluation of this 200-epoch model is not included in the
reviewed experiment outputs.

### Earlier Confusion Matrix

The saved confusion matrix shows validation predictions from an
earlier experiment using the model selected at epoch 10.

It does not represent the 200-epoch model.

## Limitations and Next Steps

- The baseline uses a small balanced subset of the available training data.
- Performance varies across classes, particularly for localized and scratch defects in the earlier evaluation.
- Future work may explore alternative models and sampling strategies.
- Results are based on a public dataset and do not demonstrate production deployment or manufacturing yield improvement.

## Files

- `wm811k_exploration.ipynb`: data exploration, label cleaning, lot-level splitting, balanced sampling, CNN training, and evaluation.
- `results/experiment_a_200epochs_history.json`: training and validation metrics for the 200-epoch experiment.
- `results/validation_confusion_matrix_epoch_10.png`: row-normalized validation confusion matrix from the earlier epoch-10 evaluation.

## Running the Notebook

1. Obtain the WM811K dataset separately.
2. Update the local dataset path in the dataset-loading cell.
3. Install the required libraries.
4. Run the notebook cells in order.

Required libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- torch

The notebook uses Apple MPS acceleration when available and otherwise
falls back to the CPU. Runtime and results may vary by environment.

## Data and Model Files

Keep datasets, compressed archives, model checkpoints, local Python
environments, and secrets out of version control.

Small experiment outputs, such as training-history JSON files and
evaluation figures, can be included to document results.
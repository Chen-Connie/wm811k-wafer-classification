# WM811K Wafer Defect Classification

Research on wafer defect classification using the WM811K dataset and a CNN.

## Current progress

- Implemented label cleaning and nine-class labeling.
- Inspected wafer map dimensions and class distributions.
- Identified 10,762 lots in the labeled data.
- Split lots into training, validation, and test groups using an 80/10/10 ratio and random seed 42.
- Created training, validation, and test DataFrames.
- Added checks for lot overlap between partitions.
- Implemented balanced training sampling with 108 wafers per class.

## Classification categories

none, Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch.

## Experiment A: Balanced CNN baseline

- Training: 108 wafers per class, totaling 972 wafers.
- Validation and test: retain their original class distributions.
- CNN training and evaluation: pending.

Planned evaluation metrics include accuracy, macro-F1, per-class recall, and a confusion matrix.

## Files

- `wm811k_exploration.ipynb`: data exploration, label cleaning, lot-based splitting, and balanced sampling.

## Running the notebook

The notebook currently uses a local absolute dataset path. Update the path in the dataset-loading cell before running it on another computer.

Required libraries for the current notebook:
- pandas
- numpy
- matplotlib
- scikit-learn

## Data and model files

The dataset, compressed archives, saved models, local Python environments, and secrets are
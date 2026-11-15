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

The saved checkpoint was evaluated on 16,714 wafers from 1,077 held-out lots.

- Test accuracy: 75.34%
- Test macro-F1: 0.5188

The test report includes per-class precision, recall, F1, and support. The test confusion matrix is normalized by true class. To reproduce the test evaluation, follow the setup instructions in the final notebook section.

### Earlier Confusion Matrix

The saved confusion matrix shows validation predictions from an
earlier experiment using the model selected at epoch 10.

It does not represent the 200-epoch model.

## Experiment B: Capped Training Sample

The notebook's final section samples up to 2,000 training wafers per class
and retains all training wafers from classes below that cap. It reuses the
lot split, preprocessing, CNN architecture, and Adam settings from Experiment A.

The three-epoch pilot selects a checkpoint using validation macro-F1. Its
validation report includes accuracy, macro-F1, the rate at which true `none`
wafers are predicted as defects, and per-class scores. The report also evaluates
the saved Experiment A model on the same validation split. The two runs have
different training budgets.

The pilot used 12,202 training wafers. Its best checkpoint was epoch 3.

| Validation metric | Experiment A (epoch 200) | Experiment B (epoch 3) |
| --- | ---: | ---: |
| Accuracy | 78.05% | 76.92% |
| Macro-F1 | 0.5637 | 0.5247 |
| False defect alerts on true `none` wafers | 20.73% | 21.03% |
| Scratch F1 | 0.0499 | 0.0000 |
| Loc F1 | 0.1840 | 0.0531 |

- `results/experiment_b_cap2000_validation.json` contains the pilot metrics.
- `checkpoints/experiment_b_cap2000_best.pt` contains the pilot weights.

### Equal-Update Comparison

Both CNNs ran for 1,146 Adam updates and used the same
validation lot split. Metrics are from the final update of each run.

| Validation metric | 108 per class | Up to 2,000 per class |
| --- | ---: | ---: |
| Accuracy | 64.85% | 76.92% |
| Macro-F1 | 0.4855 | 0.5247 |
| False defect alerts on true `none` wafers | 35.54% | 21.03% |
| Scratch F1 | 0.0307 | 0.0000 |
| Loc F1 | 0.0387 | 0.0531 |

`results/experiment_ab_matched_updates.json` contains the validation
comparison, including both confusion matrices.

### Continuation Through Epoch 10

Experiment B continued from the epoch-3 weights with freshly initialized Adam
and shuffle state. Validation used the same 17,249 wafers. The best checkpoint
was selected by validation macro-F1 across the pilot and continuation.

| Validation metric | Pilot epoch 3 | Best continuation checkpoint (epoch 9) |
| --- | ---: | ---: |
| Accuracy | 76.92% | 85.32% |
| Macro-F1 | 0.5247 | 0.5928 |
| False defect alerts on true `none` wafers | 21.03% | 12.10% |
| Scratch precision | 0.00% | 14.93% |
| Scratch recall | 0.00% | 8.62% |

- `results/experiment_b_cap2000_to10_validation.json` contains all ten epochs.
- `checkpoints/experiment_b_cap2000_to10_latest.pt` stores the last epoch and optimizer state.
- `checkpoints/experiment_b_cap2000_to10_best.pt` stores the selected model.

## Experiment C: Scratch-Only Augmentation

Both arms start from Experiment B's epoch-9 checkpoint and complete 382
optimizer updates with identical batch ordering. Only Scratch training images
are eligible for rotation or reflection in the augmented arm.

| Validation metric | Control | Scratch augmentation |
| --- | ---: | ---: |
| Accuracy | 83.85% | 84.53% |
| Macro-F1 | 0.6143 | 0.6189 |
| False defect alerts on true `none` wafers | 14.30% | 13.46% |
| Scratch precision | 15.69% | 17.39% |
| Scratch recall | 6.90% | 6.90% |
| Scratch F1 | 0.0958 | 0.0988 |

`results/experiment_c_scratch_augmentation_validation.json` contains the
per-class scores and both confusion matrices.

## Experiment D: Dual-Channel Input

Both arms start from Experiment B's epoch-9 checkpoint and complete 382
optimizer updates with identical batch ordering. The two-channel arm receives
a wafer mask and failed-die mask. Its first layer starts from the original
weights divided equally between both channels.

| Validation metric | Control | Dual-channel input |
| --- | ---: | ---: |
| Accuracy | 83.85% | 85.48% |
| Macro-F1 | 0.6143 | 0.6234 |
| False defect alerts on true `none` wafers | 14.30% | 12.30% |
| Scratch precision | 15.69% | 17.07% |
| Scratch recall | 6.90% | 6.03% |
| Scratch F1 | 0.0958 | 0.0892 |

`results/experiment_d_dual_channel_validation.json` contains the
per-class scores and both confusion matrices.

## Cross-Lot CNN vs. Cosmos Reason Pilot

How do a supervised CNN and a wafer-unadapted vision-language model classify
nine wafer patterns from previously unseen manufacturing lots?

- CNN: Experiment B, epoch 9, selected using validation macro-F1.
- VLM: `mlx-community/Cosmos-Reason1-7B-4bit` (snapshot `9921dabe46e862b71f299cc6948fa78ae8a0261e`), without
  wafer-specific fine-tuning. The format was checked on 18 validation wafers.
- Pilot: 117 test wafers from 102 held-out lots, with
  13 wafers per class. The same wafer IDs are scored for both models. Invalid
  labels count as errors.

| Paired pilot metric | CNN | Cosmos Reason |
| --- | ---: | ---: |
| Accuracy | 56.41% | 12.82% |
| Macro-F1 | 0.5441 | 0.0836 |

The class-balanced pilot scores apply to its subset. The CNN also scored
82.70% accuracy and 0.5370 macro-F1 on
all 16,714 held-out test wafers. Cosmos has pilot results.

Run the notebook's CNN evaluation and export cells. On Apple Silicon, install
`mlx-vlm==0.7.12` in a separate Python
environment, then run the following from the project folder before the paired
scoring cell. The model weights are downloaded from Hugging Face and images
are processed locally.

```sh
python run_cosmos_comparison.py data/cosmos_comparison/validation/manifest.json results/cosmos_reason_validation_pilot.json
python run_cosmos_comparison.py data/cosmos_comparison/test/manifest.json results/cosmos_reason_test_pilot.json
```

### Validation Pilot With Labeled Examples

The same 4-bit Cosmos snapshot classified 18 validation wafers using either
only the label descriptions or nine labeled examples in the prompt. The examples
came from training lots, with one wafer per class. No model weights were changed.

| Validation metric | No examples | Nine examples |
| --- | ---: | ---: |
| Accuracy | 16.67% | 11.11% |
| Macro-F1 | 0.0606 | 0.0404 |
| Scratch correct | 0 / 2 | 0 / 2 |

After running the export cell and the zero-shot validation command above, run:

```sh
python run_cosmos_comparison.py data/cosmos_comparison/validation/manifest.json results/cosmos_reason_fewshot_validation_pilot.json --examples data/cosmos_comparison/examples/manifest.json
```

The final notebook cell produces `results/cosmos_reason_fewshot_comparison.json`.
The 117-wafer held-out pilot was not used for this comparison.

## Limitations and Next Steps

- The baseline uses a small balanced subset of the available training data.
- Performance varies across classes, particularly for localized and scratch defects in the earlier evaluation.
- Future work may explore alternative models and sampling strategies.
- Results are based on a public dataset and do not demonstrate production deployment or manufacturing yield improvement.

## Files

- `wm811k_exploration.ipynb`: data exploration, label cleaning, lot-level splitting, balanced sampling, CNN training, and evaluation.
- `results/experiment_a_200epochs_history.json`: training and validation metrics for the 200-epoch experiment.
- `results/experiment_a_200epochs_test_metrics.json`: held-out test metrics and confusion counts.
- `results/experiment_a_200epochs_test_confusion_matrix.png`: row-normalized held-out test confusion matrix.
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

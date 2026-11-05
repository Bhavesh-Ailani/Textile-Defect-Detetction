# Textile Defect Classification with a CNN

Multi-class classification of small fabric defects from 64×64 image patches, using a convolutional neural network built in TensorFlow/Keras.

## Dataset

[TILDA 400 – 64×64 patches](https://www.kaggle.com/datasets/angelolmg/tilda-400-64x64-patches) (Kaggle), a patch version of the TILDA Textile Texture Database.

The 400 source fabric images are each cut into an 8×8 grid of patches, giving 25,600 RGB PNGs in five classes:

| Class | Patches | Source images |
|---|---|---|
| good | 23,170 | 400 |
| hole | 337 | 101 |
| objects | 837 | 100 |
| oil spot | 636 | 100 |
| thread error | 620 | 100 |

The dataset is not included in this repository. The notebook downloads it with `kagglehub`.

**Data notes**
- Patch filenames follow `<image_id>_patch<row>-<col>.png`. Every source image contributes defect-free patches to `good` and defect patches to one defect class.
- Source images `191` and `203` are byte-identical, so `203` is excluded. This leaves 399 source images.

![Sample patches](figures/samples.png)

## Approach

**Train / validation / test split**
- The split is made **by source image**, not by patch, so patches from the same fabric image never appear in more than one subset. The notebook checks this with an assertion.
- Source images are stratified by defect type. 10% of images are held out for test, and 20% of the rest are used for validation (287 / 72 / 40 images).

**Class imbalance**
- `good` is randomly undersampled to 3,000 patches, spread across the splits in proportion to their share of good patches. All defect patches are kept.
- Inverse-frequency class weights are applied during training.

| Split | good | hole | objects | oil spot | thread error | Total |
|---|---|---|---|---|---|---|
| Train | 2,162 | 246 | 545 | 462 | 461 | 3,876 |
| Validation | 540 | 62 | 167 | 120 | 99 | 988 |
| Test | 298 | 29 | 123 | 54 | 60 | 564 |

**Preprocessing and augmentation**
- Pixel values are scaled to [0, 1].
- Training patches are augmented with horizontal and vertical flips, random 90° rotations, and small brightness and contrast changes.

**Model**
- Four convolution blocks (32 → 64 → 128 → 256 filters), each with batch normalisation and max pooling.
- Dense layers of 256 and 128 units with batch normalisation and dropout (0.5, 0.3), followed by a softmax output.
- 1.47M parameters.

**Training**
- Adam optimiser, initial learning rate 3e-4, halved when validation loss plateaus.
- Up to 40 epochs, with early stopping on validation loss (patience 8). The best weights are restored.

## Results

Test set (564 patches from 40 unseen source images):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| good | 0.79 | 0.96 | 0.86 | 298 |
| hole | 0.65 | 0.69 | 0.67 | 29 |
| objects | 0.70 | 0.44 | 0.54 | 123 |
| oil spot | 0.52 | 0.54 | 0.53 | 54 |
| thread error | 0.58 | 0.37 | 0.45 | 60 |
| **Macro avg** | **0.65** | **0.60** | **0.61** | 564 |

- **Accuracy:** 0.727. A majority-class baseline (always predicting `good`) scores 0.528.
- **Macro F1:** 0.609.


**Observations**
- False alarms are rare: 285 of 298 good patches are classified correctly, and most of the rest are predicted as oil spots (10).
- The main failure is missed defects. 77 of 266 defect patches (29%) are predicted as `good`, including 44 of 123 foreign-object patches.
- Thread errors are the hardest class (recall 0.37). They are confused with `good`, `objects` and `hole`.
- Holes are the easiest defect type to detect (F1 0.67), and are never predicted as `good`.
- Training ran the full 40 epochs. The weights from epoch 35 (lowest validation loss) were restored.
- Validation scores fluctuate strongly between epochs, especially early on. Training accuracy stays below validation accuracy because dropout, augmentation and class weighting apply only during training.



## Limitations

- This is supervised classification of labelled defect types, not unsupervised anomaly detection. The model cannot flag defect types it has not seen.
- The test set contains only 40 source images, so the per-class scores (especially `hole`, with 29 patches) have wide uncertainty. A single split was used; grouped cross-validation would give more reliable estimates.
- Results vary between runs. Another run of the same notebook and split reached 0.672 accuracy and 0.562 macro F1, so differences of a few points should not be over-interpreted.
- Validation loss was still unstable at the end of training, so the model may not be fully converged.
- An earlier version of this project split patches at random. Almost every test patch then shared a source image with the training data, so those scores were inflated and are not reported.

## Repository structure

```
├── Textile_Defect_Detection.ipynb   # data indexing, split, training, evaluation
├── figures/                         # plots saved by the notebook
├── requirements.txt
└── README.md
```

## How to run

1. Open the notebook in Google Colab. A GPU runtime is faster, but CPU training takes about 10–15 minutes.
2. Run all cells. The dataset downloads automatically through `kagglehub`. Kaggle credentials may be required.

To run locally instead:

```bash
pip install -r requirements.txt
jupyter notebook Textile_Defect_Detection.ipynb
```

Set the `TILDA_PATH` environment variable to use an existing local copy of the dataset. Exact scores can vary slightly between runs and hardware.

## Possible extensions

- Transfer learning with a pretrained backbone (e.g. MobileNetV2 or ResNet).
- Unsupervised anomaly detection trained on `good` patches only (e.g. an autoencoder or PatchCore).
- Image-level defect localisation by aggregating patch predictions back onto the 8×8 grid.

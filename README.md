# Fruits Quality Detection

Fruit image classification for **fresh vs rotten** quality detection.

## Dataset and labels

This repository uses the published **Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality**:

- Dataset download: https://data.mendeley.com/datasets/bdd69gyhv8/1
- DOI: 10.17632/bdd69gyhv8.1
- 8 fruit types × 2 quality states = 16 classes.
- 3,200 original images and 12,335 augmented images are described by the dataset authors.
- Dataset license: CC BY 4.0 (as listed on Mendeley Data).

The original repository previously contained only a project-intent README, not an existing trained model, training pipeline, or documented engineered features. The current implementation adds a reproducible image-classification training pipeline rather than claiming to reuse unavailable model artifacts.

## Train

Download and extract **Original Image.zip** from the dataset page. Keep one folder per class, such as `Fresh Apple`, `Rotten Apple`, etc.

Install Python 3.10 or 3.11 dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python train.py --data_dir "path/to/Original Image" --epochs 12
```

The training pipeline uses MobileNetV2 transfer learning, 224×224 image input, on-the-fly training augmentation, an 80/20 training-validation split, and fine-tuning of the final feature layers.

Output files:

- `models/fruit_quality.keras`
- `models/class_names.json`

The validation accuracy is printed after training. Report the measured value from your run; do not assume a performance score before training.

## Explainable AI demo

The companion Streamlit application is here:

https://github.com/Vinodvinum/xai-lime-classroom-demo

It accepts a fruit image upload and shows the same model's prediction with and without a LIME image-region explanation. Copy the generated model and class-name files into that app's `models/` directory before deployment.

## Limitations

This is a research/education prototype. It can only classify the 16 dataset labels and cannot assess internal fruit condition, chemical contamination, or food safety. LIME provides a local approximate explanation, not proof of causation.

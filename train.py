import argparse
import json
import re
from pathlib import Path

import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def clean_class_name(name):
    return re.sub(r"[_-]+", " ", name).strip()

def main():
    parser = argparse.ArgumentParser(description="Train a fruit freshness classifier on the real Fresh and Rotten Fruits dataset.")
    parser.add_argument("--data_dir", required=True, help="Folder containing one subfolder per class, e.g. Fresh Apple, Rotten Apple.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Dataset folder not found: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    class_names = [clean_class_name(x) for x in train_ds.class_names]
    if len(class_names) < 2:
        raise SystemExit("Expected at least two class folders. Check the extracted dataset structure.")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
        num_parallel_calls=autotune
    ).prefetch(autotune)
    val_ds = val_ds.map(
        lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
        num_parallel_calls=autotune
    ).prefetch(autotune)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # Fine-tune the final feature layers for the dataset.
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=max(2, args.epochs // 3))

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "fruit_quality.keras")
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Saved model and labels to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()

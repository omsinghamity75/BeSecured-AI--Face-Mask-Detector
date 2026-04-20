import argparse
import json
from pathlib import Path

import tensorflow as tf

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a face mask detector with TensorFlow."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset folder containing with_mask/ and without_mask/ subfolders.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("artifacts/mask_detector.keras"),
        help="Path to save the trained Keras model.",
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        default=Path("artifacts/class_names.json"),
        help="Path to save the class-name mapping as JSON.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for train/validation split.",
    )
    return parser.parse_args()


def ensure_dataset(dataset_dir: Path):
    expected = [dataset_dir / "with_mask", dataset_dir / "without_mask"]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset is missing required folders: "
            + ", ".join(missing)
            + ". Expected layout: dataset/with_mask and dataset/without_mask."
        )


def build_datasets(dataset_dir: Path, batch_size: int, seed: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(learning_rate: float, num_classes: int):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plot(history, output_dir: Path):
    if plt is None:
        print("matplotlib is not installed; skipping training curve plot.")
        return

    epochs = range(1, len(history.history["accuracy"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], label="train")
    plt.plot(epochs, history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"], label="train")
    plt.plot(epochs, history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png")
    plt.close()


def main():
    args = parse_args()
    ensure_dataset(args.dataset)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_labels.parent.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(
        args.dataset, args.batch_size, args.seed
    )
    model = build_model(args.learning_rate, len(class_names))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.output_model),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.output_model)
    args.output_labels.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    save_training_plot(history, args.output_model.parent)

    print(f"Saved model to {args.output_model}")
    print(f"Saved class names to {args.output_labels}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    main()

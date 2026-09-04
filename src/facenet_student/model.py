from __future__ import annotations

import json
import os
import random
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from facenet_student.data import ImageSample, discover_dataset, stratified_split
from facenet_student.errors import FacenetStudentError

Backbone = Literal["tiny", "mobilenet_v2"]
EMBEDDING_SCHEMA_VERSION = 1


def _load_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise FacenetStudentError(
            "TensorFlow is required for this command. Install it with "
            "'python -m pip install -e .[tensorflow]'."
        ) from exc
    return tf


def _set_reproducible_seed(tf: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    with suppress(AttributeError, RuntimeError):
        tf.config.experimental.enable_op_determinism()


def _make_dataset(
    tf: Any,
    samples: list[ImageSample],
    *,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    paths = [str(sample.path) for sample in samples]
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize_with_pad(image, image_size, image_size, antialias=True)
        image = tf.cast(image, tf.float32)
        return image, label

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    if shuffle:
        dataset = dataset.shuffle(len(samples), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_model(
    tf: Any,
    *,
    image_size: int,
    num_classes: int,
    embedding_dim: int,
    backbone: Backbone,
    pretrained: bool,
):
    if image_size < 48:
        raise FacenetStudentError("image_size must be at least 48")
    if embedding_dim < 2:
        raise FacenetStudentError("embedding_dim must be at least 2")

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=101),
            tf.keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=102),
            tf.keras.layers.RandomZoom(0.08, fill_mode="reflect", seed=103),
            tf.keras.layers.RandomContrast(0.12, seed=104),
        ],
        name="augmentation",
    )
    x = augmentation(inputs)

    if backbone == "tiny":
        x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(x)
        for index, filters in enumerate((24, 48, 96), start=1):
            x = tf.keras.layers.Conv2D(
                filters,
                3,
                padding="same",
                use_bias=False,
                name=f"conv_{index}",
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{index}")(x)
            x = tf.keras.layers.ReLU(name=f"relu_{index}")(x)
            x = tf.keras.layers.MaxPooling2D(name=f"pool_{index}")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    elif backbone == "mobilenet_v2":
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="rescale_minus1_1")(x)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights="imagenet" if pretrained else None,
        )
        base.trainable = False
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    else:
        raise FacenetStudentError(f"Unsupported backbone: {backbone}")

    x = tf.keras.layers.Dropout(0.2, name="embedding_dropout")(x)
    raw_embedding = tf.keras.layers.Dense(
        embedding_dim,
        activation=None,
        name="embedding_dense",
    )(x)
    embedding = tf.keras.layers.UnitNormalization(axis=-1, name="embedding")(raw_embedding)
    identity = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="identity",
    )(embedding)
    return tf.keras.Model(inputs=inputs, outputs=identity, name=f"{backbone}_student_embedder")


def train_embedding_model(
    input_dir: Path | str,
    model_out: Path | str,
    metadata_out: Path | str,
    *,
    backbone: Backbone = "mobilenet_v2",
    image_size: int = 160,
    embedding_dim: int = 128,
    batch_size: int = 16,
    epochs: int = 20,
    validation_fraction: float = 0.2,
    learning_rate: float = 1e-3,
    seed: int = 42,
    pretrained: bool = True,
) -> dict[str, object]:
    """Train a supervised identity head and save its normalized embedding layer."""

    if batch_size < 1 or epochs < 1:
        raise FacenetStudentError("batch_size and epochs must be positive")
    if learning_rate <= 0:
        raise FacenetStudentError("learning_rate must be positive")

    samples, class_names = discover_dataset(input_dir, min_images_per_class=2)
    if len(class_names) < 2:
        raise FacenetStudentError("Training requires at least 2 identity classes")
    train_samples, validation_samples = stratified_split(
        samples,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    tf = _load_tensorflow()
    _set_reproducible_seed(tf, seed)
    train_data = _make_dataset(
        tf,
        train_samples,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_data = _make_dataset(
        tf,
        validation_samples,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    model = _build_model(
        tf,
        image_size=image_size,
        num_classes=len(class_names),
        embedding_dim=embedding_dim,
        backbone=backbone,
        pretrained=pretrained,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=max(2, min(5, epochs // 4)),
        restore_best_weights=True,
    )
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=2,
    )

    model_out = Path(model_out)
    metadata_out = Path(metadata_out)
    if model_out.suffix != ".keras":
        raise FacenetStudentError("model_out must use the .keras extension")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)

    history_values = {
        name: [float(value) for value in values] for name, values in history.history.items()
    }
    metadata: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backbone": backbone,
        "pretrained": bool(pretrained and backbone == "mobilenet_v2"),
        "image_size": image_size,
        "embedding_dim": embedding_dim,
        "seed": seed,
        "class_names": class_names,
        "num_train_images": len(train_samples),
        "num_validation_images": len(validation_samples),
        "epochs_requested": epochs,
        "epochs_completed": len(history_values["loss"]),
        "history": history_values,
        "model_path": model_out.name,
    }
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def export_embeddings(
    input_dir: Path | str,
    model_path: Path | str,
    output_path: Path | str,
    *,
    batch_size: int = 32,
    seed: int = 42,
) -> dict[str, object]:
    """Run the saved embedding layer and write a non-pickle NumPy archive."""

    if batch_size < 1:
        raise FacenetStudentError("batch_size must be positive")
    samples, class_names = discover_dataset(input_dir)
    tf = _load_tensorflow()
    _set_reproducible_seed(tf, seed)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FacenetStudentError(f"Embedding model does not exist: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    try:
        embedding_layer = model.get_layer("embedding")
    except ValueError as exc:
        raise FacenetStudentError("The model has no layer named 'embedding'") from exc
    embedding_model = tf.keras.Model(model.input, embedding_layer.output, name="embedding_export")
    input_shape = model.input_shape
    if not isinstance(input_shape, tuple) or input_shape[1] is None:
        raise FacenetStudentError("The saved model has an unsupported input shape")
    image_size = int(input_shape[1])
    dataset = _make_dataset(
        tf,
        samples,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    embeddings = np.asarray(embedding_model.predict(dataset, verbose=0), dtype=np.float32)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    relative_paths = np.asarray(
        [sample.path.relative_to(Path(input_dir)).as_posix() for sample in samples],
        dtype=np.str_,
    )
    class_array = np.asarray(class_names, dtype=np.str_)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(samples):
        raise FacenetStudentError("The embedding model returned an unexpected shape")
    if not np.all(np.isfinite(embeddings)):
        raise FacenetStudentError("The embedding model returned non-finite values")

    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise FacenetStudentError("Embedding output must use the .npz extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        labels=labels,
        class_names=class_array,
        paths=relative_paths,
        schema_version=np.asarray(EMBEDDING_SCHEMA_VERSION, dtype=np.int64),
    )
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        "output": str(output_path),
        "num_images": len(samples),
        "num_classes": len(class_names),
        "embedding_dim": int(embeddings.shape[1]),
        "minimum_norm": float(norms.min()),
        "maximum_norm": float(norms.max()),
    }

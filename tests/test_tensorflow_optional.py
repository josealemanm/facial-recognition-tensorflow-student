import numpy as np
import pytest

from facenet_student.model import _build_model


def test_tensorflow_embedding_layer_is_finite() -> None:
    tf = pytest.importorskip("tensorflow")
    model = _build_model(
        tf,
        image_size=48,
        num_classes=2,
        embedding_dim=8,
        backbone="tiny",
        pretrained=False,
    )
    embedder = tf.keras.Model(model.input, model.get_layer("embedding").output)

    vectors = embedder(np.zeros((2, 48, 48, 3), dtype=np.float32), training=False).numpy()

    assert vectors.shape == (2, 8)
    assert np.all(np.isfinite(vectors))

from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def _flatten_images(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def _normalize_unit_range(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return x_train, x_test



def load_dataset(
    name: str,
    batch_size: int,
    flatten: bool = True,
    shuffle_buffer: int = 10000,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    name = name.lower()

    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
        x_train, x_test = _normalize_unit_range(x_train, x_test)

    elif name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
        x_train, x_test = _normalize_unit_range(x_train, x_test)
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    elif name == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        num_classes = 100
        x_train, x_test = _normalize_unit_range(x_train, x_test)
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    if flatten:
        x_train = _flatten_images(x_train)
        x_test = _flatten_images(x_test)

    y_train = to_one_hot(y_train, num_classes)
    y_test = to_one_hot(y_test, num_classes)

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    info = {
        "input_dim": int(x_train.shape[1]),
        "num_classes": int(y_train.shape[1]),
        "train_size": int(x_train.shape[0]),
        "test_size": int(x_test.shape[0]),
    }
    return train_ds, test_ds, info


def mse_per_sample(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=1)


def cross_entropy_per_sample(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)


def make_loss_fn(name: str):
    name = name.lower()
    if name == "mse":
        return mse_per_sample
    if name in {"ce", "cross_entropy", "cce"}:
        return cross_entropy_per_sample
    raise ValueError(f"Unsupported loss: {name}")


def evaluate_model(model, dataset: tf.data.Dataset, loss_fn, decorrelated: bool) -> dict:
    acc = tf.keras.metrics.CategoricalAccuracy()
    loss_mean = tf.keras.metrics.Mean()

    for x, y in dataset:
        y_pred = model.forward(x, decorrelate=decorrelated)
        loss = loss_fn(y_pred, y)
        acc.update_state(y, y_pred)
        loss_mean.update_state(tf.reduce_mean(loss))

    return {
        "loss": float(loss_mean.result().numpy()),
        "acc": float(acc.result().numpy()),
    }


def algorithm_to_flags(name: str) -> Tuple[str, bool]:
    """
    Returns:
        base_algorithm in {"bp", "np", "anp", "inp"}
        decorrelated flag
    """
    name = name.lower()
    mapping = {
        "bp": ("bp", False),
        "dbp": ("bp", True),
        "np": ("np", False),
        "dnp": ("np", True),
        "anp": ("anp", False),
        "danp": ("anp", True),
        "inp": ("inp", False),
        "dinp": ("inp", True),
    }
    if name not in mapping:
        raise ValueError(f"Unknown algorithm: {name}")
    return mapping[name]
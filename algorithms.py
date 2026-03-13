from __future__ import annotations

import tensorflow as tf


def apply_decorrelation_update(model, decor_lr: float) -> None:
    """
    Per-layer decorrelation update:
        C = (X^T X / B) ⊙ (1 - I)
        R <- R - eta_dec * C R
    """
    for layer in model.layers_list:
        x = layer.inputs_clean
        batch_size = tf.cast(tf.shape(x)[0], x.dtype)
        eye = tf.eye(tf.shape(x)[1], dtype=x.dtype)
        corr = tf.einsum("ni,nj->ij", x, x) / batch_size
        corr = corr * (1.0 - eye)
        update = -tf.einsum("ij,jk->ik", corr, layer.R)
        layer.R.assign_add(tf.cast(decor_lr, layer.R.dtype) * tf.cast(update, layer.R.dtype))


def bp_gradients(model, x, y, loss_fn, decorrelated: bool):
    with tf.GradientTape() as tape:
        y_pred = model.forward(x, decorrelate=decorrelated)
        loss_per_sample = loss_fn(y_pred, y)
        loss = tf.reduce_mean(loss_per_sample)
    grads = tape.gradient(loss, model.ordered_trainable_variables())
    return grads, y_pred, loss_per_sample


def _dense_weight_and_bias_grad(layer, error: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x = layer.inputs_clean
    batch_size = tf.cast(tf.shape(x)[0], x.dtype)
    w_grad = -(1.0 / batch_size) * tf.einsum("ni,nj->ij", x, error)
    b_grad = -(1.0 / batch_size) * tf.reduce_sum(error, axis=0)
    return w_grad, b_grad


def _network_activity_stats(model):
    diffs = []
    for layer in model.layers_list:
        diff = layer.outputs_noisy - layer.outputs_clean
        diffs.append(tf.reshape(diff, [tf.shape(diff)[0], -1]))
    act_diff = tf.concat(diffs, axis=1)
    norm_sq = tf.expand_dims(tf.linalg.norm(act_diff, axis=1), axis=-1) ** 2
    n_total = tf.cast(tf.shape(act_diff)[1], tf.float32)
    return norm_sq, n_total


def _np_like_grads_from_cached_pass(
    model,
    performance_diff: tf.Tensor,
    variant: str,
    layer_idx: int | None = None,
) -> list[tf.Tensor]:
    """
    variant in {"np", "anp", "inp"}.
    Returns grads aligned with model.ordered_trainable_variables().
    """
    if variant in {"anp", "np"}:
        network_norm_sq, n_total = _network_activity_stats(model)

    grads = []
    for i, layer in enumerate(model.layers_list):
        compute_this_layer = (layer_idx is None) or (layer_idx == i)

        if not compute_this_layer:
            kernel, bias = layer.trainable_variables
            grads.extend([tf.zeros_like(kernel), tf.zeros_like(bias)])
            continue

        if variant == "np":
            error = layer.noise * performance_diff

        elif variant == "anp":
            activity_diff = layer.outputs_noisy - layer.outputs_clean
            error = activity_diff * performance_diff
            error = error / network_norm_sq
            error = error * tf.cast(n_total, error.dtype)

        elif variant == "inp":
            activity_diff = layer.outputs_noisy - layer.outputs_clean
            layer_norm_sq = tf.expand_dims(tf.linalg.norm(activity_diff, axis=1), axis=-1) ** 2
            n_layer = tf.cast(tf.shape(activity_diff)[1], activity_diff.dtype)
            error = activity_diff * performance_diff
            error = error / layer_norm_sq
            error = error * n_layer

        else:
            raise ValueError(f"Unknown variant: {variant}")

        w_grad, b_grad = _dense_weight_and_bias_grad(layer, error)
        kernel, bias = layer.trainable_variables
        grads.extend([tf.cast(w_grad, kernel.dtype), tf.cast(b_grad, bias.dtype)])

    return grads


def perturbation_gradients(
    model,
    x,
    y,
    loss_fn,
    decorrelated: bool,
    variant: str,
    noise_std: float,
    num_noise_iters: int = 1,
):
    """
    variant in {"np", "anp", "inp"}.
    Averages gradients across multiple noisy iterations.
    """
    # Clean pass first
    y_clean = model.forward(x, decorrelate=decorrelated)
    loss_clean = loss_fn(y_clean, y)

    all_iter_grads = []

    for _ in range(num_noise_iters):
        model.reset_all_noise(noise_std)

        if variant in {"np", "anp"}:
            y_noisy = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=None)
            loss_noisy = loss_fn(y_noisy, y)
            performance_diff = tf.reshape(loss_clean - loss_noisy, [-1, 1])

            if variant == "np":
                performance_diff = performance_diff / tf.cast(noise_std**2, performance_diff.dtype)

            grads = _np_like_grads_from_cached_pass(
                model=model,
                performance_diff=performance_diff,
                variant=variant,
                layer_idx=None,
            )

        elif variant == "inp":
            grads = [tf.zeros_like(v) for v in model.ordered_trainable_variables()]
            for layer_idx in range(len(model.layers_list)):
                y_noisy = model.forward_noisy(x, decorrelate=decorrelated, noise_layer_idx=layer_idx)
                loss_noisy = loss_fn(y_noisy, y)
                performance_diff = tf.reshape(loss_clean - loss_noisy, [-1, 1])
                layer_grads = _np_like_grads_from_cached_pass(
                    model=model,
                    performance_diff=performance_diff,
                    variant="inp",
                    layer_idx=layer_idx,
                )
                grads = [g0 + g1 for g0, g1 in zip(grads, layer_grads)]
        else:
            raise ValueError(f"Unknown variant: {variant}")

        all_iter_grads.append(grads)

    mean_grads = [
        tf.reduce_mean(tf.stack([glist[i] for glist in all_iter_grads], axis=0), axis=0)
        for i in range(len(all_iter_grads[0]))
    ]
    return mean_grads, y_clean, loss_clean


def optimizer_from_name(name: str, lr: float):
    name = name.lower()
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def train_step(
    model,
    optimizer,
    x,
    y,
    loss_fn,
    algorithm: str,
    decorrelated: bool,
    noise_std: float,
    num_noise_iters: int,
    decor_lr: float,
):
    if algorithm == "bp":
        grads, y_pred, loss_per_sample = bp_gradients(
            model=model,
            x=x,
            y=y,
            loss_fn=loss_fn,
            decorrelated=decorrelated,
        )
    elif algorithm in {"np", "anp", "inp"}:
        grads, y_pred, loss_per_sample = perturbation_gradients(
            model=model,
            x=x,
            y=y,
            loss_fn=loss_fn,
            decorrelated=decorrelated,
            variant=algorithm,
            noise_std=noise_std,
            num_noise_iters=num_noise_iters,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if decorrelated:
        apply_decorrelation_update(model, decor_lr)

    optimizer.apply_gradients(zip(grads, model.ordered_trainable_variables()))
    return y_pred, loss_per_sample
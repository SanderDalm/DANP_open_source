from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils import (
    algorithm_to_flags,
    evaluate_model,
    load_dataset,
    make_loss_fn,
    set_seed,
)
from models import MLP
from algorithms import optimizer_from_name, train_step


def parse_args():
    parser = argparse.ArgumentParser(description="Lean implementation of BP/NP/ANP/INP and decorrelated variants.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--algorithm", type=str, default="danp",
                        choices=["bp", "dbp", "np", "dnp", "anp", "danp", "inp", "dinp"])
    parser.add_argument("--loss", type=str, default="ce", choices=["mse", "ce"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[1024, 1024, 1024])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--decor_lr", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=1e-2)
    parser.add_argument("--num_noise_iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--save_json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    base_algorithm, decorrelated = algorithm_to_flags(args.algorithm)
    loss_fn = make_loss_fn(args.loss)

    train_ds, test_ds, info = load_dataset(
        name=args.dataset,
        batch_size=args.batch_size,
        flatten=True,
    )

    model = MLP(
        input_dim=info["input_dim"],
        hidden_sizes=args.hidden_sizes,
        output_dim=info["num_classes"],
        hidden_activation=tf.nn.leaky_relu,
        output_activation=tf.nn.softmax,
    )
    optimizer = optimizer_from_name(args.optimizer, args.lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    print("Configuration")
    print(json.dumps(vars(args), indent=2))
    print(f"Input dim: {info['input_dim']}, classes: {info['num_classes']}")

    for epoch in range(args.epochs):
        train_loss_metric = tf.keras.metrics.Mean()
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy()

        for x, y in train_ds:
            y_pred, loss_per_sample = train_step(
                model=model,
                optimizer=optimizer,
                x=x,
                y=y,
                loss_fn=loss_fn,
                algorithm=base_algorithm,
                decorrelated=decorrelated,
                noise_std=args.noise_std,
                num_noise_iters=args.num_noise_iters,
                decor_lr=args.decor_lr,
            )
            train_loss_metric.update_state(tf.reduce_mean(loss_per_sample))
            train_acc_metric.update_state(y, y_pred)

        train_stats = {
            "loss": float(train_loss_metric.result().numpy()),
            "acc": float(train_acc_metric.result().numpy()),
        }
        test_stats = evaluate_model(
            model=model,
            dataset=test_ds,
            loss_fn=loss_fn,
            decorrelated=decorrelated,
        )

        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["acc"])
        history["test_loss"].append(test_stats["loss"])
        history["test_acc"].append(test_stats["acc"])

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss {train_stats['loss']:.4f} | train acc {train_stats['acc']:.4f} | "
            f"test loss {test_stats['loss']:.4f} | test acc {test_stats['acc']:.4f}"
        )

    print("\nFinal metrics")
    print(json.dumps({k: v[-1] for k, v in history.items()}, indent=2))

    if args.save_json:
        out_dir = Path(args.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.dataset}_{args.algorithm}_seed{args.seed}.json"
        payload = {
            "config": vars(args),
            "history": history,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
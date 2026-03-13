from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from utils import (
    algorithm_to_flags,
    evaluate_model,
    load_dataset,
    make_loss_fn,
    save_experiment_results,
    set_seed,
)
from models import MLP
from algorithms import optimizer_from_name, train_step


def parse_args():
    parser = argparse.ArgumentParser(description="Lean implementation of BP/NP/ANP/INP and decorrelated variants.")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument(
        "--algorithm",
        type=str,
        default="danp",
        choices=["bp", "dbp", "np", "dnp", "anp", "danp", "inp", "dinp"],
    )
    parser.add_argument("--loss", type=str, default="ce", choices=["mse", "ce"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[1024, 1024, 1024])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decor_lr", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=1e-2)
    parser.add_argument("--num_noise_iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of random seeds to run.")
    parser.add_argument(
        "--write_results_dir",
        type=str,
        default="results",
        help="Directory where this experiment folder will be written.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment subfolder name. Defaults to dataset_algorithm.",
    )
    parser.add_argument("--save_json", action="store_true")
    return parser.parse_args()


def build_model(info: dict, hidden_sizes: list[int]) -> MLP:
    return MLP(
        input_dim=info["input_dim"],
        hidden_sizes=hidden_sizes,
        output_dim=info["num_classes"],
        hidden_activation=tf.nn.leaky_relu,
        output_activation=tf.nn.softmax,
    )


def run_single_seed(args, seed: int) -> dict:
    set_seed(seed)

    base_algorithm, decorrelated = algorithm_to_flags(args.algorithm)
    loss_fn = make_loss_fn(args.loss)

    train_ds, test_ds, info = load_dataset(
        name=args.dataset,
        batch_size=args.batch_size,
        flatten=True,
    )

    model = build_model(info=info, hidden_sizes=args.hidden_sizes)
    optimizer = optimizer_from_name(args.optimizer, args.lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    print(f"\nRunning seed {seed}")
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
            f"Seed {seed} | Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss {train_stats['loss']:.4f} | train acc {train_stats['acc']:.4f} | "
            f"test loss {test_stats['loss']:.4f} | test acc {test_stats['acc']:.4f}"
        )

    print("\nFinal metrics for seed", seed)
    print(json.dumps({k: v[-1] for k, v in history.items()}, indent=2))
    return history


def main():
    args = parse_args()

    exp_name = args.exp_name if args.exp_name is not None else f"{args.dataset}_{args.algorithm}"

    print("Configuration")
    print(json.dumps(vars(args), indent=2))

    all_histories = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    per_seed_payload = []

    for seed_offset in range(args.num_seeds):
        seed = args.seed + seed_offset
        history = run_single_seed(args, seed)

        for key in all_histories:
            all_histories[key].append(history[key])

        per_seed_payload.append(
            {
                "seed": seed,
                "history": history,
            }
        )

    save_experiment_results(
        write_results_dir=args.write_results_dir,
        exp_name=exp_name,
        histories=all_histories,
        config=vars(args),
        per_seed_payload=per_seed_payload if args.save_json else None,
    )

    print(f"\nSaved results to {Path(args.write_results_dir) / exp_name}")


if __name__ == "__main__":
    main()
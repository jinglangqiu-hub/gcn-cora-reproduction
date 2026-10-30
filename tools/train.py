"""Portable Cora GCN/MLP reproduction: official_code_last; no test labels during training."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.cora_data import load_cora_tensors
from src.gcn_model import GCN
from src.mlp_model import MLP


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def objective_parts(logits, labels, mask, model, coefficient):
    ce = F.cross_entropy(logits[mask], labels[mask])
    weight = model.w0 if isinstance(model, MLP) else model.gc1.weight
    l2 = (coefficient / 2) * weight.square().sum()
    return ce, l2, ce + l2


def forward_model(model, features, adj_norm):
    return model(features) if isinstance(model, MLP) else model(features, adj_norm)


def accuracy(logits, labels, mask):
    return (logits[mask].argmax(1) == labels[mask]).float().mean().item()


def should_stop(epoch_zero_based, validation_history, window=10):
    """Match tkipf/gcn train.py; this is not a patience counter."""
    return epoch_zero_based > window and validation_history[-1] > float(
        np.mean(validation_history[-(window + 1):-1])
    )


def save_curves(run_dir, rows, model_name="gcn"):
    os.environ["MPLCONFIGDIR"] = str(run_dir / "_matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    fig.suptitle(f"{model_name.upper()} | Cora")
    for key, label in [("eval_train_objective", "Train"), ("val_objective", "Validation")]:
        axes[0].plot(epochs, [r[key] for r in rows], label=label)
    for key, label in [("eval_train_accuracy", "Train"), ("val_accuracy", "Validation")]:
        axes[1].plot(epochs, [r[key] for r in rows], label=label)
    axes[0].set(xlabel="Epoch", ylabel="CE + first-layer L2", title="Objective (evaluation mode)")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy (evaluation mode)", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(run_dir / "curves.png", dpi=180)
    fig.savefig(run_dir / "curves.svg")
    plt.close(fig)


def build_config(run_id, model, seed, epochs, data_dir, purpose="debug"):
    return {
        "run_id": run_id, "purpose": "first_full_debug_run" if purpose == "debug" else purpose,
        "model": model, "protocol": "official_code_last", "seed": seed,
        "device": "cpu", "cpu_threads": 4, "deterministic_algorithms": True,
        "in_features": 1433, "hidden_features": 16, "out_features": 7,
        "dropout": .5, "learning_rate": .01, "max_epochs": epochs,
        "optimizer": "Adam", "adam_betas": [.9, .999], "adam_eps": 1e-8,
        "adam_amsgrad": False, "adam_foreach": False, "optimizer_weight_decay": 0,
        "first_layer_l2_coefficient": 5e-4, "early_stopping_window": 10,
        "early_stopping_metric": "validation mean CE + current first-layer L2",
        "selection": "last model at stopping epoch; best_val is diagnostic only",
        "data_directory": str(data_dir), "test_evaluation": False,
        "metric_timing": "train_*: dropout enabled before update; eval_train_* and val_*: after update, dropout disabled",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model", choices=("gcn", "mlp"), default="gcn")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/cora"))
    parser.add_argument("--config", type=Path, default=Path("configs/cora_official_last.json"))

    args = parser.parse_args()
    args.purpose = "public_reproduction"
    if not 1 <= args.epochs <= 200 or not 0 <= args.seed < 2**32:
        parser.error("Use 1..200 epochs and a seed in 0..2**32-1")
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("config.json", "manifest.json", "metrics.csv", "last.pt", "best_val.pt",
                 "result.json", "code", "status.json", "record.md", "environment.txt",
                 "curves.png", "curves.svg"):
        if (run_dir / name).exists():
            raise FileExistsError(f"Use a new run directory: {run_dir}")

    data_dir = args.data_dir.resolve()
    config = build_config(run_dir.name, args.model, args.seed, args.epochs, data_dir, args.purpose)
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    for key, value in spec.items():
        if key in config and key not in ("max_epochs",) and config[key] != value:
            raise ValueError(f"Unsupported configuration change: {key}; this release implements the fixed protocol")
    config["configuration_sha256"] = sha256(args.config)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "status.json", {"status": "running"})
    log = (run_dir / "console.log").open("x", encoding="utf-8")
    class Tee:
        def __init__(self, terminal): self.terminal = terminal
        def write(self, value): self.terminal.write(value); log.write(value); log.flush()
        def flush(self): self.terminal.flush(); log.flush()
    sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)
    try:
        freeze_manifest_hash = None
        verified = json.loads((data_dir / "verification.json").read_text(encoding="utf-8"))
        data_hashes = {name: sha256(data_dir / name) for name in verified["processed_sha256"]}
        if data_hashes != verified["processed_sha256"]:
            raise ValueError("Processed data changed since acceptance")
        sources = ["src/gcn_layer.py", "src/gcn_model.py", "src/mlp_model.py",
                   "src/cora_data.py", "tools/train.py"]
        source_hashes = {}
        for name in sources:
            source_hashes[name] = sha256(root / name)
            target = run_dir / "code" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name"))
        (run_dir / "environment.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
        write_json(run_dir / "manifest.json", {
            "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "command": sys.orig_argv,
            "working_directory": str(Path.cwd()), "python": sys.version,
            "platform": platform.platform(), "torch": str(torch.__version__),
            "source_sha256": source_hashes, "data_sha256": data_hashes,
            "data_verification_sha256": sha256(data_dir / "verification.json"),
            "freeze_manifest_sha256": freeze_manifest_hash,
        })
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        X, A_hat, y, train_mask, val_mask, _ = load_cora_tensors(data_dir)
        model_class = MLP if args.model == "mlp" else GCN
        model = model_class(1433, 16, 7, dropout=config["dropout"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
            betas=tuple(config["adam_betas"]), eps=config["adam_eps"],
            amsgrad=False, foreach=False, weight_decay=0)

        def checkpoint(epoch):
            return {"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "config": config,
                "torch_rng_state": torch.get_rng_state()}

        fields = ["epoch", "train_ce", "train_l2", "train_objective", "train_accuracy",
            "eval_train_ce", "eval_train_l2", "eval_train_objective", "eval_train_accuracy",
            "val_ce", "val_l2", "val_objective", "val_accuracy", "epoch_seconds"]
        rows, history = [], []
        best_value, best_epoch = float("inf"), None
        stop_reason = "max_epochs_reached"
        started = time.perf_counter()
        print("run directory:", run_dir, flush=True)
        print("model:", args.model, flush=True)
        print("protocol: official_code_last; device: cpu; seed:", args.seed, flush=True)
        with (run_dir / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(args.epochs):
                epoch_started = time.perf_counter()
                model.train()
                optimizer.zero_grad()
                logits = forward_model(model, X, A_hat)
                ce, l2, loss = objective_parts(logits, y, train_mask, model, 5e-4)
                train_accuracy = accuracy(logits.detach(), y, train_mask)
                loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite or missing gradients")
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    eval_logits = forward_model(model, X, A_hat)
                    tce, tl2, ttotal = objective_parts(eval_logits, y, train_mask, model, 5e-4)
                    vce, vl2, vtotal = objective_parts(eval_logits, y, val_mask, model, 5e-4)
                    row = dict(zip(fields, [epoch + 1, ce.item(), l2.item(), loss.item(), train_accuracy,
                        tce.item(), tl2.item(), ttotal.item(), accuracy(eval_logits, y, train_mask),
                        vce.item(), vl2.item(), vtotal.item(), accuracy(eval_logits, y, val_mask),
                        time.perf_counter() - epoch_started]))
                if not all(np.isfinite(value) for value in row.values()):
                    raise FloatingPointError("Non-finite metrics")
                rows.append(row)
                history.append(row["val_objective"])
                writer.writerow(row)
                stream.flush()
                torch.save(checkpoint(epoch + 1), run_dir / "last.pt")
                if row["val_objective"] < best_value:
                    best_value, best_epoch = row["val_objective"], epoch + 1
                    torch.save(checkpoint(epoch + 1), run_dir / "best_val.pt")
                print(f"epoch={epoch+1:03d} train_loss={row['train_objective']:.6f} "
                    f"val_loss={row['val_objective']:.6f} val_acc={row['val_accuracy']:.4f}", flush=True)
                if should_stop(epoch, history, config["early_stopping_window"]):
                    stop_reason = "official_code_validation_window"
                    break

        # Reload the selected last checkpoint and compare the full eval output.
        loaded = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        reloaded = model_class(1433, 16, 7, dropout=config["dropout"])
        reloaded.load_state_dict(loaded["model_state_dict"])
        reloaded.eval()
        with torch.no_grad():
            restored_logits = forward_model(reloaded, X, A_hat)
        torch.testing.assert_close(restored_logits, eval_logits, rtol=1e-5, atol=1e-7)
        reload_error = (restored_logits - eval_logits).abs().max().item()
        save_curves(run_dir, rows, args.model)
        result = {"model": args.model, "protocol": config["protocol"], "epochs_completed": len(rows),
            "stop_reason": stop_reason, "selected_checkpoint": "last.pt", "selected_epoch": loaded["epoch"],
            "last_eval_train_objective": rows[-1]["eval_train_objective"],
            "last_eval_train_accuracy": rows[-1]["eval_train_accuracy"],
            "last_validation_objective": rows[-1]["val_objective"],
            "last_validation_accuracy": rows[-1]["val_accuracy"],
            "best_validation_objective": best_value, "best_validation_epoch": best_epoch,
            "checkpoint_reload_max_abs_error": reload_error,
            "elapsed_seconds": time.perf_counter() - started, "test_evaluation": False,
            "checkpoint_sha256": {name: sha256(run_dir / name) for name in ("last.pt", "best_val.pt")}}
        write_json(run_dir / "result.json", result)
        (run_dir / "record.md").write_text(
            f"# 公开版本运行：{run_dir.name}\n\n"
            f"模型：{args.model}；身份：{args.purpose}；seed={args.seed}；CPU；协议 official_code_last。\n\n"
            f"完成 {len(rows)} 轮，停止原因：{stop_reason}。选中 last.pt（第 {loaded['epoch']} 轮）；"
            f"best_val.pt（第 {best_epoch} 轮）只作诊断。\n\n"
            f"最后 eval train accuracy={rows[-1]['eval_train_accuracy']}，"
            f"validation accuracy={rows[-1]['val_accuracy']}，validation objective={rows[-1]['val_objective']}。\n\n"
            f"checkpoint 重载全图输出最大误差={reload_error}；未计算测试指标。\n\n"
            "逐轮 CE/L2/目标/准确率见 metrics.csv；曲线见 curves.png/svg；生效配置、实际命令、"
            "代码/数据哈希与环境见 config.json、manifest.json、code/、environment.txt。"
            "console.log 由终端 Tee-Object 保存。\n\n"
            + ("本次为正式冻结批次运行；测试评估尚待单独执行。" if args.purpose == "formal"
             else "这是公开版本的新运行，不计入 2026-10-29 的原 20 项统计。") + "另存可恢复备份尚待执行。\n",
            encoding="utf-8")
        write_json(run_dir / "status.json", {"status": "completed", "epochs_completed": len(rows)})
        print("epochs completed:", len(rows))
        print("stop reason:", stop_reason)
        print("last validation accuracy:", rows[-1]["val_accuracy"])
        print("checkpoint reload max error:", reload_error)
        print("Training and checkpoint reload checks passed.", flush=True)
    except BaseException as exc:
        write_json(run_dir / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error_type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()

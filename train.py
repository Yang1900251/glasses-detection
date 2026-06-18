import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import GlassesDataset, get_class_counts
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def percent(x):
    return f"{x * 100:.2f}%"


def build_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.03
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def build_dataloaders(args):
    train_transform, eval_transform = build_transforms(args.image_size)

    train_dataset = GlassesDataset(
        data_dir=args.data_dir,
        split="train",
        transform=train_transform
    )

    val_dataset = GlassesDataset(
        data_dir=args.data_dir,
        split="val",
        transform=eval_transform
    )

    test_dataset = GlassesDataset(
        data_dir=args.data_dir,
        split="test",
        transform=eval_transform
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_loss(train_dataset, device):
    counts = get_class_counts(train_dataset)

    count_0 = max(counts[0], 1)
    count_1 = max(counts[1], 1)
    total = count_0 + count_1

    weight_0 = total / (2.0 * count_0)
    weight_1 = total / (2.0 * count_1)

    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)

    print(f"训练集类别数量：no_glasses={counts[0]}, glasses={counts[1]}", flush=True)
    print(f"损失函数类别权重：{class_weights.detach().cpu().tolist()}", flush=True)

    return nn.CrossEntropyLoss(weight=class_weights)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_num += batch_size

    avg_loss = total_loss / total_num
    avg_acc = total_correct / total_num

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_num += batch_size

        for label_value in [0, 1]:
            mask = labels == label_value
            class_total[label_value] += mask.sum().item()
            class_correct[label_value] += ((preds == labels) & mask).sum().item()

    avg_loss = total_loss / total_num
    avg_acc = total_correct / total_num

    no_glasses_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0.0
    glasses_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0.0

    return {
        "loss": avg_loss,
        "acc": avg_acc,
        "no_glasses_acc": no_glasses_acc,
        "glasses_acc": glasses_acc
    }


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, args):
    checkpoint = {
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args)
    }

    torch.save(checkpoint, path)


def save_training_history_csv(history, results_dir):
    df = pd.DataFrame(history)
    csv_path = results_dir / "training_history.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def save_final_test_csv(test_metrics_with_percent, results_dir):
    df = pd.DataFrame([test_metrics_with_percent])
    csv_path = results_dir / "final_test_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def save_final_summary_csv(summary, results_dir):
    df = pd.DataFrame([summary])
    csv_path = results_dir / "final_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/mini_celeba")
    parser.add_argument("--save_dir", type=str, default="runs/glasses_resnet18")
    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "small_cnn"])
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)

    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_dir = save_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用设备：{device}", flush=True)

    gpu_name = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU：{gpu_name}", flush=True)

    print(f"早停策略：监控 val_acc，patience={args.patience}, min_delta={args.min_delta}", flush=True)
    print(f"CSV 结果保存目录：{results_dir}", flush=True)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders(args)

    print(f"训练集数量：{len(train_dataset)}", flush=True)
    print(f"验证集数量：{len(val_dataset)}", flush=True)
    print(f"测试集数量：{len(test_dataset)}", flush=True)

    train_class_counts = get_class_counts(train_dataset)

    model = build_model(
        model_name=args.model_name,
        num_classes=2,
        pretrained=args.pretrained
    ).to(device)

    criterion = build_loss(train_dataset, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    stopped_epoch = args.epochs

    best_model_path = save_dir / "best_model.pth"
    last_model_path = save_dir / "last_model.pth"

    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch [{epoch}/{args.epochs}] 开始 ==========", flush=True)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch}/{args.epochs}] 结果：\n"
            f"  train_loss          = {train_loss:.4f}\n"
            f"  train_acc           = {percent(train_acc)}\n"
            f"  val_loss            = {val_metrics['loss']:.4f}\n"
            f"  val_acc             = {percent(val_metrics['acc'])}\n"
            f"  val_no_glasses_acc  = {percent(val_metrics['no_glasses_acc'])}\n"
            f"  val_glasses_acc     = {percent(val_metrics['glasses_acc'])}\n"
            f"  best_val_acc        = {percent(best_val_acc)}\n"
            f"  lr                  = {current_lr:.8f}",
            flush=True
        )

        improved = val_metrics["acc"] > best_val_acc + args.min_delta

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_acc_percent": percent(train_acc),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_acc_percent": percent(val_metrics["acc"]),
            "val_no_glasses_acc": val_metrics["no_glasses_acc"],
            "val_no_glasses_acc_percent": percent(val_metrics["no_glasses_acc"]),
            "val_glasses_acc": val_metrics["glasses_acc"],
            "val_glasses_acc_percent": percent(val_metrics["glasses_acc"]),
            "best_val_acc_before_update": best_val_acc,
            "best_val_acc_before_update_percent": percent(best_val_acc),
            "is_best_epoch": improved,
            "early_stop_counter_before_update": early_stop_counter,
            "lr": current_lr
        }

        history.append(record)

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        save_training_history_csv(history, results_dir)

        if improved:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            early_stop_counter = 0

            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                args=args
            )

            print(f"val_acc 提升，当前最佳模型已保存：{best_model_path}", flush=True)
            print(f"当前最佳 val_acc：{percent(best_val_acc)}", flush=True)

        else:
            early_stop_counter += 1

            print(
                f"val_acc 未提升，早停计数：{early_stop_counter}/{args.patience}",
                flush=True
            )

        save_checkpoint(
            path=last_model_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            args=args
        )

        if early_stop_counter >= args.patience:
            stopped_epoch = epoch
            print(
                f"\n触发早停：连续 {args.patience} 个 epoch 验证集 val_acc 没有提升。",
                flush=True
            )
            print(f"实际停止在 Epoch [{stopped_epoch}/{args.epochs}]", flush=True)
            print(f"最佳 val_acc：{percent(best_val_acc)}", flush=True)
            break

    print("\n========== 开始测试最佳模型 ==========", flush=True)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(
        f"测试集结果：\n"
        f"  test_loss           = {test_metrics['loss']:.4f}\n"
        f"  test_acc            = {percent(test_metrics['acc'])}\n"
        f"  test_no_glasses_acc = {percent(test_metrics['no_glasses_acc'])}\n"
        f"  test_glasses_acc    = {percent(test_metrics['glasses_acc'])}",
        flush=True
    )

    test_metrics_with_percent = {
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_acc_percent": percent(best_val_acc),
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_acc_percent": percent(test_metrics["acc"]),
        "test_no_glasses_acc": test_metrics["no_glasses_acc"],
        "test_no_glasses_acc_percent": percent(test_metrics["no_glasses_acc"]),
        "test_glasses_acc": test_metrics["glasses_acc"],
        "test_glasses_acc_percent": percent(test_metrics["glasses_acc"]),
    }

    with open(save_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics_with_percent, f, ensure_ascii=False, indent=2)

    save_final_test_csv(test_metrics_with_percent, results_dir)

    final_summary = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "device": str(device),
        "gpu_name": gpu_name,
        "data_dir": args.data_dir,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "train_no_glasses_count": train_class_counts[0],
        "train_glasses_count": train_class_counts[1],
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs_setting": args.epochs,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "amp": use_amp,
        "early_stopping_monitor": "val_acc",
        "patience": args.patience,
        "min_delta": args.min_delta,
        "best_val_acc": best_val_acc,
        "best_val_acc_percent": percent(best_val_acc),
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_acc_percent": percent(test_metrics["acc"]),
        "test_no_glasses_acc": test_metrics["no_glasses_acc"],
        "test_no_glasses_acc_percent": percent(test_metrics["no_glasses_acc"]),
        "test_glasses_acc": test_metrics["glasses_acc"],
        "test_glasses_acc_percent": percent(test_metrics["glasses_acc"]),
        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path)
    }

    save_final_summary_csv(final_summary, results_dir)

    print(f"\n训练完成，最佳模型保存在：{best_model_path}", flush=True)
    print(f"训练过程 CSV：{results_dir / 'training_history.csv'}", flush=True)
    print(f"测试结果 CSV：{results_dir / 'final_test_metrics.csv'}", flush=True)
    print(f"最终汇总 CSV：{results_dir / 'final_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
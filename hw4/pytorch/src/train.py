import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from datasets import get_dataloader
from models import LogisticRegression
from tqdm.auto import trange, tqdm
from typing import Optional, Callable
from pathlib import Path


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def train_single_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    train_data: DataLoader,
    criterion: Callable,
    step: int,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
) -> int:
    """Train the model for a single epoch. Returns the updated step count."""
    raise NotImplementedError("Your code here. Hint: about 12 lines in the answer key")
    return step


def evaluate(
    model: nn.Module,
    val_data: DataLoader,
    criterion: Callable,
    step: int,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
):
    """Evaluate the model on the validation data. Returns the accuracy and loss."""
    model.eval()
    with torch.no_grad():
        total_loss, total_acc, total_samples = 0, 0, 0
        for im, label in tqdm(val_data, desc="validation loop", position=1, leave=False):
            im, label = im.to(device), label.to(device)
            out = model(im)
            total_loss += criterion(out, label).item()
            total_acc += accuracy(out, label) * len(label)
            total_samples += len(label)
        val_loss = total_loss / len(val_data)
        val_acc = total_acc / total_samples
        if writer is not None:
            writer.add_scalar("loss/val", val_loss, step)
            writer.add_scalar("acc/val", val_acc, step)
    return val_acc, val_loss


def train(
    model: nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    num_epochs: int,
    lr: float,
    momentum: float,
    device: torch.device,
    log_dir: Optional[Path] = None,
):
    if log_dir is not None:
        if not log_dir.is_dir():
            log_dir.mkdir(parents=True)
        writer = SummaryWriter(str(log_dir))
    else:
        writer = None
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    step = 0
    for epoch in trange(num_epochs, desc="Epochs", position=0):
        # Run a single training epoch
        step = train_single_epoch(model, optimizer, train_data, loss_fn, step, device, writer)

        # Evaluate on the validation data
        val_acc, val_loss = evaluate(model, val_data, loss_fn, step, device, writer)

        # Save a checkpoint of the model
        if log_dir is not None:
            info = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(info, log_dir / f"checkpoint.pt")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str.lower, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_train = get_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        train=True,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    data_val = get_dataloader(
        args.dataset,
        batch_size=1000,
        train=False,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if args.dataset == "mnist":
        model = LogisticRegression(28 * 28, 10)
    elif args.dataset == "cifar10":
        model = LogisticRegression(32 * 32 * 3, 10)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"{args.dataset}_{model.__class__.__name__}_lr{args.lr}_m{args.momentum}_bs{args.batch_size}"
    args.log_dir = args.log_dir / run_name

    train(model, data_train, data_val, args.num_epochs, args.lr, args.momentum, args.device, args.log_dir)

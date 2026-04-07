from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn


def get_device() -> torch.device:
    """选择可用的训练设备."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _synchronize_device(device: torch.device) -> None:
    """同步设备."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True, slots=True)
class TrainDeps:
    """训练依赖项."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: nn.CrossEntropyLoss
    train_loader: torch.utils.data.DataLoader
    device: torch.device


def train(
    deps: TrainDeps,
    epochs: int,
) -> None:
    """训练卷积神经网络."""
    deps.model.train()

    _synchronize_device(deps.device)
    for epoch in range(epochs):
        running_loss = torch.zeros((), device=deps.device)
        for inputs, labels in deps.train_loader:
            inputs = inputs.to(deps.device)  # noqa: PLW2901
            labels = labels.to(deps.device)  # noqa: PLW2901
            deps.optimizer.zero_grad()
            outputs = deps.model(inputs)
            loss = deps.loss_fn(outputs, labels)
            loss.backward()
            running_loss += loss.detach()
            deps.optimizer.step()
        avg_loss = (running_loss / len(deps.train_loader)).item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    _synchronize_device(deps.device)


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """测试卷积神经网络."""
    model.eval()

    correct = torch.zeros((), dtype=torch.long, device=device)
    total = torch.zeros((), dtype=torch.long, device=device)

    _synchronize_device(device)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)  # noqa: PLW2901
            labels = labels.to(device)  # noqa: PLW2901
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().detach()
    _synchronize_device(device)

    print(f"Accuracy: {correct.item() / total.item():.4f}")


def _get_next_checkpoint_path(path: str) -> Path:
    """返回 checkpoint/ 下一个可用的 checkpoints 路径."""
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    stem = Path(path).stem
    candidate = checkpoint_dir / f"{stem}.pt"

    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = checkpoint_dir / f"{stem}_{index}.pt"
        if not candidate.exists():
            return candidate
        index += 1


def save_checkpoint(model: nn.Module, path: str) -> None:
    """保存 Module 到 checkpoint/ 下一个可用的 checkpoints 路径."""
    target_path = _get_next_checkpoint_path(path)
    torch.save(model.state_dict(), target_path)

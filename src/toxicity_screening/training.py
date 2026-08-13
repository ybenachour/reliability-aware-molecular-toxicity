from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .losses import masked_binary_cross_entropy_with_logits
from .metrics import binary_metrics


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float]]
    checkpoint_path: str | None


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _binary_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for features, target in loader:
        features = features.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, target)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels.append(target.detach().cpu().numpy())
        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(labels), np.concatenate(probabilities)


def train_binary_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    positive_weight: float | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> TrainingResult:
    device_obj = resolve_device(device)
    model.to(device_obj)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(positive_weight), device=device_obj) if positive_weight is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, patience // 3))
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = _binary_epoch(model, train_loader, device_obj, criterion, optimizer, gradient_clip_norm)
        with torch.no_grad():
            validation_loss, y_val, p_val = _binary_epoch(
                model, validation_loader, device_obj, criterion, None, gradient_clip_norm
            )
        scheduler.step(validation_loss)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_pr_auc": float(binary_metrics(y_val.astype(int), p_val)["pr_auc"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = None
    if checkpoint_path is not None:
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "best_epoch": best_epoch, "history": history}, destination)
        checkpoint = str(destination)
    return TrainingResult(best_epoch, best_loss, history, checkpoint)


def train_masked_multitask_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    positive_weights: torch.Tensor | None = None,
    task_weights: torch.Tensor | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> TrainingResult:
    device_obj = resolve_device(device)
    model.to(device_obj)
    if positive_weights is not None:
        positive_weights = positive_weights.to(device_obj)
    if task_weights is not None:
        task_weights = task_weights.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, patience // 3))
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    wait = 0
    history: list[dict[str, float]] = []

    def run(loader: DataLoader, training: bool) -> float:
        model.train(training)
        losses: list[float] = []
        for features, labels, mask in loader:
            features, labels, mask = features.to(device_obj), labels.to(device_obj), mask.to(device_obj)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = masked_binary_cross_entropy_with_logits(
                logits,
                labels,
                mask,
                positive_weights=positive_weights,
                task_weights=task_weights,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    for epoch in range(1, epochs + 1):
        train_loss = run(train_loader, True)
        with torch.no_grad():
            validation_loss = run(validation_loader, False)
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("Multitask training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = None
    if checkpoint_path is not None:
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "best_epoch": best_epoch, "history": history}, destination)
        checkpoint = str(destination)
    return TrainingResult(best_epoch, best_loss, history, checkpoint)


def train_graph_binary_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    *,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    positive_weight: float | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> TrainingResult:
    device_obj = resolve_device(device)
    model.to(device_obj)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(positive_weight), device=device_obj) if positive_weight is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, patience // 3))
    best_loss, best_epoch, wait = float("inf"), 0, 0
    best_state = None
    history: list[dict[str, float]] = []

    def run(loader, training: bool) -> float:
        model.train(training)
        losses = []
        for batch in loader:
            batch = batch.to(device_obj)
            target = batch.y.view(-1).float()
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, target)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    for epoch in range(1, epochs + 1):
        train_loss = run(train_loader, True)
        with torch.no_grad():
            val_loss = run(validation_loader, False)
        scheduler.step(val_loss)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": val_loss, "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, best_state, wait = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("Graph training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = None
    if checkpoint_path is not None:
        destination = Path(checkpoint_path); destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "best_epoch": best_epoch, "history": history}, destination)
        checkpoint = str(destination)
    return TrainingResult(best_epoch, best_loss, history, checkpoint)


def train_graph_multitask_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    *,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    positive_weights: torch.Tensor | None = None,
    task_weights: torch.Tensor | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> TrainingResult:
    device_obj = resolve_device(device)
    model.to(device_obj)
    positive_weights = positive_weights.to(device_obj) if positive_weights is not None else None
    task_weights = task_weights.to(device_obj) if task_weights is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, patience // 3))
    best_loss, best_epoch, wait = float("inf"), 0, 0
    best_state = None
    history: list[dict[str, float]] = []

    def run(loader, training: bool) -> float:
        model.train(training)
        losses = []
        for batch in loader:
            batch = batch.to(device_obj)
            labels = batch.y.view(-1, batch.y.shape[-1]).float()
            mask = batch.mask.view(-1, batch.mask.shape[-1]).float()
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = masked_binary_cross_entropy_with_logits(logits, labels, mask, positive_weights=positive_weights, task_weights=task_weights)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    for epoch in range(1, epochs + 1):
        train_loss = run(train_loader, True)
        with torch.no_grad():
            val_loss = run(validation_loader, False)
        scheduler.step(val_loss)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": val_loss, "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, best_state, wait = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("Multitask graph training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = None
    if checkpoint_path is not None:
        destination = Path(checkpoint_path); destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "best_epoch": best_epoch, "history": history}, destination)
        checkpoint = str(destination)
    return TrainingResult(best_epoch, best_loss, history, checkpoint)

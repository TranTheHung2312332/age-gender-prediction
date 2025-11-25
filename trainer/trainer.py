import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device="cuda",
        alpha=1.0,
        beta=0.2,
        early_stop_patience=10
    ):
        self.model = model.to(device)
        self.criterion = criterion         # MultiTaskLoss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stop = early_stop_patience

        self.best_val_loss = float("inf")
        self.no_improve_epochs = 0
        self.history = {
            "train_age_loss": [], "train_gender_loss": [],
            "val_age_loss": [],   "val_gender_loss": []
        }

    def run_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        age_loss_sum = 0
        gender_loss_sum = 0

        n_batches = 0

        loop = tqdm(loader, desc="Train" if train else "Valid", leave=False)

        for imgs, targets in loop:
            if imgs is None:
                continue

            imgs = imgs.to(self.device)
            age_gt = targets["age"].to(self.device)
            gender_gt = targets["gender"].float().to(self.device)

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                age_pred, gender_logit = self.model(imgs)

                loss, l_age, l_gender = self.criterion(
                    age_pred, age_gt,
                    gender_logit, gender_gt
                )

                if train:
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            age_loss_sum += l_age.item()
            gender_loss_sum += l_gender.item()
            n_batches += 1

        return (
            total_loss / n_batches,
            age_loss_sum / n_batches,
            gender_loss_sum / n_batches
        )

    def fit(self, train_loader, val_loader, n_epochs=30):
        print("Training started...")

        for epoch in range(1, n_epochs + 1):
            print(f"\n===== Epoch {epoch}/{n_epochs} =====")

            train_loss, train_age, train_gender = self.run_epoch(train_loader, train=True)
            val_loss,   val_age,   val_gender   = self.run_epoch(val_loader,   train=False)

            # Save history
            self.history["train_age_loss"].append(train_age)
            self.history["train_gender_loss"].append(train_gender)
            self.history["val_age_loss"].append(val_age)
            self.history["val_gender_loss"].append(val_gender)

            print(
                f"Train: loss={train_loss:.4f}, age={train_age:.4f}, gender={train_gender:.4f}\n"
                f"Valid: loss={val_loss:.4f}, age={val_age:.4f}, gender={val_gender:.4f}"
            )

            # Scheduler update
            if self.scheduler:
                self.scheduler.step()

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improve_epochs = 0
                torch.save(self.model.state_dict(), "best_model.pth")
                print("→ Saved new best model.")
            else:
                self.no_improve_epochs += 1
                print(f"No improvement for {self.no_improve_epochs} epochs.")

            if self.no_improve_epochs >= self.early_stop:
                print("Early stopping triggered.")
                break

        print("Training finished.")

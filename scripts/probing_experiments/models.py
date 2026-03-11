"""
Probing models for overflow detection: sklearn LinearProbe and PyTorch probes
(LinearProbeTorch, MLPProbeTorch, MLPSCLProbeTorch).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import set_seed


# -----------------------------------------------------------------------------
# Sklearn linear probe
# -----------------------------------------------------------------------------


class LinearProbe(BaseEstimator, ClassifierMixin):
    """Linear probe (Logistic Regression)."""

    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# -----------------------------------------------------------------------------
# PyTorch linear probe
# -----------------------------------------------------------------------------


class LinearProbeTorch(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        l2_lambda=1.0,
        l1_lambda=0.0,
        epochs=10,
        batch_size=256,
        normalize=True,
        device="cuda",
        random_state=42,
        verbose=True,
        early_stopping_patience=None,
        hidden_dim=None,
    ):
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.hidden_dim = hidden_dim
        self.model = None
        self.scaler = None
        self.classes_ = None
        self.history = {
            "train_loss": [], "train_bce_loss": [], "train_l2_reg": [], "train_l1_reg": [],
            "val_loss": [], "val_bce_loss": [], "val_l2_reg": [], "val_l1_reg": [], "val_auc": [],
        }
        self.best_val_auc_ = -np.inf
        self.best_epoch_ = 0

    def _create_model(self, input_dim, positive_class_proportion=0.5):
        model = nn.Linear(input_dim, 1)
        nn.init.zeros_(model.weight)
        p = np.clip(positive_class_proportion, 0.01, 0.99)
        nn.init.constant_(model.bias, np.log(p / (1 - p)))
        return model

    def _create_dataloader(self, X, y):
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def fit(self, X, y, X_val=None, y_val=None):
        set_seed(self.random_state)
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        X, y = np.asarray(X), np.asarray(y)
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val = np.asarray(X_val.cpu().numpy() if isinstance(X_val, torch.Tensor) else X_val)
            y_val = np.asarray(y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val)
        self.classes_ = np.unique(y)
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            if has_validation:
                X_val = self.scaler.transform(X_val)
        pos = np.mean(y_val if has_validation else y)
        input_dim = X.shape[1]
        self.model = self._create_model(input_dim, pos).to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)
        train_loader = self._create_dataloader(X, y)
        loss_fn = nn.BCEWithLogitsLoss()
        best_val_auc, best_model_state, patience_counter = -np.inf, None, 0
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = epoch_bce = epoch_l2 = epoch_l1 = 0.0
            it = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") if self.verbose else train_loader
            for X_batch, y_batch in it:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                batch_losses = {}

                def closure():
                    optimizer.zero_grad()
                    out = self.model(X_batch).squeeze(-1)
                    bce = loss_fn(out, y_batch)
                    loss = bce
                    batch_losses["bce"] = bce.item()
                    batch_losses["l2"] = batch_losses["l1"] = 0.0
                    if self.l2_lambda > 0:
                        l2_reg = sum(p.pow(2).sum() for n, p in self.model.named_parameters() if "bias" not in n)
                        n_params = sum(p.numel() for n, p in self.model.named_parameters() if "bias" not in n)
                        loss += 0.5 * self.l2_lambda * l2_reg / n_params
                        batch_losses["l2"] = (0.5 * self.l2_lambda * l2_reg / n_params).item()
                    if self.l1_lambda > 0:
                        l1_reg = sum(p.abs().sum() for n, p in self.model.named_parameters() if "bias" not in n)
                        n_params = sum(p.numel() for n, p in self.model.named_parameters() if "bias" not in n)
                        loss += self.l1_lambda * l1_reg / n_params
                        batch_losses["l1"] = (self.l1_lambda * l1_reg / n_params).item()
                    loss.backward()
                    return loss

                loss = closure()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_bce += batch_losses.get("bce", 0)
                epoch_l2 += batch_losses.get("l2", 0)
                epoch_l1 += batch_losses.get("l1", 0)
            n_batches = len(train_loader)
            self.history["train_loss"].append(epoch_loss / n_batches)
            self.history["train_bce_loss"].append(epoch_bce / n_batches)
            self.history["train_l2_reg"].append(epoch_l2 / n_batches)
            self.history["train_l1_reg"].append(epoch_l1 / n_batches)
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val).to(self.device)
                    y_v = torch.FloatTensor(y_val).to(self.device)
                    logits = self.model(X_v).squeeze(-1)
                    val_bce = loss_fn(logits, y_v).item()
                    val_proba = torch.sigmoid(logits).cpu().numpy()
                    val_auc = roc_auc_score(y_val, val_proba)
                self.history["val_auc"].append(val_auc)
                self.model.train()
                if val_auc > best_val_auc:
                    best_val_auc, best_model_state = val_auc, {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    self.best_val_auc_, self.best_epoch_ = best_val_auc, epoch + 1
                else:
                    patience_counter += 1
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - Val AUC: {val_auc:.4f} {'*' if val_auc == best_val_auc else ''}")
                if self.early_stopping_patience is not None and patience_counter >= self.early_stopping_patience:
                    break
        if has_validation and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X = np.asarray(X.cpu().numpy() if isinstance(X, torch.Tensor) else X)
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X).to(self.device)).squeeze(-1)
            proba_pos = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# -----------------------------------------------------------------------------
# PyTorch MLP probe
# -----------------------------------------------------------------------------


class MLPProbeTorch(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        l2_lambda=1.0,
        l1_lambda=0.0,
        epochs=10,
        batch_size=256,
        normalize=True,
        device="cuda",
        random_state=42,
        verbose=True,
        early_stopping_patience=None,
        hidden_dim=1024,
    ):
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.hidden_dim = hidden_dim
        self.model = None
        self.scaler = None
        self.classes_ = None
        self.history = {
            "train_loss": [], "train_bce_loss": [], "train_l2_reg": [], "train_l1_reg": [],
            "val_loss": [], "val_bce_loss": [], "val_l2_reg": [], "val_l1_reg": [], "val_auc": [],
        }
        self.best_val_auc_ = -np.inf
        self.best_epoch_ = 0

    def _create_model(self, input_dim, positive_class_proportion=0.5):
        model = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.xavier_uniform_(model[1].weight, 0.8)
        nn.init.zeros_(model[1].bias)
        nn.init.zeros_(model[5].weight)
        p = np.clip(positive_class_proportion, 0.01, 0.99)
        nn.init.constant_(model[5].bias, np.log(p / (1 - p)))
        return model

    def _create_dataloader(self, X, y):
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def fit(self, X, y, X_val=None, y_val=None):
        set_seed(self.random_state)
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        X, y = np.asarray(X), np.asarray(y)
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val = np.asarray(X_val.cpu().numpy() if isinstance(X_val, torch.Tensor) else X_val)
            y_val = np.asarray(y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val)
        self.classes_ = np.unique(y)
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            if has_validation:
                X_val = self.scaler.transform(X_val)
        pos = np.mean(y_val if has_validation else y)
        input_dim = X.shape[1]
        self.model = self._create_model(input_dim, pos).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        train_loader = self._create_dataloader(X, y)
        loss_fn = nn.BCEWithLogitsLoss()
        best_val_auc, best_model_state, patience_counter = -np.inf, None, 0
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = epoch_bce = epoch_l2 = epoch_l1 = 0.0
            it = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") if self.verbose else train_loader
            for X_batch, y_batch in it:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                batch_losses = {}

                def closure():
                    optimizer.zero_grad()
                    out = self.model(X_batch).squeeze(-1)
                    bce = loss_fn(out, y_batch)
                    loss = bce
                    batch_losses["bce"] = bce.item()
                    batch_losses["l2"] = batch_losses["l1"] = 0.0
                    if self.l2_lambda > 0:
                        l2_reg = sum(p.pow(2).sum() for n, p in self.model.named_parameters() if "bias" not in n)
                        n_params = sum(p.numel() for n, p in self.model.named_parameters() if "bias" not in n)
                        loss += 0.5 * self.l2_lambda * l2_reg / n_params
                        batch_losses["l2"] = (0.5 * self.l2_lambda * l2_reg / n_params).item()
                    if self.l1_lambda > 0:
                        l1_reg = sum(p.abs().sum() for n, p in self.model.named_parameters() if "bias" not in n)
                        n_params = sum(p.numel() for n, p in self.model.named_parameters() if "bias" not in n)
                        loss += self.l1_lambda * l1_reg / n_params
                        batch_losses["l1"] = (self.l1_lambda * l1_reg / n_params).item()
                    loss.backward()
                    return loss

                loss = closure()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_bce += batch_losses.get("bce", 0)
                epoch_l2 += batch_losses.get("l2", 0)
                epoch_l1 += batch_losses.get("l1", 0)
            n_batches = len(train_loader)
            self.history["train_loss"].append(epoch_loss / n_batches)
            self.history["train_bce_loss"].append(epoch_bce / n_batches)
            self.history["train_l2_reg"].append(epoch_l2 / n_batches)
            self.history["train_l1_reg"].append(epoch_l1 / n_batches)
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val).to(self.device)
                    y_v = torch.FloatTensor(y_val).to(self.device)
                    logits = self.model(X_v).squeeze(-1)
                    val_bce = loss_fn(logits, y_v).item()
                    val_proba = torch.sigmoid(logits).cpu().numpy()
                    val_auc = roc_auc_score(y_val, val_proba)
                self.history["val_auc"].append(val_auc)
                self.model.train()
                if val_auc > best_val_auc:
                    best_val_auc, best_model_state = val_auc, {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    self.best_val_auc_, self.best_epoch_ = best_val_auc, epoch + 1
                else:
                    patience_counter += 1
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - Val AUC: {val_auc:.4f} {'*' if val_auc == best_val_auc else ''}")
                if self.early_stopping_patience is not None and patience_counter >= self.early_stopping_patience:
                    break
        if has_validation and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X = np.asarray(X.cpu().numpy() if isinstance(X, torch.Tensor) else X)
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X).to(self.device)).squeeze(-1)
            proba_pos = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# -----------------------------------------------------------------------------
# Supervised contrastive loss and MLP-SCL probe
# -----------------------------------------------------------------------------


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss: same-class samples pulled together, different-class pushed apart."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size, device=device).view(-1, 1), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss


class MLPSCLModel(nn.Module):
    """MLP with backbone (for contrastive features) and classifier head."""

    def __init__(self, input_dim, hidden_dim, positive_class_proportion=0.5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
        )
        self.classifier_head = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.backbone[1].weight, 0.8)
        nn.init.zeros_(self.backbone[1].bias)
        nn.init.zeros_(self.classifier_head.weight)
        p = np.clip(positive_class_proportion, 0.01, 0.99)
        nn.init.constant_(self.classifier_head.bias, np.log(p / (1 - p)))

    def forward(self, x):
        hidden = self.backbone(x)
        logits = self.classifier_head(hidden).squeeze(-1)
        return logits, hidden


class MLPSCLProbeTorch(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        l2_lambda=1.0,
        l1_lambda=0.0,
        epochs=10,
        batch_size=256,
        normalize=True,
        device="cuda",
        random_state=42,
        verbose=True,
        early_stopping_patience=None,
        hidden_dim=1024,
        contrastive_weight=0.0,
        contrastive_temperature=0.07,
    ):
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.hidden_dim = hidden_dim
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.model = None
        self.scaler = None
        self.classes_ = None
        self.contrastive_loss_fn = None
        self.history = {
            "train_loss": [], "train_bce_loss": [], "train_contrastive_loss": [], "train_l2_reg": [], "train_l1_reg": [],
            "val_loss": [], "val_bce_loss": [], "val_contrastive_loss": [], "val_l2_reg": [], "val_l1_reg": [], "val_auc": [],
        }
        self.best_val_auc_ = -np.inf
        self.best_epoch_ = 0

    def _create_model(self, input_dim, positive_class_proportion=0.5):
        return MLPSCLModel(input_dim, self.hidden_dim, positive_class_proportion)

    def _create_dataloader(self, X, y):
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def fit(self, X, y, X_val=None, y_val=None):
        set_seed(self.random_state)
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        X, y = np.asarray(X), np.asarray(y)
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val = np.asarray(X_val.cpu().numpy() if isinstance(X_val, torch.Tensor) else X_val)
            y_val = np.asarray(y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val)
        self.classes_ = np.unique(y)
        if self.normalize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            if has_validation:
                X_val = self.scaler.transform(X_val)
        pos = np.mean(y_val if has_validation else y)
        input_dim = X.shape[1]
        self.model = self._create_model(input_dim, pos).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        if self.contrastive_weight > 0:
            self.contrastive_loss_fn = SupervisedContrastiveLoss(
                temperature=self.contrastive_temperature,
                base_temperature=self.contrastive_temperature,
            ).to(self.device)
        train_loader = self._create_dataloader(X, y)
        loss_fn = nn.BCEWithLogitsLoss()
        best_val_auc, best_model_state, patience_counter = -np.inf, None, 0
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = epoch_bce = epoch_cl = epoch_l2 = epoch_l1 = 0.0
            it = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") if self.verbose else train_loader
            for X_batch, y_batch in it:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                batch_losses = {"bce": 0.0, "contrastive": 0.0, "l2": 0.0, "l1": 0.0}

                def closure():
                    optimizer.zero_grad()
                    logits, hidden = self.model(X_batch)
                    bce = loss_fn(logits, y_batch)
                    loss = bce
                    batch_losses["bce"] = bce.item()
                    if self.contrastive_weight > 0 and self.contrastive_loss_fn is not None:
                        feats = torch.nn.functional.normalize(hidden, p=2, dim=1)
                        y_long = y_batch.long() if y_batch.dtype != torch.long else y_batch
                        cl_loss = self.contrastive_loss_fn(feats, y_long)
                        loss = loss + self.contrastive_weight * cl_loss
                        batch_losses["contrastive"] = cl_loss.item()
                    if self.l2_lambda > 0:
                        l2_reg = sum(p.pow(2).sum() for n, p in self.model.named_parameters() if "bias" not in n)
                        n_params = sum(p.numel() for n, p in self.model.named_parameters() if "bias" not in n)
                        loss += 0.5 * self.l2_lambda * l2_reg / n_params
                        batch_losses["l2"] = (0.5 * self.l2_lambda * l2_reg / n_params).item()
                    if self.l1_lambda > 0:
                        l1_reg = sum(p.abs().sum() for n, p in self.model.named_parameters() if "bias" not in n)
                        n_params = sum(p.numel() for n, p in self.model.named_parameters() if "bias" not in n)
                        loss += self.l1_lambda * l1_reg / n_params
                        batch_losses["l1"] = (self.l1_lambda * l1_reg / n_params).item()
                    loss.backward()
                    return loss

                loss = closure()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_bce += batch_losses["bce"]
                epoch_cl += batch_losses["contrastive"]
                epoch_l2 += batch_losses["l2"]
                epoch_l1 += batch_losses["l1"]
            n_batches = len(train_loader)
            self.history["train_bce_loss"].append(epoch_bce / n_batches)
            self.history["train_contrastive_loss"].append(epoch_cl / n_batches)
            self.history["train_l2_reg"].append(epoch_l2 / n_batches)
            self.history["train_l1_reg"].append(epoch_l1 / n_batches)
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val).to(self.device)
                    y_v = torch.FloatTensor(y_val).to(self.device)
                    val_logits, val_hidden = self.model(X_v)
                    val_bce = loss_fn(val_logits, y_v).item()
                    val_contrastive = 0.0
                    if self.contrastive_weight > 0 and self.contrastive_loss_fn is not None:
                        val_feats = torch.nn.functional.normalize(val_hidden, p=2, dim=1)
                        val_contrastive = self.contrastive_loss_fn(
                            val_feats, y_v.long() if y_v.dtype != torch.long else y_v
                        ).item()
                    val_proba = torch.sigmoid(val_logits).cpu().numpy()
                    val_auc = roc_auc_score(y_val, val_proba)
                self.history["val_auc"].append(val_auc)
                self.model.train()
                if val_auc > best_val_auc:
                    best_val_auc, best_model_state = val_auc, {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    self.best_val_auc_, self.best_epoch_ = best_val_auc, epoch + 1
                else:
                    patience_counter += 1
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - Val AUC: {val_auc:.4f} {'*' if val_auc == best_val_auc else ''}")
                if self.early_stopping_patience is not None and patience_counter >= self.early_stopping_patience:
                    break
        if has_validation and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        X = np.asarray(X.cpu().numpy() if isinstance(X, torch.Tensor) else X)
        if self.normalize and self.scaler is not None:
            X = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(torch.FloatTensor(X).to(self.device))
            proba_pos = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

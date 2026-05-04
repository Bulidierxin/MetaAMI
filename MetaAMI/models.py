from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def make_sklearn_model(name: str, random_state=42, n_jobs=-1):
    name = name.lower()
    if name == "lr":
        return LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=random_state)
    if name == "svm":
        return SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=random_state)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, max_features="sqrt", class_weight="balanced",
            min_samples_leaf=1, min_samples_split=2, n_jobs=n_jobs, random_state=random_state
        )
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover
            raise ImportError("Install xgboost or use: pip install -e .[full]") from exc
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=100, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=1.0, min_child_weight=1, reg_lambda=1.0,
            n_jobs=n_jobs, random_state=random_state, scale_pos_weight=5.0,
        )
    raise ValueError(f"Unknown sklearn model: {name}")


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install torch or use: pip install -e .[full]") from exc


def fit_predict_torch(model_name, X_train, y_train, X_test, *, random_state=42, epochs=40, batch_size=128, lr=1e-3, patience=8):
    """Small PyTorch trainers for MLP-like baseline models.

    The package keeps these models lightweight so the full pipeline can run from one command.
    Increase epochs in config for final paper-level experiments.
    """
    torch, nn, DataLoader, TensorDataset = _require_torch()
    import random
    random.seed(random_state); np.random.seed(random_state); torch.manual_seed(random_state); torch.cuda.manual_seed_all(random_state)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    class MLP(nn.Module):
        def __init__(self, in_dim, hidden=(128, 64), dropout=0.1):
            super().__init__()
            layers, prev = [], in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers += [nn.Linear(prev, 1)]
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x).squeeze(-1)

    class AE(nn.Module):
        def __init__(self, in_dim, latent=16, hidden=(256, 128, 64), dropout=0.1):
            super().__init__()
            enc, prev = [], in_dim
            for h in hidden:
                enc += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            self.encoder = nn.Sequential(*enc)
            self.to_latent = nn.Linear(prev, latent)
            dec, prev = [], latent
            for h in reversed(hidden):
                dec += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            dec += [nn.Linear(prev, in_dim)]
            self.decoder = nn.Sequential(*dec)
        def encode(self, x): return self.to_latent(self.encoder(x))
        def forward(self, x):
            z = self.encode(x)
            return self.decoder(z), z

    class AEMLP(nn.Module):
        def __init__(self, ae, latent=16):
            super().__init__()
            self.ae = ae
            self.head = MLP(latent, hidden=(64, 32))
        def forward(self, x): return self.head(self.ae.encode(x))

    class TransformerTab(nn.Module):
        def __init__(self, in_dim, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
            super().__init__()
            self.in_dim = in_dim
            self.feature_embed = nn.Linear(1, d_model)
            self.feat_emb = nn.Embedding(in_dim, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=2*d_model, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, 1)
        def forward(self, x):
            b, f = x.shape
            h = self.feature_embed(x.unsqueeze(-1))
            h = h + self.feat_emb(torch.arange(f, device=x.device)).unsqueeze(0)
            return self.head(self.encoder(h).mean(dim=1)).squeeze(-1)

    in_dim = X_train.shape[1]
    if model_name == "mlp":
        model = MLP(in_dim).to(dev)
    elif model_name == "ae_mlp":
        ae = AE(in_dim).to(dev)
        ae_opt = torch.optim.Adam(ae.parameters(), lr=5e-4)
        mse = nn.MSELoss()
        ae_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=max(batch_size, 256), shuffle=True)
        for _ in range(20):
            for (xb,) in ae_loader:
                xb = xb.to(dev); xh, _ = ae(xb); loss = mse(xh, xb)
                ae_opt.zero_grad(); loss.backward(); ae_opt.step()
        model = AEMLP(ae).to(dev)
    elif model_name == "ot_mlp":
        # Lightweight replacement for the uploaded OT-MLP: same MLP head; OT weighting can be extended later.
        model = MLP(in_dim, hidden=(64, 32)).to(dev)
    elif model_name == "transformer":
        model = TransformerTab(in_dim).to(dev)
    else:
        raise ValueError(f"Unknown torch model: {model_name}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.zeros(len(X_test))), batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); probs = []
    with torch.no_grad():
        for xb, _ in eval_loader:
            probs.append(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
    return np.concatenate(probs)


def fit_predict_proba(model_name, X_train, y_train, X_test, *, random_state=42, n_jobs=-1, epochs=40, batch_size=128, lr=1e-3, patience=8):
    model_name = model_name.lower()
    if model_name in {"lr", "svm", "rf", "xgb"}:
        clf = make_sklearn_model(model_name, random_state=random_state, n_jobs=n_jobs)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]
    if model_name in {"mlp", "ae_mlp", "ot_mlp", "transformer"}:
        return fit_predict_torch(model_name, X_train, y_train, X_test, random_state=random_state, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience)
    raise ValueError(f"Unknown model: {model_name}")


def make_meta_model(name="svm_rbf", random_state=42):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    name = name.lower()
    if name == "svm_rbf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=0.01, gamma=0.001, class_weight="balanced", probability=True, random_state=random_state)),
        ])
    if name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)),
        ])
    raise ValueError("meta_model must be svm_rbf or logistic")

import os
import math
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import shap

SEED = 123456
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but not available")
DEVICE = torch.device("cuda:0")
torch.cuda.set_device(0)

class TimeWindowDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, seq_len=12, dropna_strategy="median", outlier_z=3.0):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.df = df.reset_index(drop=True).copy()
        if dropna_strategy == "median":
            med = self.df[feature_cols + target_cols].median()
            self.df[feature_cols + target_cols] = self.df[feature_cols + target_cols].fillna(med)
        elif dropna_strategy == "ffill":
            self.df[feature_cols + target_cols] = self.df[feature_cols + target_cols].fillna(method="ffill").fillna(method="bfill")
        if outlier_z is not None and outlier_z > 0:
            arr = self.df[feature_cols].values
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0) + 1e-9
            z = np.abs((arr - mean) / std)
            mask = (z < outlier_z).all(axis=1)
            self.df = self.df.loc[mask].reset_index(drop=True)
        self.X, self.y = self._construct_sequences()

    def _construct_sequences(self):
        Xs, Ys = [], []
        n = len(self.df)
        for end in range(self.seq_len, n):
            start = end - self.seq_len
            xw = self.df.loc[start:end-1, self.feature_cols].values.astype(np.float32)
            yw = self.df.loc[end, self.target_cols].values.astype(np.float32)
            Xs.append(xw)
            Ys.append(yw)
        return np.stack(Xs, axis=0), np.stack(Ys, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), stride=1):
        super().__init__()
        padding = (kernel[0]//2, kernel[1]//2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch or stride != 1:
            self.down = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride), nn.BatchNorm2d(out_ch))
        else:
            self.down = nn.Identity()

    def forward(self, x):
        r = self.down(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + r
        x = self.act(x)
        return x

class ResCNNEncoder(nn.Module):
    def __init__(self, in_channels=1, channels=(32,64,128)):
        super().__init__()
        layers = []
        cin = in_channels
        for ch in channels:
            layers.append(ResidualConvBlock(cin, ch, kernel=(3,3)))
            cin = ch
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class ResCNNLSTM(nn.Module):
    def __init__(self, n_features:int, seq_len:int, lstm_hidden:int=128, lstm_layers:int=2, out_dim:int=2, dropout:float=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.encoder = ResCNNEncoder(in_channels=1, channels=(32,64,128))
        self.global_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm_input_size = 128
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(lstm_hidden, 64), nn.Tanh(), nn.Dropout(dropout), nn.Linear(64, out_dim), nn.Tanh())
    def forward(self, x):
        b, s, f = x.shape
        x2 = x.permute(0,2,1).unsqueeze(1)
        x_enc = self.encoder(x2)
        x_p = self.global_pool(x_enc)
        x_p = x_p.squeeze(2)
        x_seq = x_p.permute(0,2,1)
        lstm_out, _ = self.lstm(x_seq)
        out = lstm_out[:, -1, :]
        out = self.head(out)
        return out

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total_loss / max(1, n)

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
    return total_loss / max(1, n)

def fit_cv(df: pd.DataFrame, feature_cols, target_cols, seq_len=12, n_splits=6, batch_size=32, epochs=500, lr=1e-3, patience=10, model_dir=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    Xy_dataset = TimeWindowDataset(df, feature_cols, target_cols, seq_len=seq_len)
    indices = np.arange(len(Xy_dataset))
    fold = 0
    histories = []
    models = []
    for train_idx, val_idx in kf.split(indices):
        fold += 1
        train_subset = Subset(Xy_dataset, train_idx)
        val_subset = Subset(Xy_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)
        model = ResCNNLSTM(n_features=len(feature_cols), seq_len=seq_len, lstm_hidden=128, lstm_layers=2, out_dim=len(target_cols), dropout=0.2).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        best_val = float("inf")
        wait = 0
        history = {"train": [], "val": []}
        for ep in range(1, epochs+1):
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss = eval_one_epoch(model, val_loader, criterion, DEVICE)
            history["train"].append(tr_loss)
            history["val"].append(val_loss)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                wait = 0
                if model_dir:
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(model_dir, f"best_fold{fold}.pt"))
                    best_state = model.state_dict()
            else:
                wait += 1
            if wait >= patience:
                break
        histories.append(history)
        model.load_state_dict(best_state)
        models.append(model.cpu())
    return histories, models, Xy_dataset

def predict_with_model(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(next(model.parameters()).device)
        pred = model(X_t).cpu().numpy()
    return pred

def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred, multioutput='raw_values') if y_true.shape[1] > 1 else r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values')) if y_true.shape[1] > 1 else math.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

if __name__ == "__main__":
    csv_path = "dataset.csv"
    df = pd.read_csv(csv_path)
    feature_cols = ["precipitation","wind_speed","radiation","sunshine","air_temp","gw_depth"]
    target_cols = ["theta_depth1","theta_depth2","rwu"]
    seq_len = 12
    for c in feature_cols + target_cols:
        df[c] = df[c].astype(float)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    tmp_ds = TimeWindowDataset(df, feature_cols, target_cols, seq_len=seq_len)
    X_raw = tmp_ds.X
    y_raw = tmp_ds.y
    nsamples = X_raw.shape[0]
    X_flat = X_raw.reshape(nsamples, -1)
    scaler_X.fit(X_flat)
    scaler_y.fit(y_raw)
    X_scaled = scaler_X.transform(X_flat).reshape(nsamples, seq_len, len(feature_cols))
    y_scaled = scaler_y.transform(y_raw)
    df_scaled_rows = []
    for i in range(nsamples):
        row = {}
        for j, f in enumerate(feature_cols):
            for t in range(seq_len):
                row[f"{f}_t{t}"] = X_scaled[i, t, j]
        for k, tname in enumerate(target_cols):
            row[tname] = y_scaled[i, k]
        df_scaled_rows.append(row)
    df_scaled = pd.DataFrame(df_scaled_rows)
    histories, models, dataset_obj = fit_cv(df_scaled, feature_cols, target_cols, seq_len=seq_len, n_splits=6, batch_size=32, epochs=500, lr=0.001, patience=10, model_dir="./models")
    ensemble_preds = []
    X_all = dataset_obj.X
    for m in models:
        m.to(DEVICE)
        preds = predict_with_model(m, X_all)
        ensemble_preds.append(preds)
        m.to("cpu")
    ensemble_preds = np.stack(ensemble_preds, axis=0)
    preds_mean = ensemble_preds.mean(axis=0)
    preds_denorm = scaler_y.inverse_transform(preds_mean)
    y_denorm = scaler_y.inverse_transform(dataset_obj.y)
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    output_dim = preds_denorm.shape[1]
    xlims_list = [(-0.5, 0.5),
                  (-1.0, 1.0),
                  (-0.8, 0.8),
                  (-0.6, 0.6),
                  (-0.6, 0.6),
                  (-1.0, 1.0),
                  (-0.2, 0.2)]
    font_params = {"tick": 14, "label": 14, "title": 16, "xlabel": 14, "cbar": 14, "cbar_label": 14}
    plot_params = {"nrows": 2, "ncols": 1, "hspace": 0.4, "vspace": 0.4}
    feature_names = []
    for j, f in enumerate(feature_cols):
        for t in range(seq_len):
            feature_names.append(f"{f}_t{t}")
    X_explain_np = X_scaled[:min(200, X_scaled.shape[0])].reshape(min(200, X_scaled.shape[0]), -1).astype(np.float64)
    background = X_explain_np[np.random.choice(X_explain_np.shape[0], min(50, X_explain_np.shape[0]), replace=False)]
    def predict_fn_numpy(x_numpy):
        x_t = x_numpy.reshape(-1, seq_len, len(feature_cols)).astype(np.float32)
        x_t_scaled = torch.from_numpy(x_t).float()
        model_for_shap = ResCNNLSTM(n_features=len(feature_cols), seq_len=seq_len, lstm_hidden=128, lstm_layers=2, out_dim=len(target_cols), dropout=0.2)
        state = models[0].state_dict()
        model_for_shap.load_state_dict(state)
        model_for_shap.eval()
        with torch.no_grad():
            preds = model_for_shap(x_t_scaled).numpy()
        preds_den = scaler_y.inverse_transform(preds)
        return preds_den
    explainer = shap.KernelExplainer(predict_fn_numpy, background)
    shap_values = explainer.shap_values(X_explain_np, nsamples=100)
    shap_values_arr = np.stack(shap_values, axis=-1)
    image_files = []
    for i in range(output_dim):
        shap_vals_i = shap_values_arr[:, :, i]
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_vals_i, X_explain_np, feature_names=feature_names, show=False)
        fig = plt.gcf()
        axes = fig.axes
        main_ax = None
        for a in axes:
            yticklabels = a.get_yticklabels()
            if yticklabels and len(yticklabels) > 0:
                main_ax = a
                break
        if main_ax is None:
            main_ax = plt.gca()
        if i < len(xlims_list):
            cur_min, cur_max = xlims_list[i]
        else:
            cur_min = float(np.nanmin(shap_vals_i))
            cur_max = float(np.nanmax(shap_vals_i))
            if np.isclose(cur_min, cur_max):
                eps = max(1e-6, abs(cur_min)*0.01 + 1e-6)
                cur_min -= eps
                cur_max += eps
        xticks = np.linspace(cur_min, cur_max, 6)
        main_ax.set_xlim(cur_min, cur_max)
        main_ax.set_xticks(xticks)
        main_ax.set_xticklabels([f"{v:.2f}" for v in xticks], fontsize=font_params["tick"])
        main_ax.set_title(f"(a)" if i==0 else f"({chr(97+i)})", loc='left', fontsize=font_params["title"])
        main_ax.set_xlabel("SHAP value (impact on model output)", fontsize=font_params["xlabel"])
        for label in main_ax.get_yticklabels():
            label.set_fontsize(font_params["label"])
        for a in axes[::-1]:
            lbl = a.get_ylabel()
            if lbl is not None and ("Feature" in str(lbl) or "feature" in str(lbl).lower()):
                a.tick_params(labelsize=font_params["cbar"])
                a.set_ylabel("Feature value", fontsize=font_params["cbar_label"])
                break
        plt.tight_layout()
        filename = os.path.join(save_dir, f"shap_output_{i+1}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        image_files.append(filename)
    times = np.arange(len(preds_denorm))
    for k, tname in enumerate(target_cols):
        plt.figure(figsize=(12,5))
        plt.plot(times, y_denorm[:, k], marker='o', linestyle='-', label='真实值')
        plt.plot(times, preds_denorm[:, k], marker='s', linestyle='--', label='预测值')
        plt.xlabel('样本序号')
        plt.ylabel(tname)
        plt.title(f"{tname} 真实 vs 预测")
        plt.legend()
        plt.grid(True)
        fname = os.path.join(save_dir, f"time_series_{tname}.png")
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        image_files.append(fname)
        plt.figure(figsize=(6,6))
        plt.scatter(y_denorm[:, k], preds_denorm[:, k], alpha=0.6, label='点')
        m, b = np.polyfit(y_denorm[:, k], preds_denorm[:, k], 1)
        xs = np.array([y_denorm[:, k].min(), y_denorm[:, k].max()])
        plt.plot(xs, m*xs + b, color='r', label=f'拟合: y={m:.3f}x+{b:.3f}')
        r2v = r2_score(y_denorm[:, k], preds_denorm[:, k])
        rmsev = math.sqrt(mean_squared_error(y_denorm[:, k], preds_denorm[:, k]))
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f"{tname} 拟合散点图\nR2={r2v:.4f}  RMSE={rmsev:.4f}")
        plt.legend()
        plt.grid(True)
        fname2 = os.path.join(save_dir, f"scatter_fit_{tname}.png")
        plt.savefig(fname2, dpi=300, bbox_inches='tight')
        plt.close()
        image_files.append(fname2)
    metrics_r2, metrics_rmse = compute_metrics(y_denorm, preds_denorm)
    metrics_df = pd.DataFrame({"variable": target_cols, "R2": np.atleast_1d(metrics_r2), "RMSE": np.atleast_1d(metrics_rmse)})
    metrics_df.to_csv(os.path.join(save_dir, "metrics_summary.csv"), index=False)
    print("saved", len(image_files), "figures to", save_dir)

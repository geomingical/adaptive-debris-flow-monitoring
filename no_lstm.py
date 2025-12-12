## Ensemble strategy no LSTM
# ====================================================================
# Step 0: Install Required Packages (optional in Colab)
# ====================================================================
#!pip uninstall -y numpy pandas torch torchsummary alibi-detect optuna xgboost -q
#!pip install numpy==1.26.4 pandas==2.2.2 torch==2.3.0 torchsummary==1.5.1 alibi-detect==0.11.4 optuna==3.6.1 xgboost==2.0.3 -q

# Verify installations
import numpy
import pandas
import os
import random
import pandas as pd
import numpy as np
import torch

RANDOM_STATE = 42
REPRO_MODE = False  #  False （~1e-3～1e-2 order vibration）
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

if REPRO_MODE:
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
else:
    # 放鬆可重現：仍保留固定 seeds / generator，但不強制 deterministic 路徑
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True   # 允許 cudnn 選快路徑
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


print(f"NumPy version: {numpy.__version__}")
print(f"Pandas version: {pandas.__version__}")
print(f"PyTorch version: {torch.__version__}")

# ====================================================================
# Step 1: Import Libraries
# ====================================================================


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from alibi_detect.cd import KSDrift
from torchsummary import summary
import optuna
import re
import warnings
from tqdm.notebook import tqdm
from sklearn.model_selection import TimeSeriesSplit
import json
warnings.filterwarnings('ignore')

# ====================================================================
# Focal Loss
# ====================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# ====================================================================
# Step 2: Configuration & Temporal Helpers
# ====================================================================
STATIONS = ['ILL12','ILL13','ILL18']
BASE_PATH = "/content"
out="/content/drive/MyDrive/Colab Notebooks/ML_Mutiai_debris_flow/no_lstm/"
SEQUENCE_LENGTH = 10
EPOCHS_SEARCH = 6     # 給 Optuna 用
EPOCHS_FINAL  = 14    # 最終重訓用（小幅增加）
OPTUNA_N_TRIALS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if DEVICE == "cuda":
    torch.cuda.manual_seed(RANDOM_STATE)
print(f"Using device: {DEVICE}")

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
# ---------------- Ablation switches ----------------
# ENSEMBLE_MODE: 'full' | 'no_lstm' | 'no_xgb' | 'no_rf'
ENSEMBLE_MODE = os.getenv("ENSEMBLE_MODE", "no_lstm")
SKIP_TRAIN_UNUSED = True

#SKIP_TRAIN_UNUSED = (os.getenv("SKIP_TRAIN_UNUSED", "0") == "1")

STACK_ORDER = ["LSTM_prob", "RF_prob", "XGB_prob"]

def choose_stack_cols(mode: str):
    if mode == "full":
        idx = [0, 1, 2]
    elif mode == "no_lstm":
        idx = [1, 2]
    elif mode == "no_xgb":
        idx = [0, 1]
    elif mode == "no_rf":
        idx = [0, 2]
    else:
        idx = [0, 1, 2]
    names = [STACK_ORDER[i] for i in idx]
    return idx, names
# ---------------- EMA / Filters ----------------

def ema_1d(x, span=5):
    if span <= 1:
        return x.copy()
    alpha = 2.0 / (span + 1.0)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y


def min_run_filter(preds, k=3):
    if k <= 1:
        return preds
    out = preds.copy()
    run = 0
    start = 0
    for i, v in enumerate(np.r_[preds, 0]):
        if v == 1:
            if run == 0:
                start = i
            run += 1
        else:
            if run > 0 and run < k:
                out[start:i] = 0
            run = 0
    return out


def hysteresis_filter(probs, on_thr, off_thr):
    preds = np.zeros_like(probs, dtype=np.int32)
    state = 0
    for i, p in enumerate(probs):
        if state == 0:
            if p >= on_thr:
                state = 1
        else:
            if p < off_thr:
                state = 0
        preds[i] = state
    return preds


def hysteresis_binarize(probs, hi, lo, min_run=3):
    probs = np.asarray(probs, dtype=float)
    out = np.zeros_like(probs, dtype=int)
    state = 0
    for i, x in enumerate(probs):
        if state == 0 and x >= hi:
            state = 1
        elif state == 1 and x < lo:
            state = 0
        out[i] = state
    if int(min_run) > 1:
        out = min_run_filter(out, k=int(min_run))
    return out

# ---------------- Isotonic (by month) ----------------

def calibrate_isotonic(valid_probs, valid_labels):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(valid_probs, valid_labels)
    return ir


def fit_iso_by_month_probs(valid_probs, valid_labels, valid_months):
    models = {}
    months = np.unique(valid_months)
    for m in months:
        idx = (valid_months == m)
        y_m = valid_labels[idx]
        pos = np.sum(y_m == 1)
        neg = np.sum(y_m == 0)
        if pos < 10 or neg < 10:
            # 樣本太少 → 用全域校準，較穩
            continue
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(valid_probs[idx], y_m)
        models[int(m)] = ir
    ir_global = IsotonicRegression(out_of_bounds='clip')
    ir_global.fit(valid_probs, valid_labels)
    return models, ir_global


def transform_iso_by_month_probs(probs, months, models, ir_global):
    probs = np.asarray(probs, dtype=float)
    out = np.empty_like(probs)
    for m in np.unique(months):
        idx = (months == m)
        ir = models.get(int(m), ir_global)
        out[idx] = ir.transform(probs[idx])
    return out

# ---------------- Threshold utilities ----------------

def find_best_threshold(labels, probs, min_recall=0.90, min_precision=0.85):
    """Precision-first with recall guard; fallback = max F2."""
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    if thresholds.size == 0:
        return 0.5
    mask = (recalls[:-1] >= min_recall) & (precisions[:-1] >= min_precision)
    if np.any(mask):
        f2 = (5 * precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-12)
        return thresholds[mask][np.argmax(f2[mask])]
    f2 = (5 * precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-12)
    return thresholds[np.argmax(f2)]

# ====================================================================
# Step 3: Data Preprocessing and Feature Engineering
# ====================================================================

def safe_div(a, b, eps=1e-8):
    return a / np.clip(b, eps, None)


def add_time_features(df, col="time_window_start"):
    df[col] = pd.to_datetime(df[col])
    df['month'] = df[col].dt.month
    df['dayofyear'] = df[col].dt.dayofyear
    df['hour'] = df[col].dt.hour
    df['minute'] = df[col].dt.minute
    df['dayofweek'] = df[col].dt.dayofweek
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)
    for feat in ['FCentroid', 'MeanFFT', 'E4FFT']:
        if feat in df.columns:
            df[f'{feat}_roll_mean'] = df[feat].rolling(window=10, min_periods=1).mean()
            df[f'{feat}_roll_std'] = df[feat].rolling(window=10, min_periods=1).std()
    df.fillna(0, inplace=True)
    return df


def add_energy_features(df):
    X = df.copy()
    feats = [c for c in X.columns if c not in ["Label", "time_window_start"]]
    e_fft = [c for c in feats if re.fullmatch(r"E\d+FFT", c)]
    es = [c for c in feats if c.startswith("ES_")]
    has_bands = False
    if e_fft:
        has_bands = True
        X["EFFT_sum"] = X[e_fft].sum(axis=1)
        for c in e_fft:
            X[f"{c}_ratio"] = safe_div(X[c], X["EFFT_sum"])
        lows = [c for c in e_fft if re.findall(r"\d+", c) and int(re.findall(r"\d+", c)[0])<=2]
        highs = [c for c in e_fft if re.findall(r"\d+", c) and int(re.findall(r"\d+", c)[0])>=3]
        if lows and highs:
            X["EFFT_high_low"] = safe_div(X[highs].sum(axis=1), X[lows].sum(axis=1))
    if es:
        has_bands = True
        X["ES_sum"] = X[es].sum(axis=1)
        for c in es:
            X[f"{c}_ratio"] = safe_div(X[c], X["ES_sum"])
    if ("Fquart3" in X.columns) and ("Fquart1" in X.columns):
        X["Fquart3_over_1"] = safe_div(X["Fquart3"], X["Fquart1"])
    if "FCentroid" in X.columns:
        if has_bands:
            X["FCentroid_over_sumE"] = safe_div(X["FCentroid"], X.get("EFFT_sum", X.get("ES_sum", 1.0)))
    if "FCentroid" in X.columns and "MeanFFT" in X.columns:
        X["FCentroid_MeanFFT_interaction"] = X["FCentroid"] * X["MeanFFT"]
    if "MeanFFT" in X.columns and "E4FFT" in X.columns:
        X["MeanFFT_E4FFT_interaction"] = X["MeanFFT"] * X["E4FFT"]
    X.fillna(0, inplace=True)
    return X


def select_features(train_df, valid_df, high_drift_features=['Fquart3', 'Fquart1', 'gamma1'], max_features=25):
    time_features = ['month', 'dayofyear', 'hour', 'minute', 'dayofweek', 'month_sin', 'month_cos',
                     'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']
    X_train = train_df.drop(columns=["time_window_start", "Label"] + time_features, errors='ignore').copy()
    y_train = train_df["Label"].copy()
    features = [f for f in X_train.columns if f not in high_drift_features]
    if not features:
        raise ValueError("No features selected after drift filtering")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[features])
    X_valid_scaled = scaler.transform(valid_df[features])
    spw = np.sum(y_train==0) / (np.sum(y_train==1) + 1e-8)
    xgb = XGBClassifier(n_estimators=100,random_state=RANDOM_STATE,n_jobs=1, tree_method="hist",grow_policy="depthwise",subsample=1.0,colsample_bytree=1.0,importance_type="gain",base_score=0.5,scale_pos_weight=spw)
    xgb.fit(X_train_scaled, y_train)
    detector = KSDrift(X_train_scaled, p_val=0.05)
    drift_result = detector.predict(X_valid_scaled)
    drift_scores = drift_result['data'].get('distance', 0.0)
    if np.ndim(drift_scores) == 0:
        drift_scores = np.zeros(len(features), dtype=float)
    elif len(drift_scores) != len(features):
        drift_scores = np.zeros(len(features), dtype=float)
    feature_importance = xgb.feature_importances_ * np.exp(-3 * np.clip(drift_scores, 0, 1))
    order = np.lexsort((np.arange(len(features)), -feature_importance))

    names = np.asarray(features)
    # np.lexsort 以「最後一個鍵」為主要排序鍵；這裡用 -feature_importance（大到小）
    idx = np.lexsort((names, -feature_importance))
    top_indices = idx[:max_features]
    top_features = [features[i] for i in top_indices]

    return top_features

def preprocess_data(station, base_path=BASE_PATH):
    train_path = f"{base_path}/{station}_2017_2018.csv"
    valid_path = f"{base_path}/{station}_2019.csv"
    test_path  = f"{base_path}/test_{station}_2020.csv"

    print(f"Loading and sorting data for {station}...")
    train_df = pd.read_csv(train_path).sort_values("time_window_start").reset_index(drop=True)
    valid_df = pd.read_csv(valid_path).sort_values("time_window_start").reset_index(drop=True)
    test_df  = pd.read_csv(test_path ).sort_values("time_window_start").reset_index(drop=True)

    label_map = {'debris_flow': 1, 'no_debris_flow': 0}
    for df in [train_df, valid_df, test_df]:
        if df['Label'].dtype == 'object':
            df['Label'] = df['Label'].map(label_map)

    print(f"Adding time and energy features for {station}...")
    for i, df in enumerate([train_df, valid_df, test_df]):
        df = add_time_features(df)
        df = add_energy_features(df)
        keep = tuple(int(x) for x in os.getenv("KEEP_MONTHS", "6,7,8").split(","))
        before = len(df)
        df = df[df["month"].isin(keep)].reset_index(drop=True)
        print(f"[Filter] keep months {keep} → {before} → {len(df)} rows")
        if i == 0:   train_df = df
        elif i == 1: valid_df = df
        else:        test_df  = df

    common_cols = [c for c in train_df.columns if (c in valid_df.columns) and (c in test_df.columns)]
    common_cols = list(dict.fromkeys(common_cols))
    feature_cols = select_features(train_df[common_cols], valid_df[common_cols])
    print(f"Selected {len(feature_cols)} features for {station}.")

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    valid_df[feature_cols] = scaler.transform(valid_df[feature_cols])
    test_df [feature_cols] = scaler.transform(test_df [feature_cols])

    return train_df, valid_df, test_df, feature_cols

# ====================================================================
# Step 4: Save Results
# ====================================================================

def save_results(station, test_df, test_probs, test_preds, best_threshold, save_path):
    print(f"Saving results for {station}...")
    start_index = SEQUENCE_LENGTH - 1
    end_index = start_index + len(test_probs)
    results = pd.DataFrame({
        'time_window_start': test_df['time_window_start'].iloc[start_index:end_index].reset_index(drop=True),
        'true_label': test_df['Label'].iloc[start_index:end_index].reset_index(drop=True),
        'predicted_prob': test_probs,
        'predicted_label': test_preds,
        'decision_threshold': best_threshold
    })
    cm_types = []
    for true, pred in zip(results['true_label'], results['predicted_label']):
        if true == 1 and pred == 1: cm_types.append('TP')
        elif true == 1 and pred == 0: cm_types.append('FN')
        elif true == 0 and pred == 1: cm_types.append('FP')
        else: cm_types.append('TN')
    results['confusion_matrix_type'] = cm_types
    results.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

# ====================================================================
# Step 5: PyTorch Dataset
# ====================================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, sequence_length):
        self.features = df[feature_cols].values
        self.labels = df['Label'].values
        self.sequence_length = sequence_length
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    def __getitem__(self, idx):
        sequence = self.features[idx:idx + self.sequence_length]
        label = self.labels[idx + self.sequence_length - 1]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# ====================================================================
# Step 6: OneNet Model (LSTM-based)
# ====================================================================
class OneNetClassifier(nn.Module):
    def __init__(self, input_size, lstm_hidden, mlp_hidden, gate_hidden):
        super(OneNetClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True, num_layers=2, dropout=0.3)
        self.fc_lstm = nn.Linear(lstm_hidden, 1)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(mlp_hidden // 2, 1)
        )
        self.gating_network = nn.Sequential(
            nn.Linear(lstm_hidden + input_size, gate_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(gate_hidden, 1), nn.Sigmoid()
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_lstm_out = lstm_out[:, -1, :]
        pred_lstm = self.fc_lstm(last_lstm_out)
        last_feature_in = x[:, -1, :]
        pred_mlp = self.mlp(last_feature_in)
        gate_input = torch.cat((last_lstm_out, last_feature_in), dim=1)
        gate_weight = self.gating_network(gate_input)
        final_prediction = gate_weight * pred_lstm + (1 - gate_weight) * pred_mlp
        return final_prediction.squeeze(-1)

# ====================================================================
# Step 7: Training and Evaluation Loops
# ====================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for sequences, labels in tqdm(dataloader, desc="Training", leave=False):
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def get_predictions(model, dataloader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

# ====================================================================
# Step 8: Optuna Hyperparameter Tuning
# ====================================================================

def objective_lstm(trial, train_loader, valid_loader, input_size):
    lstm_hidden = trial.suggest_categorical('lstm_hidden', [128, 160, 192, 224, 256])
    mlp_hidden  = trial.suggest_categorical('mlp_hidden',  [32, 48, 64, 80, 96])
    gate_hidden = trial.suggest_categorical('gate_hidden', [32, 40, 48, 56, 64])
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])

    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset
    L = len(train_dataset)
    seq_len = int(train_dataset.sequence_length)
    y_full  = train_dataset.labels.astype(int)
    y_seq_end = y_full[seq_len - 1 : seq_len - 1 + L]
    assert len(y_seq_end) == L, "Sampler label length must equal dataset length"
    class_counts = np.bincount(y_seq_end, minlength=2)
    class_weights = 1.0 / (class_counts + 1e-8)
    weights = class_weights[y_seq_end]

    _nw = int(os.getenv("OPTUNA_NUM_WORKERS", "0"))
    gen = torch.Generator()
    gen.manual_seed(RANDOM_STATE + 1234)
    optuna_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(weights, num_samples=L, replacement=True, generator=gen),
        num_workers=0,
        pin_memory=False,
    )
    optuna_valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = OneNetClassifier(input_size, lstm_hidden, mlp_hidden, gate_hidden).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    criterion = FocalLoss(gamma=2.0)
    best_f1 = 0
    patience, patience_counter = 5, 0
    for epoch in range(15):
        train_one_epoch(model, optuna_train_loader, optimizer, criterion, DEVICE)
        labels, probs = get_predictions(model, optuna_valid_loader, DEVICE)
        threshold = find_best_threshold(labels, probs)
        preds = (probs > threshold).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    return best_f1


def objective_rf(trial, X_train, y_train, X_valid, y_valid):
    # 超參搜尋空間維持你原本的簡潔設計；加一個 max_features 讓模型容量更可調
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'max_depth':         trial.suggest_int('max_depth', 5, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight':      'balanced',
    }
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_valid)[:, 1]

    # 與 LSTM 的目標一致：先用 precision/recall 有下限的門檻挑選
    thr = find_best_threshold(y_valid, probs, min_recall=0.90, min_precision=0.85)
    preds = (probs > thr).astype(int)
    return f1_score(y_valid, preds)

def objective_xgb(trial, X_train, y_train, X_valid, y_valid):
    spw = float(np.sum(y_train==0) / (np.sum(y_train==1) + 1e-8))
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma':            trial.suggest_float('gamma', 0.0, 5.0),
    }
    model = XGBClassifier(
        random_state=RANDOM_STATE,
        base_score=0.5,
        eval_metric='logloss',
        scale_pos_weight=spw,
        **params
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_valid)[:, 1]
    thr = find_best_threshold(y_valid, probs)  # 對齊 LSTM 的門檻策略
    f1 = f1_score(y_valid, (probs > thr).astype(int))
    return f1

# ====================================================================
# Step 9: Drift Adaptation and Fine-Tuning (revised get_fine_tune_data)
# ====================================================================

def get_fine_tune_data(finetune_df, feature_cols, model, device, neg_to_pos_ratio=20, pos_label=1):
    """
    If model is None: early augmentation = all positives + random negatives.
    If model is trained: positives + hard negatives (prob>0.7 & label=0) + extra random negatives.
    Returns a row-wise DataFrame aligned to sequence ends (drops first SEQUENCE_LENGTH-1 rows).
    """
    df = finetune_df.copy().reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    df_seq = df.iloc[SEQUENCE_LENGTH-1:].reset_index(drop=True)
    if df_seq.empty:
        return pd.DataFrame()

    pos_df = df_seq[df_seq['Label'] == pos_label]
    if pos_df.empty:
        return pd.DataFrame()

    if model is None:
        # Early stage: no model yet → positives + random negatives only
        neg_pool = df_seq[df_seq['Label'] == 0]
        n_neg = min(len(neg_pool), len(pos_df) * neg_to_pos_ratio)
        neg_df = neg_pool.sample(n=n_neg, random_state=RANDOM_STATE) if n_neg > 0 else pd.DataFrame()
        out = pd.concat([pos_df, neg_df], axis=0).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        return out

    # With model → mine hard negatives
    dataset = TimeSeriesDataset(df, feature_cols, SEQUENCE_LENGTH)
    loader  = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    model.eval()
    probs_list = []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
            probs_list.append(p)
    probs = np.concatenate(probs_list, axis=0)
    df_seq = df_seq.assign(_prob=probs)

    hard_neg = df_seq[(df_seq['Label'] == 0) & (df_seq['_prob'] > 0.7)]
    max_hn = int(len(pos_df) * neg_to_pos_ratio * 0.8)
    hn_df  = hard_neg.sample(n=min(len(hard_neg), max_hn), random_state=RANDOM_STATE) if not hard_neg.empty else pd.DataFrame()

    remain_neg_needed = len(pos_df) * neg_to_pos_ratio - len(hn_df)
    rand_neg_pool = df_seq[(df_seq['Label'] == 0) & (~df_seq.index.isin(hn_df.index))]
    rn_df = rand_neg_pool.sample(n=max(0, min(len(rand_neg_pool), remain_neg_needed)), random_state=RANDOM_STATE) if remain_neg_needed > 0 else pd.DataFrame()

    out = pd.concat([pos_df, hn_df, rn_df], axis=0).drop(columns=['_prob'], errors='ignore')
    out = out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return out

# ====================================================================
# Step 10: Month-level Tuning (tightened defaults)
# ====================================================================

def tune_month_params(
    labels, probs,
    mode=os.getenv("TUNE_MODE", "prec_first"),
    beta=float(os.getenv("BETA", "1.5")),
    prec_floor=float(os.getenv("PREC_FLOOR", "0.85")),
    recall_floor=float(os.getenv("RECALL_FLOOR", "0.90")),
    spans=(3,5,7),
    bands=(0.03, 0.06, 0.09),
    min_runs=(3,5,7),
    gaps=(0,1,2,3,4),
    max_thr=64
):
    labels = np.asarray(labels).astype(int)
    probs  = np.asarray(probs, dtype=float)
    if labels.size == 0:
        return dict(span=3, thr=0.5, band=0.05, min_run=5, gap=0, hi=0.5, lo=0.45)
    records = []

    q = np.linspace(0.02, 0.98, num=min(max_thr, 50))
    thr_q = np.unique(np.quantile(probs, q))
    P, R, T = precision_recall_curve(labels, probs)
    knee_thr = []
    if T.size:
        mask = P[:-1] >= (prec_floor - 1e-6)
        idxs = np.where(mask)[0]
        if idxs.size:
            take = np.linspace(idxs.min(), idxs.max(), num=min(14, idxs.size)).astype(int)
            knee_thr = list(np.clip(T[take], 1e-4, 1 - 1e-4))
    thr_cands = np.linspace(0.05, 0.95, 19)
    global_thr = os.getenv("GLOBAL_THR", "")
    if global_thr:
        g = float(global_thr)
        thr_cands = np.array([t for t in thr_cands if (g - 0.03) <= t <= (g + 0.03)]) or np.array([g])

    best = None
    best_fallback = None

    def f_beta_score(p, r, b):
        return ((1+b*b) * p * r / max(1e-12, (b*b)*p + r)) if (p+r)>0 else 0.0

    for span in spans:
        p_s = ema_1d(probs, span=span)
        for thr in thr_cands:
            for band in bands:
                hi = float(thr)
                lo = max(0.01, hi - float(band))
                base = hysteresis_binarize(p_s, hi, lo, min_run=1)
                for k in min_runs:
                    b2 = min_run_filter(base, k=k)
                    for g in gaps:
                        pred = merge_small_gaps(b2, max_gap=g)
                        pr = precision_score(labels, pred, zero_division=0)
                        rc = recall_score(labels, pred,    zero_division=0)
                        f1 = f1_score(labels, pred,        zero_division=0)
                        fbeta = f_beta_score(pr, rc, beta)
                        params = dict(span=int(span), thr=float(thr), band=float(band),
                                      min_run=int(k), gap=int(g), hi=float(hi), lo=float(lo))
                        records.append((pr, rc, f1, fbeta, params))
                        if mode == "prec_first":
                            if (pr >= prec_floor) and (rc >= recall_floor):
                                score = (rc, pr)
                                if (best is None) or (score > best[0]):
                                    best = (score, params)
                            score_fb = (f1, pr, rc)
                            if (best_fallback is None) or (score_fb > best_fallback[0]):
                                best_fallback = (score_fb, params)
                        else:
                            score = (fbeta, pr, rc)
                            if (best is None) or (score > best[0]):
                                best = (score, params)

    if mode == "prec_first":
        if best is not None:
            return best[1]
        # progressive relaxation toward recall
        def pick_with_floors(records, pf, rf):
            cand = [(rc, pr, fb, f1, p) for (pr, rc, f1, fb, p) in records
                    if (pr >= pf) and (rc >= rf)]
            # 穩定且保守：先看 recall、再 precision、再 Fβ、F1；
            # 同分時偏好「較高 thr、較長 min_run、較小 gap」→ 精確率更保守
            def key(t):
                rc, pr, fb, f1, p = t
                return (rc, pr, fb, f1, float(p["thr"]), int(p["min_run"]), -int(p["gap"]))
            return max(cand, key=key)[-1] if cand else None

        # 收集所有候選（在上面內層 loop 計算時，同步存入 records）
        # --- 先把上面 loop 中的評分保存下來 ---
        # 在 for 迴圈裡每次計算完 pr/rc/f1/fbeta 之後，加上：
        # records.append((pr, rc, f1, fbeta, params))
        # 並在函式一開始先定義： records = []

        for pf in [prec_floor, max(0.0, prec_floor-0.03), max(0.0, prec_floor-0.05)]:
            sol = pick_with_floors(records, pf, recall_floor)
            if sol is not None:
                return sol
        # 最後退一步：最大 F2，但仍守住 P≥0.75
        cand = [(fb, pr, rc, p) for (pr, rc, f1, fb, p) in records if pr >= 0.82]
        if cand:
            def fb_key(t):
                fb, pr, rc, p = t
                return (fb, pr, rc, float(p["thr"]), int(p["min_run"]), -int(p["gap"]))
            return max(cand, key=fb_key)[-1]

        # 萬不得已才回傳原 fallback
        return best_fallback[1]
    else:
        return best_fallback if best_fallback is not None else dict(span=3, thr=0.5, band=0.05, min_run=5,gap=0,hi=0.5, lo=0.45)


def merge_small_gaps(preds, max_gap=0):
    if max_gap <= 0:
        return preds
    y = preds.copy()
    n = len(y)
    i = 0
    while i < n:
        while i < n and y[i] == 0:
            i += 1
        j = i
        while j < n and y[j] == 1:
            j += 1
        k = j
        while k < n and y[k] == 0:
            k += 1
        gap = k - j
        if j < n and k < n and gap > 0 and gap <= max_gap:
            y[j:k] = 1
            i = k
        else:
            i = k
    return y

def build_meta_matrix(p_lstm, p_rf, p_xgb, names):
    cols = []
    for nm in names:
        if nm == "LSTM_prob":
            cols.append(p_lstm)
        elif nm == "RF_prob":
            cols.append(p_rf)
        elif nm == "XGB_prob":
            cols.append(p_xgb)
    return np.column_stack(cols)
# ====================================================================
# Step 11: Multi-Station Execution
# ====================================================================
_nw = 0
def run_for_stations(stations, base_path=BASE_PATH):
    results = {}

    # 小工具：依當前模式選出要堆疊的欄位名稱與索引
    # 你應該已經有 choose_stack_cols(ENSEMBLE_MODE)；為保險我在這裡用名稱為主
    def build_meta_matrix(p_lstm, p_rf, p_xgb, names):
        cols = []
        for nm in names:
            if nm == "LSTM_prob":
                cols.append(p_lstm)
            elif nm == "RF_prob":
                cols.append(p_rf)
            elif nm == "XGB_prob":
                cols.append(p_xgb)
        return np.column_stack(cols)

    def to_logit_feats(P, eps=1e-6):
        Q = np.clip(P, eps, 1 - eps)
        return np.log(Q / (1 - Q))

    for station in stations:
        print(f"\n{'='*20} Processing station: {station} {'='*20}")

        # ----------------------- Data & split -----------------------
        train_df, valid_df, test_df, feature_cols = preprocess_data(station, base_path)

        valid_df_sorted = valid_df.sort_values('time_window_start').reset_index(drop=True)
        split_index = int(len(valid_df_sorted) * 0.8)
        valid_split_df   = valid_df_sorted.iloc[:split_index]
        finetune_split_df = valid_df_sorted.iloc[split_index:]
        train_split_df   = train_df
        test_split_df    = test_df
        print(f"Validation set split: {len(valid_split_df)} for tuning/meta-learning, {len(finetune_split_df)} for fine-tuning.")

        # 早期增強（不依賴模型）：正樣本 + 隨機負樣本
        aug_valid_df = get_fine_tune_data(valid_split_df, feature_cols, model=None, device=DEVICE, neg_to_pos_ratio=10)
        if not aug_valid_df.empty:
            train_df = pd.concat([train_df, aug_valid_df]).reset_index(drop=True)
            print(f"[Early Augment] Added {len(aug_valid_df)} samples (pos + random neg) into training.")

        # 平鋪資料（給樹模型）
        X_train_flat = train_df.iloc[SEQUENCE_LENGTH-1:][feature_cols].values.astype(np.float32)
        y_train_flat = train_df.iloc[SEQUENCE_LENGTH-1:]['Label'].values.astype(int)
        X_valid_flat = valid_split_df.iloc[SEQUENCE_LENGTH-1:][feature_cols].values.astype(np.float32)
        y_valid_flat = valid_split_df.iloc[SEQUENCE_LENGTH-1:]['Label'].values.astype(int)
        X_test_flat  = test_split_df.iloc[SEQUENCE_LENGTH-1:][feature_cols].values.astype(np.float32)

        # ----------------------- Optuna blocks（視模式可選） -----------------------
        # ----------------------- Optuna blocks（依 ENSEMBLE_MODE 有選擇地執行） -----------------------
        # LSTM
        if ENSEMBLE_MODE == "no_lstm" and SKIP_TRAIN_UNUSED:
            print("[Ablation] ENSEMBLE_MODE=no_lstm → skip LSTM training/Optuna")
            lstm_best_params = None
        else:
            print("Running Optuna for LSTM...")
            train_dataset = TimeSeriesDataset(train_df, feature_cols, SEQUENCE_LENGTH)
            valid_dataset = TimeSeriesDataset(valid_split_df, feature_cols, SEQUENCE_LENGTH)
            labels = train_df['Label'].values[SEQUENCE_LENGTH-1:]
            class_counts   = np.bincount(labels.astype(int), minlength=2)
            class_weights  = 1.0 / (class_counts + 1e-8)
            sample_weights = class_weights[labels.astype(int)]

            gen = torch.Generator(); gen.manual_seed(RANDOM_STATE)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=gen)
            optuna_train_loader = DataLoader(train_dataset, batch_size=1024, sampler=sampler,
                                             num_workers=0, generator=gen, pin_memory=(DEVICE=="cuda"))
            optuna_valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False,
                                             num_workers=0, pin_memory=(DEVICE=="cuda"))

            study_lstm = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True, group=True),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
            )
            study_lstm.optimize(lambda trial: objective_lstm(trial, optuna_train_loader, optuna_valid_loader, len(feature_cols)),
                                n_trials=OPTUNA_N_TRIALS)
            lstm_best_params = study_lstm.best_params
            print(f"Best params for LSTM: {lstm_best_params}")

        # RF
        if ENSEMBLE_MODE == "no_rf" and SKIP_TRAIN_UNUSED:
            print("[Ablation] ENSEMBLE_MODE=no_rf → skip RF training/Optuna")
            rf_best_params = None
        else:
            print("Running Optuna for Random Forest...")
            study_rf = optuna.create_study(direction='maximize',
                                           sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
            study_rf.optimize(lambda t: objective_rf(t,
                                                     X_train_flat, y_train_flat,
                                                     X_valid_flat, y_valid_flat),
                              n_trials=OPTUNA_N_TRIALS)
            rf_best_params = study_rf.best_params
            print(f"Best params for RF: {rf_best_params}")

        # XGB
        if ENSEMBLE_MODE == "no_xgb" and SKIP_TRAIN_UNUSED:
            print("[Ablation] ENSEMBLE_MODE=no_xgb → skip XGB training/Optuna")
            xgb_best_params = None
        else:
            print("Running Optuna for XGBoost...")
            study_xgb = optuna.create_study(direction='maximize',
                                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
            study_xgb.optimize(lambda t: objective_xgb(t,
                                                       X_train_flat, y_train_flat,
                                                       X_valid_flat, y_valid_flat),
                               n_trials=OPTUNA_N_TRIALS)
            xgb_best_params = study_xgb.best_params
            print(f"Best params for XGBoost: {xgb_best_params}")

        # ----------------------- 依模式選擇要疊的欄位 -----------------------
        stack_idx, stack_names = choose_stack_cols(ENSEMBLE_MODE)  # 例：no_lstm → ["RF_prob","XGB_prob"]
        meta_dim = len(stack_names)
        print(f"[Ablation] ENSEMBLE_MODE={ENSEMBLE_MODE}, stack={stack_names}")

        # ----------------------- 產 OOF（valid，不洩漏） -----------------------
        tscv = TimeSeriesSplit(n_splits=5)
        valid_split_flat = valid_split_df.iloc[SEQUENCE_LENGTH-1:].reset_index(drop=True)
        oof_labels = valid_split_flat['Label'].values
        oof_meta_features = np.zeros((len(valid_split_flat), meta_dim), dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(valid_split_flat)):
            print(f"Processing Fold {fold+1}/5 for OOF...")
            fold_train_flat = valid_split_flat.iloc[tr_idx]
            fold_val_flat   = valid_split_flat.iloc[va_idx]

            # RF OOF
            if "RF_prob" in stack_names:
                rf_col = stack_names.index("RF_prob")
                if rf_best_params is None:
                    oof_meta_features[va_idx, rf_col] = 0.0
                else:
                    fold_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                                     class_weight='balanced', **rf_best_params
                                                    ).fit(fold_train_flat[feature_cols], fold_train_flat['Label'])
                    if len(fold_rf.classes_) < 2:
                        oof_meta_features[va_idx, rf_col] = 0.0
                    else:
                        oof_meta_features[va_idx, rf_col] = fold_rf.predict_proba(fold_val_flat[feature_cols])[:, 1]

            # XGB OOF
            if "XGB_prob" in stack_names:
                xgb_col = stack_names.index("XGB_prob")
                if xgb_best_params is None or len(np.unique(fold_train_flat['Label'])) < 2:
                    oof_meta_features[va_idx, xgb_col] = 0.0
                else:
                    spw = np.sum(fold_train_flat['Label']==0) / (np.sum(fold_train_flat['Label']==1) + 1e-8)
                    fold_xgb = XGBClassifier(random_state=RANDOM_STATE, base_score=0.5,
                                             scale_pos_weight=spw, **xgb_best_params
                                            ).fit(fold_train_flat[feature_cols], fold_train_flat['Label'])
                    oof_meta_features[va_idx, xgb_col] = fold_xgb.predict_proba(fold_val_flat[feature_cols])[:, 1]

        # ----------------------- LSTM（僅當需要） -----------------------
        valid_lstm_probs = None
        test_lstm_probs  = None
        if "LSTM_prob" in stack_names and (lstm_best_params is not None):
            train_dataset = TimeSeriesDataset(train_df,       feature_cols, SEQUENCE_LENGTH)
            valid_dataset = TimeSeriesDataset(valid_split_df, feature_cols, SEQUENCE_LENGTH)
            test_dataset  = TimeSeriesDataset(test_split_df,  feature_cols, SEQUENCE_LENGTH)

            labels = train_df['Label'].values[SEQUENCE_LENGTH-1:]
            class_counts   = np.bincount(labels.astype(int), minlength=2)
            class_weights  = 1.0 / (class_counts + 1e-8)
            sample_weights = class_weights[labels.astype(int)]
            gen = torch.Generator(); gen.manual_seed(RANDOM_STATE)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=gen)

            train_loader = DataLoader(train_dataset, batch_size=lstm_best_params['batch_size'],
                                      sampler=sampler, num_workers=0, generator=gen, pin_memory=(DEVICE=="cuda"))
            valid_loader = DataLoader(valid_dataset, batch_size=lstm_best_params['batch_size'],
                                      shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))
            test_loader  = DataLoader(test_dataset,  batch_size=lstm_best_params['batch_size'],
                                      shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))

            lstm_model = OneNetClassifier(len(feature_cols),
                                          lstm_best_params['lstm_hidden'],
                                          lstm_best_params['mlp_hidden'],
                                          lstm_best_params['gate_hidden']).to(DEVICE)
            optimizer = optim.Adam(lstm_model.parameters(), lr=lstm_best_params['lr'])
            criterion = FocalLoss(gamma=2.0)

            # 簡易 early stop（F2）
            best_f2, best_state, patience, wait = -1.0, None, 5, 0
            for _ in range(EPOCHS):
                train_one_epoch(lstm_model, train_loader, optimizer, criterion, DEVICE)
                _, cur = get_predictions(lstm_model, valid_loader, DEVICE)
                pr = precision_score(oof_labels, (cur>=0.5).astype(int), zero_division=0)
                rc = recall_score(  oof_labels, (cur>=0.5).astype(int), zero_division=0)
                f2 = (5*pr*rc)/(4*pr+rc+1e-12)
                if f2 > best_f2:
                    best_f2, best_state, wait = f2, {k:v.cpu().clone() for k,v in lstm_model.state_dict().items()}, 0
                else:
                    wait += 1
                    if wait >= patience: break
            if best_state is not None:
                lstm_model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})

            # valid/test 機率
            _, valid_lstm_probs = get_predictions(lstm_model, valid_loader, DEVICE)
            _, test_lstm_probs  = get_predictions(lstm_model, test_loader,  DEVICE)

            # 寫回 OOF 的 LSTM 欄
            lstm_col = stack_names.index("LSTM_prob")
            oof_meta_features[:, lstm_col] = valid_lstm_probs

        # ----------------------- valid/test 融合特徵 -----------------------
        n_valid = len(valid_split_df) - (SEQUENCE_LENGTH-1)
        n_test  = len(test_split_df)  - (SEQUENCE_LENGTH-1)

        # final RF/XGB → test 機率
        test_rf_probs  = np.zeros(n_test, dtype=float)
        test_xgb_probs = np.zeros(n_test, dtype=float)
        final_rf_model = None
        final_xgb_model = None
        if "RF_prob" in stack_names and (rf_best_params is not None):
            final_rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                                    class_weight='balanced', **rf_best_params
                                                   ).fit(X_train_flat, y_train_flat)
            proba = final_rf_model.predict_proba(X_test_flat)
            test_rf_probs = proba[:,1] if proba.shape[1]==2 else np.zeros(n_test)
        if "XGB_prob" in stack_names and (xgb_best_params is not None):
            spw_final = np.sum(y_train_flat==0) / (np.sum(y_train_flat==1)+1e-8)
            final_xgb_model = XGBClassifier(random_state=RANDOM_STATE, base_score=0.5,
                                            scale_pos_weight=spw_final, **xgb_best_params
                                           ).fit(X_train_flat, y_train_flat)
            proba = final_xgb_model.predict_proba(X_test_flat)
            test_xgb_probs = proba[:,1] if proba.shape[1]==2 else np.zeros(n_test)

        # LSTM 若被排除 → 以 0 向量代替
        if valid_lstm_probs is None: valid_lstm_probs = np.zeros(n_valid)
        if test_lstm_probs  is None: test_lstm_probs  = np.zeros(n_test)

        # valid 用 OOF，test 用 final base
        valid_rf_probs  = np.zeros(n_valid); valid_xgb_probs = np.zeros(n_valid)
        if "RF_prob"  in stack_names:
            valid_rf_probs  = oof_meta_features[:, stack_names.index("RF_prob")]
        if "XGB_prob" in stack_names:
            valid_xgb_probs = oof_meta_features[:, stack_names.index("XGB_prob")]

        def build_meta_features(p_lstm, p_rf, p_xgb, names):
            cols = []
            for nm in names:
                if nm == "LSTM_prob": cols.append(p_lstm)
                elif nm == "RF_prob": cols.append(p_rf)
                elif nm == "XGB_prob": cols.append(p_xgb)
            return np.column_stack(cols)

        def to_logit_feats(P, eps=1e-6):
            Q = np.clip(P, eps, 1 - eps); return np.log(Q / (1 - Q))

        valid_meta_features = build_meta_features(valid_lstm_probs, valid_rf_probs, valid_xgb_probs, stack_names)
        test_meta_features  = build_meta_features(test_lstm_probs,  test_rf_probs,  test_xgb_probs,  stack_names)

        # ----------------------- Meta-learner（LR with logit） -----------------------
        meta_learner = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced',
                                          C=0.3, penalty='l2', solver='lbfgs', max_iter=1000)
        Z_train = to_logit_feats(valid_meta_features)
        meta_learner.fit(Z_train, oof_labels)

        valid_meta_probs = meta_learner.predict_proba(to_logit_feats(valid_meta_features))[:, 1]
        all_probs        = meta_learner.predict_proba(to_logit_feats(test_meta_features))[:, 1]

        # ----------------------- Stacking 權重輸出 -----------------------
        coef = meta_learner.coef_.ravel()
        intercept = float(meta_learner.intercept_.ravel()[0])
        w_abs = np.abs(coef); w_norm = w_abs / (w_abs.sum() + 1e-12)
        if len(stack_names) > 0:
            stack_df = pd.DataFrame({"feature": stack_names, "coef": coef,
                                     "abs_coef": w_abs, "abs_coef_norm": w_norm})
            stack_df.loc[len(stack_df)] = ["(intercept)", intercept, np.nan, np.nan]
            suffix = {"full":"full","no_lstm":"noLSTM","no_xgb":"noXGB","no_rf":"noRF"}.get(ENSEMBLE_MODE,"full")
            stack_out_a = f"{out}/stack_weights_{station}_{suffix}.csv"
            stack_df.to_csv(stack_out_a, index=False)
            print(f"[{ENSEMBLE_MODE}] Saved stacking weights to {stack_out_a}")
            stack_out_b = f"{out}/stack_weights_{station}.csv"
            stack_df.to_csv(stack_out_b, index=False)
            print(f"Saved stacking weights to {stack_out_b}")

        # ----------------------- Drift & LSTM fine-tune（僅當含 LSTM） -----------------------
        use_lstm = ("LSTM_prob" in stack_names) and (lstm_best_params is not None)
        if use_lstm:
            detector = KSDrift(valid_df[feature_cols].values, p_val=0.05)
            drift_result = detector.predict(test_df[feature_cols].values)
            if drift_result['data']['is_drift']:
                print(f"Drift detected for {station}! Fine-tuning LSTM...")
                final_lstm_model = lstm_model
                fine_tune_df = get_fine_tune_data(finetune_split_df, feature_cols, final_lstm_model, DEVICE)
                if not fine_tune_df.empty and len(fine_tune_df['Label'].unique()) > 1:
                    fine_ds = TimeSeriesDataset(fine_tune_df, feature_cols, SEQUENCE_LENGTH)
                    fine_loader = DataLoader(fine_ds, batch_size=lstm_best_params['batch_size'],
                                             shuffle=True, num_workers=0)
                    opt_ft = optim.Adam(final_lstm_model.parameters(), lr=lstm_best_params['lr']/10)
                    crit_ft = FocalLoss(gamma=2.0)
                    for ep in range(10):
                        loss = train_one_epoch(final_lstm_model, fine_loader, opt_ft, crit_ft, DEVICE)
                        print(f"Fine-tune Epoch {ep+1}/10: Loss {loss:.4f}")
                else:
                    print("Skipping fine-tuning due to insufficient or single-class data.")
            else:
                print("No drift detected; skip fine-tune.")
                final_lstm_model = lstm_model
        else:
            print("[Ablation] LSTM excluded → skip drift detection and fine-tune.")

        # ----------------------- PR/Calibration 輸出 -----------------------
        val_months  = valid_split_df['month'].values[SEQUENCE_LENGTH-1:]
        test_months = test_df['month'].values[SEQUENCE_LENGTH-1:]
        iso_map, iso_global = fit_iso_by_month_probs(valid_meta_probs, oof_labels, val_months)
        valid_meta_probs = transform_iso_by_month_probs(valid_meta_probs, val_months, iso_map, iso_global)
        all_probs        = transform_iso_by_month_probs(all_probs,        test_months, iso_map, iso_global)

        P_v, R_v, T_v = precision_recall_curve(oof_labels, valid_meta_probs)
        pd.DataFrame({"precision": P_v[:-1], "recall": R_v[:-1], "threshold": T_v}).to_csv(
            f"{out}/pr_curve_valid_{station}.csv", index=False)
        P_t, R_t, T_t = precision_recall_curve(test_df['Label'].values[SEQUENCE_LENGTH-1:], all_probs)
        pd.DataFrame({"precision": P_t[:-1], "recall": R_t[:-1], "threshold": T_t}).to_csv(
            f"{out}/pr_curve_test_{station}.csv", index=False)

        # ----------------------- 月別參數搜尋與後處理（保持你的函式） -----------------------
        PREC_FLOOR   = float(os.getenv("PREC_FLOOR",   "0.85"))
        RECALL_FLOOR = float(os.getenv("RECALL_FLOOR", "0.90"))
        BAND_CANDS   = tuple(float(x) for x in os.getenv("BAND_CANDS", "0.05").split(","))
        RUN_CANDS    = tuple(int(x)    for x in os.getenv("RUN_CANDS",  "5,7").split(","))
        GAP_CANDS    = tuple(int(x)    for x in os.getenv("GAP_CANDS",  "0,1").split(","))
        SPAN_CANDS   = tuple(int(x)    for x in os.getenv("SPAN_CANDS", "5").split(","))

        tuned = {}
        for m in (6,7,8):
            mask = (val_months == m)
            if not np.any(mask): continue
            tuned[m] = tune_month_params(
                oof_labels[mask], valid_meta_probs[mask],
                mode=os.getenv("TUNE_MODE", "prec_first"),
                beta=float(os.getenv("BETA", "1.5")),
                prec_floor=PREC_FLOOR, recall_floor=RECALL_FLOOR,
                spans=SPAN_CANDS, bands=BAND_CANDS, min_runs=RUN_CANDS, gaps=GAP_CANDS,
                max_thr=64
            )
        with open(f"{out}/tuned_params_{station}.json", "w") as f:
            json.dump({int(k):{kk:(round(v,4) if isinstance(v,float) else v) for kk,v in d.items()}
                       for k,d in tuned.items()}, f, indent=2)

        # 應用月別規則
        y_test_flat = test_df['Label'].values[SEQUENCE_LENGTH-1:]
        test_preds = np.zeros_like(y_test_flat, dtype=int)
        for m, params in tuned.items():
            idx = (test_months == m)
            if not np.any(idx): continue
            span = int(params["span"]); thr = float(params["thr"])
            band = float(params["band"]); k = int(params["min_run"]); g = int(params["gap"])
            hi, lo = thr, max(0.01, thr - band)
            p_sm = ema_1d(all_probs[idx], span=span)
            base = hysteresis_binarize(p_sm, hi, lo, min_run=1)
            test_preds[idx] = merge_small_gaps(min_run_filter(base, k=k), max_gap=g)

        if test_preds.size and test_preds.max()==0:
            if T_v.size:
                f = 2*P_v[:-1]*R_v[:-1] / np.maximum(P_v[:-1]+R_v[:-1], 1e-12)
                ok = (R_v[:-1] >= RECALL_FLOOR)
                ix = np.argmax(np.where(ok, f, -1))
                thr_global = float(T_v[ix]) if ix >= 0 else 0.5
            else:
                thr_global = 0.5
            test_preds = hysteresis_binarize(all_probs, thr_global, max(0.05, thr_global-0.05), min_run=5)

        best_threshold = float(np.mean([v["thr"] for v in tuned.values()])) if tuned else 0.5
        print(f"Effective (avg) threshold: {best_threshold:.4f}")

        # ----------------------- 指標與輸出 -----------------------
        test_metrics = {
            "precision": precision_score(y_test_flat, test_preds, zero_division=0),
            "recall":    recall_score(y_test_flat,    test_preds, zero_division=0),
            "f1_score":  f1_score(y_test_flat,        test_preds, zero_division=0),
            "auprc":     average_precision_score(y_test_flat, all_probs),
            "roc_auc":   roc_auc_score(y_test_flat,   all_probs),
            "best_threshold": best_threshold
        }
        by_m_rows = []
        for m in (6,7,8):
            idx = (test_months == m)
            if not np.any(idx): continue
            by_m_rows.append({
                "month": m,
                "precision": precision_score(y_test_flat[idx], test_preds[idx], zero_division=0),
                "recall":    recall_score(y_test_flat[idx],    test_preds[idx], zero_division=0),
                "f1":        f1_score(y_test_flat[idx],        test_preds[idx], zero_division=0),
                "support":   int(np.sum(idx))
            })
        pd.DataFrame(by_m_rows).to_csv(f"{out}/metrics_by_month_{station}.csv", index=False)

        print(f"\n--- Test Results for {station} ---")
        print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1-Score: {test_metrics['f1_score']:.4f}")
        _suffix = {"full":"full","no_lstm":"noLSTM","no_xgb":"noXGB","no_rf":"noRF"}.get(ENSEMBLE_MODE,"full")
        save_results(station, test_df, all_probs, test_preds, test_metrics['best_threshold'],
                     f"{out}/results_{station}_{_suffix}.csv")
        results[station] = test_metrics
    return results

# ====================================================================
# Step 12: Main Execution
# ====================================================================
if __name__ == "__main__":
    results = run_for_stations(STATIONS)
    print("\n--- Summary of Results ---")
    for station, metrics in results.items():
        print(f"{station}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}, AUPRC={metrics['auprc']:.4f}")

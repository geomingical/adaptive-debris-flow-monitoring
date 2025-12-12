## No Drift detection

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
out="/content/drive/MyDrive/Colab Notebooks/ML_Mutiai_debris_flow/no_tuned/"
SEQUENCE_LENGTH = 10
EPOCHS_SEARCH = 6     # 給 Optuna 用
EPOCHS_FINAL  = 15    # 最終重訓用（小幅增加）
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

# ====================================================================
# Step 11: Multi-Station Execution
# ====================================================================
_nw = 0
def run_for_stations(stations, base_path=BASE_PATH):
    results = {}
    for station in stations:
        print(f"\n{'='*20} Processing station: {station} {'='*20}")
        train_df, valid_df, test_df, feature_cols = preprocess_data(station, base_path)

        # ----- Validation split
        valid_df_sorted = valid_df.sort_values('time_window_start').reset_index(drop=True)
        split_index = int(len(valid_df_sorted) * 0.8)
        valid_split_df = valid_df_sorted.iloc[:split_index]
        finetune_split_df = valid_df_sorted.iloc[split_index:]
        print(f"Validation set split: {len(valid_split_df)} for tuning/meta-learning, {len(finetune_split_df)} for fine-tuning.")
        train_split_df = train_df
        test_split_df  = test_df
        # ----- EARLY AUGMENTATION (no model yet): positives + random negatives
        aug_valid_df = get_fine_tune_data(valid_split_df, feature_cols, model=None, device=DEVICE, neg_to_pos_ratio=10)
        if not aug_valid_df.empty:
            train_df = pd.concat([train_df, aug_valid_df]).reset_index(drop=True)
            print(f"[Early Augment] Added {len(aug_valid_df)} samples (pos + random neg) into training.")

        # ----- Flattened features for RF/XGB
        X_train_flat = train_df.iloc[SEQUENCE_LENGTH-1:][feature_cols].values
        y_train_flat = train_df.iloc[SEQUENCE_LENGTH-1:]['Label'].values
        X_valid_flat = valid_split_df.iloc[SEQUENCE_LENGTH-1:][feature_cols].values
        y_valid_flat = valid_split_df.iloc[SEQUENCE_LENGTH-1:]['Label'].values

        # ----- Optuna for LSTM
        print(f"Running Optuna for LSTM...")
        train_dataset = TimeSeriesDataset(train_df, feature_cols, SEQUENCE_LENGTH)
        valid_dataset = TimeSeriesDataset(valid_split_df, feature_cols, SEQUENCE_LENGTH)
        test_dataset  = TimeSeriesDataset(test_split_df,   feature_cols, SEQUENCE_LENGTH)
        labels = train_df['Label'].values[SEQUENCE_LENGTH-1:]
        class_counts = np.bincount(labels.astype(int))
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[labels.astype(int)]

        gen = torch.Generator()                # 一律 CPU generator（避免裝置不匹配）
        gen.manual_seed(RANDOM_STATE)

        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights),
            replacement=True,
            generator=gen
        )
        optuna_train_loader = DataLoader(
            train_dataset,
            batch_size=1024,
            sampler=sampler,
            num_workers=0,
            generator=gen,
            pin_memory=(DEVICE == "cuda")
        )

        optuna_valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False, num_workers=0)

        study_lstm = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_STATE,
                multivariate=True,      # 參數聯合建模 → 收斂更快
                group=True,             # 同名超參成組處理（有助穩定）
                n_startup_trials=8,     # 前期探索
                n_ei_candidates=64      # 提升候選探索品質（20 trials 仍可負擔）
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,     # 前 5 次不裁
                n_warmup_steps=3        # 至少跑 3 個 epoch 再裁
            )
        )
        study_lstm.optimize(lambda trial: objective_lstm(trial, optuna_train_loader, optuna_valid_loader, len(feature_cols)), n_trials=OPTUNA_N_TRIALS)
        lstm_best_params = study_lstm.best_params
        print(f"Best params for LSTM: {lstm_best_params}")

        # ----- Optuna for RF
        print(f"Running Optuna for Random Forest...")
        study_rf = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study_rf.optimize(lambda trial: objective_rf(trial, X_train_flat, y_train_flat, X_valid_flat, y_valid_flat), n_trials=OPTUNA_N_TRIALS)
        rf_best_params = study_rf.best_params
        print(f"Best params for RF: {rf_best_params}")

        # ----- Optuna for XGB
        print(f"Running Optuna for XGBoost...")
        study_xgb = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study_xgb.optimize(lambda trial: objective_xgb(trial, X_train_flat, y_train_flat, X_valid_flat, y_valid_flat), n_trials=OPTUNA_N_TRIALS)
        xgb_best_params = study_xgb.best_params
        print(f"Best params for XGBoost: {xgb_best_params}")

        # ----- Train final base models + OOF for meta
        print("\nTraining final base models and generating OOF predictions...")
        tscv = TimeSeriesSplit(n_splits=5)
        valid_split_flat = valid_split_df.iloc[SEQUENCE_LENGTH-1:].reset_index(drop=True)

        # OOF for meta (valid 全部覆蓋到，且無洩漏)
        oof_meta_features = np.zeros((len(valid_split_flat), 3))
        oof_labels = valid_split_flat['Label'].values

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(valid_split_flat)):
            print(f"Processing Fold {fold+1}/5 for RF/XGB OOF...")
            fold_train_flat = valid_split_flat.iloc[tr_idx]
            fold_val_flat   = valid_split_flat.iloc[va_idx]
            # RF OOF
            fold_rf = RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced', **rf_best_params
            ).fit(fold_train_flat[feature_cols], fold_train_flat['Label'])
            if len(fold_rf.classes_) < 2:
                oof_meta_features[va_idx, 1] = 0.0
            else:
                oof_meta_features[va_idx, 1] = fold_rf.predict_proba(fold_val_flat[feature_cols])[:, 1]
            # XGB OOF
            if len(np.unique(fold_train_flat['Label'])) < 2:
                oof_meta_features[va_idx, 2] = 0.0
            else:
                spw = np.sum(fold_train_flat['Label'] == 0) / (np.sum(fold_train_flat['Label'] == 1) + 1e-8)
                fold_xgb = XGBClassifier(
                    random_state=RANDOM_STATE, base_score=0.5, scale_pos_weight=spw, **xgb_best_params
                ).fit(fold_train_flat[feature_cols], fold_train_flat['Label'])
                oof_meta_features[va_idx, 2] = fold_xgb.predict_proba(fold_val_flat[feature_cols])[:, 1]
        gen = torch.Generator()
        gen.manual_seed(RANDOM_STATE)
        # Final LSTM fit（用你 optuna 選出的最佳參數）
        train_loader = DataLoader(train_dataset,
                                  batch_size=lstm_best_params['batch_size'],
                                  sampler=sampler, num_workers=0,
                                  generator=gen, pin_memory=(DEVICE == "cuda"))
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=lstm_best_params['batch_size'],
                                  shuffle=False, num_workers=0,
                                  generator=gen, pin_memory=(DEVICE == "cuda"))
        test_loader  = DataLoader(test_dataset,
                                  batch_size=lstm_best_params['batch_size'],
                                  shuffle=False, num_workers=0,
                                  generator=gen, pin_memory=(DEVICE == "cuda"))
        # 你的訓練資料集類別若不是 SequenceDataset，請用你原本用在 train_dataset 的那個類別名


        lstm_model = OneNetClassifier(len(feature_cols),
                                      lstm_best_params['lstm_hidden'],
                                      lstm_best_params['mlp_hidden'],
                                      lstm_best_params['gate_hidden']).to(DEVICE)
        optimizer = optim.Adam(lstm_model.parameters(), lr=lstm_best_params['lr'])
        criterion = FocalLoss(gamma=2.0)

        best_f2, best_state, patience, wait = -1.0, None, 5, 0
        for epoch in range(EPOCHS_FINAL):
            train_one_epoch(lstm_model, train_loader, optimizer, criterion, DEVICE)
            _, cur = get_predictions(lstm_model, valid_loader, DEVICE)
            # 以 valid 的 Precision/Recall 算 F2
            pr, rc = precision_score(oof_labels, (cur>=0.5).astype(int), zero_division=0), \
                     recall_score(oof_labels, (cur>=0.5).astype(int),  zero_division=0)
            f2 = (5*pr*rc)/(4*pr+rc+1e-12)
            if f2 > best_f2:
                best_f2, best_state, wait = f2, {k:v.cpu().clone() for k,v in lstm_model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        # 回滾到最佳
        if best_state is not None:
            lstm_model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
        _, oof_lstm_probs = get_predictions(lstm_model, valid_loader, DEVICE)
        _, test_lstm_probs = get_predictions(lstm_model, test_loader,  DEVICE)

        # 把 LSTM 的 OOF 機率放入第 0 欄
        oof_meta_features[:, 0] = oof_lstm_probs

        # ====== 這裡建「valid/test 的融合特徵」 ======
        def build_meta_features(p_lstm, p_rf, p_xgb):
            return np.column_stack([p_lstm, p_rf, p_xgb])

        def to_logit_feats(P, eps=1e-6):
            Q = np.clip(P, eps, 1 - eps)
            return np.log(Q / (1 - Q))

        # （A）valid 的融合特徵：用 OOF（最乾淨、無洩漏）
        valid_meta_features = oof_meta_features.copy()

        # （B）test 的融合特徵：需要 RF/XGB 的「final」模型來出測試集機率
        #     final RF/XGB → 用 train_split（平鋪）來訓練，再對 test_flat 做 predict_proba
        X_train_flat = train_split_df.iloc[SEQUENCE_LENGTH-1:][feature_cols].values.astype(np.float32)
        y_train_flat = train_split_df.iloc[SEQUENCE_LENGTH-1:]['Label'].values.astype(int)
        X_valid_flat = valid_split_df.iloc[SEQUENCE_LENGTH-1:][feature_cols].values.astype(np.float32)
        X_test_flat  = test_split_df.iloc[SEQUENCE_LENGTH-1:][feature_cols].values.astype(np.float32)

        # Final RF/XGB
        final_rf_model = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced', **rf_best_params
        ).fit(X_train_flat, y_train_flat)

        spw_final = np.sum(y_train_flat == 0) / (np.sum(y_train_flat == 1) + 1e-8)
        final_xgb_model = XGBClassifier(
            random_state=RANDOM_STATE, base_score=0.5, scale_pos_weight=spw_final, **xgb_best_params
        ).fit(X_train_flat, y_train_flat)

        def safe_pos_proba(clf, X):
            proba = clf.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
            if hasattr(clf, "classes_") and len(clf.classes_) == 1:
                return np.ones(len(X)) if clf.classes_[0] == 1 else np.zeros(len(X))
            return np.zeros(len(X))

        # RF/XGB 的 test 機率
        test_rf_probs   = safe_pos_proba(final_rf_model,  X_test_flat)
        test_xgb_probs  = safe_pos_proba(final_xgb_model, X_test_flat)

        # 組成 test 融合特徵（N×3）
        test_meta_features = build_meta_features(test_lstm_probs, test_rf_probs, test_xgb_probs)

        # ====== 疊加器（LR）：一致的 logit 前處理 + 輕度 L2（若你要更保守，可把 C=0.5→0.3） ======
        meta_learner = LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            C=0.3, penalty='l2', solver='lbfgs', max_iter=1000
        )

        Z_train = to_logit_feats(oof_meta_features)       # OOF → 訓練堆疊器
        meta_learner.fit(Z_train, oof_labels)

        valid_Z = to_logit_feats(valid_meta_features)     # valid（用 OOF 特徵）→ 產生 valid meta 機率
        test_Z  = to_logit_feats(test_meta_features)      # test（用 final base 機率）→ 產生 test meta 機率
        valid_meta_probs = meta_learner.predict_proba(valid_Z)[:, 1]
        all_probs        = meta_learner.predict_proba(test_Z)[:, 1]

        # （可保留）輸出 stacking 權重以供論文使用
        stack_feat_names = ["LSTM_prob", "RF_prob", "XGB_prob"]
        coef = meta_learner.coef_.ravel()
        intercept = float(meta_learner.intercept_.ravel()[0])
        w_abs = np.abs(coef)
        w_norm = w_abs / (w_abs.sum() + 1e-12)
        stack_df = pd.DataFrame({
            "feature": stack_feat_names,
            "coef": coef,
            "abs_coef": w_abs,
            "abs_coef_norm": w_norm
        })
        stack_df.loc[len(stack_df)] = ["(intercept)", intercept, np.nan, np.nan]
        stack_out = f"{out}/stack_weights_{station}.csv"
        stack_df.to_csv(stack_out, index=False)
        print(f"Saved stacking weights to {stack_out}")

        final_lstm_model = lstm_model
        torch.save(final_lstm_model.state_dict(), f"{out}/best_onenet_{station}.pth")

        final_rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced', **rf_best_params).fit(X_train_flat, y_train_flat)
        final_xgb_model = XGBClassifier(random_state=RANDOM_STATE, base_score=0.5, scale_pos_weight=np.sum(y_train_flat==0)/np.sum(y_train_flat==1), **xgb_best_params).fit(X_train_flat, y_train_flat)

        # 1) 存下使用的 25 個特徵列表
        feat_list_out = f"{out}/features_used_{station}.csv"
        pd.Series(feature_cols, name="feature").to_csv(feat_list_out, index=False)
        print(f"Saved features used to {feat_list_out}")

        # 2) RF / XGB 的特徵重要度
        rf_imp = getattr(final_rf_model, "feature_importances_", None)
        xgb_imp = getattr(final_xgb_model, "feature_importances_", None)
        if rf_imp is not None:
            rf_imp_df = pd.DataFrame({"feature": feature_cols, "rf_importance": rf_imp})
            rf_imp_df.sort_values("rf_importance", ascending=False, inplace=True)
            rf_out = f"{out}/feature_importance_rf_{station}.csv"
            rf_imp_df.to_csv(rf_out, index=False)
            print(f"Saved RF importances to {rf_out}")
        if xgb_imp is not None:
            xgb_imp_df = pd.DataFrame({"feature": feature_cols, "xgb_importance": xgb_imp})
            xgb_imp_df.sort_values("xgb_importance", ascending=False, inplace=True)
            xgb_out = f"{out}/feature_importance_xgb_{station}.csv"
            xgb_imp_df.to_csv(xgb_out, index=False)
            print(f"Saved XGB importances to {xgb_out}")

        # ----- Test fast path
        start_idx = SEQUENCE_LENGTH - 1
        X_test_flat = test_df[feature_cols].values[start_idx:]
        y_test_flat = test_df['Label'].values[start_idx:]
        n_test = len(X_test_flat)

        ckpt_path = f"best_onenet_{station}.pth"
        final_lstm_model.eval()

        test_dataset = TimeSeriesDataset(test_df, feature_cols, SEQUENCE_LENGTH)
        test_loader  = DataLoader(test_dataset, batch_size=lstm_best_params['batch_size'], shuffle=False, num_workers=0, pin_memory=False)

        lstm_probs_list = []
        with torch.inference_mode():
            for sequences, _labels in tqdm(test_loader, desc="Testing(LSTM)", leave=False):
                sequences = sequences.to(DEVICE, non_blocking=True)
                logits = final_lstm_model(sequences)
                lstm_probs_list.append(torch.sigmoid(logits).cpu().numpy())
        lstm_probs_all = np.concatenate(lstm_probs_list, axis=0)

        def safe_pos_proba(clf, X):
            proba = clf.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
            # 只訓到單一類別時：若那類是 1 則回傳全 1，否則全 0
            if hasattr(clf, "classes_") and len(clf.classes_) == 1:
                return np.ones(len(X)) if clf.classes_[0] == 1 else np.zeros(len(X))
            # 不確定情況，保底回 0
            return np.zeros(len(X))

        rf_probs_all  = safe_pos_proba(final_rf_model,  X_test_flat)
        xgb_probs_all = safe_pos_proba(final_xgb_model, X_test_flat)

        test_meta_features = np.column_stack([lstm_probs_all, rf_probs_all, xgb_probs_all])
        all_labels = y_test_flat
        all_probs = meta_learner.predict_proba(test_meta_features)[:, 1]

        valid_meta_probs = meta_learner.predict_proba(oof_meta_features)[:, 1]

        # ----- Isotonic by month (fit on validation; apply to test)
        val_months  = valid_split_df['month'].values[SEQUENCE_LENGTH-1:]
        test_months = test_df['month'].values[SEQUENCE_LENGTH-1:]
        iso_map, iso_global = fit_iso_by_month_probs(valid_meta_probs, oof_labels, val_months)
        valid_meta_probs = transform_iso_by_month_probs(valid_meta_probs, val_months, iso_map, iso_global)
        all_probs        = transform_iso_by_month_probs(all_probs,        test_months, iso_map, iso_global)
        # Validation PR 曲線（以 OOF 校準後的 valid 為基礎）
        P_v, R_v, T_v = precision_recall_curve(oof_labels, valid_meta_probs)
        pr_valid_out = f"{out}/pr_curve_valid_{station}.csv"
        pd.DataFrame({"precision": P_v[:-1], "recall": R_v[:-1], "threshold": T_v}).to_csv(pr_valid_out, index=False)

        # Test PR 曲線
        P_t, R_t, T_t = precision_recall_curve(all_labels, all_probs)
        pr_test_out = f"{out}/pr_curve_test_{station}.csv"
        pd.DataFrame({"precision": P_t[:-1], "recall": R_t[:-1], "threshold": T_t}).to_csv(pr_test_out, index=False)

        # Calibration（reliability）資料：分 10 bin
        def calibration_table(y, p, n_bins=10):
            bins = np.linspace(0.0, 1.0, n_bins+1)
            idx = np.digitize(p, bins) - 1
            rows = []
            for b in range(n_bins):
                mask = (idx == b)
                if np.any(mask):
                    rows.append({
                        "bin_left": bins[b],
                        "bin_right": bins[b+1],
                        "count": int(np.sum(mask)),
                        "avg_prob": float(np.mean(p[mask])),
                        "empirical_pos": float(np.mean(y[mask]))
                    })
            return pd.DataFrame(rows)

        cal_valid = calibration_table(oof_labels, valid_meta_probs, n_bins=10)
        cal_test  = calibration_table(all_labels, all_probs, n_bins=10)
        cal_valid.to_csv(f"{out}/calibration_valid_{station}.csv", index=False)
        cal_test.to_csv(f"{out}/calibration_test_{station}.csv",  index=False)
        print("Saved PR curves and calibration tables.")

        # ----- Month-wise tuning (precision-first, tightened candidates)
        PREC_FLOOR = float(os.getenv("PREC_FLOOR", "0.85"))
        RECALL_FLOOR = float(os.getenv("RECALL_FLOOR", "0.90"))
        BAND_CANDS = tuple(float(x) for x in os.getenv("BAND_CANDS", "0.05").split(","))
        RUN_CANDS  = tuple(int(x)    for x in os.getenv("RUN_CANDS",  "5,7").split(","))
        GAP_CANDS  = tuple(int(x)    for x in os.getenv("GAP_CANDS",  "0,1").split(","))
        SPAN_CANDS = tuple(int(x)    for x in os.getenv("SPAN_CANDS", "5").split(","))

        tuned = {}
        for m in (6, 7, 8):
            mask = (val_months == m)
            if not np.any(mask):
                continue
            tuned[m] = tune_month_params(
                oof_labels[mask], valid_meta_probs[mask],
                mode=os.getenv("TUNE_MODE", "prec_first"),
                beta=float(os.getenv("BETA", "1.5")),
                prec_floor=PREC_FLOOR, recall_floor=RECALL_FLOOR,
                spans=SPAN_CANDS, bands=BAND_CANDS, min_runs=RUN_CANDS, gaps=GAP_CANDS,
                max_thr=64
            )

        tuned_pretty = {int(m): {k: (round(v,4) if isinstance(v, float) else v) for k, v in d.items()}
                        for m, d in tuned.items()}
        print("Month-wise tuned params:", tuned_pretty)
        import json
        tuned_out = f"{out}/tuned_params_{station}.json"
        with open(tuned_out, "w") as f:
            json.dump(tuned_pretty, f, indent=2)
        print(f"Saved tuned params to {tuned_out}")

        # ----- Apply month-specific smoothing & hysteresis
        test_preds = np.zeros_like(all_labels, dtype=int)
        for m, params in tuned.items():
            idx = (test_months == m)
            if not np.any(idx):
                continue
            span = int(params["span"])
            thr  = float(params["thr"])
            band = float(params["band"])
            k    = int(params["min_run"])
            g    = int(params["gap"])
            hi, lo = float(thr), max(0.01, float(thr) - float(band))
            p_sm = ema_1d(all_probs[idx], span=span)
            base = hysteresis_binarize(p_sm, hi, lo, min_run=1)
            b2   = min_run_filter(base, k=k)
            test_preds[idx] = merge_small_gaps(b2, max_gap=g)

        # Fallback (shouldn't happen with JJA-only)
        if test_preds.size and test_preds.max() == 0:
            P_, R_, T_ = precision_recall_curve(oof_labels, valid_meta_probs)
            if T_.size:
                f = 2 * P_[:-1] * R_[:-1] / np.maximum(P_[:-1] + R_[:-1], 1e-12)
                ok = (R_[:-1] >= RECALL_FLOOR)
                idx = np.argmax(np.where(ok, f, -1))
                thr_global = float(T_[idx]) if idx >= 0 else 0.5
            else:
                thr_global = 0.5
            test_preds = hysteresis_binarize(all_probs, thr_global, max(0.05, thr_global - 0.05), min_run=5)

        best_threshold = float(np.mean([v["thr"] for v in tuned.values()])) if len(tuned) else 0.5
        print(f"Effective (avg) threshold: {best_threshold:.4f}")

        test_metrics = {
            "precision": precision_score(all_labels, test_preds, zero_division=0),
            "recall": recall_score(all_labels, test_preds, zero_division=0),
            "f1_score": f1_score(all_labels, test_preds, zero_division=0),
            "auprc": average_precision_score(all_labels, all_probs),
            "roc_auc": roc_auc_score(all_labels, all_probs),
            "best_threshold": best_threshold
        }
        # 月別指標
        by_m_rows = []
        for m in (6,7,8):
            idx = (test_months == m)
            if not np.any(idx):
                continue
            pr_m = precision_score(all_labels[idx], test_preds[idx], zero_division=0)
            rc_m = recall_score(all_labels[idx],    test_preds[idx], zero_division=0)
            f1_m = f1_score(all_labels[idx],        test_preds[idx], zero_division=0)
            supp = int(np.sum(idx))
            by_m_rows.append({"month": m, "precision": pr_m, "recall": rc_m, "f1": f1_m, "support": supp})
        by_m_df = pd.DataFrame(by_m_rows)
        by_m_out = f"{out}/metrics_by_month_{station}.csv"
        by_m_df.to_csv(by_m_out, index=False)
        print(f"Saved by-month metrics to {by_m_out}")
        print(f"\n--- Test Results for {station} ---")
        print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1-Score: {test_metrics['f1_score']:.4f}")

        save_path = f"{out}/results_{station}.csv"
        save_results(station, test_df, all_probs, test_preds, test_metrics['best_threshold'], save_path)
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


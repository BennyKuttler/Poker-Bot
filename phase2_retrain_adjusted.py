# Phase 2 (Retrain Adjusted) — Safeguarded with Action Class Check

import pandas as pd
import numpy as np
import pickle
import time
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from treys import Card, Evaluator
from tqdm import tqdm

# === Treys-Based Strength ===
def compute_hand_strength_treys(hole_cards_str, board_cards_str):
    if not hole_cards_str:
        return 0.0
    hole = hole_cards_str.split("|")
    board = board_cards_str.split("|") if board_cards_str else []
    if len(hole) + len(board) < 5:
        return 0.0
    try:
        evaluator = Evaluator()
        hole_treys = [Card.new(c) for c in hole]
        board_treys = [Card.new(c) for c in board]
        score = evaluator.evaluate(board_treys, hole_treys)
        return (7463 - score) / 7462.0
    except:
        return 0.0

# === Load Data ===
start_time = time.time()
df = pd.read_csv("/Users/bennykuttler/Downloads/Poker Bot/poker_data.csv", on_bad_lines="skip")
print(f"Data loaded in {time.time() - start_time:.2f} seconds")

for col in ["hole_cards", "board_cards", "player_id"]:
    df[col] = df.get(col, "").fillna("")
if "timestamp" not in df.columns:
    df["timestamp"] = np.arange(len(df))

# === Compute Hand Strength ===
print("\nComputing hand strength...")
df["hand_strength"] = [compute_hand_strength_treys(h, b) for h, b in tqdm(zip(df["hole_cards"], df["board_cards"]), total=len(df))]
print("\n=== Hand Strength Distribution ===")
print(df["hand_strength"].describe())

# === Action Label Encoding ===
unique_actions = df["action"].unique().tolist()
expected_actions = ["cbr", "cc", "f"]

missing = set(expected_actions) - set(unique_actions)
if missing:
    raise ValueError(f"❌ Missing expected actions in data: {missing}")

le_action = LabelEncoder()
df["action_encoded"] = le_action.fit_transform(df["action"])
print("Action mapping:", dict(zip(le_action.classes_, range(len(le_action.classes_)))))

# === Feature Engineering ===
def compute_rolling_stats(df, window=100):
    df = df.sort_values(["player_id", "timestamp"], ignore_index=True)
    df["cbr_indicator"] = (df["action"] == "cbr").astype(float)
    df["fold_indicator"] = (df["action"] == "f").astype(float)
    df["pass_indicator"] = (df["action"] == "cc").astype(float)

    df["player_agg_freq_roll"] = df.groupby("player_id")["cbr_indicator"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df["player_fold_freq_roll"] = df.groupby("player_id")["fold_indicator"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df["player_pass_freq_roll"] = df.groupby("player_id")["pass_indicator"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df["player_agg_freq"] = df.groupby("player_id")["cbr_indicator"].transform("mean")

    folds = df.groupby("player_id")["fold_indicator"].transform("sum")
    calls = df.groupby("player_id")["pass_indicator"].transform("sum").replace(0, np.nan)
    df["player_fold_freq"] = (folds / calls).fillna(0.0)

    df.drop(["cbr_indicator", "fold_indicator", "pass_indicator"], axis=1, inplace=True)
    return df

df = compute_rolling_stats(df)

# === Encode Categorical Features ===
df["street"] = df["street"].fillna("flop")
df["position"] = df["position"].fillna("OOP")
df["board_texture"] = df["board_texture"].fillna("none")

le_street = LabelEncoder()
le_position = LabelEncoder()
le_texture = LabelEncoder()

df["street_encoded"] = le_street.fit_transform(df["street"])
df["position_encoded"] = le_position.fit_transform(df["position"])
df["board_texture_encoded"] = le_texture.fit_transform(df["board_texture"])

# === Clean Numeric Columns ===
clip_map = {
    "pot_size": (1e-3, None),
    "stack_to_pot_ratio": (0, 5000),
    "effective_stack": (0, 20000),
    "board_connectedness": (0, 1),
    "is_preflop_aggressor": (0, 1),
    "bet_size_to_pot_ratio": (0, 5),
    "prev_bet_size": (0, None),
    "player_agg_freq": (0, 1),
    "player_agg_freq_roll": (0, 1),
    "player_fold_freq": (0, 10),
    "player_fold_freq_roll": (0, 1),
    "player_pass_freq_roll": (0, 1),
    "hand_strength": (0, 1)
}

for col, (low, high) in clip_map.items():
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    if high is not None:
        df[col] = df[col].clip(lower=low, upper=high)
    else:
        df[col] = df[col].clip(lower=low)

# === Scale Target for Bet Size Regression ===
bet_size_scaler = MinMaxScaler((0, 1))
df["bet_size_to_pot_ratio_scaled"] = bet_size_scaler.fit_transform(df[["bet_size_to_pot_ratio"]])

# === Model Features ===
features = [
    "street_encoded", "pot_size", "position_encoded", "stack_to_pot_ratio",
    "effective_stack", "board_texture_encoded", "board_connectedness",
    "is_preflop_aggressor", "bet_size_to_pot_ratio", "prev_bet_size",
    "player_agg_freq", "player_agg_freq_roll", "player_fold_freq_roll",
    "player_pass_freq_roll", "player_fold_freq", "hand_strength"
]
X = df[features]
y = df["action_encoded"]

print(f"\nData shape for training: {X.shape[0]} rows, {X.shape[1]} features")

# === Train Action Model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'mlogloss'
}

bst = xgb.train(
    params, dtrain, num_boost_round=200,
    evals=[(dtest, 'test')],
    early_stopping_rounds=20,
    verbose_eval=10
)

with open("/Users/bennykuttler/Downloads/Poker Bot/action_model_phase2_retrained.pkl", "wb") as f:
    pickle.dump(bst, f)

# === Train Bet Size Model ===
cbr_mask = (df["action"] == "cbr")
X_bet = X[cbr_mask]
y_bet = df["bet_size_to_pot_ratio_scaled"][cbr_mask]

X_bet_train, X_bet_test, y_bet_train, y_bet_test = train_test_split(X_bet, y_bet, test_size=0.2, random_state=42)
dtrain_bet = xgb.DMatrix(X_bet_train, label=y_bet_train, feature_names=features)
dtest_bet = xgb.DMatrix(X_bet_test, label=y_bet_test, feature_names=features)

params_bet = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse'
}

reg = xgb.train(
    params_bet, dtrain_bet, num_boost_round=200,
    evals=[(dtest_bet, 'test')],
    early_stopping_rounds=20,
    verbose_eval=10
)

with open("/Users/bennykuttler/Downloads/Poker Bot/bet_model_phase2_retrained.pkl", "wb") as f:
    pickle.dump(reg, f)
with open("/Users/bennykuttler/Downloads/Poker Bot/bet_size_scaler.pkl", "wb") as f:
    pickle.dump(bet_size_scaler, f)

print("\n✅ Phase 2 retraining complete. Models saved.")

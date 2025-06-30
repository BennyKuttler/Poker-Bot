# Phase 2 (Updated)
import pandas as pd
import numpy as np
import pickle
import time
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess full data
start_time = time.time()
df = pd.read_csv("/Users/bennykuttler/Downloads/Poker Bot/poker_data.csv", on_bad_lines="skip")
print(f"Data loaded in {time.time() - start_time:.2f} seconds")

# Verify action mapping
le_action = LabelEncoder()
df["action_encoded"] = le_action.fit_transform(df["action"])
print("Action mapping:", dict(zip(le_action.classes_, range(len(le_action.classes_)))))

# Compute fold frequency
fold_freq = (df["action"] == "f").mean()
print(f"Actual Fold Frequency in Data: {fold_freq:.2f}")

def compute_rolling_stats(df, window=100):
    df['timestamp'] = np.arange(len(df))
    grouped = df.groupby('player_id')
    rolling_stats = []
    for player, player_df in grouped:
        player_df = player_df.sort_values('timestamp')
        player_df['agg_count'] = (player_df['action'] == 'cbr').cumsum()
        player_df['fold_count'] = (player_df['action'] == 'f').cumsum()
        player_df['pass_count'] = (player_df['action'] == 'cc').cumsum()
        player_df['player_agg_freq_roll'] = player_df['agg_count'].rolling(window, min_periods=1).mean() / (player_df['timestamp'] - player_df['timestamp'].min() + 1)
        player_df['player_fold_freq_roll'] = player_df['fold_count'].rolling(window, min_periods=1).mean() / (player_df['timestamp'] - player_df['timestamp'].min() + 1)
        player_df['player_pass_freq_roll'] = player_df['pass_count'].rolling(window, min_periods=1).mean() / (player_df['timestamp'] - player_df['timestamp'].min() + 1)
        player_df['player_fold_freq'] = player_df['fold_count'] / player_df['pass_count'].replace(0, 1)
        rolling_stats.append(player_df[['player_agg_freq_roll', 'player_fold_freq_roll', 'player_pass_freq_roll', 'player_fold_freq']])
    rolling_df = pd.concat(rolling_stats)
    df = df.merge(rolling_df, left_index=True, right_index=True, how='left')
    return df

start_time = time.time()
df = compute_rolling_stats(df)
print(f"Rolling stats computed in {time.time() - start_time:.2f} seconds")

le_street = LabelEncoder()
le_position = LabelEncoder()
le_texture = LabelEncoder()
df["street_encoded"] = le_street.fit_transform(df["street"])
df["position_encoded"] = le_position.fit_transform(df["position"])
df["board_texture_encoded"] = le_texture.fit_transform(df["board_texture"])

numeric_features = ["pot_size", "stack_to_pot_ratio", "effective_stack", "board_connectedness",
                    "is_preflop_aggressor", "bet_size_to_pot_ratio", "prev_bet_size",
                    "player_agg_freq", "player_agg_freq_roll", "player_fold_freq_roll",
                    "player_pass_freq_roll", "player_fold_freq"]
for feature in numeric_features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.0)

features = ["street_encoded", "pot_size", "position_encoded", "stack_to_pot_ratio",
            "effective_stack", "board_texture_encoded", "board_connectedness",
            "is_preflop_aggressor", "bet_size_to_pot_ratio", "prev_bet_size",
            "player_agg_freq", "player_agg_freq_roll", "player_fold_freq_roll",
            "player_pass_freq_roll", "player_fold_freq"]

X = df[features]
y = df["action_encoded"]

# Split and train action model
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
    'scale_pos_weight': 10,
    'eval_metric': 'mlogloss'
}

start_time = time.time()
bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, 'test')], early_stopping_rounds=20, verbose_eval=10)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Save action model
with open("/Users/bennykuttler/Downloads/Poker Bot/action_model_phase2_retrained.pkl", "wb") as f:
    pickle.dump(bst, f)

# Train bet size model (predict bet_size_to_pot_ratio)
cbr_mask = df["action"] == "cbr"
X_bet = X[cbr_mask]
y_bet_ratio = df["bet_size_to_pot_ratio"][cbr_mask]

X_bet_train, X_bet_test, y_bet_train, y_bet_test = train_test_split(X_bet, y_bet_ratio, test_size=0.2, random_state=42)
dtrain_bet = xgb.DMatrix(X_bet_train, label=y_bet_train, feature_names=features)
dtest_bet = xgb.DMatrix(X_bet_test, label=y_bet_test, feature_names=features)

params_bet = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 2.0,
    'alpha': 1.0,
    'seed': 42
}

start_time = time.time()
reg = xgb.train(params_bet, dtrain_bet, num_boost_round=200, evals=[(dtest_bet, 'test')], early_stopping_rounds=20, verbose_eval=10)
print(f"Bet size model training completed in {time.time() - start_time:.2f} seconds")

# Save bet size model
with open("/Users/bennykuttler/Downloads/Poker Bot/bet_model_phase2.pkl", "wb") as f:
    pickle.dump(reg, f)

# Verify probabilities
pred_probs = bst.predict(dtest)
fold_idx = list(le_action.classes_).index('f')
print(f"Mean Predicted Fold Prob: {np.mean(pred_probs[:, fold_idx]):.2f}")
print(f"Mean Predicted Call Prob: {np.mean(pred_probs[:, list(le_action.classes_).index('cc')]):.2f}")
print(f"Mean Predicted Raise Prob: {np.mean(pred_probs[:, list(le_action.classes_).index('cbr')]):.2f}")
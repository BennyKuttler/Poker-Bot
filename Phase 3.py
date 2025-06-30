import pandas as pd
import numpy as np
import pickle
import time
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# --- Step 1: Load Data and Models ---
start_time = time.time()
df = pd.read_csv("/Users/bennykuttler/Downloads/Poker Bot/poker_data.csv", on_bad_lines="skip")
sim_df = df.sample(10000, random_state=42)
print(f"Data loaded and sampled in {time.time() - start_time:.2f} seconds")

with open("/Users/bennykuttler/Downloads/Poker Bot/action_model_phase2_retrained.pkl", "rb") as f:
    clf = pickle.load(f)
with open("/Users/bennykuttler/Downloads/Poker Bot/bet_model_phase2.pkl", "rb") as f:
    reg = pickle.load(f)

# --- Step 2: Prepare Data ---
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

def prepare_data(df):
    df = compute_rolling_stats(df)
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
    return X, df

start_time = time.time()
X_sim, sim_df = prepare_data(sim_df)
print(f"Data prepared in {time.time() - start_time:.2f} seconds")
print(f"Features used: {list(X_sim.columns)}")

dmat_sim = xgb.DMatrix(X_sim, feature_names=X_sim.columns.tolist())
cbr_mask = sim_df["action"] == "cbr"
X_bet_sim = X_sim[cbr_mask]
dmat_bet_sim = xgb.DMatrix(X_bet_sim, feature_names=X_sim.columns.tolist())

# --- Step 3: Decision Engine ---
def estimate_ev(pred_probs, pred_bet_size, pot_size, effective_stack, board_texture, street):
    fold_prob, call_prob, raise_prob = pred_probs
    
    evs = {}
    evs["f"] = 0.0
    
    call_cost = pred_bet_size if pred_bet_size > 0 else pot_size * 0.25
    win_prob = 0.15 if street == "preflop" else (0.1 if board_texture == "wet" else 0.2)
    ev_call_win = (pot_size + call_cost) * win_prob
    ev_call_loss = -call_cost * (1 - win_prob)
    evs["cc"] = (ev_call_win * call_prob + ev_call_loss * (1 - call_prob - raise_prob)) - (call_cost * raise_prob * 0.5)
    
    bet_size = pred_bet_size if pred_bet_size > 0 else pot_size * 0.5
    if bet_size > effective_stack:
        bet_size = effective_stack
    
    ev_fold = pot_size * fold_prob
    ev_call = (pot_size + bet_size) * call_prob * win_prob - bet_size * call_prob * (1 - win_prob)
    ev_raise = 0  # No penalty—max cbr
    evs["cbr"] = min(ev_fold + ev_call + ev_raise, pot_size)  # Looser cap
    
    return evs

def decide_action(pred_probs, pred_bet_size, row):
    evs = estimate_ev(pred_probs, pred_bet_size, row["pot_size"], row["effective_stack"],
                      row["board_texture"], row["street"])
    best_action = max(evs, key=evs.get)
    bet_size = pred_bet_size if best_action == "cbr" else 0.0
    if best_action == "cbr" and bet_size <= 0:
        bet_size = row["pot_size"] * 0.5
    return best_action, bet_size, evs[best_action]

# --- Step 4: Simulate Decisions ---
start_time = time.time()
pred_probs = clf.predict(dmat_sim)
pred_bet_sizes = reg.predict(dmat_bet_sim) if len(X_bet_sim) > 0 else np.zeros(len(X_sim))

decisions = []
for i, (probs, row) in enumerate(zip(pred_probs, sim_df.iterrows())):
    pred_bet = pred_bet_sizes[i] if i < len(pred_bet_sizes) else 0.0
    action, bet_size, ev = decide_action(probs, pred_bet, row[1])
    decisions.append((action, bet_size, ev))

actions, bet_sizes, evs = zip(*decisions)
print(f"Phase 3 Action Distribution:\n{pd.Series(actions).value_counts()}")
print(f"Average Bet Size: {np.mean([b for b, action in zip(bet_sizes, actions) if action == 'cbr']):.2f} BB")
print(f"Average EV per Hand: {np.mean(evs):.2f} BB")
print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
print(f"Mean Predicted Fold Prob: {np.mean(pred_probs[:, 2]):.2f}")
print(f"Mean Predicted Call Prob: {np.mean(pred_probs[:, 1]):.2f}")
print(f"Mean Predicted Raise Prob: {np.mean(pred_probs[:, 0]):.2f}")
print(f"Mean Player Fold Freq Roll: {sim_df['player_fold_freq_roll'].mean():.2f}")
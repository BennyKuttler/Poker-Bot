import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

features = [
    "street_encoded", "pot_size", "position_encoded", "stack_to_pot_ratio",
    "effective_stack", "board_texture_encoded", "board_connectedness",
    "is_preflop_aggressor", "bet_size_to_pot_ratio", "prev_bet_size",
    "player_agg_freq", "player_agg_freq_roll", "player_fold_freq_roll", "player_pass_freq_roll",
    "player_fold_freq"
]

def compute_rolling_stats(df, window=100):
    df = df.copy()
    df['timestamp'] = df.index
    grouped = df.groupby('player_id')
    rolling_dfs = []
    
    for player, player_df in grouped:
        player_df = player_df.sort_values('timestamp')
        player_df['agg_count'] = (player_df['action'] == 'cbr').cumsum()
        player_df['fold_count'] = (player_df['action'] == 'f').cumsum()
        player_df['pass_count'] = (player_df['action'] == 'cc').cumsum()
        
        player_df['player_agg_freq_roll'] = player_df['agg_count'].rolling(window, min_periods=1).mean() / (player_df.index - player_df.index.min() + 1)
        player_df['player_fold_freq_roll'] = player_df['fold_count'].rolling(window, min_periods=1).mean() / (player_df.index - player_df.index.min() + 1)
        player_df['player_pass_freq_roll'] = player_df['pass_count'].rolling(window, min_periods=1).mean() / (player_df.index - player_df.index.min() + 1)
        player_df['player_fold_freq'] = player_df['fold_count'] / player_df['pass_count'].replace(0, 1)
        
        rolling_dfs.append(player_df[['player_agg_freq_roll', 'player_fold_freq_roll', 'player_pass_freq_roll', 'player_fold_freq']])
    
    rolling_df = pd.concat(rolling_dfs)
    for col in ['player_agg_freq_roll', 'player_fold_freq_roll', 'player_pass_freq_roll', 'player_fold_freq']:
        df[col] = rolling_df[col].reindex(df.index, fill_value=0.0)
    return df

def preprocess_data(df):
    le_street = LabelEncoder()
    le_position = LabelEncoder()
    le_texture = LabelEncoder()

    df["street_encoded"] = le_street.fit_transform(df["street"])
    df["position_encoded"] = le_position.fit_transform(df["position"])
    df["board_texture_encoded"] = le_texture.fit_transform(df["board_texture"])

    numeric_features = ["pot_size", "stack_to_pot_ratio", "effective_stack", "board_connectedness",
                        "is_preflop_aggressor", "bet_size", "bet_size_to_pot_ratio", "prev_bet_size",
                        "player_agg_freq", "player_agg_freq_roll", "player_fold_freq_roll", "player_pass_freq_roll",
                        "player_fold_freq"]

    for feature in numeric_features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    df["pot_size"] = df["pot_size"].fillna(1.0)
    df["stack_to_pot_ratio"] = df["stack_to_pot_ratio"].fillna(1000.0).clip(upper=1000.0)
    df["effective_stack"] = df["effective_stack"].fillna(100.0)
    df["board_connectedness"] = df["board_connectedness"].fillna(0.0)
    df["is_preflop_aggressor"] = df["is_preflop_aggressor"].fillna(0).astype(int)
    df["bet_size"] = df["bet_size"].fillna(0.0)
    df["bet_size_to_pot_ratio"] = df["bet_size_to_pot_ratio"].fillna(0.0)
    df["prev_bet_size"] = df["prev_bet_size"].fillna(0.0)
    df["player_agg_freq"] = df["player_agg_freq"].fillna(0.0)
    df["player_agg_freq_roll"] = df["player_agg_freq_roll"].fillna(0.0)
    df["player_fold_freq_roll"] = df["player_fold_freq_roll"].fillna(0.0)
    df["player_pass_freq_roll"] = df["player_pass_freq_roll"].fillna(0.0)
    df["player_fold_freq"] = df["player_fold_freq"].fillna(0.0)

    return df
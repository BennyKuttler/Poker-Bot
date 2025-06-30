# Phase 1 (Updated)
import re
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import xgboost as xgb
import os
import pickle
import logging

# Set up logging
logging.basicConfig(filename='/Users/bennykuttler/Downloads/Poker Bot/processing.log', 
                    level=logging.WARNING, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Step 1: Data Processing and Feature Engineering ---
POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]

def parse_value(value):
    value = value.strip()
    if value == "true":
        return True
    elif value == "false":
        return False
    elif value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(", ")
        return [parse_item(item) for item in items]
    elif value.replace(".", "").isdigit():
        return float(value)
    return value

def parse_item(item):
    item = item.strip()
    if item.startswith("'") and item.endswith("'"):
        return item.strip("'")
    try:
        return float(item) if item.replace(".", "").isdigit() else item
    except (ValueError, TypeError):
        return item

def board_texture(board):
    if not board:
        return "none"
    ranks = [card[:-1] for card in board]
    suits = [card[-1] for card in board]
    if len(set(ranks)) < len(ranks):
        return "paired"
    rank_values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    nums = sorted([rank_values[r] for r in ranks])
    flush_draw = len(set(suits)) == 1 or (len(board) == 4 and len(set(suits)) == 2)
    straight_draw = any(nums[i+2] - nums[i] <= 4 for i in range(len(nums)-2)) if len(nums) >= 3 else False
    return "wet" if flush_draw or straight_draw else "dry"

def board_connectedness(board):
    if not board:
        return 0.0
    rank_values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    nums = sorted([rank_values[card[:-1]] for card in board])
    if len(nums) < 2:
        return 0.0
    gaps = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    max_gap = max(gaps)
    return 1 - (sum(gaps) / (len(gaps) * 5)) if max_gap <= 5 else 1.0

def parse_prev_actions(prev_actions):
    stats = {"cbr": 0, "f": 0, "cc": 0, "total": 0}
    last_bet_size = 0.0
    for action in prev_actions:
        if action.startswith("cbr"):
            stats["cbr"] += 1
            try:
                bet_parts = action.split()
                if len(bet_parts) > 1:
                    last_bet_size = float(bet_parts[1])
            except (ValueError, IndexError):
                pass
        elif action == "f":
            stats["f"] += 1
        elif action == "cc":
            stats["cc"] += 1
        stats["total"] += 1
    return stats, last_bet_size

def parse_hand(hand_text):
    try:
        hand = {}
        for line in hand_text.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                hand[key.strip()] = parse_value(value)
        
        hand_id = hand.get("hand", "unknown")
        players = hand.get("players", [])
        seats = hand.get("seats", [])
        blinds = hand.get("blinds_or_straddles", [])
        stacks = hand.get("starting_stacks", [])
        actions = hand.get("actions", [])

        numeric_blinds = [float(b) if isinstance(b, (int, float, str)) and str(b).replace(".", "").isdigit() else 0 for b in blinds]
        big_blind = max(numeric_blinds) if numeric_blinds else 1
        if big_blind <= 0:
            big_blind = 1

        position_map = {}
        for i, seat in enumerate(seats):
            position_map[f"p{i + 1}"] = POSITIONS[i % len(POSITIONS)]

        decision_points = []
        street = "preflop"
        pot_size = sum(numeric_blinds[:2]) / big_blind if len(numeric_blinds) >= 2 else 0
        board = []
        current_stacks = [float(stack) / big_blind if stack and str(stack).replace(".", "").isdigit() else 0 for stack in stacks]
        effective_stack = min(current_stacks) if current_stacks else 0
        
        prev_actions = []
        preflop_aggressor = None
        street_actions = {"preflop": [], "flop": [], "turn": [], "river": []}

        for action in actions:
            if action.startswith("d db"):
                card_str = action.split(" ", 2)[2]
                cards = re.findall(r"[2-9TJQKA][shdc]", card_str)
                if len(cards) == 3:
                    street = "flop"
                    board = cards
                elif len(cards) == 1:
                    street = "turn" if len(board) == 3 else "river"
                    board.append(cards[0])
                prev_actions = []
                continue
            elif action.startswith("d dh"):
                continue
            
            parts = action.split(" ", 1)
            if len(parts) < 2:
                logging.warning(f"Skipping action due to insufficient parts: {action}")
                continue
            
            player, act = parts
            bet_size = 0
            action_type = "unknown"
            
            if "cbr" in act:
                try:
                    bet_parts = act.split()
                    if len(bet_parts) > 1:
                        bet_size = float(bet_parts[1]) / big_blind
                    action_type = "cbr"
                    if street == "preflop" and bet_size > 1:
                        player_idx = int(player[1]) - 1 if player[1:].isdigit() else 0
                        if 0 <= player_idx < len(players):
                            preflop_aggressor = players[player_idx]
                except (ValueError, TypeError, IndexError):
                    bet_size = 0
                    action_type = "cbr"
            elif act == "f":
                action_type = "f"
            elif act == "cc":
                action_type = "cc"
            else:
                logging.warning(f"Skipping unrecognized action: {action}")
                continue

            try:
                player_idx = int(player[1]) - 1 if player[1:].isdigit() else 0
                if 0 <= player_idx < len(players) and player_idx < len(current_stacks):
                    player_id = players[player_idx]
                    stack_before = current_stacks[player_idx]
                    
                    # Compute stats from prior actions
                    prev_stats, prev_bet_size = parse_prev_actions(prev_actions)
                    
                    if bet_size > 0:
                        current_stacks[player_idx] -= bet_size
                        pot_size += bet_size
                    effective_stack = min(current_stacks) if current_stacks else 0

                    # Ensure bet_size_to_pot_ratio is calculated correctly
                    bet_size_to_pot_ratio = bet_size / pot_size if pot_size > 0 else 0.0

                    dp = {
                        "street": street,
                        "pot_size": pot_size,
                        "board_texture": board_texture(board),
                        "board_connectedness": board_connectedness(board),
                        "position": position_map.get(player, "OOP"),
                        "stack_to_pot_ratio": stack_before / pot_size if pot_size > 0 else 1000.0,
                        "effective_stack": effective_stack,
                        "player_id": player_id,
                        "action": action_type,
                        "bet_size": bet_size,
                        "bet_size_to_pot_ratio": bet_size_to_pot_ratio,
                        "prev_bet_size": prev_bet_size,
                        "prev_actions": "|".join(prev_actions),
                        "is_preflop_aggressor": int(player_id == preflop_aggressor),
                        "player_agg_freq": prev_stats["cbr"] / max(1, prev_stats["total"])
                    }
                    decision_points.append(dp)
                    prev_actions.append(action_type)
                    street_actions[street].append(action_type)
            except (ValueError, IndexError, KeyError, ZeroDivisionError) as e:
                logging.warning(f"Skipping malformed action: {action} - {str(e)}")
                continue

        return {"hand_id": hand_id, "decision_points": decision_points}
    except Exception as e:
        logging.warning(f"Failed to parse hand: {str(e)}")
        return {"hand_id": "error", "decision_points": []}

# In Phase1.py, after creating the DataFrame in process_folder
def process_folder(base_folder_path, output_csv):
    all_rows = []
    total_hands = 0
    errors = 0
    
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    for root, dirs, files in os.walk(base_folder_path):
        for filename in sorted(files):
            if filename.endswith(".phhs"):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r") as f:
                        text = f.read()
                    
                    hand_blocks = re.split(r"\[\d+\]", text)[1:]
                    hands = []
                    for block in hand_blocks:
                        hand_text = block.strip()
                        if hand_text:
                            try:
                                parsed_hand = parse_hand(hand_text)
                                if parsed_hand and parsed_hand.get("decision_points"):
                                    hands.append(parsed_hand)
                            except Exception as e:
                                errors += 1
                                logging.warning(f"Error parsing hand in {file_path}: {str(e)}")
                    
                    total_hands += len(hands)
                    print(f"Processed {len(hands)} hands from {file_path}")
                    
                    for hand in hands:
                        for dp in hand["decision_points"]:
                            all_rows.append(dp)
                    
                    if len(all_rows) >= 10000:
                        df_chunk = pd.DataFrame(all_rows)
                        df_chunk.to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False)
                        all_rows = []
                except Exception as e:
                    logging.warning(f"Error processing file {file_path}: {str(e)}")
    
    if all_rows:
        df_chunk = pd.DataFrame(all_rows)
        df_chunk.to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False)
    
    # Analyze the distribution of bet_size_to_pot_ratio for cbr actions
    df = pd.DataFrame(all_rows)
    if "bet_size_to_pot_ratio" in df.columns and "action" in df.columns:
        cbr_df = df[df["action"] == "cbr"]
        if not cbr_df.empty:
            print("\n=== Distribution of bet_size_to_pot_ratio for cbr actions ===")
            print(cbr_df["bet_size_to_pot_ratio"].describe())
            print("==========================================\n")
    
    print(f"Total hands processed: {total_hands}")
    if errors > 0:
        print(f"Hands with errors: {errors} (see processing.log for details)")
    return total_hands

# --- Step 2: Data Preparation ---
def prepare_data(df):
    df = df.dropna(subset=["action"])
    df["action_type"] = df["action"].apply(lambda x: x)
    
    le_street = LabelEncoder()
    le_position = LabelEncoder()
    le_texture = LabelEncoder()
    le_action = LabelEncoder()
    
    df["street_encoded"] = le_street.fit_transform(df["street"])
    df["position_encoded"] = le_position.fit_transform(df["position"])
    df["board_texture_encoded"] = le_texture.fit_transform(df["board_texture"])
    df["action_type_encoded"] = le_action.fit_transform(df["action_type"])
    
    numeric_features = ["pot_size", "stack_to_pot_ratio", "effective_stack", "board_connectedness",
                        "is_preflop_aggressor", "bet_size", "bet_size_to_pot_ratio", "prev_bet_size",
                        "player_agg_freq"]
    
    for feature in numeric_features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    df["pot_size"] = df["pot_size"].fillna(1.0)
    df["stack_to_pot_ratio"] = df["stack_to_pot_ratio"].fillna(1000.0)
    df["effective_stack"] = df["effective_stack"].fillna(100.0)
    df["board_connectedness"] = df["board_connectedness"].fillna(0.0)
    df["is_preflop_aggressor"] = df["is_preflop_aggressor"].fillna(0).astype(int)
    df["bet_size"] = df["bet_size"].fillna(0.0)
    df["bet_size_to_pot_ratio"] = df["bet_size_to_pot_ratio"].fillna(0.0)
    df["prev_bet_size"] = df["prev_bet_size"].fillna(0.0)
    df["player_agg_freq"] = df["player_agg_freq"].fillna(0.0)
    
    df["stack_to_pot_ratio"] = df["stack_to_pot_ratio"].clip(upper=1000.0)
    df["bet_size_to_pot_ratio"] = df["bet_size_to_pot_ratio"].clip(lower=0.0, upper=2.0)  # Cap at 200% of pot
    
    features = [
        "street_encoded", "pot_size", "position_encoded", "stack_to_pot_ratio",
        "effective_stack", "board_texture_encoded", "board_connectedness",
        "is_preflop_aggressor", "bet_size_to_pot_ratio", "prev_bet_size",
        "player_agg_freq"
    ]
    
    X = df[features]
    y_action = df["action_type_encoded"]
    y_bet_ratio = df["bet_size_to_pot_ratio"]  # Use bet_size_to_pot_ratio as target
    
    cbr_mask = df["action_type"] == "cbr"
    X_bet = X[cbr_mask]
    y_bet_ratio = y_bet_ratio[cbr_mask]
    
    return X, y_action, X_bet, y_bet_ratio, le_action.classes_

# --- Step 3: Model Training with Cross-Validation ---
def train_models(X, y_action, X_bet, y_bet_ratio, action_classes):
    if len(X) < 100 or len(X_bet) < 100:
        print("Not enough data to train models. Need at least 100 samples.")
        return None, None
    
    # Balanced regularization for action model
    params = {
        'objective': 'multi:softprob',
        'num_class': len(action_classes),
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 2.0,
        'alpha': 1.0,
        'seed': 42
    }
    
    # 5-Fold CV for Action Classification
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    action_accuracy = []
    action_f1 = []
    action_rmse = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_action.iloc[train_idx], y_action.iloc[test_idx]
        
        X_train_temp, X_val, y_train_temp, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
        
        dtrain = xgb.DMatrix(X_train_temp, label=y_train_temp)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        clf = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, 'train'), (dval, 'val')], 
                        early_stopping_rounds=10, verbose_eval=False)
        
        y_test_pred = clf.predict(dtest).argmax(axis=1)
        y_test_prob = clf.predict(dtest)
        
        report = classification_report(y_test, y_test_pred, target_names=action_classes, digits=6, output_dict=True)
        action_accuracy.append(report['accuracy'])
        action_f1.append(report['macro avg']['f1-score'])
        action_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_prob.argmax(axis=1))))
    
    print("5-Fold CV Action Classification Results:")
    print(f"Mean Accuracy: {np.mean(action_accuracy):.6f} (+/- {np.std(action_accuracy) * 2:.6f})")
    print(f"Mean Macro F1-Score: {np.mean(action_f1):.6f} (+/- {np.std(action_f1) * 2:.6f})")
    print(f"Mean RMSE Loss: {np.mean(action_rmse):.6f} (+/- {np.std(action_rmse) * 2:.6f})")
    
    # Train final action model
    X_train_temp, X_val, y_train_temp, y_val = train_test_split(X, y_action, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train_temp, label=y_train_temp)
    dval = xgb.DMatrix(X_val, label=y_val)
    clf = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, 'train'), (dval, 'val')], 
                    early_stopping_rounds=10, verbose_eval=False)
    
    # Bet size regression (predict bet_size_to_pot_ratio) with 5-Fold CV
    if len(X_bet) > 100:
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
        
        bet_mse = []
        for train_idx, test_idx in kf.split(X_bet):
            X_bet_train, X_bet_test = X_bet.iloc[train_idx], X_bet.iloc[test_idx]
            y_bet_train, y_bet_test = y_bet_ratio.iloc[train_idx], y_bet_ratio.iloc[test_idx]
            
            X_bet_train_temp, X_bet_val, y_bet_train_temp, y_bet_val = train_test_split(X_bet_train, y_bet_train, test_size=0.125, random_state=42)
            
            dtrain_bet = xgb.DMatrix(X_bet_train_temp, label=y_bet_train_temp)
            dval_bet = xgb.DMatrix(X_bet_val, label=y_bet_val)
            dtest_bet = xgb.DMatrix(X_bet_test, label=y_bet_test)
            
            reg = xgb.train(params_bet, dtrain_bet, num_boost_round=100, evals=[(dtrain_bet, 'train'), (dval_bet, 'val')], 
                            early_stopping_rounds=10, verbose_eval=False)
            
            y_bet_test_pred = reg.predict(dtest_bet)
            bet_mse.append(mean_squared_error(y_bet_test, y_bet_test_pred))
        
        print(f"5-Fold CV Bet Size (Ratio) MSE: {np.mean(bet_mse):.6f} (+/- {np.std(bet_mse) * 2:.6f})")
        
        # Train final bet size model
        X_bet_train_temp, X_bet_val, y_bet_train_temp, y_bet_val = train_test_split(X_bet, y_bet_ratio, test_size=0.2, random_state=42)
        dtrain_bet = xgb.DMatrix(X_bet_train_temp, label=y_bet_train_temp)
        dval_bet = xgb.DMatrix(X_bet_val, label=y_bet_val)
        reg = xgb.train(params_bet, dtrain_bet, num_boost_round=100, evals=[(dtrain_bet, 'train'), (dval_bet, 'val')], 
                        early_stopping_rounds=10, verbose_eval=False)
    else:
        reg = None
        print("Not enough betting actions to train bet size model")
    
    print("Feature Importance (Action Classifier):")
    importance = clf.get_score(importance_type='gain')
    for feature, imp in importance.items():
        print(f"{feature}: {imp:.6f}")
    
    return clf, reg

# --- Main Execution ---
if __name__ == "__main__":
    folder_path = "/Users/bennykuttler/Downloads/Poker Bot/hand_histories"
    output_csv = "/Users/bennykuttler/Downloads/Poker Bot/poker_data.csv"
    
    try:
        total_hands = process_folder(folder_path, output_csv)
        print(f"Total hands processed: {total_hands}")
        
        try:
            df = pd.read_csv(output_csv, on_bad_lines="skip")
            print(f"Total decision points: {len(df)}")
            X, y_action, X_bet, y_bet_ratio, action_classes = prepare_data(df)
            clf, reg = train_models(X, y_action, X_bet, y_bet_ratio, action_classes)
            
            if clf is not None:
                with open("/Users/bennykuttler/Downloads/Poker Bot/action_model.pkl", "wb") as f:
                    pickle.dump(clf, f)
                if reg is not None:
                    with open("/Users/bennykuttler/Downloads/Poker Bot/bet_model.pkl", "wb") as f:
                        pickle.dump(reg, f)
                print("Models saved successfully")
        except pd.errors.ParserError as e:
            print(f"Error loading CSV: {e}")
        except Exception as e:
            print(f"Error in data processing or model training: {e}")
    except Exception as e:
        print(f"Fatal error: {e}")
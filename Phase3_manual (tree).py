# Phase 3 Manual Version (Tree) — Integrated with strategy_tree.py for EV Lookahead
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import json

try:
    from treys import Card, Evaluator
    evaluator = Evaluator()
    USE_TREYS = True
except ImportError:
    USE_TREYS = False

from strategy_tree import simulate_strategy_tree

# --- Compute Treys Strength ---
def compute_hand_strength_treys(hole_cards_str, board_cards_list):
    if not USE_TREYS or not hole_cards_str:
        return 0.5
    hole = hole_cards_str.split()
    if len(hole) + len(board_cards_list) < 5:
        return 0.5
    try:
        hole_treys = [Card.new(c) for c in hole]
        board_treys = [Card.new(c) for c in board_cards_list]
        score = evaluator.evaluate(board_treys, hole_treys)
        return (7463 - score) / 7462.0
    except:
        return 0.5

# --- Model Wrapper ---
class XGBWrapper:
    def __init__(self, booster, feature_names):
        self.booster = booster
        self.feature_names = feature_names

    def predict(self, X_df):
        dmatrix = xgb.DMatrix(X_df[self.feature_names], feature_names=self.feature_names)
        return self.booster.predict(dmatrix)

# --- Main CLI Loop ---
def main_loop_exploit_tree():
    with open("/Users/bennykuttler/Downloads/Poker Bot/action_model_phase2_retrained.pkl", "rb") as f:
        booster = pickle.load(f)
    with open("/Users/bennykuttler/Downloads/Poker Bot/bet_model_phase2_retrained.pkl", "rb") as f:
        bet_model = pickle.load(f)
    with open("/Users/bennykuttler/Downloads/Poker Bot/bet_size_scaler.pkl", "rb") as f:
        bet_scaler = pickle.load(f)

    feature_names = [
        "street_encoded", "pot_size", "position_encoded", "stack_to_pot_ratio",
        "effective_stack", "board_texture_encoded", "board_connectedness",
        "is_preflop_aggressor", "bet_size_to_pot_ratio", "prev_bet_size",
        "player_agg_freq", "player_agg_freq_roll", "player_fold_freq_roll",
        "player_pass_freq_roll", "player_fold_freq", "hand_strength"
    ]
    wrapped_model = XGBWrapper(booster, feature_names)

    while True:
        print("\n=== Exploitative Decision (EV Tree) ===")
        pot = float(input("Pot size (BB): ").strip())
        stack = float(input("Effective stack (BB): ").strip())
        current_bet = float(input("Current bet size (BB): ").strip())
        hole = input("Enter your hole cards (e.g. 'As Kd'): ").strip()
        board = input("Enter board cards (e.g. '7h 9c 2d') or press Enter: ").strip()
        board_cards = board.split() if board else []
        strength = compute_hand_strength_treys(hole, board_cards)
        print(f"Hand strength = {strength:.3f}")

        state = {
            "street_encoded": 1,
            "pot_size": pot,
            "position_encoded": 0,
            "stack_to_pot_ratio": stack / pot if pot > 0 else 1000,
            "effective_stack": stack,
            "board_texture_encoded": 0,
            "board_connectedness": 0.5,
            "is_preflop_aggressor": 1,
            "bet_size_to_pot_ratio": current_bet / pot if pot > 0 else 0.0,
            "prev_bet_size": current_bet,
            "player_agg_freq": 0.3,
            "player_agg_freq_roll": 0.25,
            "player_fold_freq_roll": 0.3,
            "player_pass_freq_roll": 0.2,
            "player_fold_freq": 0.35,
            "hand_strength": strength,
            "hero_strength": strength,
            "bet_size": current_bet
        }

        ev_trace = []
        try:
            action, ev_value = simulate_strategy_tree(state, wrapped_model, bet_model, bet_scaler, depth=1, max_depth=3, ev_trace=ev_trace)
        except IndexError as e:
            print(f"[ERROR] Action model returned insufficient class probabilities: {e}")
            print("Please ensure your model was trained with the correct number of action classes.")
            return

        print(f"\nExploitative EV (tree-simulated): {ev_value:.2f}")
        print(f"Recommended Action: {action.upper()}")

        for trace in reversed(ev_trace):
            if trace[1] == 'cbr':
                suggested_ratio = trace[3] if isinstance(trace[3], float) else 0.05
                break
        else:
            suggested_ratio = 0.05

        suggested_bet = max(min(suggested_ratio * pot, stack), 1.0)
        print(f"Suggested bet size: {suggested_bet:.2f} BB ({suggested_ratio:.2f}x pot)")

        with open("ev_trace_log.json", "w") as f:
            json.dump([{
                "depth": d, "action": a, "ev": ev, "probability": prob
            } for d, a, ev, prob in ev_trace], f, indent=2)

        print("EV trace saved to ev_trace_log.json")
        again = input("\nTry another hand? (y/n): ").strip().lower()
        if again != 'y':
            break

if __name__ == "__main__":
    main_loop_exploit_tree()

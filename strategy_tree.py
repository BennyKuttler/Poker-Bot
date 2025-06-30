# strategy_tree.py

import pandas as pd
import numpy as np
import xgboost as xgb

def simulate_strategy_tree(current_state, model, bet_model, bet_scaler, depth=1, max_depth=3, ev_trace=None):
    """
    Recursively simulate EV from current state using action + bet size models.
    """
    if depth > max_depth:
        return current_state, 0.0

    input_df = pd.DataFrame([current_state])
    dmat = xgb.DMatrix(input_df[model.feature_names], feature_names=model.feature_names)

    # Predict action distribution
    action_probs = model.predict(input_df)[0]
    
    # Infer available actions dynamically
    available_actions = model.booster.classes_ if hasattr(model.booster, "classes_") else ['cc', 'cbr'][:len(action_probs)]

    # Predict bet size for CBR
    scaled_bet = bet_model.predict(dmat)[0]
    bet_ratio = bet_scaler.inverse_transform([[scaled_bet]])[0][0]
    bet_ratio = max(bet_ratio, 0.05)

    strength = current_state.get("hand_strength", 0.5)
    pot = current_state.get("pot_size", 10)
    stack = current_state.get("effective_stack", 100)
    bet_size = max(min(bet_ratio * pot, stack), 1.0)

    evs = {}
    for i, action in enumerate(available_actions):
        prob = float(action_probs[i])
        next_state = current_state.copy()

        if action == "f":
            ev = 0.0
        elif action == "cc":
            next_state["pot_size"] += current_state.get("bet_size", 0)
            next_state["effective_stack"] -= current_state.get("bet_size", 0)
            ev = strength * next_state["pot_size"]
        elif action == "cbr":
            next_state["pot_size"] += bet_size
            next_state["effective_stack"] -= bet_size
            next_state["bet_size"] = bet_size
            ev = strength * next_state["pot_size"] - bet_size
        else:
            ev = 0.0

        # Recursive lookahead
        if depth < max_depth:
            _, future_ev = simulate_strategy_tree(next_state, model, bet_model, bet_scaler, depth+1, max_depth, ev_trace)
            ev += future_ev * prob

        evs[action] = ev
        if ev_trace is not None:
            ev_trace.append((depth, action, ev, prob))

    best_action = max(evs, key=evs.get)
    return best_action, evs[best_action]
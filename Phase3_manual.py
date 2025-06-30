# -*- coding: utf-8 -*-
"""exploitative_poker_bot.py

End‑to‑end implementation of a **purely data‑driven, exploitative** No‑Limit‑Hold‑em bot.
The file contains three logical sections that can be run independently:

1.  **Phase 1 – Hand‑history parsing & feature extraction**
2.  **Phase 2 – Model training** (action classifier + bet‑size regressor)
3.  **Phase 3 – Interactive live bot**

All critical unit‑mismatches and leakage problems in the original code have been fixed.
The script uses *XGBoost*; install dependencies with:

```bash
pip install pandas numpy scikit‑learn xgboost treys
```

© 2025  – Feel free to reuse / adapt.
"""

###############################################################################
#  Imports & constants                                                         #
###############################################################################

from __future__ import annotations
import os, re, json, time, random, pickle, logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import xgboost as xgb

try:
    from treys import Card, Evaluator  # type: ignore
    TREYS_AVAILABLE = True
except ImportError:
    TREYS_AVAILABLE = False

POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
DISCRETE_SIZES = [0.33, 0.50, 0.66, 1.00, 1.25, 1.50, 2.00]  # pot‑ratios

###############################################################################
#  Utility helpers                                                             #
###############################################################################

RANK_VALUES = {**{str(n): n for n in range(2, 10)}, **{"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}}

logging.basicConfig(
    filename=str(Path.home() / "poker_bot_processing.log"),
    level=logging.INFO,
    format="%(asctime)s – %(levelname)s – %(message)s",
)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


###############################################################################
#  Phase 1 – hand‑history parsing                                             #
###############################################################################

def _parse_value(raw: str):
    raw = raw.strip()
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw.startswith("[") and raw.endswith("]"):
        return [_parse_value(x) for x in re.split(r",\s*", raw[1:-1]) if x]
    try:
        return float(raw) if raw.replace(".", "").isdigit() else raw
    except ValueError:
        return raw


def _board_texture(board: List[str]) -> str:
    if not board:
        return "none"
    ranks = [c[:-1] for c in board]
    suits = [c[-1] for c in board]
    if len(set(ranks)) < len(ranks):
        return "paired"
    flush_draw = len(set(suits)) == 1 or (len(board) == 4 and len(set(suits)) <= 2)
    nums = sorted(RANK_VALUES[r] for r in ranks)
    straight_draw = any(nums[i + 2] - nums[i] <= 4 for i in range(len(nums) - 2)) if len(nums) >= 3 else False
    return "wet" if flush_draw or straight_draw else "dry"


def _board_connectedness(board: List[str]) -> float:
    if len(board) < 2:
        return 0.0
    nums = sorted(RANK_VALUES[c[:-1]] for c in board)
    gaps = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
    return 1 - (sum(gaps) / (len(gaps) * 5))  # 0 (dry) .. 1 (straighty)


def _parse_prev_actions(prev: List[str]):
    stats = Counter()
    last_bet = 0.0
    for act in prev:
        kind, *rest = act.split()
        stats[kind] += 1
        if kind == "cbr" and rest:
            last_bet = _safe_float(rest[0])
    stats["total"] = max(stats["total"], len(prev))
    return stats, last_bet


def parse_hand(hand_text: str) -> Dict[str, Any]:
    """Parse a single *.phhs* hand block into decision‑point rows."""
    raw: Dict[str, Any] = {}
    for line in hand_text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            raw[k.strip()] = _parse_value(v)

    players: List[str] = raw.get("players", [])
    seats: List[str] = raw.get("seats", [])
    blinds: List[float] = [_safe_float(b) for b in raw.get("blinds_or_straddles", [])]
    stacks: List[float] = [_safe_float(s) for s in raw.get("starting_stacks", [])]
    actions: List[str] = raw.get("actions", [])

    bb = max(blinds) if blinds else 1.0
    pos_map = {f"p{i+1}": POSITIONS[i % len(POSITIONS)] for i in range(len(seats))}

    street = "preflop"
    board: List[str] = []
    pot = sum(blinds[:2]) / bb if len(blinds) >= 2 else 0.0
    cur_stacks = [s / bb for s in stacks]

    prev_actions: List[str] = []
    decision_points: List[Dict[str, Any]] = []
    preflop_agg: str | None = None

    for line in actions:
        # Deal lines ---------------------------------------------------------
        if line.startswith("d db"):
            cards = re.findall(r"[2-9TJQKA][shdc]", line)
            if len(cards) == 3:
                street = "flop"
                board = cards
            elif len(cards) == 1:
                street = "turn" if len(board) == 3 else "river"
                board.append(cards[0])
            prev_actions.clear()
            continue
        if line.startswith("d dh"):
            continue  # ignore hole‑card info (not available online)

        # Action lines -------------------------------------------------------
        pl_raw, act_raw = line.split(" ", 1)
        player_idx = int(pl_raw[1:]) - 1  # p1‑indexed
        if player_idx >= len(players):
            continue  # malformed
        player_id = players[player_idx]
        kind, *rest = act_raw.split()
        bet_bb = _safe_float(rest[0]) / bb if rest else 0.0

        if kind == "cbr" and street == "preflop" and bet_bb > 1:
            preflop_agg = player_id

        stats_prev, last_bet = _parse_prev_actions(prev_actions)
        pot_before = pot
        if bet_bb:
            cur_stacks[player_idx] -= bet_bb
            pot += bet_bb
        spr = cur_stacks[player_idx] / pot if pot else 1000.0

        dp = {
            "street": street,
            "pot_size": pot_before,  # pot BEFORE action
            "board_texture": _board_texture(board),
            "board_connectedness": _board_connectedness(board),
            "position": pos_map.get(pl_raw, "OOP"),
            "stack_to_pot_ratio": spr,
            "effective_stack": min(cur_stacks),
            "player_id": player_id,
            "action": kind,
            "bet_size": bet_bb,
            "bet_ratio": bet_bb / pot_before if pot_before else 0.0,
            "prev_bet_size": last_bet,
            "is_preflop_aggressor": int(player_id == preflop_agg),
            "player_agg_freq": stats_prev["cbr"] / max(1, stats_prev["total"]),
        }
        decision_points.append(dp)
        prev_actions.append(kind + (f" {bet_bb}" if bet_bb else ""))

    return {"hand_id": raw.get("hand", "unknown"), "decision_points": decision_points}


# ---------------------------------------------------------------------------
#  Bulk processing                                                             
# ---------------------------------------------------------------------------

def process_folder(folder: str | Path, out_csv: str | Path, flush_every: int = 10000) -> None:
    folder = Path(folder)
    out_csv = Path(out_csv)
    if out_csv.exists():
        out_csv.unlink()

    rows: List[Dict[str, Any]] = []
    for file in sorted(folder.rglob("*.phhs")):
        text = file.read_text(errors="ignore")
        for block in re.split(r"\[\d+\]", text)[1:]:
            parsed = parse_hand(block.strip())
            for dp in parsed["decision_points"]:
                rows.append(dp)
            if len(rows) >= flush_every:
                pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
                rows.clear()
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)

###############################################################################
#  Phase 2 – data prep & model training                                        #
###############################################################################

def _label_encode(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    df[col] = df[col].fillna("missing")
    df[col + "_enc"] = le.fit_transform(df[col])
    return df[col + "_enc"].values, le


def prepare_dataset(csv_path: str | Path):
    df = pd.read_csv(csv_path, on_bad_lines="skip")

    # Encode categorical ----------------------------------------------------
    _, le_street = _label_encode(df, "street")
    _, le_pos = _label_encode(df, "position")
    _, le_tex = _label_encode(df, "board_texture")

    # Numerics --------------------------------------------------------------
    num_cols = [
        "pot_size",
        "stack_to_pot_ratio",
        "effective_stack",
        "board_connectedness",
        "is_preflop_aggressor",
        "prev_bet_size",
        "player_agg_freq",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Clip / normalise main ratios -----------------------------------------
    df["stack_to_pot_ratio"] = df["stack_to_pot_ratio"].clip(0, 1000)

    # Final feature list (NO leakage of bet_ratio!) ------------------------
    FEATURES = [
        "street_enc",
        "pot_size",
        "position_enc",
        "stack_to_pot_ratio",
        "effective_stack",
        "board_texture_enc",
        "board_connectedness",
        "is_preflop_aggressor",
        "prev_bet_size",
        "player_agg_freq",
    ]

    X = df[FEATURES]
    y_action = LabelEncoder().fit_transform(df["action"].astype(str))

    # Bet‑size regressor uses only C‑bets ----------------------------------
    cbr_mask = df["action"] == "cbr"
    X_bet = X[cbr_mask]
    y_bet_raw = df.loc[cbr_mask, "bet_ratio"].clip(0.0, 2.0)  # 0 .. 2×pot

    # Scale bet target ------------------------------------------------------
    scaler = MinMaxScaler((0, 1))
    y_bet = scaler.fit_transform(y_bet_raw.to_numpy().reshape(-1, 1)).ravel()

    return X, y_action, X_bet, y_bet, scaler, FEATURES


def train_models(
    X: pd.DataFrame,
    y_action: np.ndarray,
    X_bet: pd.DataFrame,
    y_bet: np.ndarray,
    scaler: MinMaxScaler,
    FEATURES: List[str],
    save_dir: str | Path = "models",
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Action classifier -----------------------------------------------------
    params_cls = dict(
        objective="multi:softprob",
        num_class=len(np.unique(y_action)),
        max_depth=6,
        eta=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        seed=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y_action, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURES)

    clf = xgb.train(params_cls, dtrain, num_boost_round=400, evals=[(dval, "val")], early_stopping_rounds=20)
    clf.save_model(Path(save_dir) / "action_model.json")

    # Bet‑size regressor ----------------------------------------------------
    if len(X_bet) < 100:
        logging.warning("Not enough c‑bet samples for bet‑size model. Skipping …")
        reg = None
    else:
        params_reg = dict(
            objective="reg:squarederror",
            max_depth=6,
            eta=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="rmse",
            seed=42,
        )
        Xb_tr, Xb_val, yb_tr, yb_val = train_test_split(X_bet, y_bet, test_size=0.2, random_state=42)
        dtr = xgb.DMatrix(Xb_tr, label=yb_tr, feature_names=FEATURES)
        dval = xgb.DMatrix(Xb_val, label=yb_val, feature_names=FEATURES)
        reg = xgb.train(params_reg, dtr, 400, evals=[(dval, "val")], early_stopping_rounds=20)
        reg.save_model(Path(save_dir) / "bet_model.json")
        pickle.dump(scaler, open(Path(save_dir) / "bet_scaler.pkl", "wb"))

    print("Models saved to", save_dir)
    return clf, reg, scaler

###############################################################################
#  Phase 3 – Live bot                                                         #
###############################################################################

class LiveBot:
    """Interactive console bot using the trained models."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.clf = xgb.Booster()
        self.clf.load_model(model_dir / "action_model.json")
        self.reg = None
        reg_path = model_dir / "bet_model.json"
        if reg_path.exists():
            self.reg = xgb.Booster()
            self.reg.load_model(reg_path)
            self.scaler: MinMaxScaler = pickle.load(open(model_dir / "bet_scaler.pkl", "rb"))
        else:
            print("[!] Bet‑size model not found – bot will default to ⅔‑pot.")
        self.history = pd.DataFrame(columns=["player_id", "action", "timestamp"])
        self.feature_names = [
            "street_enc",
            "pot_size",
            "position_enc",
            "stack_to_pot_ratio",
            "effective_stack",
            "board_texture_enc",
            "board_connectedness",
            "is_preflop_aggressor",
            "prev_bet_size",
            "player_agg_freq",
        ]
        # encoders (frozen to training order) -------------------------------
        self.le_street = LabelEncoder().fit(["preflop", "flop", "turn", "river", "missing"])
        self.le_pos = LabelEncoder().fit(POSITIONS + ["OOP", "missing"])
        self.le_tex = LabelEncoder().fit(["none", "dry", "wet", "paired", "missing"])
        self.eval = Evaluator() if TREYS_AVAILABLE else None

    # ---------------------------------------------------------------------
    #  Opponent stats
    # ---------------------------------------------------------------------
    def _rolling_stats(self, pid: str, window: int = 50):
        df = self.history[self.history.player_id == pid].sort_values("timestamp")
        if df.empty:
            return dict(agg=0.2, agg_roll=0.2, fold_roll=0.3)
        recent = df.tail(window)
        agg_roll = (recent.action == "cbr").mean()
        fold_roll = (recent.action == "f").mean()
        agg = (df.action == "cbr").mean()
        return dict(agg=agg, agg_roll=agg_roll, fold_roll=fold_roll)

    # ---------------------------------------------------------------------
    #  Hand strength (7‑card)
    # ---------------------------------------------------------------------
    def _hand_strength(self, hole: str, board: List[str]):
        if not (TREYS_AVAILABLE and hole):
            return 0.0
        cards_h = [Card.new(c) for c in hole.split()]
        cards_b = [Card.new(c) for c in board] if board else []
        if len(cards_h) + len(cards_b) < 5:
            return 0.0
        score = self.eval.evaluate(cards_b, cards_h)
        return (7463 - score) / 7462.0

    # ---------------------------------------------------------------------
    #  Feature row builder
    # ---------------------------------------------------------------------
    def _build_row(self, state: Dict[str, Any]):
        row = {
            "street_enc": self.le_street.transform([state["street"]])[0],
            "pot_size": state["pot"],
            "position_enc": self.le_pos.transform([state["position"]])[0],
            "stack_to_pot_ratio": state["spr"],
            "effective_stack": state["eff_stack"],
            "board_texture_enc": self.le_tex.transform([state["texture"]])[0],
            "board_connectedness": state["conn"],
            "is_preflop_aggressor": int(state["pfa"]),
            "prev_bet_size": state["prev_bet"],
            "player_agg_freq": state["opp_stats"]["agg"],
        }
        return pd.DataFrame([row])

    # ---------------------------------------------------------------------
    #  Discrete bet‑size helper
    # ---------------------------------------------------------------------
    def _nearest_bucket(self, ratio: float, pot: float, eff: float):
        if ratio * pot >= eff * 0.95:
            return "allin", eff
        best = min(DISCRETE_SIZES, key=lambda x: abs(x - ratio))
        return best, best * pot

    # ---------------------------------------------------------------------
    #  Main decision routine
    # ---------------------------------------------------------------------
    def decide(self, state: Dict[str, Any]):
        X = self._build_row(state)
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        probs = self.clf.predict(dmat)[0]  # order preserved from training (cbr, cc, f)

        # Bet‑size prediction ------------------------------------------------
        if self.reg is not None:
            ratio_scaled = self.reg.predict(dmat)[0]
            ratio = float(self.scaler.inverse_transform([[ratio_scaled]])[0][0])
        else:
            ratio = 0.66  # default ⅔‑pot
        ratio = float(np.clip(ratio, 0.10, 2.00))

        # Expected value (very coarse) --------------------------------------
        win_prob = max(0.15, state["hero_hs"])
        pot = state["pot"]
        bet_amt = ratio * pot
        call_ev = win_prob * (pot + bet_amt) - (1 - win_prob) * bet_amt
        evs = {
            "cbr": probs[0] * call_ev,  # oversimplified
            "cc": probs[1] * win_prob * pot,
            "f": 0.0,
        }
        best_action = max(evs, key=evs.get)
        if best_action == "cbr":
            bucket, amt = self._nearest_bucket(ratio, pot, state["eff_stack"])
            return "cbr", bucket, amt, probs
        return best_action, None, 0.0, probs

    # ---------------------------------------------------------------------
    #  Console loop
    # ---------------------------------------------------------------------
    def cli(self):
        ts = 0
        while True:
            print("\n--- New spot (blank line to quit) ---")
            brd = input("Board cards (e.g. 'Js 9s 2d' – empty = preflop): ").strip()
            if not brd and input("Exit? (y/N) ").lower().startswith("y"):
                break
            board = brd.split() if brd else []
            street = ["preflop", "flop", "turn", "river"][len(board)]
            pot = _safe_float(input("Pot in BB: "), 1.0)
            eff = _safe_float(input("Effective stack in BB: "), 50.0)
            prev = _safe_float(input("Previous bet (0 if none): "), 0.0)
            pos = input("Your position (BB/SB/UTG/MP/CO/BTN): ").strip().upper() or "BB"
            pfa = input("Are you pre‑flop aggressor? (y/N): ").lower().startswith("y")
            hole = input("Your hole cards (optional): ").strip()
            hero_hs = self._hand_strength(hole, board) if hole else 0.0

            opp_stats = self._rolling_stats("villain")
            state = dict(
                street=street,
                pot=pot,
                eff_stack=eff,
                spr=eff / pot if pot else 1000,
                position=pos,
                pfa=pfa,
                prev_bet=prev,
                texture=_board_texture(board),
                conn=_board_connectedness(board),
                hero_hs=hero_hs,
                opp_stats=opp_stats,
            )
            act, bucket, amt, probs = self.decide(state)
            print(f"\nModel probs  cbr={probs[0]:.3f}  cc={probs[1]:.3f}  f={probs[2]:.3f}")
            if act == "cbr":
                print(f"=> Bet {bucket if bucket=='allin' else f'{bucket:.2f}×pot'}  (≈ {amt:.1f} BB)")
            elif act == "cc":
                print("=> Call / check")
            else:
                print("=> Fold")

            # Log to history -------------------------------------------------
            self.history = pd.concat(
                [self.history, pd.DataFrame([{"player_id": "hero", "action": act, "timestamp": ts}])]
            )
            ts += 1


###############################################################################
#  __main__                                                                   #
###############################################################################

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Exploitative poker bot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("parse", help="Parse .phhs hand histories → CSV")
    p1.add_argument("folder", help="Folder with .phhs files")
    p1.add_argument("out", help="Output CSV path")

    p2 = sub.add_parser("train", help="Train models from CSV")
    p2.add_argument("csv", help="CSV produced by 'parse'")
    p2.add_argument("modeldir", help="Directory to store models")

    p3 = sub.add_parser("play", help="Run interactive bot")
    p3.add_argument("modeldir", help="Directory with saved models")

    args = ap.parse_args()

    if args.cmd == "parse":
        process_folder(args.folder, args.out)
        print("Parsing complete →", args.out)

    elif args.cmd == "train":
        X, y_act, Xb, yb, scaler, feats = prepare_dataset(args.csv)
        train_models(X, y_act, Xb, yb, scaler, feats, args.modeldir)

    elif args.cmd == "play":
        LiveBot(args.modeldir).cli()

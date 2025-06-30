# Phase3_live.py (Calibration Mode for Card Troubleshooting)
import pandas as pd
import numpy as np
import pickle
import time
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pyautogui
import cv2
import pytesseract
import re
import os

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Screen capture and OCR
def capture_screen_region(x, y, width, height):
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Enhance for dark background, white text (ClubGG)
    gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    # Try adaptive thresholding for graphical text
    gray_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Use neural net OCR and single block mode for better accuracy
    text = pytesseract.image_to_string(gray_adaptive, config='--psm 6 --oem 3')
    return text.strip()

# Template matching for card ranks and suits (optional—requires training images)
def match_template(image, templates, threshold=0.7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = {}
    for name, template in templates.items():
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w, h = template_gray.shape[::-1]
        res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            results[name] = True
    return max(results.keys(), key=lambda k: results.get(k, 0)) if results else ""

# Load card templates (example—create these manually or programmatically)
def load_card_templates():
    templates = {}
    # Example: Add paths to pre-cropped images of ranks (A, K, Q, J, 10-2) and suits (♠, ♥, ♦, ♣)
    # For simplicity, assume you’ve saved images in /Users/bennykuttler/Downloads/Poker Bot/card_templates/
    template_dir = "/Users/bennykuttler/Downloads/Poker Bot/card_templates"
    if os.path.exists(template_dir):
        for filename in os.listdir(template_dir):
            if filename.endswith('.png'):
                name = filename.split('.')[0]  # e.g., '10', 'spades'
                path = os.path.join(template_dir, filename)
                templates[name] = cv2.imread(path)
    return templates

# Load models
with open("/Users/bennykuttler/Downloads/Poker Bot/action_model_phase2_retrained.pkl", "rb") as f:
    clf = pickle.load(f)
with open("/Users/bennykuttler/Downloads/Poker Bot/bet_model_phase2.pkl", "rb") as f:
    reg = pickle.load(f)

# Prepare data
def compute_rolling_stats(df, window=100):
    if 'action' not in df.columns:
        for col in ['player_agg_freq_roll', 'player_fold_freq_roll', 'player_pass_freq_roll', 'player_fold_freq']:
            if col not in df.columns:
                df[col] = 0.0
        return df
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
    return df.merge(rolling_df, left_index=True, right_index=True, how='left')

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

# EV and decision logic
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
    ev_raise = 0
    evs["cbr"] = min(ev_fold + ev_call + ev_raise, pot_size)
    return evs

def decide_action(pred_probs, pred_bet_size, row):
    evs = estimate_ev(pred_probs, pred_bet_size, row["pot_size"], row["effective_stack"],
                      row["board_texture"], row["street"])
    best_action = max(evs, key=evs.get)
    bet_size = pred_bet_size if best_action == "cbr" else 0.0
    if best_action == "cbr" and bet_size <= 0:
        bet_size = row["pot_size"] * 0.5
    return best_action, bet_size, evs[best_action]

# Define regions and buttons (calibrated for ClubGG based on your refined coordinates)
regions = {
    "pot_size": (483, 535, 150, 50),  # Pot (e.g., "7.9 BB")—centered at 483, 535
    "effective_stack": (328, 736, 150, 50),  # Your stack (e.g., "3 BB")—centered at 328, 736
    "prev_bet_size": (481, 230, 150, 50),  # Previous bet (e.g., "Raise to 7.5 BB")—centered at 481, 230
    "position": (423, 668, 200, 50),  # Position (e.g., "BB")—centered at 423, 668
    # Individual card regions for rank and suit
    "card1_rank": (368, 411, 70, 50),  # Card 1 rank (e.g., "4")—centered at 368, 411, enlarged for graphics
    "card1_suit": (370, 433, 70, 50),  # Card 1 suit (e.g., "♦")—centered at 370, 433, enlarged for graphics
    "card2_rank": (421, 409, 70, 50),  # Card 2 rank (e.g., "Q")—centered at 421, 409
    "card2_suit": (423, 431, 70, 50),  # Card 2 suit (e.g., "♠")—centered at 423, 431
    "card3_rank": (475, 407, 70, 50),  # Card 3 rank (e.g., "4")—centered at 475, 407
    "card3_suit": (476, 432, 70, 50),  # Card 3 suit (e.g., "♠")—centered at 476, 432
    "card4_rank": (529, 410, 70, 50),  # Card 4 rank (e.g., "3")—centered at 529, 410
    "card4_suit": (530, 431, 70, 50),  # Card 4 suit (e.g., "♥")—centered at 530, 431
    "card5_rank": (582, 409, 70, 50),  # Card 5 rank (e.g., "6")—centered at 582, 409
    "card5_suit": (583, 432, 70, 50),  # Card 5 suit (e.g., "♥")—centered at 583, 432
}
buttons = {
    "fold": (550, 750),  # Fold button—near bottom center, adjusted
    "call": (650, 750),  # Call button—near bottom center, adjusted
    "raise": (750, 750),  # Raise button—near bottom center, adjusted
    "bet_input": (700, 800),  # Bet input field (adjust for half-pot area)—revised
}

# Load card templates for template matching
card_templates = load_card_templates()

def get_live_game_state():
    pot_size = 1.0
    effective_stack = 50.0
    board_texture = "dry"
    board_connectedness = 0.2
    street = "preflop"  # Default to preflop, update based on cards
    prev_bet_size = 0.0
    position = "BB"
    stack_to_pot_ratio = 1000.0
    bet_size_to_pot_ratio = 0.0
    stats = {"player_agg_freq": 0.3, "player_agg_freq_roll": 0.2, "player_fold_freq_roll": 0.1,
             "player_pass_freq_roll": 0.05, "player_fold_freq": 0.15}

    try:
        # Capture and extract—save images for debugging
        pot_image = capture_screen_region(*regions["pot_size"])
        cv2.imwrite('pot_debug.png', pot_image)  # Save for inspection
        pot_size_text = extract_text_from_image(pot_image)
        print(f"Pot size text: '{pot_size_text}'")
        pot_size_match = re.search(r'(\d+\.?\d*)', pot_size_text)
        pot_size = float(pot_size_match.group(1)) if pot_size_match else 1.0

        stack_image = capture_screen_region(*regions["effective_stack"])
        cv2.imwrite('stack_debug.png', stack_image)
        stack_text = extract_text_from_image(stack_image)
        print(f"Stack text: '{stack_text}'")
        stack_match = re.search(r'(\d+\.?\d*)', stack_text)
        effective_stack = float(stack_match.group(1)) if stack_match else 50.0

        prev_bet_image = capture_screen_region(*regions["prev_bet_size"])
        cv2.imwrite('prev_bet_debug.png', prev_bet_image)
        prev_bet_text = extract_text_from_image(prev_bet_image)
        print(f"Prev bet text: '{prev_bet_text}'")
        prev_bet_match = re.search(r'(\d+\.?\d*)', prev_bet_text)
        prev_bet_size = float(prev_bet_match.group(1)) if prev_bet_match else 0.0

        position_image = capture_screen_region(*regions["position"])
        cv2.imwrite('position_debug.png', position_image)
        position_text = extract_text_from_image(position_image)
        print(f"Position text: '{position_text}'")
        position = position_text or "BB"

        # Process individual community cards
        board_cards = []
        card_regions = [
            ("card1_rank", "card1_suit"), ("card2_rank", "card2_suit"),
            ("card3_rank", "card3_suit"), ("card4_rank", "card4_suit"),
            ("card5_rank", "card5_suit")
        ]
        for rank_key, suit_key in card_regions:
            rank_image = capture_screen_region(*regions[rank_key])
            cv2.imwrite(f'{rank_key}_debug.png', rank_image)
            rank_text = extract_text_from_image(rank_image)
            print(f"{rank_key} text: '{rank_text}'")
            # Try template matching for rank
            rank_match = match_template(rank_image, {k: v for k, v in card_templates.items() if k in ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']})
            rank = rank_match if rank_match else rank_text if re.match(r'[AKQJT98765432]', rank_text) else ""

            suit_image = capture_screen_region(*regions[suit_key])
            cv2.imwrite(f'{suit_key}_debug.png', suit_image)
            suit_text = extract_text_from_image(suit_image)
            print(f"{suit_key} text: '{suit_text}'")
            # Try template matching for suit
            suit_match = match_template(suit_image, {k: v for k, v in card_templates.items() if k in ['spades', 'hearts', 'diamonds', 'clubs']})
            suit = suit_match if suit_match else suit_text if suit_text in '♠♥♦♣shdc' else ""
            # Map suit abbreviations if needed
            suit_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
            suit = suit_map.get(suit.lower(), suit)

            # Combine rank and suit—handle graphical cards
            card = f"{rank}{suit}" if rank and suit else ""
            if re.match(r'[AKQJT98765432](?:♠|♥|♦|♣|[shdc])', card) or (rank in 'AKQJT98765432' and suit in '♠♥♦♣shdc'):
                board_cards.append(card)

        # Determine street from number of cards
        num_cards = len(board_cards)
        street = "preflop" if num_cards == 0 else ("flop" if num_cards == 3 else "turn" if num_cards == 4 else "river")

        # Calculate board texture and connectedness for the given board (4dQs4s3h6h)
        board_texture = "wet" if num_cards >= 3 and len(set(card[-1] for card in board_cards if len(card) > 1 and card[-1] in '♠♥♦♣shdc')) < 3 else "dry"
        board_connectedness = 0.8 if board_texture == "wet" else 0.2

        # Derived features
        stack_to_pot_ratio = effective_stack / pot_size if pot_size > 0 else 1000.0
        bet_size_to_pot_ratio = prev_bet_size / pot_size if pot_size > 0 else 0.0

    except ValueError as e:
        print(f"ValueError in game state extraction: {e}")
    except Exception as e:
        print(f"Unexpected error in game state extraction: {e}")

    return {
        "street": street, "pot_size": pot_size, "position": position, "stack_to_pot_ratio": stack_to_pot_ratio,
        "effective_stack": effective_stack, "board_texture": board_texture, "board_connectedness": board_connectedness,
        "is_preflop_aggressor": 1, "bet_size_to_pot_ratio": bet_size_to_pot_ratio, "prev_bet_size": prev_bet_size,
        **stats, "player_id": "player1", "board_cards": board_cards
    }

def print_action(game_state):
    if game_state:
        sim_df = pd.DataFrame([game_state])
        X_sim, sim_df = prepare_data(sim_df)
        dmat_sim = xgb.DMatrix(X_sim, feature_names=X_sim.columns.tolist())
        pred_probs = clf.predict(dmat_sim)
        pred_bet_size = reg.predict(dmat_sim)[0] if len(X_sim) > 0 else 0.0
        action, bet_size, ev = decide_action(pred_probs[0], pred_bet_size, sim_df.iloc[0])
        print(f"Action: {action}, Bet Size: {bet_size} BB, EV: {ev:.2f} BB, Board Cards: {', '.join(game_state.get('board_cards', []))}")
        return action, bet_size

# Calibration loop (manual mode—no clicking)
while True:
    game_state = get_live_game_state()
    if game_state:
        action, bet_size = print_action(game_state)
        with open("bot_log.txt", "a") as log:
            log.write(f"Action: {action}, Bet Size: {bet_size}, Board Cards: {', '.join(game_state.get('board_cards', []))}, {time.ctime()}\n")
    time.sleep(5)  # Slower loop for calibration—adjust as needed
# ev_rollouts.py
import numpy as np

def rollout_ev(pred_probs, hero_strength, pot_size, bet_size, samples=100):
    """
    Simulates multiple action rollouts to approximate EV.
    """
    actions = ['f', 'cc', 'cbr']
    evs = []
    for _ in range(samples):
        action = np.random.choice(actions, p=pred_probs)
        if action == 'f':
            ev = 0
        elif action == 'cc':
            ev = (pot_size + bet_size) * hero_strength - bet_size * (1 - hero_strength)
        else:  # raise back
            ev = -bet_size  # assume fold if re-raised for now
        evs.append(ev)
    return np.mean(evs)
import numpy as np
from itertools import combinations
import random

nInst = 50
currentPos = np.zeros(nInst)

# Global cache
lastComputedDay = -1
cachedTopGroups = []

def compute_spread_zscore_group(prices):
    prices = np.array(prices)
    y = prices[0]
    X = prices[1:].T
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    spread = y - X.dot(beta)
    # Inside compute_spread_zscore_group
    # Replace the z-score line with a rolling calculation
    window = 30 # Example rolling window size
    rolling_mean = np.array([np.mean(spread[max(0, i-window):i]) for i in range(1, len(spread) + 1)])
    rolling_std = np.array([np.std(spread[max(0, i-window):i]) for i in range(1, len(spread) + 1)])

    # Avoid division by zero if std is 0 in the initial period
    rolling_std[rolling_std == 0] = 1 

    z = (spread - rolling_mean) / rolling_std
    # z = (spread - np.mean(spread)) / np.std(spread)
    return spread, z, beta

def simulate_trades_group(z, entry=1.0, exit=0.0):
    positions = np.zeros(len(z))
    for t in range(1, len(z)):
        if z[t-1] < -entry:
            positions[t] = 1
        elif z[t-1] > entry:
            positions[t] = -1
        elif abs(z[t-1]) < exit:
            positions[t] = 0
        else:
            positions[t] = positions[t-1]
    return positions

def get_asset_positions_from_spread_signal(signal, beta, prices, max_dollar=10000, base_shares=1000):
    m = len(prices)
    positions = np.zeros(m, dtype=int)
    if signal == 0:
        return positions
    pos0 = base_shares if signal == 1 else -base_shares
    pos_others = -pos0 * beta if signal == 1 else -pos0 * beta * -1
    pos_others = np.round(pos_others).astype(int)
    positions[0] = pos0
    positions[1:] = pos_others
    for i in range(m):
        max_shares = int(max_dollar / prices[i])
        if abs(positions[i]) > max_shares:
            positions[i] = np.sign(positions[i]) * max_shares
    return positions

def find_top_correlated_groups_sampled(cor_matrix, min_size=2, max_size=5, top_n=10, n_samples=1000):
    n = cor_matrix.shape[0]
    group_scores = []
    for size in range(min_size, max_size + 1):
        sampled = random.sample(list(combinations(range(n), size)), min(n_samples, len(list(combinations(range(n), size)))))
        for group in sampled:
            sub_corr = cor_matrix[np.ix_(group, group)]
            off_diag_sum = np.sum(np.abs(sub_corr)) - np.sum(np.abs(np.diag(sub_corr)))
            group_scores.append((group, off_diag_sum))
    group_scores.sort(key=lambda x: x[1], reverse=True)
    return group_scores[:top_n]

def getMyPosition(prcSoFar):
    global currentPos, lastComputedDay, cachedTopGroups

    n_assets, n_days = prcSoFar.shape
    if n_days < 60:
        return np.zeros(n_assets, dtype=int)

    if n_days % 30 == 0 or lastComputedDay == -1:
        returns = np.diff(np.log(prcSoFar), axis=1)
        cor_matrix = np.corrcoef(returns)
        cachedTopGroups = find_top_correlated_groups_sampled(cor_matrix, min_size=2, max_size=5, top_n=10, n_samples=1000)
        lastComputedDay = n_days

        print(f"\nDay {n_days}: Selected stat arb groups:")
        for group, score in cachedTopGroups:
            print(f"  Group: {group}, Corr Score: {score:.4f}")

    position_vector = np.zeros(n_assets, dtype=int)

    for group, _ in cachedTopGroups:
        try:
            group_prices = [prcSoFar[i] for i in group]
            spread, z, beta = compute_spread_zscore_group(group_prices)
            signal = simulate_trades_group(z)[-1]
            current_prices = np.array([p[-1] for p in group_prices])
            group_positions = get_asset_positions_from_spread_signal(signal, beta, current_prices)
            for idx, inst_id in enumerate(group):
                position_vector[inst_id] += group_positions[idx]
        except Exception:
            continue

    currentPos = position_vector
    return currentPos

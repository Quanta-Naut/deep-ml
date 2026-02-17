def exp_weighted_average(Q1: float, rewards: list[float], alpha: float) -> float:
    k = len(rewards)
    if k == 0:
        return Q1

    initial_term = ((1 - alpha) ** k) * Q1

    weighted_sum = 0
    for i_idx, r in enumerate(rewards):
        i = i_idx + 1
        weight = alpha * ((1 - alpha) ** (k - i))
        weighted_sum += weight * r
        
    return initial_term + weighted_sum
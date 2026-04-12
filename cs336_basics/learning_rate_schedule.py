import math


def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    if t < T_w:
        return alpha_max * t / T_w
    if t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (alpha_max - alpha_min)
    return alpha_min

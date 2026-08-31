import numpy as np


CLEAN_CURRENT_PROFILES = {
    "square": [
        (300, 1.0),
        (300, -1.0),
    ],
    "pulse_rest": [
        (180, 1.0),
        (120, 0.0),
        (180, -1.0),
        (120, 0.0),
    ],
    "variable": [
        (240, 0.5),
        (120, 1.0),
        (60, 0.0),
        (240, -0.5),
        (120, -1.0),
        (60, 0.0),
    ],
    "mixed": [
        (120, 1.0),
        (120, 0.5),
        (120, 0.0),
        (120, -0.5),
        (120, -1.0),
        (120, 0.0),
    ],
}


def generate_clean_current_profile(
    profile_name,
    num_steps,
    dt_s,
    current_magnitude_a,
):
    """Generate a deterministic, zero-net-charge current profile."""
    if profile_name not in CLEAN_CURRENT_PROFILES:
        available = ", ".join(CLEAN_CURRENT_PROFILES)
        raise ValueError(
            f"Unknown current profile '{profile_name}'. Available: {available}"
        )

    pattern = CLEAN_CURRENT_PROFILES[profile_name]
    cycle_duration_s = sum(duration_s for duration_s, _ in pattern)
    cycle_time_s = (np.arange(num_steps) * dt_s) % cycle_duration_s
    current = np.zeros(num_steps, dtype=float)

    segment_start_s = 0.0
    for duration_s, magnitude_fraction in pattern:
        segment_end_s = segment_start_s + duration_s
        in_segment = (
            (cycle_time_s >= segment_start_s)
            & (cycle_time_s < segment_end_s)
        )
        current[in_segment] = magnitude_fraction * current_magnitude_a
        segment_start_s = segment_end_s

    return current


# Charge and discharge direction parameters.
def switch_mode(soc, direction, current_magnitude_a):
    if soc >= 0.7:
        direction = -1
    elif soc <= 0.1:
        direction = 1

    current = direction * current_magnitude_a
    return current, direction

def generate_current_noise(num_steps, capacity_ah, noise_std_c, seed):
    noise_std_a = noise_std_c * capacity_ah
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=noise_std_a, size=num_steps)

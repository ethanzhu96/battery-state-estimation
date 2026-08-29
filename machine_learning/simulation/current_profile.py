import numpy as np

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

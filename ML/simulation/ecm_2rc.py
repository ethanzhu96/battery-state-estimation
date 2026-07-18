import numpy as np

def update_soc(soc, current_a, dt_s, capacity_ah):
    soc_next = soc + (current_a * dt_s) / (capacity_ah * 3600)
    return soc_next
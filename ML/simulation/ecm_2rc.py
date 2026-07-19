import numpy as np

def update_soc(soc, current_a, dt_s, capacity_ah):
    soc_next = soc + (current_a * dt_s) / (capacity_ah * 3600)
    return soc_next

def update_rc_voltage(v_rc, current_a, resistance_ohm, capacitance_f, dt_s):
    tau_s = resistance_ohm * capacitance_f
    decay = np.exp(-dt_s / tau_s)

    v_rc_next = (decay * v_rc + (1.0 - decay) * resistance_ohm * current_a)

    return v_rc_next

def ocv_from_soc(soc, voltage_empty=3.0, voltage_full=4.2):
    voltage_range = voltage_full - voltage_empty
    ocv = voltage_empty + soc * voltage_range
    return ocv


def terminal_voltage(ocv, current_a, r0, v1, v2):
    voltage = ocv + current_a * r0 + v1 + v2
    return voltage

def update_states(
    soc,
    v1,
    v2,
    current_a,
    dt_s,
    capacity_ah,
    r1,
    c1,
    r2,
    c2):

    soc_next = update_soc(
        soc,
        current_a,
        dt_s,
        capacity_ah,
    )

    v1_next = update_rc_voltage(
        v1,
        current_a,
        r1,
        c1,
        dt_s,
    )

    v2_next = update_rc_voltage(
        v2,
        current_a,
        r2,
        c2,
        dt_s,
    )

    return soc_next, v1_next, v2_next

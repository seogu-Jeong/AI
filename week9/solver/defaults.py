# Default parameters for each topic
DEFAULTS = {
    'euler_rk4': {'omega': 2.0, 'dt': 0.05, 't_end': 10.0, 'x0': 1.0, 'v0': 0.0},
    'planetary': {'semi_major_axis': 1.0, 'eccentricity': 0.2, 'mass_star': 1.989e30},
    'double_pendulum': {
        'theta1_deg': 120.0, 'theta2_deg': -30.0, 
        'l1': 1.0, 'l2': 1.0, 'm1': 1.0, 'm2': 1.0, 
        't_end': 20.0, 'compare_chaos': True
    },
    'lagrangian': {'length': 1.0, 'mass': 1.0, 'theta0_deg': 60.0, 'omega0': 0.0, 't_end': 10.0}
}

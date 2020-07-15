from .__head__ import *

def decision_kwargs(
    scenario : int,
) -> Dict[str, object]:
    if scenario == 1:
        p = 0.06
        q = 0.80

        p_val = np.array([
            # State 1
            [p],
            # State 2
            [p],
            # State 3
            [p],
        ])
        q_val = np.array([
            [ # State 1
                [     q     , (1 - q) / 2, (1 - q) / 2],
            ],
            [ # State 2
                [(1 - q) / 2,      q     , (1 - q) / 2],
            ],
            [ # State 3
                [(1 - q) / 2, (1 - q) / 2,      q     ],
            ],
        ])
        eta_a = [-0.30, -0.40, -0.50]
        eta_b = [-0.55, -0.60, -0.65]
        eta_c = [-0.10]
        costs = [ 0.10]
        eta_d = [ 0.00] * 4

    if scenario == 2:
        p_hi = 0.13
        p_lo = 0.06

        q_hi = 0.90
        q_lo = 0.80

        unif = 0.50

        p_val = np.array([
            # State 1
            [p_hi, p_lo],
            # State 2
            [p_hi, p_lo],
            # State 3
            [p_hi, p_lo],
        ])
        q_val = np.array([
            [ # State 1
                [    q_hi, 1 - q_hi],
                [    unif,     unif],
            ],
            [ # State 2
                [1 - q_hi,     q_hi],
                [    q_lo, 1 - q_lo],
            ],
            [ # State 3
                [1 - q_hi,     q_hi],
                [1 - q_lo,     q_lo],
            ],
        ])
        eta_a = [-0.30, -0.40, -0.50]
        eta_b = [-0.55, -0.60, -0.65]
        eta_c = [-0.10] * 2
        costs = [ 0.10] * 2
        eta_d = [ 0.00] * 4

    if scenario == 3: # NOTE: This is Example #1
        p_hi = 0.13
        p_lo = 0.06

        q_hi = 0.90
        q_lo = 0.80

        unif = 0.50

        p_val = np.array([
            # State 1
            [p_hi, p_hi, p_hi, p_lo, p_lo, p_lo],
            # State 2
            [p_hi, p_hi, p_hi, p_lo, p_lo, p_lo],
            # State 3
            [p_hi, p_hi, p_hi, p_lo, p_lo, p_lo],
        ])
        q_val = np.array([
            [ # State 1
                [    q_hi, 1 - q_hi],
                [1 - q_hi,     q_hi],
                [1 - q_hi,     q_hi],
                [    unif,     unif],
                [1 - q_lo,     q_lo],
                [    q_lo, 1 - q_lo],
            ],
            [ # State 2
                [1 - q_hi,     q_hi],
                [    q_hi, 1 - q_hi],
                [1 - q_hi,     q_hi],
                [    q_lo, 1 - q_lo],
                [    unif,     unif],
                [1 - q_lo,     q_lo],
            ],
            [ # State 3
                [1 - q_hi,     q_hi],
                [1 - q_hi,     q_hi],
                [    q_hi, 1 - q_hi],
                [1 - q_lo,     q_lo],
                [    q_lo, 1 - q_lo],
                [    unif,     unif],
            ],
        ])
        eta_a = [-0.30, -0.40, -0.50]
        eta_b = [-0.55, -0.60, -0.65]
        eta_c = [-0.10] * 6
        costs = [ 0.125, 0.115, 0.105, 0.095, 0.085, 0.075]
        eta_d = [ 0.00] * 4

    if scenario == 4: # NOTE: This is Example #3(a)
        p = 0.06
        q_hi = 0.85
        q_lo = 0.80

        p_val = np.array([
            # State 1
            [p, p],
            # State 2
            [p, p],
        ])
        q_val = np.array([
            [ # State 1
                [q_lo, 1 - q_lo],
                [q_hi, 1 - q_hi],
            ],
            [ # State 2
                [1 - q_lo, q_lo],
                [1 - q_hi, q_hi],
            ],
        ])
        eta_a = [-0.50, -0.50]
        eta_b = [-0.50, -0.50]
        eta_c = [-0.25, -0.85]
        costs = [ 0.05,  0.10]
        eta_d = [ 0.00] * 3

    if scenario == 5: # NOTE: This is Example #2
        p = 0.06
        q = 0.80

        p_val = np.array([
            # State 1
            [p],
            # State 2
            [p],
        ])
        q_val = np.array([
            [ # State 1
                [    q, 1 - q],
            ],
            [ # State 2
                [1 - q,     q],
            ],
        ])
        eta_a = [-0.25, -0.75]
        eta_b = [-1.00, -1.00]
        eta_c = [-0.25]
        costs = [ 0.10]
        eta_d = [ 0.00] * 3

    if scenario == 6: # NOTE: This is Example #3(b)
        p = 0.06
        q_hi = 0.85
        q_lo = 0.80

        p_val = np.array([
            # State 1
            [p, p],
            # State 2
            [p, p],
        ])
        q_val = np.array([
            [ # State 1
                [q_lo, 1 - q_lo],
                [q_hi, 1 - q_hi],
            ],
            [ # State 2
                [1 - q_lo, q_lo],
                [1 - q_hi, q_hi],
            ],
        ])
        eta_a = [-0.50, -0.50]
        eta_b = [-0.50, -0.50]
        eta_c = [-0.50, -0.50]
        costs = [ 0.05,  0.10]
        eta_d = [ 0.00] * 3

    if scenario == 7: # NOTE: This is Example #4
        p = 0.06
        q_hi = 0.85
        q_lo = 0.80

        p_val = np.array([
            # State 1
            [p, p],
            # State 2
            [p, p],
        ])
        q_val = np.array([
            [ # State 1
                [q_lo, 1 - q_lo],
                [q_hi, 1 - q_hi],
            ],
            [ # State 2
                [1 - q_lo, q_lo],
                [1 - q_hi, q_hi],
            ],
        ])
        eta_a = [-0.25, -0.75]
        eta_b = [-1.00, -1.00]
        eta_c = [-0.50, -0.50]
        costs = [ 0.00,  0.00]
        eta_d = [ 0.00] * 3

    return {
        'p_val' : p_val,
        'q_val' : q_val,
        'eta_a' : eta_a,
        'eta_b' : eta_b,
        'eta_c' : eta_c,
        'costs' : costs,
        'eta_d' : eta_d,
    }

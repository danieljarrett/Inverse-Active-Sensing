from .__head__ import *

def diagnosis_kwargs(
    tree_factor     : int  ,
    tree_height     : int  ,
    base_prob_comp  : float,
    base_prob_fail  : float,
    base_theta_test : float,
    base_theta_comp : float,
    base_theta_mark : float,
    base_omega      : float,
) -> Dict[str, object]:
    np.random.seed(0)

    tree_factor  = tree_factor
    tree_height  = tree_height

    num_diseases = tree_factor ** (tree_height - 1)
    num_tests    = int((tree_factor ** (tree_height - 1) - 1) / (tree_factor - 1))

    prob_comp    = [base_prob_comp] * num_diseases + [1.0]
    prob_fail    = [base_prob_fail] * num_tests

    phi_test     = 1
    phi_comp     = 1
    phi_mark     = 1

    theta_test   = [-base_theta_test] * num_tests
    theta_comp   = [-base_theta_comp] * num_diseases
    theta_mark   = [-base_theta_mark] * num_diseases


    omega        = [ base_omega] * num_diseases + [0.0]

    return {
        'tree_factor' : tree_factor,
        'tree_height' : tree_height,
        'prob_comp'   : prob_comp  ,
        'prob_fail'   : prob_fail  ,
        'phi_test'    : phi_test   ,
        'phi_comp'    : phi_comp   ,
        'phi_mark'    : phi_mark   ,
        'theta_test'  : theta_test ,
        'theta_comp'  : theta_comp ,
        'theta_mark'  : theta_mark ,
        'omega'       : omega      ,
    }

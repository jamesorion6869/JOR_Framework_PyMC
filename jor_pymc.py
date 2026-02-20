import pymc as pm
import pandas as pd
import numpy as np
import pytensor.tensor as pt

# Global Parameters from JOR-V3
PRIOR_NH_MU = 0.20
CALIBRATION_K = 0.20
WEIGHTS = [0.4, 0.3, 0.3]

def calc_beta_params_vec(mu, sigma):
    """Vectorized calculation of Beta parameters."""
    var = sigma ** 2
    alpha = ((1 - mu) / var - 1 / mu) * mu ** 2
    beta = alpha * (1 / mu - 1)
    return alpha, beta

def run_jor_pymc_safe(data, draws=1000, tune=1000, chains=4, cores=4, target_accept=0.95):
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    # Vectorize inputs (Convert columns to numpy arrays)
    c_scores = df['C_score'].values
    e_scores = df['E_score'].values
    p_scores = df['P_score'].values
    f_mods = df['flight_mod'].values
    num_cases = len(df)

    # Pre-calculate Beta parameters for the priors
    c_a, c_b = calc_beta_params_vec(c_scores, 0.05)
    e_a, e_b = calc_beta_params_vec(e_scores, 0.05)
    p_a, p_b = calc_beta_params_vec(p_scores, 0.05)
    prior_a, prior_b = calc_beta_params_vec(PRIOR_NH_MU, 0.02)

    with pm.Model() as model:
        # Priors (Now vectors of length num_cases)
        C = pm.Beta('Witness', alpha=c_a, beta=c_b, shape=num_cases)
        E = pm.Beta('Environment', alpha=e_a, beta=e_b, shape=num_cases)
        P = pm.Beta('Physical', alpha=p_a, beta=p_b, shape=num_cases)
        
        # SOP & NHP logic
        SOP = pm.Deterministic('SOP', WEIGHTS[0]*C + WEIGHTS[1]*E + WEIGHTS[2]*P)
        NHP_raw = SOP + f_mods
        NHP = pm.Deterministic('NHP', pt.minimum(1.0, NHP_raw))

        # Likelihood equivalents
        like_h = pm.Deterministic('Like_H', pt.minimum(1.0, 1.0 - NHP + (CALIBRATION_K * SOP)))
        like_nh = NHP

        # Bayesian update (Prior is a single distribution applied to all)
        prior = pm.Beta('Prior', alpha=prior_a, beta=prior_b)
        
        posterior = pm.Deterministic(
            'Posterior_NH',
            (prior * like_nh) / ((prior * like_nh) + ((1 - prior) * like_h))
        )

        # One sampling run for ALL rows
        trace = pm.sample(
            draws=draws, 
            tune=tune, 
            chains=chains, 
            cores=cores, 
            target_accept=target_accept,
            progressbar=True
        )

    # Extract and reshape results
    post_samples = trace.posterior['Posterior_NH'].values  # Shape: (chains, draws, num_cases)
    stacked_samples = post_samples.reshape(-1, num_cases) # Flatten chains and draws
    
    df_results = pd.DataFrame({
        'case_name': df['case_name'],
        'Posterior_Mean': np.mean(stacked_samples, axis=0),
        'CI_2.5%': np.percentile(stacked_samples, 2.5, axis=0),
        'CI_97.5%': np.percentile(stacked_samples, 97.5, axis=0)
    })
    
    return df_results
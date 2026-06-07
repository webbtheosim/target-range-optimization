"""Shared math utilities for acquisition functions."""

import autograd.numpy as np
from autograd.scipy.special import erf


def ncx2_cdf_approximation(t, dof, non_centrality):
    """Non-central chi-squared CDF approximation (Sankaran 1963)."""
    t = np.array(t, dtype=float)
    dof = np.array(dof, dtype=float)
    non_centrality = np.array(non_centrality, dtype=float)

    dof = np.maximum(dof, 1e-8)
    non_centrality = np.clip(non_centrality, 0.0, 1e12)
    t = np.clip(t, 1e-12, 1e12)

    r1 = dof + non_centrality
    r2 = 2.0 * (dof + 2.0 * non_centrality)
    r3 = 8.0 * (dof + 3.0 * non_centrality)

    r1_safe = np.maximum(r1, 1e-12)
    r2_safe = np.maximum(r2, 1e-12)
    r1_squared = r1_safe ** 2
    r1_fourth = r1_squared ** 2

    m_numerator = r1 * r3
    m_denominator = 3.0 * r2_safe ** 2
    m = 1.0 - m_numerator / m_denominator
    m = np.clip(m, 0.05, 2.5)

    ratio = t / r1_safe
    ratio_safe = np.clip(ratio, 1e-300, 1e300)
    log_z = m * np.log(ratio_safe)
    z = np.exp(log_z)
    z = np.clip(z, 1e-8, 1e8)

    term1 = r2_safe / (2.0 * r1_squared)
    term2_numerator = (2.0 - m) * (1.0 - 3.0 * m) * r2_safe ** 2
    term2_denominator = 8.0 * r1_fourth + 1e-300
    term2 = term2_numerator / term2_denominator

    alpha = 1.0 + m * (m - 1.0) * (term1 - term2)
    alpha = np.clip(alpha, -1e6, 1e6)

    rho_inner_numerator = (1.0 - m) * (1.0 - 3.0 * m) * r2_safe
    rho_inner_denominator = 4.0 * r1_squared + 1e-300
    rho_inner = rho_inner_numerator / rho_inner_denominator

    rho = (m * np.sqrt(r2_safe) / r1_safe) * (1.0 - rho_inner)
    rho = np.maximum(rho, 1e-12)

    erf_argument = (z - alpha) / (rho * np.sqrt(2.0))
    erf_argument = np.clip(erf_argument, -40.0, 40.0)

    cdf = 0.5 * (1.0 + erf(erf_argument))
    return cdf


def broadcast_variances(variances, output_dim):
    """Broadcast variances to output_dim if needed."""
    if variances.shape[1] < output_dim:
        return np.repeat(variances, output_dim, axis=1)
    return variances


def reshape_variance_gradients(d_variance_dx, output_dim):
    """Ensure variance gradients have shape (n, d, output_dim)."""
    if d_variance_dx.ndim == 2:
        d_variance_dx = np.atleast_3d(d_variance_dx)
        return np.repeat(d_variance_dx, output_dim, axis=2)
    return d_variance_dx


def assemble_gradient(d_mean_dx, d_variance_dx, d_acq_d_mean, d_acq_d_variance):
    """Assemble full gradient from mean and variance partials."""
    n_points = d_mean_dx.shape[0]
    d_input = d_mean_dx.shape[1]
    grad = np.zeros((n_points, d_input), dtype=float)
    for i in range(n_points):
        grad[i] = (
            np.dot(d_mean_dx[i], d_acq_d_mean[i]).ravel()
            + np.dot(d_variance_dx[i], d_acq_d_variance[i]).ravel()
        )
    return grad

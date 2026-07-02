"""
Discrete-time LQR and Riccati solvers for **lifted** EDMD dynamics ``x' = A x + B u``.

Builds structured ``Q`` (physical tracking-error block + small weights on lifted coords), solves the
discrete-time algebraic Riccati equation, and returns ``P``, ``K``, closed-loop ``A_cl``, etc.
"""

import numpy as np
import control as ctrl
from scipy.linalg import solve_discrete_are


def compute_lqr_gains(A_lifted, B_lifted, n_error=6, q_x=1.0, q_phi=1e-6, normalize_A=True):
    """
    Compute discrete-time LQR gains using Riccati equation.

    Args:
        A_lifted: Lifted space A matrix
        B_lifted: Lifted space B matrix
        n_error: Dimension of physical tracking error in lifted coordinates
        q_x: Weight for original tracking error components
        q_phi: Weight for lifted-only coordinates
        normalize_A: If True, scale A by 1/rho(A) when rho(A) > 1 before the DARE solve.
            An unstable A (rho > 1) amplifies V_next exponentially in forward prediction.
            The same normalised A is returned in "A_normalized" and must be saved to disk
            so the RL agents use the same A that P was designed for.

    Returns:
        Dictionary with P, K, Q, R, A_cl, A_normalized matrices
    """
    lifted_dim = A_lifted.shape[0]
    if not (0 < n_error <= lifted_dim):
        raise ValueError(
            f"n_error must satisfy 0 < n_error <= lifted_dim; got n_error={n_error}, lifted_dim={lifted_dim}"
        )
    m = lifted_dim - n_error

    # Normalise unstable A so rho(A) <= 1 before solving the Riccati equation.
    # Dividing by spectral radius preserves controllability of (A, B) while capping
    # the open-loop amplification to exactly 1.0. B is left unchanged.
    rho_raw = float(np.max(np.abs(np.linalg.eigvals(A_lifted))))
    if normalize_A and rho_raw > 1.0:
        A_lifted = A_lifted / rho_raw
        print(f"A normalised: raw rho={rho_raw:.4f} -> scaled rho=1.000 (divided by {rho_raw:.4f})")

    Q = np.diag([q_x] * n_error + [q_phi] * m)
    R = np.eye(B_lifted.shape[1])

    print("\n--- Pre-DARE Diagnostics ---")
    print("Checking system properties for DARE solvability...")

    A_lifted_eigenvals = np.linalg.eigvals(A_lifted)
    A_lifted_eigenvals_real = np.real(A_lifted_eigenvals)
    A_lifted_eigenvals_imag = np.imag(A_lifted_eigenvals)
    print(
        f"A_lifted eigenvalues (real part): min={np.min(A_lifted_eigenvals_real):.4f}, "
        f"max={np.max(A_lifted_eigenvals_real):.4f}"
    )
    print(
        f"A_lifted eigenvalues (imag part): min={np.min(np.abs(A_lifted_eigenvals_imag)):.4f}, "
        f"max={np.max(np.abs(A_lifted_eigenvals_imag)):.4f}"
    )

    max_eigenval_magnitude = np.max(np.abs(A_lifted_eigenvals))
    print(f"A_lifted max eigenvalue magnitude: {max_eigenval_magnitude:.4f}")

    controllability_matrix_lifted = ctrl.ctrb(A_lifted, B_lifted)
    controllability_rank_lifted = np.linalg.matrix_rank(controllability_matrix_lifted)
    print(f"Lifted system controllability rank: {controllability_rank_lifted} / {lifted_dim}")

    try:
        print("\n--- Computing P matrix for lifted space ---")
        P_lifted = solve_discrete_are(A_lifted, B_lifted, Q, R)
        # Enforce exact symmetry to eliminate floating-point asymmetry from DARE solver
        P_lifted = (P_lifted + P_lifted.T) / 2
        print(f"Computed P_lifted shape: {P_lifted.shape}")
        print(f"P_lifted trace: {np.trace(P_lifted):.4f}")

        P_lifted_eigenvals = np.linalg.eigvals(P_lifted)
        P_lifted_eigenvals_real = np.real(P_lifted_eigenvals)
        min_eig = float(np.min(P_lifted_eigenvals_real))
        if min_eig < 0:
            n_neg = int(np.sum(P_lifted_eigenvals_real < 0))
            raise ValueError(
                f"DARE solution P_lifted is NOT positive definite: {n_neg} negative eigenvalue(s), "
                f"min eigenvalue = {min_eig:.4e}. "
                f"V(z) = z^T P z requires P > 0. "
                f"Re-tune EDMD (--n-rbf-centers, --rbf-width, --regularization) or increase data coverage "
                f"so the fitted (A, B) is controllable and the Riccati equation has a valid solution."
            )
        if np.max(np.abs(P_lifted_eigenvals_real)) > 1e4:
            print(
                f"  ⚠️  WARNING: P_lifted has very large eigenvalues "
                f"(max: {np.max(P_lifted_eigenvals_real):.2e}) — Lyapunov function may be ill-conditioned."
            )
    except Exception as e:
        print(f"Warning: Could not solve DARE for lifted space: {e}")
        print("Falling back to extended identity matrix")
        P_lifted = np.eye(lifted_dim)
        print(f"Using identity P_lifted: shape {P_lifted.shape}")

    K = np.linalg.inv((R + B_lifted.T @ P_lifted @ B_lifted)) @ (B_lifted.T @ P_lifted @ A_lifted)
    print(f"LQR gain K shape: {K.shape}")

    A_cl = A_lifted - B_lifted @ K

    eig_A = np.linalg.eigvals(A_lifted)
    eig_Acl = np.linalg.eigvals(A_cl)
    print(f"\nOpen-loop spectral radius: {np.max(np.abs(eig_A)):.4f}")
    print(f"Closed-loop spectral radius: {np.max(np.abs(eig_Acl)):.4f}")

    return {
        "P": P_lifted,
        "K": K,
        "Q": Q,
        "R": R,
        "A_cl": A_cl,
        "A_normalized": A_lifted,  # may differ from the original if rho > 1 was normalised
    }


def analyze_p_matrix(P_lifted, A_lifted, B_lifted):
    """Console diagnostics for the Riccati solution P and lifted dynamics."""
    print("\n" + "=" * 60)
    print("P_MATRIX DIAGNOSTICS")
    print("=" * 60)

    P_to_analyze = P_lifted.copy()
    P_name = "P_lifted"

    is_symmetric = np.allclose(P_to_analyze, P_to_analyze.T, rtol=1e-5, atol=1e-8)
    print(f" Symmetry Check ({P_name}):")
    print(f"   P is symmetric: {is_symmetric}")
    if not is_symmetric:
        max_asymmetry = np.max(np.abs(P_to_analyze - P_to_analyze.T))
        print(f"   WARNING: Max asymmetry = {max_asymmetry:.2e}")
        P_to_analyze = (P_to_analyze + P_to_analyze.T) / 2
        print(f"   Fixed: Symmetrized {P_name} as (P + P.T) / 2")

    eigenvals = np.linalg.eigvals(P_to_analyze)
    eigenvals_real = np.real(eigenvals)
    eigenvals_imag = np.imag(eigenvals)

    print(f" Magnitude Check ({P_name}):")
    print(f"   P matrix stats:")
    print(f"     - Min value: {np.min(P_to_analyze):.2e}")
    print(f"     - Max value: {np.max(P_to_analyze):.2e}")
    print(f"     - Mean |value|: {np.mean(np.abs(P_to_analyze)):.2e}")
    print(f"     - Trace: {np.trace(P_to_analyze):.2e}")
    print(f"     - Frobenius norm: {np.linalg.norm(P_to_analyze, 'fro'):.2e}")

    print(f" Eigenvalue Check ({P_name}):")
    print(f"     - Min eigenvalue (real): {np.min(eigenvals_real):.2e}")
    print(f"     - Max eigenvalue (real): {np.max(eigenvals_real):.2e}")
    condition_num = np.max(eigenvals_real) / (np.min(eigenvals_real) + 1e-10)
    print(f"     - Condition number: {condition_num:.2e}")
    if np.any(eigenvals_real < 0):
        print(f"     - WARNING: {np.sum(eigenvals_real < 0)} negative eigenvalues found!")
        print(f"       P should be positive definite for Lyapunov function")

    print(f" Update Matrices Magnitude:")
    print(
        f"     - A_lifted: min={np.min(A_lifted):.2e}, max={np.max(A_lifted):.2e}, "
        f"mean|value|={np.mean(np.abs(A_lifted)):.2e}"
    )
    print(
        f"     - B_lifted: min={np.min(B_lifted):.2e}, max={np.max(B_lifted):.2e}, "
        f"mean|value|={np.mean(np.abs(B_lifted)):.2e}"
    )

    print("\n" + "=" * 60)

# ==============================================================================
# UNIFIED VALIDATION SCRIPT (Version 5.2 - Corrected Plotting Logic)
#
# This script validates the CMD engine and includes a final demonstration
# of its robustness against simulated experimental noise.
#
# Changes:
# - Corrected the plotting logic in validate_j0_correlated_beam() to show
#   the solver's true output for degenerate modes, matching the paper's text.
# - Increased all font sizes for publication-quality figures.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, jv, genlaguerre
from math import factorial
import time
import os

# --- Matplotlib Font Settings for Publication Quality ---
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=16)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=18)   # fontsize of the figure title

# --- Global Simulation Parameters ---
N_POINTS = 64
XY_MAX = 7.0

# --- Ensure a directory exists for the figures ---
FIGURE_DIR = "paper_figures"
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

# ==============================================================================
# SECTION 1: CORE NUMERICAL ENGINE & UTILITIES
# ==============================================================================

def decompose_csd(csd_matrix):
    """Solves the eigenvalue problem and returns PHYSICALLY NORMALIZED modes."""
    print("Performing eigenvalue decomposition...", end='', flush=True)
    start_time = time.time()
    eigenvalues, eigenvectors = np.linalg.eigh(csd_matrix)
    end_time = time.time()
    print(f" done in {end_time - start_time:.2f} seconds.")

    sort_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_eigenvalues_norm = sorted_eigenvalues / np.max(sorted_eigenvalues)

    N = int(np.sqrt(csd_matrix.shape[0]))
    dx = (2 * XY_MAX) / N
    
    sorted_modes = []
    physical_norm_factor = np.sqrt(dx**2)

    for i in sort_indices:
        mode_vector = eigenvectors[:, i]
        mode_2d = (mode_vector / physical_norm_factor).reshape((N, N)).astype(np.complex128)
        
        max_abs_idx = np.argmax(np.abs(mode_2d))
        phase_correction = np.angle(mode_2d.flat[max_abs_idx])
        mode_2d *= np.exp(-1j * phase_correction)
        
        sorted_modes.append(mode_2d)
        
    return sorted_eigenvalues_norm, sorted_modes

def calculate_fidelity(mode1, mode2):
    """Calculates fidelity after ensuring both modes are properly normalized."""
    dx = (2 * XY_MAX) / N_POINTS
    norm1 = np.sqrt(np.sum(np.abs(mode1)**2)*dx**2)
    norm2 = np.sqrt(np.sum(np.abs(mode2)**2)*dx**2)
    if norm1 < 1e-9 or norm2 < 1e-9: return 0.0
    mode1_norm = mode1 / norm1
    mode2_norm = mode2 / norm2
    overlap = np.sum(np.conj(mode1_norm) * mode2_norm) * dx**2
    return np.abs(overlap)**2

# ==============================================================================
# SECTION 2: BEAM DEFINITIONS
# ==============================================================================

def hg_mode(n, m, w, x, y):
    xx, yy = np.meshgrid(x, y)
    norm = np.sqrt(2 / (np.pi * w**2 * (2**n * factorial(n)) * (2**m * factorial(m))))
    Hn = hermite(n)(np.sqrt(2) * xx / w)
    Hm = hermite(m)(np.sqrt(2) * yy / w)
    gauss = np.exp(-(xx**2 + yy**2) / w**2)
    return norm * Hn * Hm * gauss

def lg_mode(p, l, w0, x, y):
    xx, yy = np.meshgrid(x, y)
    rr, alpha = np.sqrt(xx**2 + yy**2), np.arctan2(yy, xx)
    laguerre_poly = genlaguerre(p, np.abs(l))
    Lpl = laguerre_poly(2 * rr**2 / w0**2)
    norm = np.sqrt(2 * factorial(p) / (np.pi * factorial(p + np.abs(l)))) / w0
    field = norm * (np.sqrt(2) * rr / w0)**np.abs(l) * \
            np.exp(-rr**2 / w0**2) * Lpl * np.exp(1j * l * alpha)
    return field

def calculate_gsm_analytical_solution(w0, sigma_g, x, y, max_order=5):
    A = 0.5/w0**2 + 0.5/sigma_g**2
    B = 1.0/sigma_g**2
    w_mode = np.sqrt(2.0 / np.sqrt(4*A**2 - B**2))
    q = B / (2*A + np.sqrt(4*A**2 - B**2))
    eigenvalues, modes, labels = [], [], []
    for n in range(max_order):
        for m in range(max_order):
            eigenvalues.append((q**n) * (q**m))
            modes.append(hg_mode(n, m, w_mode, x, y))
            labels.append(f'HG({n},{m})')
    eigenvalues = np.array(eigenvalues)
    sort_indices = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sort_indices], [modes[i] for i in sort_indices], [labels[i] for i in sort_indices]

def construct_gsm_csd_matrix(w0, sigma_g, x, y):
    N = len(x)
    xx, yy = np.meshgrid(x, y)
    coords_flat = np.vstack([xx.ravel(), yy.ravel()]).T
    A = 0.5/w0**2 + 0.5/sigma_g**2
    B = 1.0/sigma_g**2
    print(f"Constructing {N**2}x{N**2} GSM CSD matrix...")
    start_time = time.time()
    r_sq = np.sum(coords_flat**2, axis=1)
    r1_dot_r2 = np.dot(coords_flat, coords_flat.T)
    csd_matrix = np.exp(-A * (r_sq[:, np.newaxis] + r_sq) + B * r1_dot_r2)
    end_time = time.time()
    print(f"CSD matrix constructed in {end_time - start_time:.2f} seconds.")
    return csd_matrix

def calculate_j0_analytical_solution(R, beta, x, y, max_order=15):
    xx, yy = np.meshgrid(x, y)
    rr, alpha = np.sqrt(xx**2 + yy**2), np.arctan2(yy, xx)
    dx = x[1]-x[0]
    T_r = (rr <= R).astype(float)
    eigenvalues, modes, labels = [], [], []
    intensity = 1.0
    for n in range(max_order):
        arg = beta * R
        eigenvalue = np.pi * R**2 * intensity * (jv(n, arg)**2 - jv(n - 1, arg) * jv(n + 1, arg))
        
        mode_plus = T_r * jv(n, beta * rr) * np.exp(1j * n * alpha)
        norm_factor_plus = np.sqrt(np.sum(np.abs(mode_plus)**2) * dx**2)
        modes.append(mode_plus / norm_factor_plus if norm_factor_plus > 1e-9 else mode_plus)
        eigenvalues.append(eigenvalue)
        labels.append(f'J0 Mode n={n}')
        
        if n > 0:
            mode_minus = T_r * jv(n, beta * rr) * np.exp(-1j * n * alpha)
            norm_factor_minus = np.sqrt(np.sum(np.abs(mode_minus)**2) * dx**2)
            modes.append(mode_minus / norm_factor_minus if norm_factor_minus > 1e-9 else mode_minus)
            eigenvalues.append(eigenvalue)
            labels.append(f'J0 Mode n=-{n}')
    
    eigenvalues = np.array(eigenvalues)
    sort_indices = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sort_indices], [modes[i] for i in sort_indices], [labels[i] for i in sort_indices]

def construct_j0_csd_matrix(R, beta, x, y):
    N = len(x)
    xx, yy = np.meshgrid(x, y)
    coords_flat = np.vstack([xx.ravel(), yy.ravel()]).T
    print(f"Constructing {N**2}x{N**2} J0-Correlated CSD matrix...")
    start_time = time.time()
    r_flat = np.sqrt(np.sum(coords_flat**2, axis=1))
    T_flat = (r_flat <= R).astype(float)
    T_term = np.outer(T_flat, T_flat)
    x_coords_flat, y_coords_flat = coords_flat[:, 0], coords_flat[:, 1]
    dist_sq_x = (x_coords_flat[:, np.newaxis] - x_coords_flat)**2
    dist_sq_y = (y_coords_flat[:, np.newaxis] - y_coords_flat)**2
    r1_minus_r2 = np.sqrt(dist_sq_x + dist_sq_y)
    j0_term = jv(0, beta * r1_minus_r2)
    csd_matrix = T_term * j0_term
    end_time = time.time()
    print(f"CSD matrix constructed in {end_time - start_time:.2f} seconds.")
    return csd_matrix

def construct_lg01_csd_matrix(w0, x, y):
    print(f"Constructing CSD matrix for coherent LG01 beam...", end='', flush=True)
    start_time = time.time()
    E_lg01 = lg_mode(p=0, l=1, w0=w0, x=x, y=y)
    v_lg01 = E_lg01.ravel()
    csd_matrix = np.outer(v_lg01, np.conj(v_lg01))
    end_time = time.time()
    print(f" done in {end_time - start_time:.2f} seconds.")
    return csd_matrix, E_lg01

# ==============================================================================
# SECTION 3: VALIDATION RUNNERS (WITH PUBLICATION-QUALITY PLOTTING)
# ==============================================================================

def validate_gsm_beam():
    print("\n" + "="*80 + "\nRUNNING VALIDATION FOR: GAUSSIAN SCHELL-MODEL (GSM) BEAM\n" + "="*80)
    w0, sigma_g = 2.0, 1.5
    x, y = np.linspace(-XY_MAX, XY_MAX, N_POINTS), np.linspace(-XY_MAX, XY_MAX, N_POINTS)
    lambda_theory_unnorm, modes_theory, labels_theory = calculate_gsm_analytical_solution(w0, sigma_g, x, y, max_order=4)
    lambda_theory = lambda_theory_unnorm / np.max(lambda_theory_unnorm)
    csd_matrix = construct_gsm_csd_matrix(w0, sigma_g, x, y)
    lambda_num, modes_num = decompose_csd(csd_matrix)
    
    print("\n--- Eigenvalue Comparison ---")
    for i in range(5): print(f"{labels_theory[i]:<9} | Theory: {lambda_theory[i]:.6f} | Numerical: {lambda_num[i]:.6f}")
    print("\n--- Modal Shape Fidelity Comparison ---")
    for i in range(5):
        best_fid = max(calculate_fidelity(modes_theory[j], modes_num[i]) for j in range(len(modes_theory)))
        print(f"Numerical Mode {i} | Best Fidelity with a Theory Mode: {best_fid:.6f}")
    
    # --- Plotting for Figure 1 ---
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(10), lambda_theory[:10], 'bo', markersize=8, label='Analytical Theory')
    plt.semilogy(range(10), lambda_num[:10], 'r.', markersize=12, label='Numerical Result')
    plt.xlabel('Mode Index')
    plt.ylabel('Normalized Eigenvalue')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "eigenvalue_spectrum_gsm.png"), dpi=200)
    plt.show()

    # --- Plotting for Figure 2 ---
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    theory_labels_to_plot = ['HG(0,0)', 'HG(1,0)', 'HG(0,1)', 'HG(2,0)']
    plot_indices_th = [labels_theory.index(lbl) for lbl in theory_labels_to_plot]
    
    for i, idx in enumerate(plot_indices_th):
        axes[0, i].imshow(np.abs(modes_theory[idx]), cmap='inferno', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX])
        axes[0, i].set_title(f"Theory: {labels_theory[idx]}", fontsize=16)
        axes[1, i].imshow(np.abs(modes_num[i]), cmap='inferno', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX])
        axes[1, i].set_title(f"Numerical: Mode {i}", fontsize=16)
        for ax in [axes[0,i], axes[1,i]]: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "mode_comparison_gsm.png"), dpi=200)
    plt.show()

def validate_j0_correlated_beam():
    """
    CORRECTED: This function now plots the true, ordered output of the numerical
    solver to correctly illustrate the basis change for degenerate modes.
    """
    print("\n" + "="*80 + "\nRUNNING VALIDATION FOR: J0-CORRELATED SCHELL-MODEL BEAM\n" + "="*80)
    R, beta = 3.0, 1.0
    x, y = np.linspace(-XY_MAX, XY_MAX, N_POINTS), np.linspace(-XY_MAX, XY_MAX, N_POINTS)
    lambda_theory_unnorm, modes_theory, labels_theory = calculate_j0_analytical_solution(R, beta, x, y)
    lambda_theory = lambda_theory_unnorm / np.max(lambda_theory_unnorm)
    csd_matrix = construct_j0_csd_matrix(R, beta, x, y)
    lambda_num, modes_num = decompose_csd(csd_matrix)
    
    print("\n--- Eigenvalue Comparison ---")
    for i in range(5): print(f"{labels_theory[i]:<12} | Theory: {lambda_theory[i]:.6f} | Numerical: {lambda_num[i]:.6f}")
    print("\n--- Modal Shape Fidelity Comparison ---")
    for i in range(5):
        best_fid = max(calculate_fidelity(modes_theory[j], modes_num[i]) for j in range(len(modes_theory)))
        print(f"Numerical Mode {i} | Best Fidelity with a Theory Mode: {best_fid:.6f}")
    
    # --- Plotting for Figure 3 ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(10), lambda_theory[:10], 'bo', markersize=8, label='Analytical Theory')
    plt.plot(range(10), lambda_num[:10], 'r.', markersize=12, label='Numerical Result')
    plt.xlabel('Mode Index')
    plt.ylabel('Normalized Eigenvalue')
    plt.xlim(-1, 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "eigenvalue_spectrum_j0.png"), dpi=200)
    plt.show()

    # --- CORRECTED Plotting for Figure 4 ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Define the theoretical modes to display
    theory_labels_to_plot = ['J0 Mode n=0', 'J0 Mode n=1', 'J0 Mode n=-1', 'J0 Mode n=2']
    plot_indices_th = [labels_theory.index(lbl) for lbl in theory_labels_to_plot]
    
    # Plot theory modes in the top row
    for i, idx in enumerate(plot_indices_th):
        axes[0, i].imshow(np.abs(modes_theory[idx]), cmap='inferno', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX])
        axes[0, i].set_title(f"Theory: {labels_theory[idx]}", fontsize=16)

    # Plot the first 4 numerical modes in the bottom row IN ORDER
    # This shows the actual output of the solver, which is what the paper text explains
    for i in range(4):
        axes[1, i].imshow(np.abs(modes_num[i]), cmap='inferno', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX])
        axes[1, i].set_title(f"Numerical: Mode {i}", fontsize=16)

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "mode_comparison_j0.png"), dpi=200)
    plt.show()


def validate_coherent_vortex_beam():
    print("\n" + "="*80 + "\nRUNNING VALIDATION FOR: COHERENT LG01 VORTEX BEAM\n" + "="*80)
    w0 = 2.0
    x, y = np.linspace(-XY_MAX, XY_MAX, N_POINTS), np.linspace(-XY_MAX, XY_MAX, N_POINTS)
    csd_matrix, E_lg01_theory = construct_lg01_csd_matrix(w0, x, y)
    lambda_num, modes_num = decompose_csd(csd_matrix)
    
    print("\n--- Eigenvalue Spectrum Analysis ---")
    print(f"Normalized Eigenvalues: {lambda_num[:5]}")
    print("\n--- Modal Shape Fidelity Comparison ---")
    fidelity = calculate_fidelity(E_lg01_theory, modes_num[0])
    print(f"Fidelity of recovered mode with theoretical LG01 mode: {fidelity:.6f}")
    
    # --- Plotting for Figure 5 ---
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0,0].imshow(np.abs(E_lg01_theory), cmap='inferno', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX])
    axes[0,0].set_title('Theory: Amplitude', fontsize=16)
    axes[0,1].imshow(np.angle(E_lg01_theory), cmap='twilight_shifted', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX])
    axes[0,1].set_title('Theory: Phase', fontsize=16)
    axes[1,0].imshow(np.abs(modes_num[0]), cmap='inferno', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX])
    axes[1,0].set_title('Numerical: Amplitude', fontsize=16)
    axes[1,1].imshow(np.angle(modes_num[0]), cmap='twilight_shifted', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX])
    axes[1,1].set_title('Numerical: Phase', fontsize=16)
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "vortex_phase_comparison.png"), dpi=200)
    plt.show()

def validate_noise_robustness():
    """
    Demonstrates the framework's resilience to noise.
    """
    print("\n" + "="*80 + "\nDEMONSTRATION OF NOISE ROBUSTNESS\n" + "="*80)
    w0 = 2.0
    x, y = np.linspace(-XY_MAX, XY_MAX, N_POINTS), np.linspace(-XY_MAX, XY_MAX, N_POINTS)
    
    csd_clean, E_lg01_theory = construct_lg01_csd_matrix(w0, x, y)
    
    noise_std_dev_fraction = 0.20 
    noise_std_dev = noise_std_dev_fraction * np.max(np.abs(csd_clean))
    print(f"Adding complex Gaussian noise with std. dev. = {noise_std_dev_fraction*100:.0f}% of max|CSD|")

    rng = np.random.default_rng()
    noise_real = rng.normal(0, noise_std_dev / np.sqrt(2), csd_clean.shape)
    noise_imag = rng.normal(0, noise_std_dev / np.sqrt(2), csd_clean.shape)
    noise_matrix = noise_real + 1j * noise_imag
    hermitian_noise = (noise_matrix + np.conj(noise_matrix).T) / 2
    
    csd_noisy = csd_clean + hermitian_noise

    lambda_noisy, modes_noisy = decompose_csd(csd_noisy)

    fidelity = calculate_fidelity(E_lg01_theory, modes_noisy[0])
    print("\n--- Noise Robustness Result ---")
    print(f"Fidelity of dominant mode recovered from noisy CSD: {fidelity:.6f}")
    if fidelity > 0.99:
        print("Result: Framework successfully recovered the mode with high fidelity.")
    else:
        print("Result: Noise significantly impacted the mode recovery.")

# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    validate_gsm_beam()
    validate_j0_correlated_beam()
    validate_coherent_vortex_beam()
    validate_noise_robustness()
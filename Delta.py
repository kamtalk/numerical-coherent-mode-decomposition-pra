# ==============================================================================
# PROJECT DELTA: DIRECT M-SQUARED VALIDATION BY VARYING BEAM COMPLEXITY
#
# This script keeps the grid size fixed and varies the number of modes
# used to construct the beam, providing a clean validation of the M^2 methods.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from math import factorial
import time
import os

# --- Global Simulation Parameters ---
N_POINTS = 64  # Keep grid fixed for a fair test
XY_MAX = 8.0

# --- Ensure a directory exists for the figures ---
FIGURE_DIR = "paper_figures"
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

# (Core functions are identical to the previous script, so they are included here for completeness)
# ==============================================================================
# SECTION 1: CORE FUNCTIONS (UNCHANGED)
# ==============================================================================

def lg_mode(p, l, w0, x, y):
    xx, yy = np.meshgrid(x, y)
    rr, alpha = np.sqrt(xx**2 + yy**2), np.arctan2(yy, xx)
    laguerre_poly = genlaguerre(p, np.abs(l))
    Lpl = laguerre_poly(2 * rr**2 / w0**2)
    norm_const = np.sqrt(2 * factorial(p) / (np.pi * factorial(p + np.abs(l)))) / w0
    field = norm_const * (np.sqrt(2) * rr / w0)**np.abs(l) * \
            np.exp(-rr**2 / w0**2) * Lpl * np.exp(1j * l * alpha)
    return field

def calculate_fidelity(mode1, mode2, dx):
    norm1 = np.sqrt(np.sum(np.abs(mode1)**2)*dx**2)
    norm2 = np.sqrt(np.sum(np.abs(mode2)**2)*dx**2)
    if norm1 < 1e-9 or norm2 < 1e-9: return 0.0
    mode1_norm = mode1 / norm1
    mode2_norm = mode2 / norm2
    overlap = np.sum(np.conj(mode1_norm) * mode2_norm) * dx**2
    return np.abs(overlap)**2

def calculate_m2_from_lg_coeffs(coeffs):
    powers = np.abs(coeffs)**2
    numerator = 0
    for p in range(len(coeffs)):
        m2_mode = 2 * p + 1
        numerator += powers[p] * m2_mode
    total_power = np.sum(powers)
    return numerator / total_power if total_power > 1e-9 else 1.0

def calculate_m2_from_cmd_results(eigenvalues, eigenvectors, basis_modes, n_points, dx):
    sort_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_eigenvectors = eigenvectors[:, sort_indices]
    
    significant_mask = sorted_eigenvalues > 1e-9 * sorted_eigenvalues[0]
    filtered_eigenvalues = sorted_eigenvalues[significant_mask]
    filtered_eigenvectors = sorted_eigenvectors[:, significant_mask]
    
    normalized_powers = filtered_eigenvalues / np.sum(filtered_eigenvalues)
    
    m2_total = 0.0
    
    for i in range(len(normalized_powers)):
        mode_num_vec = filtered_eigenvectors[:, i]
        mode_num_2d = (mode_num_vec / np.sqrt(dx**2)).reshape((n_points, n_points))

        fidelities = [calculate_fidelity(mode_num_2d, basis_mode, dx) for basis_mode in basis_modes]
        best_match_idx = np.argmax(fidelities)
        m2_of_this_mode = 2 * best_match_idx + 1
        m2_total += normalized_powers[i] * m2_of_this_mode
        
    return m2_total

# ==============================================================================
# SECTION 2: MODULAR SIMULATION FUNCTION FOR "PROJECT DELTA"
# ==============================================================================

def run_m2_test(num_modes):
    """
    Runs the M2 validation for a specific number of modes.
    Returns the ground truth M2 and the measured M2.
    """
    print(f"\n--- Testing with {num_modes} modes ---")
    
    x = np.linspace(-XY_MAX, XY_MAX, N_POINTS)
    y = np.linspace(-XY_MAX, XY_MAX, N_POINTS)
    dx = x[1]-x[0]
    
    sg_order, sg_radius, basis_waist = 20, 3.0, 2.5

    # Step 1: Decompose Super-Gaussian using `num_modes`
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    E_sg_phys = np.exp(- (rr / sg_radius)**sg_order)
    E_sg_phys /= np.sqrt(np.sum(np.abs(E_sg_phys)**2) * dx**2)

    secret_coeffs = []
    basis_modes_phys = []
    for p in range(num_modes): # The only variable part
        lg_basis_mode = lg_mode(p, 0, basis_waist, x, y)
        norm_factor = np.sqrt(np.sum(np.abs(lg_basis_mode)**2)*dx**2)
        lg_basis_phys_norm = lg_basis_mode / norm_factor
        coeff = np.sum(np.conj(lg_basis_phys_norm) * E_sg_phys) * dx**2
        secret_coeffs.append(coeff)
        basis_modes_phys.append(lg_basis_phys_norm)
    
    secret_coeffs = np.array(secret_coeffs)
    secret_powers = np.abs(secret_coeffs)**2

    # Step 2: Calculate Ground Truth M2 from the known recipe
    m2_truth = calculate_m2_from_lg_coeffs(secret_coeffs)
    print(f"Ground Truth M²: {m2_truth:.4f}")

    # Step 3: Synthesize and Analyze the CSD
    W_trojan = np.zeros((N_POINTS**2, N_POINTS**2), dtype=np.complex128)
    for i in range(num_modes):
        mode_flat = basis_modes_phys[i].ravel()
        W_trojan += secret_powers[i] * np.outer(mode_flat, np.conj(mode_flat))
    
    eigenvalues, eigenvectors = np.linalg.eigh(W_trojan)

    # Step 4: Calculate Measured M2 from CMD results
    m2_measured = calculate_m2_from_cmd_results(eigenvalues, eigenvectors, basis_modes_phys, N_POINTS, dx)
    print(f"Measured M²    : {m2_measured:.4f}")
    
    return m2_truth, m2_measured

# ==============================================================================
# SECTION 3: MAIN EXECUTION BLOCK FOR "PROJECT DELTA"
# ==============================================================================

def main():
    """
    Main function to run the M2 validation by varying beam complexity.
    """
    # Test a range of beam complexities, from simple to complex
    modes_to_test = [2, 4, 6, 8, 10, 15, 20]
    results = []
    
    total_start_time = time.time()
    
    for num in modes_to_test:
        m2_t, m2_m = run_m2_test(num)
        results.append({
            "Num Modes": num,
            "Truth M2": m2_t,
            "Measured M2": m2_m,
            "Discrepancy (%)": 100 * abs(m2_t - m2_m) / m2_t,
        })

    total_end_time = time.time()
    
    print("\n" + "="*80)
    print("  DIRECT M-SQUARED VALIDATION SUMMARY (FIXED GRID, VARYING COMPLEXITY)")
    print("="*80)
    print(f"{'Num Modes':<15} | {'Truth M²':<15} | {'Measured M²':<15} | {'Discrepancy (%)':<18}")
    print("-"*80)
    
    for res in results:
        print(f"{res['Num Modes']:<15} | {res['Truth M2']:<15.4f} | {res['Measured M2']:<15.4f} | {res['Discrepancy (%)']:<18.2f}")
    
    print("-"*80)
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
# ==============================================================================
# PROJECT CHARLIE: M-SQUARED CONVERGENCE TEST
#
# This script runs the "Hidden Top-Hat" M-squared validation for multiple
# grid resolutions (16, 32, 64, 128) and produces a convergence table.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from math import factorial
import time
import os

# --- Global Simulation Parameters ---
XY_MAX = 8.0 # Physical extent of the simulation grid

# --- Ensure a directory exists for the figures ---
FIGURE_DIR = "paper_figures"
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

# ==============================================================================
# SECTION 1: CORE FUNCTIONS (FROM VERIFIED CODE)
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
    """Calculates M-squared from a set of LG(p,0) modal coefficients."""
    powers = np.abs(coeffs)**2
    numerator = 0
    # For LG(p,l) modes, M^2 = 2p + |l| + 1. Here l=0.
    for p in range(len(coeffs)):
        m2_mode = 2 * p + 1
        numerator += powers[p] * m2_mode
    
    total_power = np.sum(powers)
    if total_power < 1e-9:
        return 1.0
    
    return numerator / total_power

def calculate_m2_from_cmd_results(eigenvalues, eigenvectors, basis_modes, n_points, dx):
    """Calculates M^2 from the results of a CMD by identifying each mode."""
    # Ensure eigenvalues are sorted descending and normalize to probabilities
    sort_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_eigenvectors = eigenvectors[:, sort_indices]
    
    # Filter out numerically insignificant eigenvalues before normalization
    significant_mask = sorted_eigenvalues > 1e-9 * sorted_eigenvalues[0]
    filtered_eigenvalues = sorted_eigenvalues[significant_mask]
    filtered_eigenvectors = sorted_eigenvectors[:, significant_mask]
    
    normalized_powers = filtered_eigenvalues / np.sum(filtered_eigenvalues)
    
    m2_total = 0.0
    
    # Identify each numerical mode by finding its best match in the theoretical basis
    for i in range(len(normalized_powers)):
        mode_num_vec = filtered_eigenvectors[:, i]
        mode_num_2d = (mode_num_vec / np.sqrt(dx**2)).reshape((n_points, n_points))

        fidelities = [calculate_fidelity(mode_num_2d, basis_mode, dx) for basis_mode in basis_modes]
        best_match_idx = np.argmax(fidelities) # This is the 'p' index for the LG_p,0 mode
        
        # The M^2 of a single LG_p,0 mode is 2*p + 1
        m2_of_this_mode = 2 * best_match_idx + 1
        
        m2_total += normalized_powers[i] * m2_of_this_mode
        
    return m2_total

# ==============================================================================
# SECTION 2: MODULAR SIMULATION FUNCTION
# ==============================================================================

def run_simulation(n_points, generate_figure=False):
    """
    Runs the entire Hidden Top-Hat simulation for a given grid size.
    Returns the ground truth M2, the measured M2, and the execution time.
    """
    print("\n" + "="*80)
    print(f"  RUNNING SIMULATION FOR GRID SIZE: {n_points} x {n_points}")
    print("="*80)
    
    sim_start_time = time.time()
    
    x = np.linspace(-XY_MAX, XY_MAX, n_points)
    y = np.linspace(-XY_MAX, XY_MAX, n_points)
    dx = x[1]-x[0]
    
    sg_order, sg_radius, basis_waist, num_recipe_modes = 20, 3.0, 2.5, 20

    # --- Step 1: Get the "Magic Recipe" ---
    print(f"[N={n_points}] Calculating LG decomposition of super-Gaussian...")
    
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    E_sg_phys = np.exp(- (rr / sg_radius)**sg_order)
    E_sg_phys /= np.sqrt(np.sum(np.abs(E_sg_phys)**2) * dx**2)

    secret_coeffs = []
    basis_modes_phys = []
    for p in range(num_recipe_modes):
        lg_basis_mode = lg_mode(p, 0, basis_waist, x, y)
        # We need physically normalized modes for the overlap integral
        norm_factor = np.sqrt(np.sum(np.abs(lg_basis_mode)**2)*dx**2)
        lg_basis_phys_norm = lg_basis_mode / norm_factor if norm_factor > 1e-9 else lg_basis_mode
        coeff = np.sum(np.conj(lg_basis_phys_norm) * E_sg_phys) * dx**2
        secret_coeffs.append(coeff)
        basis_modes_phys.append(lg_basis_phys_norm)
    
    secret_coeffs = np.array(secret_coeffs)
    secret_powers = np.abs(secret_coeffs)**2

    # --- Calculate Ground Truth M2 from the known recipe ---
    m2_truth = calculate_m2_from_lg_coeffs(secret_coeffs)
    print(f"[N={n_points}] Ground Truth M-squared (from recipe): {m2_truth:.6f}")

    # --- Step 2: Synthesize the Deceptive Beam ---
    print(f"[N={n_points}] Synthesizing CSD matrix ({n_points**2}x{n_points**2})...")
    W_trojan = np.zeros((n_points**2, n_points**2), dtype=np.complex128)
    for i in range(num_recipe_modes):
        mode_flat = basis_modes_phys[i].ravel()
        W_trojan += secret_powers[i] * np.outer(mode_flat, np.conj(mode_flat))
    
    # --- Step 3: Analyze with the CMD Framework ---
    print(f"[N={n_points}] Performing eigenvalue decomposition...")
    start_eigh = time.time()
    eigenvalues, eigenvectors = np.linalg.eigh(W_trojan)
    end_eigh = time.time()
    print(f"[N={n_points}] Eigensolver finished in {end_eigh - start_eigh:.2f} seconds.")

    # --- Calculate Measured M2 from the CMD results ---
    m2_measured = calculate_m2_from_cmd_results(eigenvalues, eigenvectors, basis_modes_phys, n_points, dx)
    print(f"[N={n_points}] Measured M-squared (from CMD results): {m2_measured:.6f}")
    
    if generate_figure:
        print(f"[N={n_points}] Generating summary figure...")
        # (Code to reconstruct and plot the figure, same as before)
        intensity_total = np.real(np.diag(W_trojan)).reshape((n_points, n_points))
        sort_indices = np.argsort(eigenvalues)[::-1]
        
        modes_num_phys = []
        for j in range(num_recipe_modes):
            mode_vector = eigenvectors[:, sort_indices[j]]
            mode_2d = (mode_vector / np.sqrt(dx**2)).reshape((n_points, n_points))
            max_abs_idx = np.argmax(np.abs(mode_2d))
            phase_corr = np.angle(mode_2d.ravel()[max_abs_idx])
            mode_2d *= np.exp(-1j * phase_corr)
            modes_num_phys.append(mode_2d)

        E_reconstructed_sg = np.zeros_like(E_sg_phys, dtype=np.complex128)
        used_indices = set()
        for i in range(num_recipe_modes):
            mode_num_phys_i = modes_num_phys[i]
            fids = [calculate_fidelity(sm, mode_num_phys_i, dx) if k not in used_indices else 0.0 for k, sm in enumerate(basis_modes_phys)]
            best_match_idx = np.argmax(fids)
            used_indices.add(best_match_idx)
            E_reconstructed_sg += secret_coeffs[best_match_idx] * mode_num_phys_i

        fig = plt.figure(figsize=(20, 7))
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 2], hspace=0.1, wspace=0.3)
        ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(intensity_total, cmap='inferno', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX]); ax0.set_title('(a) The Deception: Total Intensity', fontsize=18); ax0.set_xticks([]); ax0.set_yticks([])
        ax0_slice = fig.add_subplot(gs[1, 0]); center_slice = intensity_total[n_points//2, :]; ax0_slice.plot(x, center_slice/center_slice.max(), 'r-', linewidth=2.5); ax0_slice.grid(True, linestyle='--'); ax0_slice.set_yticks([]); ax0_slice.set_xlim(-XY_MAX, XY_MAX)
        gs_inner = gs[0, 1].subgridspec(2, 2, wspace=0.05, hspace=0.05); axes_b = gs_inner.subplots(); axes_b[0,0].text(0.5, 1.25, '(b) The Hidden Modes', transform=axes_b[0,0].transAxes, ha='center', va='center', fontsize=18)
        axes_b[0,0].imshow(np.abs(modes_num_phys[0]), cmap='inferno'); axes_b[0,0].set_title('Mode 0 (LG$_{0,0}$)')
        axes_b[0,1].imshow(np.abs(modes_num_phys[1]), cmap='inferno'); axes_b[0,1].set_title('Mode 1 (LG$_{1,0}$)')
        axes_b[1,0].imshow(np.abs(modes_num_phys[2]), cmap='inferno'); axes_b[1,0].set_title('Mode 2 (LG$_{2,0}$)')
        axes_b[1,1].imshow(np.abs(modes_num_phys[3]), cmap='inferno'); axes_b[1,1].set_title('Mode 3 (LG$_{3,0}$)')
        for ax in axes_b.flat: ax.axis('off')
        ax_dummy = fig.add_subplot(gs[1, 1]); ax_dummy.axis('off')
        ax2 = fig.add_subplot(gs[0, 2]); ax2.imshow(np.abs(E_reconstructed_sg)**2, cmap='inferno', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX]); ax2.set_title('(c) The Hidden Potential', fontsize=18); ax2.set_xticks([]); ax2.set_yticks([])
        ax2_slice = fig.add_subplot(gs[1, 2]); theory_slice_sg = np.abs(E_sg_phys[n_points//2, :])**2; recon_slice_sg = np.abs(E_reconstructed_sg[n_points//2, :])**2; ax2_slice.plot(x, theory_slice_sg/theory_slice_sg.max(), 'b-', linewidth=2.5, label='Theoretical Target'); ax2_slice.plot(x, recon_slice_sg/recon_slice_sg.max(), 'r--', linewidth=2.5, label='Reconstructed'); ax2_slice.grid(True, linestyle='--'); ax2_slice.set_yticks([]); ax2_slice.set_xlim(-XY_MAX, XY_MAX); ax2_slice.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "hidden_tophat_figure_final.png"), dpi=200)
        plt.show()

    sim_end_time = time.time()
    exec_time = sim_end_time - sim_start_time
    
    return m2_truth, m2_measured, exec_time

# ==============================================================================
# SECTION 3: MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """
    Main function to run the convergence test and print the summary.
    """
    grid_sizes_to_test = [16, 32, 64, 128]
    results = []
    
    total_start_time = time.time()
    
    for n in grid_sizes_to_test:
        # Generate the figure only for the last (highest resolution) run
        should_generate_figure = (n == grid_sizes_to_test[-1])
        
        m2_t, m2_m, ex_time = run_simulation(n, generate_figure=should_generate_figure)
        
        results.append({
            "N": n,
            "Truth M2": m2_t,
            "Measured M2": m2_m,
            "Discrepancy (%)": 100 * abs(m2_t - m2_m) / m2_t,
            "Time (s)": ex_time
        })

    total_end_time = time.time()
    
    print("\n" + "="*80)
    print("  M-SQUARED CONVERGENCE TEST SUMMARY")
    print("="*80)
    print(f"{'Grid Size':<15} | {'Truth M²':<15} | {'Measured M²':<15} | {'Discrepancy (%)':<18} | {'Time (s)':<10}")
    print("-"*80)
    
    for res in results:
        grid_str = f"{res['N']}x{res['N']}"
        print(f"{grid_str:<15} | {res['Truth M2']:<15.4f} | {res['Measured M2']:<15.4f} | {res['Discrepancy (%)']:<18.2f} | {res['Time (s)']:<10.2f}")
    
    print("-"*80)
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
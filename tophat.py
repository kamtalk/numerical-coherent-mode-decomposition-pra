# ==============================================================================
# PROJECT BRAVO: THE "HIDDEN TOP-HAT" DEMONSTRATION
#
# This script generates the 4-panel figure for the "Hidden Coherence"
# application, showing the deception, the recovered modes, the eigenvalue
# spectrum, and the reconstructed potential.
#
# MODIFIED VERSION: Includes the eigenvalue spectrum plot.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from math import factorial
import time
import os

# --- Global Simulation Parameters ---
N_POINTS = 128
XY_MAX = 8.0

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

# ==============================================================================
# SECTION 2: THE "HIDDEN TOP-HAT" DEMONSTRATION
# ==============================================================================

def main():
    print("--- Project Bravo: The Hidden Top-Hat Demonstration ---")
    
    x = np.linspace(-XY_MAX, XY_MAX, N_POINTS)
    y = np.linspace(-XY_MAX, XY_MAX, N_POINTS)
    dx = x[1]-x[0]
    
    sg_order, sg_radius, basis_waist, num_recipe_modes = 20, 3.0, 2.5, 20

    # --- Step 1: Get the "Magic Recipe" ---
    print(f"Calculating LG decomposition of super-Gaussian (p={sg_order})...", end='', flush=True)
    start_time = time.time()
    
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    E_sg_phys = np.exp(- (rr / sg_radius)**sg_order)
    E_sg_phys /= np.sqrt(np.sum(np.abs(E_sg_phys)**2) * dx**2)

    secret_coeffs = []
    basis_modes_phys = []
    for p in range(num_recipe_modes):
        lg_basis_mode = lg_mode(p, 0, basis_waist, x, y)
        lg_basis_phys_norm = lg_basis_mode / np.sqrt(np.sum(np.abs(lg_basis_mode)**2)*dx**2)
        coeff = np.sum(np.conj(lg_basis_phys_norm) * E_sg_phys) * dx**2
        secret_coeffs.append(coeff)
        basis_modes_phys.append(lg_basis_phys_norm)
    
    end_time = time.time()
    print(f" done in {end_time - start_time:.2f} seconds.")
    
    secret_coeffs = np.array(secret_coeffs)
    secret_powers = np.abs(secret_coeffs)**2

    # --- Step 2: Synthesize the Deceptive Beam ---
    print("\n[Step 2] Synthesizing deceptive partially coherent beam...", end='', flush=True)
    start_time = time.time()
    W_trojan = np.zeros((N_POINTS**2, N_POINTS**2), dtype=np.complex128)
    for i in range(num_recipe_modes):
        mode_flat = basis_modes_phys[i].ravel()
        W_trojan += secret_powers[i] * np.outer(mode_flat, np.conj(mode_flat))
    end_time = time.time()
    print(f" done in {end_time - start_time:.2f} seconds.")
    
    intensity_total = np.real(np.diag(W_trojan)).reshape((N_POINTS, N_POINTS))

    # --- Step 3: Analyze with the CMD Framework ---
    print("\n[Step 3] Analyzing with the CMD framework...")
    print("Performing eigenvalue decomposition...", end='', flush=True)
    start_time = time.time()
    eigenvalues, eigenvectors = np.linalg.eigh(W_trojan)
    end_time = time.time()
    print(f" done in {end_time - start_time:.2f} seconds.")

    sort_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_modal_powers = sorted_eigenvalues * dx**2
    modes_num_l2 = [eigenvectors[:, sort_indices[j]] for j in range(num_recipe_modes)]

    # Physical modes with phase convention
    modes_num_phys = []
    for j in range(num_recipe_modes):
        mode_vector = modes_num_l2[j]
        mode_2d = (mode_vector / np.sqrt(dx**2)).reshape((N_POINTS, N_POINTS))
        max_abs_idx = np.argmax(np.abs(mode_2d))
        phase_corr = np.angle(mode_2d.ravel()[max_abs_idx])
        mode_2d *= np.exp(-1j * phase_corr)
        modes_num_phys.append(mode_2d)
        
    # --- Step 4: Reveal the Hidden Truth ---
    print("\n[Step 4] Revealing the hidden structure...")
    theory_powers_sorted = np.sort(secret_powers)[::-1]
    print("\n--- Recovered Eigenvalue Spectrum vs. Theory ---")
    print("Mode Idx | Theory Power | Numerical Power")
    for i in range(5):
        print(f"   {i}     | {theory_powers_sorted[i]:.6f}   | {sorted_modal_powers[i]:.6f}")
    
    print("\n--- Recovered Mode Fidelity ---")
    for i in range(5):
        best_fid = max(calculate_fidelity(modes_num_phys[i], sm, dx) for sm in basis_modes_phys)
        print(f"Fidelity of Recovered Mode {i}: {best_fid:.6f}")

    # --- Step 5: Reconstruct the Hidden Potential ---
    print("\n[Step 5] Reconstructing hidden coherent potential...")
    E_reconstructed_sg = np.zeros_like(E_sg_phys, dtype=np.complex128)
    used_indices = set()
    for i in range(num_recipe_modes):
        mode_num_phys = modes_num_phys[i]
        fids = [calculate_fidelity(sm, mode_num_phys, dx) if k not in used_indices else 0.0 for k, sm in enumerate(basis_modes_phys)]
        best_match_idx = np.argmax(fids)
        used_indices.add(best_match_idx)
        overlap = np.sum(np.conj(basis_modes_phys[best_match_idx]) * mode_num_phys) * dx**2
        phase_correction = np.angle(overlap)
        mode_aligned = mode_num_phys * np.exp(-1j * phase_correction)
        E_reconstructed_sg += secret_coeffs[best_match_idx] * mode_aligned
    
    # --- Step 6: Generate the Definitive Figure ---
    print("\n[Step 6] Generating the summary figure...")
    fig = plt.figure(figsize=(20, 7))
    # Increased hspace for better title spacing
    gs = fig.add_gridspec(2, 3, height_ratios=[5, 2], hspace=0.3, wspace=0.3)
    
    # Panel (a): Deception
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(intensity_total, cmap='inferno', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX])
    ax0.set_title('(a) The Deception: Total Intensity', fontsize=18)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0_slice = fig.add_subplot(gs[1, 0])
    center_slice = intensity_total[N_POINTS//2, :]
    ax0_slice.plot(x, center_slice/center_slice.max(), 'r-', linewidth=2.5)
    ax0_slice.grid(True, linestyle='--'); ax0_slice.set_yticks([])
    ax0_slice.set_xlim(-XY_MAX, XY_MAX)

    # Panel (b): Hidden Modes
    gs_inner = gs[0, 1].subgridspec(2, 2, wspace=0.05, hspace=0.05)
    axes_b = gs_inner.subplots()
    axes_b[0,0].text(0.5, 1.25, '(b) The Hidden Modes', transform=axes_b[0,0].transAxes, ha='center', va='center', fontsize=18)
    
    axes_b[0,0].imshow(np.abs(modes_num_phys[0]), cmap='inferno'); axes_b[0,0].set_title('Mode 0 (LG$_{0,0}$)')
    axes_b[0,1].imshow(np.abs(modes_num_phys[1]), cmap='inferno'); axes_b[0,1].set_title('Mode 1 (LG$_{1,0}$)')
    axes_b[1,0].imshow(np.abs(modes_num_phys[2]), cmap='inferno'); axes_b[1,0].set_title('Mode 2 (LG$_{2,0}$)')
    axes_b[1,1].imshow(np.abs(modes_num_phys[3]), cmap='inferno'); axes_b[1,1].set_title('Mode 3 (LG$_{3,0}$)')
    for ax in axes_b.flat: ax.axis('off')
    
    # >>> NEW PLOT: The Eigenvalue Spectrum <<<
    ax_eigen = fig.add_subplot(gs[1, 1])
    mode_indices = np.arange(num_recipe_modes)
    powers_to_plot = sorted_modal_powers[:num_recipe_modes]
    # Normalize powers for the plot title if you want
    total_power = np.sum(powers_to_plot)
    ax_eigen.semilogy(mode_indices, powers_to_plot / total_power, 'o-', color='darkorange', markersize=4, mfc='white')
    ax_eigen.set_title("Recovered Eigenvalue Spectrum", fontsize=14)
    ax_eigen.set_xlabel("Mode Index", fontsize=12)
    ax_eigen.set_ylabel("Normalized Power", fontsize=12)
    ax_eigen.grid(True, which='both', linestyle=':')
    ax_eigen.set_xlim(left=-0.5, right=num_recipe_modes - 0.5)
    ax_eigen.set_ylim(bottom=1e-5, top=1.0) # Set reasonable y-limits
    ax_eigen.tick_params(axis='both', which='major', labelsize=10)


    # Panel (c): Verification
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(np.abs(E_reconstructed_sg)**2, cmap='inferno', extent=[-XY_MAX, XY_MAX, -XY_MAX, XY_MAX])
    ax2.set_title('(c) The Hidden Potential', fontsize=18)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2_slice = fig.add_subplot(gs[1, 2])
    
    theory_slice_sg = np.abs(E_sg_phys[N_POINTS//2, :])**2
    recon_slice_sg = np.abs(E_reconstructed_sg[N_POINTS//2, :])**2
    ax2_slice.plot(x, theory_slice_sg/theory_slice_sg.max(), 'b-', linewidth=3, label='Theoretical Target')
    ax2_slice.plot(x, recon_slice_sg/recon_slice_sg.max(), 'r--', linewidth=2.5, label='Reconstructed')
    ax2_slice.grid(True, linestyle='--'); ax2_slice.set_yticks([])
    ax2_slice.set_xlim(-XY_MAX, XY_MAX)
    ax2_slice.legend(fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Add a bit of padding for main title if needed
    plt.savefig(os.path.join(FIGURE_DIR, "hidden_tophat_figure_with_spectrum.png"), dpi=200)
    plt.show()

    print("\nDemonstration complete.")

if __name__ == '__main__':
    main()
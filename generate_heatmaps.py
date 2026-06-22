import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_heatmaps():
    csv_path = os.path.join(BASE_DIR, 'sensitivity_results.csv')
    df = pd.read_csv(csv_path)

    # Slice for a fixed c_rej (e.g., 500)
    df_slice = df[df['c_rej'] == 500]
    grouped = df_slice.groupby(['p', 'c_esc']).mean().reset_index()

    # Coordinates of the experimental grid
    x = grouped['p'].values
    y = grouped['c_esc'].values
    z_i = grouped['i_opt'].values
    z_f = grouped['f_opt'].values

    # Create a dense regular grid for smooth visualization
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate using cubic method for smooth transitions
    Zi_i = griddata((x, y), z_i, (Xi, Yi), method='cubic')
    Zi_f = griddata((x, y), z_f, (Xi, Yi), method='cubic')
    # Clip sampling fraction to valid range [0,1] (interpolation may overshoot)
    Zi_f = np.clip(Zi_f, 0.0, 1.0)

    # --- Plot i* heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(
        Zi_i,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis',
        interpolation='bilinear'
    )

    # Overlay original grid points with text labels
    for _, row in grouped.iterrows():
        ax.text(row['p'], row['c_esc'], f"{int(row['i_opt'])}",
                ha='center', va='center', color='white' if row['i_opt'] > 250 else 'black',
                fontsize=9, fontweight='bold')

    fig.colorbar(cax, label='Optimal clearance number $i^*$')
    ax.set_xlabel("Incoming Defect Rate ($p'$)")
    ax.set_ylabel("Escape Cost ($c_{esc}$)")
    ax.set_title("Optimal clearance number $i^*$ vs. defect rate ($p'$) and escape cost ($c_{esc}$)")
    plt.savefig(os.path.join(BASE_DIR, 'heatmap_i_opt.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'heatmap_i_opt.eps'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot f* heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(
        Zi_f,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='auto',
        cmap='viridis',
        interpolation='bilinear'
    )

    # Overlay original grid points with text labels
    for _, row in grouped.iterrows():
        ax.text(row['p'], row['c_esc'], f"{row['f_opt']:.2f}",
                ha='center', va='center', color='white' if row['f_opt'] > 0.5 else 'black',
                fontsize=9, fontweight='bold')

    fig.colorbar(cax, label='Optimal sampling fraction $f^*$')
    ax.set_xlabel("Incoming Defect Rate ($p'$)")
    ax.set_ylabel("Escape Cost ($c_{esc}$)")
    ax.set_title("Optimal sampling fraction $f^*$ vs. defect rate ($p'$) and escape cost ($c_{esc}$)")
    plt.savefig(os.path.join(BASE_DIR, 'heatmap_f_opt.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'heatmap_f_opt.eps'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Heatmaps generated and saved (interpolated, 300 dpi).")

if __name__ == "__main__":
    generate_heatmaps()

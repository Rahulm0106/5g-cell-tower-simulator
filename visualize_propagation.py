"""
Visualization of Path Loss Models

This script creates plots to help visualize how radio signals degrade with distance.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.propagation import (
    PropagationEnvironment,
    FreeSpacePathLoss,
    OkumuraHataModel,
    ThreeGPP_38_901_UMa,
    calculate_received_power
)

def plot_path_loss_comparison():
    """
    Compare different propagation models.
    
    This helps understand:
    1. How optimistic/pessimistic each model is
    2. Which model to use for which scenario
    3. The impact of environment on signal propagation
    """
    # Setup
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    # Initialize models
    fspl = FreeSpacePathLoss(env)
    okumura = OkumuraHataModel(env)
    threeGpp = ThreeGPP_38_901_UMa(env)
    
    # Distance range: 10m to 2000m
    distances = np.linspace(10, 2000, 200)
    
    # Calculate path loss for each model
    fspl_losses = [fspl.calculate_path_loss(d) for d in distances]
    okumura_losses = [okumura.calculate_path_loss(d) for d in distances]
    threeGpp_los_losses = [threeGpp.calculate_path_loss(d, force_los=True) for d in distances]
    threeGpp_nlos_losses = [threeGpp.calculate_path_loss(d, force_los=False) for d in distances]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Path Loss vs Distance
    ax1.plot(distances, fspl_losses, 'b-', label='Free Space (FSPL)', linewidth=2)
    ax1.plot(distances, okumura_losses, 'r-', label='Okumura-Hata (Urban)', linewidth=2)
    ax1.plot(distances, threeGpp_los_losses, 'g-', label='3GPP 38.901 (LoS)', linewidth=2)
    ax1.plot(distances, threeGpp_nlos_losses, 'm-', label='3GPP 38.901 (NLoS)', linewidth=2)
    
    ax1.set_xlabel('Distance (meters)', fontsize=12)
    ax1.set_ylabel('Path Loss (dB)', fontsize=12)
    ax1.set_title('Path Loss Models Comparison\n(Urban, 3.5 GHz, BS Height=25m)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add annotations
    ax1.annotate('Lower loss =\nBetter signal', 
                xy=(500, 90), fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax1.annotate('Higher loss =\nWeaker signal', 
                xy=(500, 140), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # Plot 2: Received Power vs Distance
    tx_power = 43  # dBm
    rx_powers_fspl = [calculate_received_power(tx_power, pl) for pl in fspl_losses]
    rx_powers_okumura = [calculate_received_power(tx_power, pl) for pl in okumura_losses]
    rx_powers_3gpp_los = [calculate_received_power(tx_power, pl) for pl in threeGpp_los_losses]
    rx_powers_3gpp_nlos = [calculate_received_power(tx_power, pl) for pl in threeGpp_nlos_losses]
    
    ax2.plot(distances, rx_powers_fspl, 'b-', label='Free Space', linewidth=2)
    ax2.plot(distances, rx_powers_okumura, 'r-', label='Okumura-Hata', linewidth=2)
    ax2.plot(distances, rx_powers_3gpp_los, 'g-', label='3GPP LoS', linewidth=2)
    ax2.plot(distances, rx_powers_3gpp_nlos, 'm-', label='3GPP NLoS', linewidth=2)
    
    # Add signal quality thresholds
    ax2.axhline(y=-80, color='green', linestyle='--', alpha=0.5, label='Excellent (>-80 dBm)')
    ax2.axhline(y=-90, color='orange', linestyle='--', alpha=0.5, label='Good (>-90 dBm)')
    ax2.axhline(y=-100, color='red', linestyle='--', alpha=0.5, label='Fair (>-100 dBm)')
    
    ax2.set_xlabel('Distance (meters)', fontsize=12)
    ax2.set_ylabel('Received Power (dBm)', fontsize=12)
    ax2.set_title(f'Received Signal Strength (Tx Power = {tx_power} dBm)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('/home/claude/5g-cell-tower-sim/path_loss_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Plot saved: path_loss_comparison.png")
    
    return fig


def plot_coverage_map():
    """
    Create a 2D coverage heatmap showing signal strength around a cell tower.
    
    This visualizes:
    - Coverage area of a cell
    - Dead zones (poor signal)
    - Where handovers might occur
    """
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    model = ThreeGPP_38_901_UMa(env)
    
    # Create grid around tower (tower at origin)
    x = np.linspace(-1000, 1000, 100)
    y = np.linspace(-1000, 1000, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate received power at each point
    tx_power = 43  # dBm
    Z = np.zeros_like(X)
    
    print("Calculating coverage map... (this may take a moment)")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            distance = np.sqrt(X[i, j]**2 + Y[i, j]**2)
            if distance < 10:  # Avoid tower location
                distance = 10
            path_loss = model.calculate_path_loss(distance, force_los=False)
            Z[i, j] = calculate_received_power(tx_power, path_loss)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Plot heatmap
    levels = np.arange(-120, -40, 5)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlGn')
    contour = ax.contour(X, Y, Z, levels=[-100, -90, -80], colors='black', linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=10, fmt='%d dBm')
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, label='Received Power (dBm)')
    
    # Mark tower location
    ax.plot(0, 0, 'r^', markersize=20, label='Cell Tower', markeredgecolor='black', markeredgewidth=2)
    
    # Add labels and formatting
    ax.set_xlabel('Distance East-West (meters)', fontsize=12)
    ax.set_ylabel('Distance North-South (meters)', fontsize=12)
    ax.set_title('5G Cell Tower Coverage Map\n(Urban, 3.5 GHz, Tx Power=43 dBm)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add text annotations
    ax.text(0, -900, 'Green = Good Signal\nYellow = Fair Signal\nRed = Poor Signal', 
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/claude/5g-cell-tower-sim/coverage_map.png', dpi=150, bbox_inches='tight')
    print("✅ Plot saved: coverage_map.png")
    
    return fig


def plot_frequency_comparison():
    """
    Compare how different 5G frequency bands behave.
    
    Key learning:
    - Low band (Sub-1 GHz): Better coverage, lower speed
    - Mid band (1-6 GHz): Balanced
    - High band (mmWave, >24 GHz): High speed, poor coverage
    """
    frequencies = {
        '600 MHz (Low Band)': 600,
        '3.5 GHz (Mid Band)': 3500,
        '28 GHz (mmWave)': 28000
    }
    
    distances = np.linspace(10, 2000, 200)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for label, freq in frequencies.items():
        env = PropagationEnvironment(
            environment_type='urban',
            carrier_frequency=freq
        )
        model = ThreeGPP_38_901_UMa(env)
        
        losses = [model.calculate_path_loss(d, force_los=True) for d in distances]
        rx_powers = [calculate_received_power(43, pl) for pl in losses]
        
        ax.plot(distances, rx_powers, linewidth=2.5, label=label)
    
    # Add threshold lines
    ax.axhline(y=-80, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=-100, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Distance (meters)', fontsize=12)
    ax.set_ylabel('Received Power (dBm)', fontsize=12)
    ax.set_title('5G Frequency Band Comparison\n(Why lower frequencies have better coverage)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    
    # Add annotation
    ax.annotate('Higher frequency =\nMore path loss =\nShorter range', 
                xy=(1500, -80), fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/claude/5g-cell-tower-sim/frequency_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Plot saved: frequency_comparison.png")
    
    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING EDUCATIONAL VISUALIZATIONS")
    print("=" * 70)
    print()
    
    # Generate all plots
    print("1. Generating path loss comparison plot...")
    plot_path_loss_comparison()
    print()
    
    print("2. Generating coverage map...")
    plot_coverage_map()
    print()
    
    print("3. Generating frequency comparison plot...")
    plot_frequency_comparison()
    print()
    
    print("=" * 70)
    print("✅ All visualizations generated successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - path_loss_comparison.png")
    print("  - coverage_map.png")
    print("  - frequency_comparison.png")
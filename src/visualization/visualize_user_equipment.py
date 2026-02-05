"""
Visualization: Multi-Cell Network with UE Trajectory

Generates a comprehensive 2x2 figure:
  Top-Left:  Coverage map + UE path + handover points
  Top-Right: SINR over time (the metric that determines your speed)
  Bot-Left:  RSRP over time (raw signal strength)
  Bot-Right: Throughput over time (what the user actually experiences)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
# Ensure the project `src/` directory is on sys.path when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.propagation import PropagationEnvironment
from core.cell_tower import CellTower
from core.user_equipment import UserEquipment, GaussMarkovMobility


def run_simulation():
    """Run a 3-tower, 1-UE simulation for 180 seconds."""

    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )

    # 3 towers in a triangle arrangement
    towers = [
        CellTower.create_standard_3sector("Tower_A", x=-500,  y= 0,   environment=env),
        CellTower.create_standard_3sector("Tower_B", x= 500,  y= 0,   environment=env),
        CellTower.create_standard_3sector("Tower_C", x=  0,   y= 800, environment=env),
    ]

    # UE starts near Tower A, drifts generally East then North
    mobility = GaussMarkovMobility(
        alpha=0.75,
        mean_speed=4.0,
        speed_std=1.2,
        mean_direction_deg=60.0,          # NE-ish
        bounds=(-700, 700, -300, 1000)
    )

    ue = UserEquipment(
        ue_id="UE_01",
        x=-450, y=-50,
        mobility=mobility,
        serving_tower=towers[0]
    )

    # Run 180 ticks @ 1 s each
    results = []
    for _ in range(180):
        results.append(ue.tick(towers, dt=1.0))

    return towers, ue, results


def plot_all(towers, ue, results):
    """Generate the 2x2 summary figure."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Multi-Cell 5G Network — UE Movement, SINR & Handover",
                 fontsize=15, fontweight='bold', y=0.98)

    # Colour map for towers
    tower_colors = {"Tower_A": "#e74c3c", "Tower_B": "#2ecc71", "Tower_C": "#3498db"}

    # =========================================================
    # TOP-LEFT: Coverage map + trajectory
    # =========================================================
    ax = axes[0, 0]

    # Light coverage circles (just for visual context)
    for t in towers:
        circle = plt.Circle((t.x, t.y), 600, color=tower_colors[t.tower_id],
                            alpha=0.08, linewidth=0)
        ax.add_patch(circle)
        # Tower marker
        ax.plot(t.x, t.y, '^', color=tower_colors[t.tower_id],
                markersize=18, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax.annotate(t.tower_id, (t.x, t.y), textcoords="offset points",
                    xytext=(0, 14), ha='center', fontsize=10, fontweight='bold')

    # UE trajectory, coloured by serving tower
    xs = [r['x'] for r in results]
    ys = [r['y'] for r in results]
    servings = [r['serving'] for r in results]

    # Draw segment by segment so colour changes at handovers
    for i in range(1, len(xs)):
        color = tower_colors.get(servings[i], 'gray')
        ax.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]], color=color, linewidth=2.5, zorder=3)

    # Handover markers
    for tick_idx, from_id, to_id in ue.handover_log:
        if tick_idx < len(results):
            hx, hy = results[tick_idx]['x'], results[tick_idx]['y']
            ax.plot(hx, hy, 'D', color='gold', markersize=12,
                    markeredgecolor='black', markeredgewidth=1.5, zorder=6)
            ax.annotate(f"HO→{to_id[-1]}", (hx, hy), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='gold', alpha=0.8))

    # Start marker
    ax.plot(xs[0], ys[0], 'o', color='white', markersize=10,
            markeredgecolor='black', markeredgewidth=2, zorder=7)

    ax.set_xlim(-750, 750)
    ax.set_ylim(-400, 1100)
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")
    ax.set_title("Network Topology + UE Trajectory")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Legend
    legend_elements = [
        mpatches.Patch(color=tower_colors["Tower_A"], label="Serving Tower A"),
        mpatches.Patch(color=tower_colors["Tower_B"], label="Serving Tower B"),
        mpatches.Patch(color=tower_colors["Tower_C"], label="Serving Tower C"),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gold',
                   markersize=10, markeredgecolor='black', label="Handover event"),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # =========================================================
    # TOP-RIGHT: SINR over time
    # =========================================================
    ax = axes[0, 1]
    time_s = np.arange(len(results))
    sinr_vals = [r['sinr_db'] for r in results]

    ax.plot(time_s, sinr_vals, color='#8e44ad', linewidth=1.8)
    ax.fill_between(time_s, sinr_vals, alpha=0.15, color='#8e44ad')

    # Handover vertical lines
    for tick_idx, from_id, to_id in ue.handover_log:
        ax.axvline(tick_idx, color='gold', linewidth=2, linestyle='--', alpha=0.8)
        ax.text(tick_idx + 1, max(sinr_vals) * 0.9, f"HO→{to_id[-1]}",
                fontsize=8, color='#b7950b', fontweight='bold')

    # Quality bands
    ax.axhline(y=20,  color='green',  linestyle=':', alpha=0.4)
    ax.axhline(y=10,  color='orange', linestyle=':', alpha=0.4)
    ax.axhline(y=0,   color='red',    linestyle=':', alpha=0.4)
    ax.text(182, 21, 'Excellent', fontsize=7, color='green')
    ax.text(182, 11, 'Good',      fontsize=7, color='orange')
    ax.text(182, 1,  'Fair',      fontsize=7, color='red')

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("SINR (dB)")
    ax.set_title("SINR Over Time\n(drops at cell edges where interference is high)")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, len(results))

    # =========================================================
    # BOTTOM-LEFT: RSRP over time
    # =========================================================
    ax = axes[1, 0]
    rsrp_vals = [r['rsrp_dbm'] for r in results]

    # Colour the line by serving tower
    for i in range(1, len(time_s)):
        color = tower_colors.get(results[i]['serving'], 'gray')
        ax.plot([time_s[i-1], time_s[i]], [rsrp_vals[i-1], rsrp_vals[i]],
                color=color, linewidth=2)

    # Handover lines
    for tick_idx, from_id, to_id in ue.handover_log:
        ax.axvline(tick_idx, color='gold', linewidth=2, linestyle='--', alpha=0.8)

    # Signal quality thresholds
    ax.axhline(y=-80,  color='green',  linestyle=':', alpha=0.4)
    ax.axhline(y=-90,  color='orange', linestyle=':', alpha=0.4)
    ax.axhline(y=-100, color='red',    linestyle=':', alpha=0.4)
    ax.text(182, -79,  'Excellent', fontsize=7, color='green')
    ax.text(182, -89,  'Good',      fontsize=7, color='orange')
    ax.text(182, -99,  'Fair',      fontsize=7, color='red')

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("RSRP (dBm)")
    ax.set_title("RSRP Over Time\n(colour = which tower is serving)")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, len(results))

    # =========================================================
    # BOTTOM-RIGHT: Throughput over time
    # =========================================================
    ax = axes[1, 1]
    tput_vals = [r['throughput_mbps'] for r in results]

    ax.plot(time_s, tput_vals, color='#16a085', linewidth=1.8)
    ax.fill_between(time_s, tput_vals, alpha=0.15, color='#16a085')

    # Handover lines
    for tick_idx, from_id, to_id in ue.handover_log:
        ax.axvline(tick_idx, color='gold', linewidth=2, linestyle='--', alpha=0.8)

    # Average throughput line
    avg_tput = np.mean(tput_vals)
    ax.axhline(y=avg_tput, color='#16a085', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(5, avg_tput + 20, f'Avg: {avg_tput:.0f} Mbps', fontsize=9,
            color='#16a085', fontweight='bold')

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Estimated Throughput\n(Shannon capacity × 0.7 efficiency)")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, len(results))
    ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save to results/plots/ folder
    output_dir = os.path.join(os.path.dirname(__file__), '../../results/plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'multi_cell_overview.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("✅ Saved: multi_cell_overview.png")


if __name__ == "__main__":
    print("Running 3-tower simulation (180 s)...")
    towers, ue, results = run_simulation()

    print(f"\n📊 Simulation summary:")
    print(f"   Handovers executed: {len(ue.handover_log)}")
    for tick_idx, from_id, to_id in ue.handover_log:
        print(f"     Tick {tick_idx:>4}s: {from_id} → {to_id}")

    avg_sinr  = np.mean([r['sinr_db'] for r in results])
    avg_tput  = np.mean([r['throughput_mbps'] for r in results])
    print(f"   Avg SINR:       {avg_sinr:.1f} dB")
    print(f"   Avg Throughput: {avg_tput:.0f} Mbps")

    print("\nGenerating visualization...")
    plot_all(towers, ue, results)
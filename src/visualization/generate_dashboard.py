"""
Automated Dashboard Generator

Runs all scenarios and generates a comprehensive comparison dashboard.
Perfect for presentations and portfolio demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Ensure the project `src/` directory is on sys.path when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.scenarios import (
    scenario_baseline,
    scenario_peak_hour,
    scenario_tower_failure,
    scenario_traffic_surge
)


def generate_comprehensive_dashboard():
    """
    Generate complete dashboard with all scenarios.
    
    Layout: 4x4 grid
      - Row 1: Scenario 1 (Baseline)
      - Row 2: Scenario 2 (Peak Hour)
      - Row 3: Scenario 3 (Tower Failure)
      - Row 4: Scenario 4 (Traffic Surge)
      
    Columns: SINR, Throughput, Cell Load, Summary Stats
    """
    
    # Define scenarios to run
    scenarios = [
        ('Baseline (20 UEs)', lambda: scenario_baseline(num_ues=20), 120),
        ('Peak Hour (80 UEs)', lambda: scenario_peak_hour(num_ues=80), 180),
        ('Tower Failure (40 UEs)', lambda: scenario_tower_failure(num_ues=40, failure_time=60), 180),
        ('Traffic Surge (30+30 UEs)', lambda: scenario_traffic_surge(baseline_ues=30, surge_time=60, surge_ues=30), 150),
    ]
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('5G Network Simulator — Comprehensive Dashboard', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Run each scenario
    for row_idx, (name, factory, duration) in enumerate(scenarios):
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")
        
        # Create network and run
        net = factory()
        net.run(duration_s=duration, dt=1.0, verbose=False)
        
        kpis = net.kpi_history
        times = [k.timestamp for k in kpis]
        
        # Column 1: SINR
        ax = plt.subplot(4, 4, row_idx*4 + 1)
        avg_sinr = [k.avg_sinr_db for k in kpis]
        p5_sinr = [k.p5_sinr_db for k in kpis]
        
        ax.plot(times, avg_sinr, 'b-', linewidth=2, label='Avg')
        ax.plot(times, p5_sinr, 'r--', linewidth=1.5, label='P5')
        ax.axhline(y=10, color='green', linestyle=':', alpha=0.5)
        ax.fill_between(times, p5_sinr, alpha=0.15, color='red')
        
        if row_idx == 0:
            ax.set_title('SINR (dB)', fontsize=12, fontweight='bold')
        ax.set_ylabel(name, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        if row_idx < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=9)
        
        # Column 2: Throughput
        ax = plt.subplot(4, 4, row_idx*4 + 2)
        total_tput = [k.total_throughput_gbps for k in kpis]
        
        ax.plot(times, total_tput, 'g-', linewidth=2)
        ax.fill_between(times, total_tput, alpha=0.2, color='green')
        
        if row_idx == 0:
            ax.set_title('Throughput (Gbps)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if row_idx < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=9)
        
        # Column 3: Cell Load
        ax = plt.subplot(4, 4, row_idx*4 + 3)
        
        all_tower_ids = sorted(set(tid for k in kpis for tid in k.tower_loads.keys()))
        load_matrix = np.zeros((len(times), len(all_tower_ids)))
        for i, k in enumerate(kpis):
            for j, tid in enumerate(all_tower_ids):
                load_matrix[i, j] = k.tower_loads.get(tid, 0)
        
        ax.stackplot(times, load_matrix.T, labels=all_tower_ids, alpha=0.7)
        
        if row_idx == 0:
            ax.set_title('Cell Load (UEs)', fontsize=12, fontweight='bold')
            ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        if row_idx < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=9)
        
        # Column 4: Summary Stats
        ax = plt.subplot(4, 4, row_idx*4 + 4)
        ax.axis('off')
        
        # Calculate stats
        avg_sinr_val = np.mean(avg_sinr)
        min_sinr_val = np.min(p5_sinr)
        avg_tput_val = np.mean(total_tput)
        max_load = max(k.max_cell_load for k in kpis)
        total_hos = kpis[-1].total_handovers
        
        # Determine tower failure if applicable
        num_towers_final = len(all_tower_ids)
        
        summary = f"""
SUMMARY STATISTICS

Network Performance:
  • Avg SINR:      {avg_sinr_val:>6.2f} dB
  • Min SINR (P5): {min_sinr_val:>6.2f} dB
  • Avg Tput:      {avg_tput_val:>6.2f} Gbps
  • Peak Load:     {max_load:>6} UEs/cell

Mobility:
  • Total HOs:     {total_hos:>6}
  • HO Rate:       {total_hos/duration:>6.2f} /s

Configuration:
  • Duration:      {duration:>6} s
  • Towers:        {num_towers_final:>6}
  • Final UEs:     {kpis[-1].num_ues:>6}
        """
        
        if row_idx == 0:
            ax.text(0.5, 1.0, 'Statistics', fontsize=12, fontweight='bold',
                   ha='center', va='top', transform=ax.transAxes)
        
        ax.text(0.5, 0.5, summary, fontsize=9, family='monospace',
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        print(f"✅ {name} complete")
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    # Save to results/plots directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'comprehensive_dashboard.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n{'='*70}")
    print("✅ Comprehensive dashboard saved: comprehensive_dashboard.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  5G NETWORK SIMULATOR — COMPREHENSIVE DASHBOARD")
    print("=" * 70)
    print("\n📊 Generating dashboard with all scenarios...")
    print("   This will take ~2 minutes to run all simulations.\n")
    
    generate_comprehensive_dashboard()
    
    print("\n" + "=" * 70)
    print("  ✅ DASHBOARD GENERATION COMPLETE!")
    print("=" * 70)
    print("\n📁 Output: comprehensive_dashboard.png")
    print("   • 4 scenarios × 4 metrics = 16 panels")
    print("   • Ready for presentations/portfolio")
    print("=" * 70)
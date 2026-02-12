"""
5G Network Simulator Dashboard

Interactive dashboard for running scenarios and visualizing results.
Uses matplotlib for offline compatibility (no web server needed).

Features:
  - Scenario selection (baseline, peak hour, failure, etc.)
  - Parameter tuning (num UEs, tower power, handover policy)
  - Real-time visualization (4-panel KPI dashboard)
  - Export results to PNG/CSV
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.propagation import PropagationEnvironment
from core.cell_tower import CellTower
from core.user_equipment import UserEquipment, GaussMarkovMobility
from simulation.network import Network
from simulation.scenarios import (
    scenario_baseline,
    scenario_peak_hour,
    scenario_tower_failure,
    scenario_traffic_surge
)


class Dashboard:
    """
    Interactive dashboard for 5G network simulation.
    
    Layout:
      - Left panel: Controls (scenario, parameters)
      - Right panel: 4-panel visualization (SINR, throughput, load, handovers)
    """
    
    def __init__(self):
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('5G Network Simulator Dashboard', fontsize=16, fontweight='bold')
        
        # Left panel for controls (20% width)
        self.ax_controls = plt.subplot2grid((10, 5), (0, 0), rowspan=10, colspan=1)
        self.ax_controls.axis('off')
        
        # Right panel for plots (80% width)
        self.ax_sinr = plt.subplot2grid((10, 5), (0, 1), rowspan=5, colspan=2)
        self.ax_tput = plt.subplot2grid((10, 5), (0, 3), rowspan=5, colspan=2)
        self.ax_load = plt.subplot2grid((10, 5), (5, 1), rowspan=5, colspan=2)
        self.ax_ho = plt.subplot2grid((10, 5), (5, 3), rowspan=5, colspan=2)
        
        # State
        self.current_scenario = 'baseline'
        self.num_ues = 30
        self.duration = 180
        self.network = None
        self.simulation_running = False
        
        # Initialize controls
        self._setup_controls()
        
        # Initial message
        self._show_welcome()
    
    def _setup_controls(self):
        """Create interactive controls."""
        # Title
        self.ax_controls.text(0.5, 0.98, '⚙️ CONTROLS', 
                            ha='center', va='top', fontsize=14, fontweight='bold',
                            transform=self.ax_controls.transAxes)
        
        # Scenario selector
        self.ax_controls.text(0.1, 0.90, 'Scenario:', fontsize=11, fontweight='bold',
                            transform=self.ax_controls.transAxes)
        
        ax_radio = plt.axes([0.05, 0.60, 0.18, 0.28], facecolor='lightgray')
        self.radio_scenario = RadioButtons(
            ax_radio,
            ('Baseline\n(20 UEs)', 
             'Peak Hour\n(100 UEs)', 
             'Tower Failure\n(at 120s)',
             'Traffic Surge\n(+40 at 60s)'),
            active=0
        )
        self.radio_scenario.on_clicked(self._on_scenario_change)
        
        # Run button
        ax_run = plt.axes([0.05, 0.50, 0.18, 0.06])
        self.btn_run = Button(ax_run, '▶️ RUN SIMULATION', color='lightgreen', hovercolor='green')
        self.btn_run.on_clicked(self._on_run_clicked)
        
        # Duration slider
        self.ax_controls.text(0.1, 0.45, 'Duration (s):', fontsize=10,
                            transform=self.ax_controls.transAxes)
        ax_duration = plt.axes([0.05, 0.40, 0.18, 0.03])
        self.slider_duration = Slider(ax_duration, '', 60, 300, valinit=180, valstep=30)
        self.slider_duration.on_changed(self._on_duration_change)
        
        # UE count slider
        self.ax_controls.text(0.1, 0.35, 'UE Count:', fontsize=10,
                            transform=self.ax_controls.transAxes)
        ax_ues = plt.axes([0.05, 0.30, 0.18, 0.03])
        self.slider_ues = Slider(ax_ues, '', 10, 150, valinit=30, valstep=10)
        self.slider_ues.on_changed(self._on_ues_change)
        
        # Export button
        ax_export = plt.axes([0.05, 0.20, 0.18, 0.06])
        self.btn_export = Button(ax_export, '💾 EXPORT RESULTS', color='lightblue', hovercolor='blue')
        self.btn_export.on_clicked(self._on_export_clicked)
        
        # Status text
        self.ax_controls.text(0.5, 0.10, '', fontsize=9, ha='center',
                            transform=self.ax_controls.transAxes,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.status_text = self.ax_controls.texts[-1]
        
        # Info panel
        info_text = """
📊 Dashboard Features:
  • 4 scenarios pre-configured
  • Adjustable parameters
  • Real-time KPI visualization
  • Export to PNG/CSV
  
💡 Quick Start:
  1. Select scenario
  2. Adjust parameters
  3. Click RUN
  4. View results
        """
        self.ax_controls.text(0.5, 0.05, info_text, fontsize=8, ha='center', va='bottom',
                            family='monospace',
                            transform=self.ax_controls.transAxes,
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    def _show_welcome(self):
        """Display welcome message."""
        for ax in [self.ax_sinr, self.ax_tput, self.ax_load, self.ax_ho]:
            ax.clear()
            ax.text(0.5, 0.5, '👋 Welcome!\n\nSelect a scenario and click RUN to start',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)
            ax.axis('off')
        plt.draw()
    
    def _on_scenario_change(self, label):
        """Handle scenario selection."""
        scenario_map = {
            'Baseline\n(20 UEs)': 'baseline',
            'Peak Hour\n(100 UEs)': 'peak_hour',
            'Tower Failure\n(at 120s)': 'tower_failure',
            'Traffic Surge\n(+40 at 60s)': 'traffic_surge'
        }
        self.current_scenario = scenario_map[label]
        self._update_status(f"Scenario: {self.current_scenario}")
    
    def _on_duration_change(self, val):
        """Handle duration slider."""
        self.duration = int(val)
    
    def _on_ues_change(self, val):
        """Handle UE count slider."""
        self.num_ues = int(val)
    
    def _on_run_clicked(self, event):
        """Run simulation when button clicked."""
        if self.simulation_running:
            self._update_status("⚠️ Simulation already running!")
            return
        
        self._update_status(f"🚀 Running {self.current_scenario}...")
        plt.pause(0.1)  # Update display
        
        # Run simulation
        self._run_simulation()
        
        self._update_status("✅ Simulation complete!")
    
    def _on_export_clicked(self, event):
        """Export results."""
        if self.network is None:
            self._update_status("⚠️ Run simulation first!")
            return
        
        # Export plot to results directory
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots'))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'dashboard_export_{self.current_scenario}.png')
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        
        # Export CSV
        import pandas as pd
        csv_filename = os.path.join(output_dir, f'dashboard_export_{self.current_scenario}.csv')
        df = pd.DataFrame([
            {
                'time': k.timestamp,
                'num_ues': k.num_ues,
                'avg_sinr': k.avg_sinr_db,
                'p5_sinr': k.p5_sinr_db,
                'total_tput': k.total_throughput_gbps,
                'handovers': k.handovers_this_tick
            }
            for k in self.network.kpi_history
        ])
        df.to_csv(csv_filename, index=False)
        
        self._update_status(f"💾 Exported to {filename.split('/')[-1]}")
    
    def _update_status(self, message):
        """Update status text."""
        self.status_text.set_text(message)
        plt.draw()
    
    def _run_simulation(self):
        """Execute the simulation."""
        self.simulation_running = True
        
        # Create network based on scenario
        if self.current_scenario == 'baseline':
            self.network = scenario_baseline(num_ues=self.num_ues)
        elif self.current_scenario == 'peak_hour':
            self.network = scenario_peak_hour(num_ues=self.num_ues)
        elif self.current_scenario == 'tower_failure':
            self.network = scenario_tower_failure(num_ues=self.num_ues, failure_time=self.duration/2)
        elif self.current_scenario == 'traffic_surge':
            self.network = scenario_traffic_surge(baseline_ues=self.num_ues//2, 
                                                  surge_time=self.duration/3,
                                                  surge_ues=self.num_ues//2)
        
        # Run simulation (silent)
        self.network.run(duration_s=self.duration, dt=1.0, verbose=False)
        
        # Update visualizations
        self._update_plots()
        
        self.simulation_running = False
    
    def _update_plots(self):
        """Update all plots with simulation results."""
        kpis = self.network.kpi_history
        times = [k.timestamp for k in kpis]
        
        # Clear all axes
        for ax in [self.ax_sinr, self.ax_tput, self.ax_load, self.ax_ho]:
            ax.clear()
            ax.axis('on')
        
        # Plot 1: SINR
        avg_sinr = [k.avg_sinr_db for k in kpis]
        p5_sinr = [k.p5_sinr_db for k in kpis]
        
        self.ax_sinr.plot(times, avg_sinr, 'b-', linewidth=2, label='Average')
        self.ax_sinr.plot(times, p5_sinr, 'r--', linewidth=2, label='5th Percentile')
        self.ax_sinr.axhline(y=10, color='green', linestyle=':', alpha=0.5, label='Good threshold')
        self.ax_sinr.fill_between(times, p5_sinr, alpha=0.2, color='red')
        
        self.ax_sinr.set_ylabel('SINR (dB)', fontsize=10)
        self.ax_sinr.set_title('Signal Quality', fontsize=11, fontweight='bold')
        self.ax_sinr.legend(fontsize=8)
        self.ax_sinr.grid(True, alpha=0.3)
        
        # Plot 2: Throughput
        total_tput = [k.total_throughput_gbps for k in kpis]
        
        self.ax_tput.plot(times, total_tput, 'g-', linewidth=2)
        self.ax_tput.fill_between(times, total_tput, alpha=0.3, color='green')
        
        avg_tput = np.mean(total_tput)
        self.ax_tput.axhline(y=avg_tput, color='darkgreen', linestyle='--', alpha=0.7)
        self.ax_tput.text(times[-1]*0.7, avg_tput*1.1, f'Avg: {avg_tput:.1f} Gbps',
                         fontsize=9, color='darkgreen', fontweight='bold')
        
        self.ax_tput.set_ylabel('Total Throughput (Gbps)', fontsize=10)
        self.ax_tput.set_title('Network Capacity', fontsize=11, fontweight='bold')
        self.ax_tput.grid(True, alpha=0.3)
        
        # Plot 3: Cell Load (stacked area)
        all_tower_ids = sorted(set(
            tid for k in kpis for tid in k.tower_loads.keys()
        ))
        
        load_matrix = np.zeros((len(times), len(all_tower_ids)))
        for i, k in enumerate(kpis):
            for j, tid in enumerate(all_tower_ids):
                load_matrix[i, j] = k.tower_loads.get(tid, 0)
        
        self.ax_load.stackplot(times, load_matrix.T, labels=all_tower_ids, alpha=0.7)
        self.ax_load.set_xlabel('Time (s)', fontsize=10)
        self.ax_load.set_ylabel('UEs Connected', fontsize=10)
        self.ax_load.set_title('Cell Load Distribution', fontsize=11, fontweight='bold')
        self.ax_load.legend(fontsize=8, loc='upper left')
        self.ax_load.grid(True, alpha=0.3)
        
        # Plot 4: Handovers
        ho_rate = [k.handovers_this_tick for k in kpis]
        
        self.ax_ho.bar(times, ho_rate, width=1.0, color='orange', alpha=0.7)
        
        total_hos = sum(ho_rate)
        self.ax_ho.text(0.95, 0.95, f'Total: {total_hos} HOs\nRate: {total_hos/self.duration:.2f} HO/s',
                       transform=self.ax_ho.transAxes, fontsize=9,
                       ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.ax_ho.set_xlabel('Time (s)', fontsize=10)
        self.ax_ho.set_ylabel('Handovers', fontsize=10)
        self.ax_ho.set_title('Handover Activity', fontsize=11, fontweight='bold')
        self.ax_ho.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.draw()
    
    def run(self):
        """Display dashboard."""
        plt.show()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  5G NETWORK SIMULATOR DASHBOARD")
    print("=" * 70)
    print("\n📊 Launching interactive dashboard...")
    print("\n💡 Instructions:")
    print("   1. Select a scenario from the left panel")
    print("   2. Adjust duration and UE count sliders")
    print("   3. Click 'RUN SIMULATION' button")
    print("   4. View real-time results in 4-panel display")
    print("   5. Click 'EXPORT RESULTS' to save PNG + CSV")
    print("\n⚠️  Note: Dashboard runs in matplotlib (offline mode)")
    print("=" * 70)
    
    dashboard = Dashboard()
    dashboard.run()
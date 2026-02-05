"""
Network Simulation Engine

============================================================
CONCEPT: What is a Network Simulation?
============================================================
A network simulator is a discrete-event system that:
  1. Maintains STATE (towers, UEs, connections)
  2. Advances TIME in fixed steps (ticks)
  3. Computes METRICS (throughput, handovers, load per cell)
  4. Allows SCENARIOS (inject events like failures, traffic surges)

Why we need this:
  - Test "what-if" scenarios WITHOUT deploying real hardware
  - Validate optimization algorithms before production
  - Train ML models on synthetic but realistic data

This is the core of a DIGITAL TWIN.

============================================================
CONCEPT: Discrete-Event Simulation
============================================================
Real networks are CONTINUOUS — radio signals, movement, all
happen smoothly. But simulating continuous systems is slow.

Instead we use DISCRETE-EVENT SIMULATION:
  - Time advances in fixed steps (e.g., 1 second ticks)
  - All state updates happen atomically per tick
  - Between ticks, the world is "frozen"

Trade-off:
  ✅ Fast, deterministic, easy to debug
  ❌ Misses sub-tick events (e.g., 10ms bursts)
  
For network planning (seconds to minutes scale), this is perfect.

============================================================
CONCEPT: Network-Wide KPIs
============================================================
Per-UE metrics (SINR, throughput) are useful, but operators
care about AGGREGATE metrics:

  • Average SINR across all UEs
  • Total network throughput (Gbps)
  • Load per cell (how many UEs connected)
  • Handover success rate
  • Coverage holes (% of area with poor signal)
  • 5th percentile throughput (worst 5% of users)

These KPIs drive network investment decisions.

============================================================
CONCEPT: Scenarios — The Real Power
============================================================
A digital twin isn't useful if it just runs one configuration.
The value is in testing SCENARIOS:

  ① Peak Hour:        Spike UE count 3×, cluster in hot spots
  ② Tower Failure:    One tower goes offline → load redistributes
  ③ Weather:          Rain/fog increases path loss
  ④ Event:            Stadium game → 50K users in 1 km radius
  ⑤ New Deployment:   Add a tower → measure coverage improvement

We'll build a scenario API that makes these easy to set up.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import sys
import os
# Ensure the project `src/` directory is on sys.path when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.propagation import PropagationEnvironment
from core.cell_tower import CellTower
from core.user_equipment import UserEquipment, GaussMarkovMobility


# ===========================================================================
# Network-Wide KPIs
# ===========================================================================
@dataclass
class NetworkKPIs:
    """
    Snapshot of network performance at one instant.
    
    These are the metrics operators track on dashboards.
    """
    timestamp: float                # Simulation time (seconds)
    
    # UE metrics
    num_ues: int = 0
    avg_sinr_db: float = 0.0
    p5_sinr_db: float = 0.0         # 5th percentile (worst 5%)
    avg_throughput_mbps: float = 0.0
    total_throughput_gbps: float = 0.0
    
    # Cell metrics
    tower_loads: Dict[str, int] = field(default_factory=dict)  # {tower_id: ue_count}
    max_cell_load: int = 0
    min_cell_load: int = 0
    
    # Handover metrics
    handovers_this_tick: int = 0
    total_handovers: int = 0
    
    # Coverage metrics
    ues_below_fair_sinr: int = 0    # SINR < 0 dB


# ===========================================================================
# Network — The Main Simulation Engine
# ===========================================================================
class Network:
    """
    Complete RAN simulation with multiple towers and UEs.
    
    Responsibilities:
      - Initialize towers and UEs
      - Run time-stepped simulation loop
      - Aggregate KPIs each tick
      - Execute scenario events (failures, load changes)
      - Export results for analysis
      
    Usage:
        net = Network(environment=env)
        net.add_tower(CellTower(...))
        net.add_ue(UserEquipment(...))
        net.run(duration_s=300, dt=1.0)
        net.plot_kpis()
    """
    
    def __init__(self, environment: PropagationEnvironment):
        self.environment = environment
        
        # Network components
        self.towers: List[CellTower] = []
        self.ues: List[UserEquipment] = []
        
        # Simulation state
        self.time: float = 0.0          # Current simulation time
        self.tick_count: int = 0
        
        # KPI history (one entry per tick)
        self.kpi_history: List[NetworkKPIs] = []
        
        # Event queue (for scenario injection)
        self.events: List[tuple] = []   # [(time, callable), ...]
    
    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    def add_tower(self, tower: CellTower) -> None:
        """Add a cell tower to the network."""
        self.towers.append(tower)
    
    def add_ue(self, ue: UserEquipment) -> None:
        """Add a user equipment to the network."""
        self.ues.append(ue)
    
    def schedule_event(self, time_s: float, callback: Callable) -> None:
        """
        Schedule a scenario event to fire at a specific time.
        
        Example:
            net.schedule_event(120, lambda: net.fail_tower("Tower_B"))
        """
        self.events.append((time_s, callback))
        self.events.sort(key=lambda x: x[0])  # Keep sorted by time
    
    # -----------------------------------------------------------------------
    # Scenario Actions
    # -----------------------------------------------------------------------
    def fail_tower(self, tower_id: str) -> None:
        """
        Simulate a tower failure.
        
        All UEs connected to this tower will lose signal and must
        immediately reconnect to the next-best neighbor.
        """
        print(f"[{self.time:.1f}s] ⚠️  Tower {tower_id} FAILED")
        
        for tower in self.towers:
            if tower.tower_id == tower_id:
                # Disconnect all UEs
                for ue_id in list(tower.connected_ue_ids):
                    for ue in self.ues:
                        if ue.ue_id == ue_id:
                            tower.disconnect_ue(ue_id)
                            ue.serving_tower = None  # Force reconnect next tick
                            break
                
                # Mark tower as offline (we'll just remove it from tower list)
                self.towers.remove(tower)
                print(f"         {len(tower.connected_ue_ids)} UEs disconnected")
                break
    
    def restore_tower(self, tower: CellTower) -> None:
        """Bring a failed tower back online."""
        self.towers.append(tower)
        print(f"[{self.time:.1f}s] ✅ Tower {tower.tower_id} RESTORED")
    
    def inject_traffic_surge(self, center_x: float, center_y: float,
                             radius: float, num_new_ues: int) -> None:
        """
        Inject a burst of new UEs in a specific area.
        
        Use case: Stadium event, concert, rush hour at a train station.
        """
        print(f"[{self.time:.1f}s] 📈 Traffic surge: +{num_new_ues} UEs near ({center_x}, {center_y})")
        
        for i in range(num_new_ues):
            # Random position within radius
            angle = np.random.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(np.random.uniform(0, 1))
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            
            # Stationary or slow-moving
            mobility = GaussMarkovMobility(
                alpha=0.9,
                mean_speed=1.0,
                speed_std=0.5,
                mean_direction_deg=np.random.uniform(0, 360)
            )
            
            ue = UserEquipment(
                ue_id=f"UE_surge_{self.tick_count}_{i}",
                x=x, y=y,
                mobility=mobility,
                serving_tower=None  # Auto-connects on first tick
            )
            self.add_ue(ue)
    
    # -----------------------------------------------------------------------
    # Simulation Loop
    # -----------------------------------------------------------------------
    def tick(self, dt: float) -> NetworkKPIs:
        """
        Execute one simulation time step.
        
        Order of operations:
          1. Process any scheduled events for this time
          2. Tick all UEs (move, measure, handover)
          3. Aggregate network-wide KPIs
          4. Increment time
        
        Returns:
            NetworkKPIs for this tick
        """
        # ── 1. Process events ──────────────────────────────────────
        while self.events and self.events[0][0] <= self.time:
            event_time, callback = self.events.pop(0)
            callback()
        
        # ── 2. Tick all UEs ────────────────────────────────────────
        ue_results = []
        for ue in self.ues:
            result = ue.tick(self.towers, dt=dt)
            ue_results.append(result)
        
        # ── 3. Aggregate KPIs ──────────────────────────────────────
        kpis = self._compute_kpis(ue_results)
        self.kpi_history.append(kpis)
        
        # ── 4. Increment time ──────────────────────────────────────
        self.time += dt
        self.tick_count += 1
        
        return kpis
    
    def run(self, duration_s: float, dt: float = 1.0,
            verbose: bool = True, log_interval: int = 30) -> None:
        """
        Run the simulation for a specified duration.
        
        Args:
            duration_s:   Total simulation time (seconds)
            dt:           Time step (seconds)
            verbose:      Print progress updates
            log_interval: Print every N seconds
        """
        num_ticks = int(duration_s / dt)
        
        if verbose:
            print("=" * 70)
            print(f"  NETWORK SIMULATION")
            print("=" * 70)
            print(f"  Duration:  {duration_s} s")
            print(f"  Time step: {dt} s")
            print(f"  Towers:    {len(self.towers)}")
            print(f"  UEs:       {len(self.ues)}")
            print("=" * 70)
        
        for i in range(num_ticks):
            kpis = self.tick(dt)
            
            if verbose and (i % log_interval == 0 or i == num_ticks - 1):
                print(f"[{self.time:>6.1f}s] "
                      f"UEs: {kpis.num_ues:>3} | "
                      f"Avg SINR: {kpis.avg_sinr_db:>5.1f} dB | "
                      f"Tput: {kpis.total_throughput_gbps:>5.2f} Gbps | "
                      f"HOs: {kpis.handovers_this_tick:>2}")
        
        if verbose:
            print("=" * 70)
            print("  Simulation complete!")
            self._print_summary()
    
    # -----------------------------------------------------------------------
    # KPI Computation
    # -----------------------------------------------------------------------
    def _compute_kpis(self, ue_results: List[dict]) -> NetworkKPIs:
        """
        Aggregate network-wide KPIs from per-UE results.
        """
        kpis = NetworkKPIs(timestamp=self.time)
        
        if not ue_results:
            return kpis
        
        # ── UE metrics ─────────────────────────────────────────────
        sinrs = [r['sinr_db'] for r in ue_results if r['sinr_db'] > -999]
        tputs = [r['throughput_mbps'] for r in ue_results]
        
        kpis.num_ues = len(ue_results)
        kpis.avg_sinr_db = np.mean(sinrs) if sinrs else 0.0
        kpis.p5_sinr_db  = np.percentile(sinrs, 5) if sinrs else 0.0
        kpis.avg_throughput_mbps = np.mean(tputs)
        kpis.total_throughput_gbps = sum(tputs) / 1000.0
        
        kpis.ues_below_fair_sinr = sum(1 for s in sinrs if s < 0)
        
        # ── Cell load ──────────────────────────────────────────────
        for tower in self.towers:
            kpis.tower_loads[tower.tower_id] = tower.stats.connected_ues
        
        if kpis.tower_loads:
            kpis.max_cell_load = max(kpis.tower_loads.values())
            kpis.min_cell_load = min(kpis.tower_loads.values())
        
        # ── Handovers ──────────────────────────────────────────────
        hos_this_tick = sum(
            1 for ue in self.ues
            if ue.handover_log and ue.handover_log[-1][0] == self.tick_count
        )
        kpis.handovers_this_tick = hos_this_tick
        kpis.total_handovers = sum(len(ue.handover_log) for ue in self.ues)
        
        return kpis
    
    def _print_summary(self) -> None:
        """Print final simulation statistics."""
        print("\n📊 SIMULATION SUMMARY")
        print("-" * 70)
        
        if not self.kpi_history:
            print("  No data collected.")
            return
        
        # Aggregate across entire run
        avg_sinr_overall = np.mean([k.avg_sinr_db for k in self.kpi_history])
        avg_tput_overall = np.mean([k.avg_throughput_mbps for k in self.kpi_history])
        total_hos = self.kpi_history[-1].total_handovers
        
        print(f"  Average SINR:       {avg_sinr_overall:.2f} dB")
        print(f"  Average Throughput: {avg_tput_overall:.1f} Mbps per UE")
        print(f"  Total Handovers:    {total_hos}")
        print(f"  Handover rate:      {total_hos / self.time:.2f} HOs/second")
        
        # Per-tower stats
        print("\n  Per-Tower Load (final):")
        final_kpis = self.kpi_history[-1]
        for tid, count in sorted(final_kpis.tower_loads.items()):
            bar = "█" * count
            print(f"    {tid:<12} {count:>3} UEs  {bar}")
        
        print("-" * 70)
    
    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------
    def plot_kpis(self, filename: str = "network_kpis.png") -> None:
        """
        Generate time-series plots of network KPIs.
        
        4-panel figure:
          - SINR over time (avg + p5)
          - Throughput over time
          - Cell load over time (stacked area)
          - Handover rate over time
        """
        import matplotlib.pyplot as plt
        
        if not self.kpi_history:
            print("No KPI data to plot.")
            return
        
        times = [k.timestamp for k in self.kpi_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Network Performance — Time Series KPIs", fontsize=14, fontweight='bold')
        
        # ── Top-Left: SINR ─────────────────────────────────────────
        ax = axes[0, 0]
        avg_sinr = [k.avg_sinr_db for k in self.kpi_history]
        p5_sinr  = [k.p5_sinr_db for k in self.kpi_history]
        
        ax.plot(times, avg_sinr, label="Average SINR", linewidth=2, color='#3498db')
        ax.plot(times, p5_sinr, label="5th Percentile", linewidth=2, color='#e74c3c', linestyle='--')
        ax.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        ax.fill_between(times, p5_sinr, alpha=0.2, color='#e74c3c')
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SINR (dB)")
        ax.set_title("SINR Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ── Top-Right: Throughput ──────────────────────────────────
        ax = axes[0, 1]
        total_tput = [k.total_throughput_gbps for k in self.kpi_history]
        
        ax.plot(times, total_tput, linewidth=2, color='#2ecc71')
        ax.fill_between(times, total_tput, alpha=0.2, color='#2ecc71')
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Total Throughput (Gbps)")
        ax.set_title("Network Capacity")
        ax.grid(True, alpha=0.3)
        
        # ── Bottom-Left: Cell Load ─────────────────────────────────
        ax = axes[1, 0]
        
        # Extract all unique tower IDs
        all_tower_ids = sorted(set(
            tid for k in self.kpi_history for tid in k.tower_loads.keys()
        ))
        
        # Build matrix: time × tower
        load_matrix = np.zeros((len(times), len(all_tower_ids)))
        for i, k in enumerate(self.kpi_history):
            for j, tid in enumerate(all_tower_ids):
                load_matrix[i, j] = k.tower_loads.get(tid, 0)
        
        # Stacked area plot
        ax.stackplot(times, load_matrix.T, labels=all_tower_ids, alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("UEs Connected")
        ax.set_title("Cell Load Distribution")
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # ── Bottom-Right: Handover Rate ────────────────────────────
        ax = axes[1, 1]
        ho_rate = [k.handovers_this_tick for k in self.kpi_history]
        
        ax.plot(times, ho_rate, linewidth=1.5, color='#f39c12', marker='o', markersize=3)
        ax.fill_between(times, ho_rate, alpha=0.2, color='#f39c12')
        
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Handovers per Tick")
        ax.set_title("Handover Activity")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        # Save to project results/plots directory
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots'))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")


# ===========================================================================
# Demo: Multi-UE simulation with a tower failure event
# ===========================================================================
if __name__ == "__main__":
    from core.propagation import PropagationEnvironment
    
    print("\n" + "=" * 70)
    print("  NETWORK SIMULATION ENGINE — DEMO")
    print("=" * 70)
    
    # Environment
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    # Create network
    net = Network(environment=env)
    
    # Add 3 towers
    net.add_tower(CellTower.create_standard_3sector("Tower_A", -500, 0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_B", 500, 0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_C", 0, 800, env))
    
    # Add 20 UEs scattered randomly
    for i in range(20):
        x = np.random.uniform(-600, 600)
        y = np.random.uniform(-200, 900)
        
        mobility = GaussMarkovMobility(
            alpha=0.75,
            mean_speed=np.random.uniform(2, 6),
            speed_std=1.0,
            mean_direction_deg=np.random.uniform(0, 360),
            bounds=(-700, 700, -300, 1000)
        )
        
        ue = UserEquipment(f"UE_{i:02d}", x, y, mobility, serving_tower=None)
        net.add_ue(ue)
    
    # Schedule a tower failure at t=120s
    net.schedule_event(120, lambda: net.fail_tower("Tower_B"))
    
    # Run simulation for 5 minutes
    net.run(duration_s=300, dt=1.0, log_interval=30)
    
    # Generate KPI plots
    net.plot_kpis("demo_network_kpis.png")
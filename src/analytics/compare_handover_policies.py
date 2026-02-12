"""
Final Comparison: Classical vs AI-Powered Handover

Compares two handover policies:
  1. Classical: 3GPP A3 event (RSRP threshold)
  2. AI-Powered: RL-trained Q-learning agent

Metrics compared:
  - Average SINR
  - Handover count
  - Ping-pong ratio
  - Throughput stability
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Ensure the project `src/` directory is on sys.path when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.propagation import PropagationEnvironment
from core.cell_tower import CellTower
from core.user_equipment import UserEquipment, GaussMarkovMobility
from core.network import Network
from analytics.rl_handover import RLHandoverAgent, extract_state


def run_classical_handover(duration_s: int = 180) -> dict:
    """
    Run simulation with classical A3 handover.
    
    This is the baseline — standard 3GPP behavior.
    """
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    net = Network(environment=env)
    
    # 3 towers
    net.add_tower(CellTower.create_standard_3sector("Tower_A", -500, 0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_B", 500, 0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_C", 0, 800, env))
    
    # 20 UEs with various mobility patterns
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
    
    # Run simulation
    print("\n🔵 Running CLASSICAL handover simulation...")
    net.run(duration_s=duration_s, dt=1.0, verbose=False)
    
    # Collect stats
    total_handovers = sum(len(ue.handover_log) for ue in net.ues)
    avg_sinr = np.mean([k.avg_sinr_db for k in net.kpi_history])
    avg_tput = np.mean([k.avg_throughput_mbps for k in net.kpi_history])
    
    # Count ping-pongs (immediate reversals)
    ping_pongs = 0
    for ue in net.ues:
        for i in range(1, len(ue.handover_log)):
            _, from_prev, to_prev = ue.handover_log[i-1]
            _, from_curr, to_curr = ue.handover_log[i]
            if to_prev == from_curr and from_prev == to_curr:
                ping_pongs += 1
    
    return {
        'type': 'Classical A3',
        'total_handovers': total_handovers,
        'ping_pongs': ping_pongs,
        'avg_sinr': avg_sinr,
        'avg_tput': avg_tput,
        'sinr_history': [k.avg_sinr_db for k in net.kpi_history],
        'ho_per_ue': total_handovers / len(net.ues)
    }


def run_rl_handover(agent: RLHandoverAgent, duration_s: int = 180) -> dict:
    """
    Run simulation with RL-based handover.
    
    Uses trained Q-learning agent to make handover decisions.
    """
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    net = Network(environment=env)
    
    # Same setup as classical
    net.add_tower(CellTower.create_standard_3sector("Tower_A", -500, 0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_B", 500, 0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_C", 0, 800, env))
    
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
    
    # Run with RL decisions (override classical handover)
    print("\n🟢 Running RL-BASED handover simulation...")
    
    sinr_history = []
    for tick in range(duration_s):
        # Normal tick (but we'll override handover decisions)
        for ue in net.ues:
            ue.move(1.0)
            
            # Populate cache
            ue._tick_cache = ue._measure_all_cached(net.towers)
            
            # Auto-connect if needed
            if ue.serving_tower is None:
                best_id = max(ue._tick_cache, key=ue._tick_cache.get)
                for t in net.towers:
                    if t.tower_id == best_id:
                        ue.serving_tower = t
                        t.connect_ue(ue.ue_id)
                        break
            
            # RL decision (instead of A3 event)
            state = extract_state(ue, net.towers)
            action = agent.choose_action(state, training=False)  # Exploit only
            
            if action == 1:  # RL says handover
                # Find best neighbor
                best_neighbor = None
                best_rsrp = -999
                for t in net.towers:
                    if t.tower_id != ue.serving_tower.tower_id:
                        rsrp = ue._tick_cache.get(t.tower_id, -999)
                        if rsrp > best_rsrp:
                            best_rsrp = rsrp
                            best_neighbor = t
                
                if best_neighbor:
                    ue.serving_tower.disconnect_ue(ue.ue_id)
                    best_neighbor.connect_ue(ue.ue_id)
                    ue.serving_tower = best_neighbor
                    ue.handover_log.append((tick, ue.serving_tower.tower_id, best_neighbor.tower_id))
            
            # Compute SINR and log
            sinr = ue.calculate_sinr(net.towers)
            ue.sinr_history.append(sinr)
        
        # Aggregate network SINR
        avg_sinr = np.mean([ue.sinr_history[-1] for ue in net.ues if ue.sinr_history])
        sinr_history.append(avg_sinr)
        
        net.time += 1.0
        net.tick_count += 1
    
    # Collect stats
    total_handovers = sum(len(ue.handover_log) for ue in net.ues)
    avg_sinr = np.mean(sinr_history)
    avg_tput = np.mean([ue.sinr_to_throughput_mbps(s) for ue in net.ues 
                        for s in ue.sinr_history])
    
    ping_pongs = 0
    for ue in net.ues:
        for i in range(1, len(ue.handover_log)):
            _, from_prev, to_prev = ue.handover_log[i-1]
            _, from_curr, to_curr = ue.handover_log[i]
            if to_prev == from_curr and from_prev == to_curr:
                ping_pongs += 1
    
    return {
        'type': 'RL-Based',
        'total_handovers': total_handovers,
        'ping_pongs': ping_pongs,
        'avg_sinr': avg_sinr,
        'avg_tput': avg_tput,
        'sinr_history': sinr_history,
        'ho_per_ue': total_handovers / 20
    }


def plot_comparison(classical: dict, rl: dict):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Classical vs RL-Based Handover Comparison", fontsize=14, fontweight='bold')
    
    # Panel 1: SINR over time
    ax = axes[0, 0]
    ax.plot(classical['sinr_history'], label='Classical A3', linewidth=2, alpha=0.8)
    ax.plot(rl['sinr_history'], label='RL-Based', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average SINR (dB)')
    ax.set_title('SINR Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Metrics comparison (bar chart)
    ax = axes[0, 1]
    metrics = ['Avg SINR\n(dB)', 'Avg Tput\n(Mbps)', 'HO per UE', 'Ping-Pongs']
    classical_vals = [classical['avg_sinr'], classical['avg_tput']/10, 
                     classical['ho_per_ue'], classical['ping_pongs']]
    rl_vals = [rl['avg_sinr'], rl['avg_tput']/10, 
              rl['ho_per_ue'], rl['ping_pongs']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, classical_vals, width, label='Classical', alpha=0.8)
    ax.bar(x + width/2, rl_vals, width, label='RL-Based', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_title('Performance Metrics')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Handover count comparison
    ax = axes[1, 0]
    categories = ['Total\nHandovers', 'Ping-Pongs']
    classical_hos = [classical['total_handovers'], classical['ping_pongs']]
    rl_hos = [rl['total_handovers'], rl['ping_pongs']]
    
    x = np.arange(len(categories))
    ax.bar(x - width/2, classical_hos, width, label='Classical', alpha=0.8, color='#3498db')
    ax.bar(x + width/2, rl_hos, width, label='RL-Based', alpha=0.8, color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Count')
    ax.set_title('Handover Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
COMPARISON SUMMARY

Classical A3:
  • Avg SINR: {classical['avg_sinr']:.2f} dB
  • Handovers: {classical['total_handovers']}
  • Ping-pongs: {classical['ping_pongs']}
  • HO per UE: {classical['ho_per_ue']:.1f}

RL-Based:
  • Avg SINR: {rl['avg_sinr']:.2f} dB
  • Handovers: {rl['total_handovers']}
  • Ping-pongs: {rl['ping_pongs']}
  • HO per UE: {rl['ho_per_ue']:.1f}

Improvement:
  • SINR: {((rl['avg_sinr'] - classical['avg_sinr'])/classical['avg_sinr']*100):+.1f}%
  • HO reduction: {((classical['total_handovers'] - rl['total_handovers'])/classical['total_handovers']*100):+.1f}%
  • Ping-pong reduction: {((classical['ping_pongs'] - rl['ping_pongs'])/max(classical['ping_pongs'], 1)*100):+.1f}%
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save to results/plots directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'classical_vs_rl_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print("\n✅ Comparison plot saved: classical_vs_rl_comparison.png")


if __name__ == "__main__":
    print("=" * 70)
    print("  FINAL COMPARISON: CLASSICAL vs RL HANDOVER")
    print("=" * 70)
    
    # Run classical
    classical_results = run_classical_handover(duration_s=180)
    
    # Load trained RL agent
    print("\n📦 Loading trained RL agent...")
    try:
        agent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'rl_handover_agent.pkl'))
        agent = RLHandoverAgent.load(agent_path)
        agent.epsilon = 0.0  # Pure exploitation
    except FileNotFoundError:
        print("   ⚠️  Trained agent not found. Using untrained agent...")
        agent = RLHandoverAgent(epsilon=0.0)
    
    # Run RL-based
    rl_results = run_rl_handover(agent, duration_s=180)
    
    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Classical':<15} {'RL-Based':<15} {'Difference'}")
    print("-" * 70)
    print(f"{'Avg SINR (dB)':<25} {classical_results['avg_sinr']:<15.2f} {rl_results['avg_sinr']:<15.2f} "
          f"{rl_results['avg_sinr'] - classical_results['avg_sinr']:+.2f}")
    print(f"{'Total Handovers':<25} {classical_results['total_handovers']:<15} {rl_results['total_handovers']:<15} "
          f"{rl_results['total_handovers'] - classical_results['total_handovers']:+}")
    print(f"{'Ping-pongs':<25} {classical_results['ping_pongs']:<15} {rl_results['ping_pongs']:<15} "
          f"{rl_results['ping_pongs'] - classical_results['ping_pongs']:+}")
    print(f"{'HO per UE':<25} {classical_results['ho_per_ue']:<15.2f} {rl_results['ho_per_ue']:<15.2f} "
          f"{rl_results['ho_per_ue'] - classical_results['ho_per_ue']:+.2f}")
    print("-" * 70)
    
    # Generate plots
    plot_comparison(classical_results, rl_results)
    
    print("\n" + "=" * 70)
    print("  ✅ COMPARISON COMPLETE!")
    print("=" * 70)
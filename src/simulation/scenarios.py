"""
Scenario Library for Network Simulation

Pre-configured test scenarios for different network conditions.
Each scenario is a factory function that returns a configured Network.

============================================================
WHY SCENARIOS MATTER
============================================================
A digital twin is only useful if you can ask "what if?" questions:

  • "What if we lose Tower B during rush hour?"
  • "What if there's a concert at (0, 500)?"
  • "What if we add a 4th tower at (-200, 400)?"

Scenarios let you:
  ① Test before deploying (validate network plans)
  ② Train ML models on edge cases
  ③ Generate synthetic data for analysis tools

"""

import numpy as np
import sys
import os
# Ensure the project `src/` directory is on sys.path when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.propagation import PropagationEnvironment
from core.cell_tower import CellTower
from core.user_equipment import UserEquipment, GaussMarkovMobility
from core.network import Network


# ===========================================================================
# Helper: Generate Random UEs
# ===========================================================================
def generate_random_ues(count: int,
                       x_range: tuple = (-600, 600),
                       y_range: tuple = (-300, 1000),
                       speed_range: tuple = (2, 6),
                       bounds: tuple = (-700, 700, -300, 1000)) -> list:
    """
    Generate N UEs with random positions and mobility.
    
    Args:
        count: Number of UEs to create
        x_range, y_range: Initial position bounds
        speed_range: Min/max speed in m/s
        bounds: Reflection boundaries
        
    Returns:
        List of UserEquipment instances
    """
    ues = []
    for i in range(count):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        
        mobility = GaussMarkovMobility(
            alpha=0.75,
            mean_speed=np.random.uniform(*speed_range),
            speed_std=1.0,
            mean_direction_deg=np.random.uniform(0, 360),
            bounds=bounds
        )
        
        ue = UserEquipment(f"UE_{i:03d}", x, y, mobility, serving_tower=None)
        ues.append(ue)
    
    return ues


def generate_clustered_ues(count: int,
                           center_x: float,
                           center_y: float,
                           radius: float,
                           speed: float = 1.0) -> list:
    """
    Generate UEs clustered around a hotspot (e.g., stadium, mall).
    
    Args:
        count: Number of UEs
        center_x, center_y: Cluster center
        radius: Cluster radius in metres
        speed: Mean UE speed (typically slow in crowds)
    """
    ues = []
    for i in range(count):
        # Uniform distribution in circle
        angle = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)
        
        mobility = GaussMarkovMobility(
            alpha=0.9,  # Low mobility (crowded)
            mean_speed=speed,
            speed_std=0.5,
            mean_direction_deg=np.random.uniform(0, 360)
        )
        
        ue = UserEquipment(f"UE_cluster_{i:03d}", x, y, mobility, serving_tower=None)
        ues.append(ue)
    
    return ues


# ===========================================================================
# SCENARIO 1: Baseline (Light Load)
# ===========================================================================
def scenario_baseline(num_ues: int = 20) -> Network:
    """
    Baseline: 3 towers, light UE load, normal operation.
    
    Use case:
      - Validate simulator correctness
      - Establish baseline KPIs for comparison
      - Debug before running complex scenarios
    """
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    net = Network(environment=env)
    
    # Standard 3-tower triangle
    net.add_tower(CellTower.create_standard_3sector("Tower_A", -500,  0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_B",  500,  0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_C",    0, 800, env))
    
    # Add UEs
    for ue in generate_random_ues(num_ues):
        net.add_ue(ue)
    
    return net


# ===========================================================================
# SCENARIO 2: Peak Hour (High Load)
# ===========================================================================
def scenario_peak_hour(num_ues: int = 100) -> Network:
    """
    Peak hour: 3 towers, 100+ UEs, some clustered in hot spots.
    
    Use case:
      - Test network under high load
      - Identify bottlenecks (overloaded cells)
      - Validate capacity planning
      
    Expected behavior:
      - Some cells hit max capacity
      - SINR degrades (more interference)
      - Handover rate increases
    """
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    net = Network(environment=env)
    
    # Towers
    net.add_tower(CellTower.create_standard_3sector("Tower_A", -500,  0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_B",  500,  0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_C",    0, 800, env))
    
    # 70% random distribution
    for ue in generate_random_ues(int(num_ues * 0.7)):
        net.add_ue(ue)
    
    # 30% clustered near Tower A (downtown office area)
    for ue in generate_clustered_ues(int(num_ues * 0.3), -400, 100, 200, speed=2.0):
        net.add_ue(ue)
    
    return net


# ===========================================================================
# SCENARIO 3: Tower Failure
# ===========================================================================
def scenario_tower_failure(num_ues: int = 50, failure_time: float = 120.0) -> Network:
    """
    Tower failure: Tower_B goes offline mid-simulation.
    
    Use case:
      - Test network resilience
      - Validate automatic failover
      - Measure impact on KPIs (coverage holes, dropped calls)
      
    Expected behavior:
      - UEs connected to Tower_B immediately lose signal
      - They handover to Tower_A or Tower_C
      - Load on remaining towers spikes
      - Some UEs may fall below SINR threshold (coverage gap)
    """
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    net = Network(environment=env)
    
    # Towers
    net.add_tower(CellTower.create_standard_3sector("Tower_A", -500,  0, env))
    tower_b = CellTower.create_standard_3sector("Tower_B",  500,  0, env)
    net.add_tower(tower_b)
    net.add_tower(CellTower.create_standard_3sector("Tower_C",    0, 800, env))
    
    # UEs evenly distributed
    for ue in generate_random_ues(num_ues):
        net.add_ue(ue)
    
    # Schedule failure
    net.schedule_event(failure_time, lambda: net.fail_tower("Tower_B"))
    
    # Optional: restore after 60s
    # net.schedule_event(failure_time + 60, lambda: net.restore_tower(tower_b))
    
    return net


# ===========================================================================
# SCENARIO 4: Traffic Surge (Event)
# ===========================================================================
def scenario_traffic_surge(baseline_ues: int = 30,
                           surge_time: float = 100.0,
                           surge_ues: int = 70) -> Network:
    """
    Traffic surge: Sudden influx of users at a specific location.
    
    Use case:
      - Simulate stadium event, concert, protest
      - Test how network handles localized congestion
      - Validate QoS under extreme load
      
    Expected behavior:
      - Cell near surge location gets overloaded
      - SINR drops sharply in that cell
      - Throughput per UE decreases
      - May need temporary cell (COW - Cell on Wheels)
    """
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    net = Network(environment=env)
    
    # Towers
    net.add_tower(CellTower.create_standard_3sector("Tower_A", -500,  0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_B",  500,  0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_C",    0, 800, env))
    
    # Baseline UEs
    for ue in generate_random_ues(baseline_ues):
        net.add_ue(ue)
    
    # Schedule surge near Tower C (stadium)
    net.schedule_event(surge_time, lambda: net.inject_traffic_surge(
        center_x=0, center_y=700, radius=150, num_new_ues=surge_ues
    ))
    
    return net


# ===========================================================================
# SCENARIO 5: New Tower Deployment
# ===========================================================================
def scenario_new_deployment(num_ues: int = 60) -> Network:
    """
    New tower deployment: Add Tower_D mid-simulation.
    
    Use case:
      - Test impact of network densification
      - Measure coverage improvement
      - Validate capital expenditure ROI
      
    Expected behavior:
      - Load redistributes across 4 towers (more balanced)
      - Average SINR improves (less interference per cell)
      - Handover rate increases initially (UEs re-select)
    """
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    net = Network(environment=env)
    
    # Initial 3 towers
    net.add_tower(CellTower.create_standard_3sector("Tower_A", -500,   0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_B",  500,   0, env))
    net.add_tower(CellTower.create_standard_3sector("Tower_C",    0, 800, env))
    
    # UEs
    for ue in generate_random_ues(num_ues):
        net.add_ue(ue)
    
    # Add new tower at 100s (fills coverage gap in south)
    tower_d = CellTower.create_standard_3sector("Tower_D", 0, -400, env)
    net.schedule_event(100.0, lambda: net.restore_tower(tower_d))
    
    return net


# ===========================================================================
# SCENARIO 6: Weather Impact (Increased Path Loss)
# ===========================================================================
def scenario_weather_degradation(num_ues: int = 40) -> Network:
    """
    Weather event: Heavy rain/fog increases path loss.
    
    Implementation note:
      We can't dynamically change the propagation model mid-sim easily,
      so this is a simplified version. In practice you'd:
        - Add a rain attenuation factor to path loss
        - Or temporarily increase noise floor
        
    For now we'll just reduce all tower transmit power to simulate
    degraded propagation.
    
    Use case:
      - Test network in adverse conditions
      - Validate link budget margins
      - Plan for seasonal variations
    """
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    net = Network(environment=env)
    
    # Towers (initially normal power)
    net.add_tower(CellTower.create_standard_3sector("Tower_A", -500,  0, env, tx_power_dbm=43))
    net.add_tower(CellTower.create_standard_3sector("Tower_B",  500,  0, env, tx_power_dbm=43))
    net.add_tower(CellTower.create_standard_3sector("Tower_C",    0, 800, env, tx_power_dbm=43))
    
    # UEs
    for ue in generate_random_ues(num_ues):
        net.add_ue(ue)
    
    # At t=120, reduce power by 6 dB (simulate weather attenuation)
    def apply_weather():
        print("[120s] ☔ Weather event: Heavy rain (effective +6 dB path loss)")
        for tower in net.towers:
            for sector in tower.sectors.values():
                sector.tx_power_dbm -= 6  # Reduce by 6 dB
    
    net.schedule_event(120.0, apply_weather)
    
    return net


# ===========================================================================
# Demo: Run all scenarios
# ===========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    scenarios = [
        ("baseline", scenario_baseline, 180),
        ("peak_hour", scenario_peak_hour, 300),
        ("tower_failure", scenario_tower_failure, 240),
        ("traffic_surge", scenario_traffic_surge, 240),
        ("new_deployment", scenario_new_deployment, 240),
        ("weather", scenario_weather_degradation, 240),
    ]
    
    print("=" * 70)
    print("  SCENARIO LIBRARY DEMO")
    print("=" * 70)
    print(f"\nRunning {len(scenarios)} scenarios...\n")
    
    for name, factory, duration in scenarios:
        print(f"\n{'='*70}")
        print(f"  SCENARIO: {name.upper()}")
        print(f"{'='*70}")
        
        net = factory()
        net.run(duration_s=duration, dt=1.0, verbose=True, log_interval=60)
        net.plot_kpis(f"scenario_{name}_kpis.png")
        
        print()
    
    print("=" * 70)
    print("  All scenarios complete! Check the PNG files.")
    print("=" * 70)
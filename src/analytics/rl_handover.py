"""
Reinforcement Learning for Handover Optimization

============================================================
CONCEPT: Why RL for Handovers?
============================================================
Classical handover (A3 event) uses FIXED rules:
  "Handover when neighbor RSRP > serving RSRP + 3 dB"

Problems:
  ① One-size-fits-all (doesn't adapt to context)
  ② Can't balance multiple objectives (SINR vs stability)
  ③ Parameters tuned manually (trial & error)

RL learns an OPTIMAL POLICY from experience:
  - Try different handover decisions
  - Observe outcomes (throughput, ping-pong)
  - Learn which actions work best in which situations

Result: Context-aware, adaptive handover decisions.

============================================================
CONCEPT: Q-Learning (Simplified RL)
============================================================
Q-Learning learns a Q-table: Q(state, action) = expected reward

The algorithm:
  1. Observe current state (RSRP, SINR, neighbor signals)
  2. Choose action (handover to neighbor A/B/C, or stay)
  3. Execute action, observe reward
  4. Update Q-value:
     Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
     
Where:
  α = learning rate (how fast to update)
  γ = discount factor (value future rewards)
  r = immediate reward

After training, policy = argmax_a Q(s,a) (pick best action).

============================================================
CONCEPT: State/Action/Reward Design
============================================================
This is the ART of RL. Bad design = agent never learns.

State (what agent observes):
  - RSRP from serving cell
  - RSRP from best neighbor
  - RSRP difference (serving - neighbor)
  - Current SINR
  - Time since last handover (prevent ping-pong)

Action (what agent can do):
  - Stay on current cell
  - Handover to neighbor

Reward (what agent optimizes):
  + SINR improvement (better signal = better)
  + Throughput gain
  - Handover cost (switching has overhead)
  - Ping-pong penalty (rapid back-and-forth is bad)

Get this right = agent learns good policy.
Get it wrong = agent does random stuff forever.

============================================================
CONCEPT: xApp Architecture (O-RAN)
============================================================
In real 5G networks, AI runs as "xApps" in the RIC (RAN
Intelligent Controller):

  ┌──────────────────────────────────────┐
  │  Non-RT RIC (>1s)                    │
  │  - rApps (policy, optimization)      │
  └──────────────┬───────────────────────┘
                 │
  ┌──────────────▼───────────────────────┐
  │  Near-RT RIC (10-1000ms)             │
  │  - xApps (handover, scheduling)      │
  │  - Our RL agent runs here ←──────────│
  └──────────────┬───────────────────────┘
                 │
  ┌──────────────▼───────────────────────┐
  │  gNodeB (<10ms)                      │
  │  - PHY/MAC layer                     │
  └──────────────────────────────────────┘

xApps are containerized, modular, vendor-agnostic.
This is the O-RAN Alliance vision for AI-RAN.

We'll implement a simplified xApp for handover control.
"""

import numpy as np
from typing import Dict, Tuple, List
import pickle
from collections import defaultdict
import sys
import os
# Ensure the project `src/` directory is on sys.path when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.user_equipment import UserEquipment
from core.cell_tower import CellTower


# ===========================================================================
# State Space Discretization
# ===========================================================================
def discretize_rsrp(rsrp: float) -> int:
    """
    Discretize RSRP into bins for Q-table.
    
    Bins: Excellent (>-80), Good (-80 to -90), Fair (-90 to -100), Poor (<-100)
    """
    if rsrp > -80:
        return 0  # Excellent
    elif rsrp > -90:
        return 1  # Good
    elif rsrp > -100:
        return 2  # Fair
    else:
        return 3  # Poor


def discretize_sinr(sinr: float) -> int:
    """
    Discretize SINR into bins.
    
    Bins: Excellent (>20), Good (10-20), Fair (0-10), Poor (<0)
    """
    if sinr > 20:
        return 0
    elif sinr > 10:
        return 1
    elif sinr > 0:
        return 2
    else:
        return 3


def extract_state(ue: UserEquipment, towers: List[CellTower]) -> Tuple[int, int, int, int]:
    """
    Extract discrete state representation for RL agent.
    
    State = (serving_rsrp_bin, neighbor_rsrp_bin, rsrp_diff_bin, sinr_bin)
    
    Returns:
        Tuple of 4 integers (state representation)
    """
    if not ue.serving_tower:
        return (3, 3, 2, 3)  # Poor state
    
    # Get measurements from cache (populated by tick)
    serving_rsrp = ue._tick_cache.get(ue.serving_tower.tower_id, -999)
    
    # Find best neighbor
    best_neighbor_rsrp = -999
    for tid, rsrp in ue._tick_cache.items():
        if tid != ue.serving_tower.tower_id and rsrp > best_neighbor_rsrp:
            best_neighbor_rsrp = rsrp
    
    # Discretize
    s_rsrp_bin = discretize_rsrp(serving_rsrp)
    n_rsrp_bin = discretize_rsrp(best_neighbor_rsrp)
    
    # RSRP difference
    rsrp_diff = best_neighbor_rsrp - serving_rsrp
    if rsrp_diff > 5:
        diff_bin = 0  # Neighbor much better
    elif rsrp_diff > 0:
        diff_bin = 1  # Neighbor slightly better
    else:
        diff_bin = 2  # Serving better
    
    # SINR
    sinr = ue.sinr_history[-1] if ue.sinr_history else 0
    sinr_bin = discretize_sinr(sinr)
    
    return (s_rsrp_bin, n_rsrp_bin, diff_bin, sinr_bin)


# ===========================================================================
# RL Handover Agent (Q-Learning)
# ===========================================================================
class RLHandoverAgent:
    """
    Q-Learning agent for handover decisions.
    
    Policy:
      Given state s, choose action a that maximizes Q(s, a).
      
    Actions:
      0 = Stay on current cell
      1 = Handover to best neighbor
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1):
        """
        Args:
            learning_rate: How fast to update Q-values (α)
            discount_factor: How much to value future rewards (γ)
            epsilon: Exploration rate (ε-greedy)
        """
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-table: {state: {action: Q-value}}
        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: {0: 0.0, 1: 0.0})
        
        # Training stats
        self.episodes = 0
        self.total_reward = 0.0
        self.rewards_history = []
    
    def choose_action(self, state: Tuple, training: bool = True) -> int:
        """
        ε-greedy action selection.
        
        With probability ε: explore (random action)
        With probability 1-ε: exploit (best known action)
        
        Args:
            state: Current state tuple
            training: If False, always exploit (for evaluation)
            
        Returns:
            0 (stay) or 1 (handover)
        """
        if training and np.random.random() < self.epsilon:
            # Explore
            return np.random.choice([0, 1])
        else:
            # Exploit
            q_values = self.q_table[state]
            return max(q_values, key=q_values.get)
    
    def calculate_reward(self, 
                        sinr_before: float, sinr_after: float,
                        tput_before: float, tput_after: float,
                        did_handover: bool,
                        ping_pong: bool) -> float:
        """
        Reward function for handover decision.
        
        Components:
          + SINR improvement (better signal)
          + Throughput improvement (better performance)
          - Handover cost (switching overhead)
          - Ping-pong penalty (rapid reversals)
          
        Args:
            sinr_before, sinr_after: SINR in dB before/after action
            tput_before, tput_after: Throughput in Mbps before/after
            did_handover: Whether agent chose to handover
            ping_pong: Whether this was a reversal of recent HO
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # SINR improvement (main objective)
        sinr_gain = (sinr_after - sinr_before) / 10.0  # Normalize
        reward += sinr_gain * 10.0
        
        # Throughput improvement
        tput_gain = (tput_after - tput_before) / 100.0  # Normalize
        reward += tput_gain * 5.0
        
        # Handover cost (switching has overhead)
        if did_handover:
            reward -= 2.0  # Penalty for handover
        
        # Ping-pong penalty (big penalty!)
        if ping_pong:
            reward -= 20.0
        
        return reward
    
    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """
        Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        # TD update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Track stats
        self.total_reward += reward
    
    def train_episode(self, ue: UserEquipment, towers: List[CellTower], 
                     num_steps: int = 100) -> float:
        """
        Train for one episode (sequence of steps).
        
        Returns:
            Total reward for episode
        """
        episode_reward = 0.0
        
        for step in range(num_steps):
            # Observe state
            state = extract_state(ue, towers)
            
            # Measure current performance
            sinr_before = ue.sinr_history[-1] if ue.sinr_history else 0
            tput_before = ue.sinr_to_throughput_mbps(sinr_before)
            
            # Choose action
            action = self.choose_action(state, training=True)
            
            # Execute action
            did_handover = False
            ping_pong = False
            
            if action == 1:  # Handover
                # Find best neighbor
                best_neighbor = None
                best_rsrp = -999
                for t in towers:
                    if t.tower_id != ue.serving_tower.tower_id:
                        rsrp = ue._tick_cache.get(t.tower_id, -999)
                        if rsrp > best_rsrp:
                            best_rsrp = rsrp
                            best_neighbor = t
                
                if best_neighbor and best_rsrp > -999:
                    old_tower = ue.serving_tower
                    ue.serving_tower.disconnect_ue(ue.ue_id)
                    best_neighbor.connect_ue(ue.ue_id)
                    ue.serving_tower = best_neighbor
                    did_handover = True
                    
                    # Check for ping-pong (reversed recent HO)
                    if (ue.handover_log and 
                        len(ue.handover_log) >= 2 and
                        ue.handover_log[-1][2] == old_tower.tower_id):
                        ping_pong = True
            
            # Observe next state and performance
            ue.tick(towers, dt=1.0)
            next_state = extract_state(ue, towers)
            sinr_after = ue.sinr_history[-1] if ue.sinr_history else 0
            tput_after = ue.sinr_to_throughput_mbps(sinr_after)
            
            # Calculate reward
            reward = self.calculate_reward(
                sinr_before, sinr_after,
                tput_before, tput_after,
                did_handover, ping_pong
            )
            
            # Update Q-table
            self.update(state, action, reward, next_state)
            episode_reward += reward
        
        self.episodes += 1
        self.rewards_history.append(episode_reward)
        
        return episode_reward
    
    def save(self, filepath: str):
        """Save trained agent."""
        agent_dict = {
            'q_table': dict(self.q_table),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episodes': self.episodes
        }
        with open(filepath, 'wb') as f:
            pickle.dump(agent_dict, f)
        print(f"✅ RL agent saved: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load trained agent."""
        with open(filepath, 'rb') as f:
            agent_dict = pickle.load(f)
        
        agent = cls(
            learning_rate=agent_dict['alpha'],
            discount_factor=agent_dict['gamma'],
            epsilon=agent_dict['epsilon']
        )
        agent.q_table = defaultdict(lambda: {0: 0.0, 1: 0.0}, agent_dict['q_table'])
        agent.episodes = agent_dict['episodes']
        
        print(f"✅ RL agent loaded: {filepath} ({agent.episodes} episodes trained)")
        return agent


# ===========================================================================
# Training & Evaluation
# ===========================================================================
def train_rl_agent(num_episodes: int = 100) -> RLHandoverAgent:
    """
    Train RL agent on a simple 2-tower scenario.
    
    Setup:
      - 2 towers 1000m apart
      - UE walks between them
      - Agent learns optimal handover policy
    """
    from core.propagation import PropagationEnvironment
    from core.user_equipment import GaussMarkovMobility
    
    print("=" * 70)
    print("  TRAINING RL HANDOVER AGENT")
    print("=" * 70)
    
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )
    
    towers = [
        CellTower.create_standard_3sector("Tower_A", -500, 0, env),
        CellTower.create_standard_3sector("Tower_B", 500, 0, env)
    ]
    
    agent = RLHandoverAgent(learning_rate=0.1, discount_factor=0.95, epsilon=0.2)
    
    for ep in range(num_episodes):
        # Reset UE to random position
        x = np.random.uniform(-400, 400)
        mobility = GaussMarkovMobility(
            alpha=0.8,
            mean_speed=5.0,
            speed_std=1.0,
            mean_direction_deg=np.random.uniform(0, 360),
            bounds=(-600, 600, -300, 300)
        )
        
        ue = UserEquipment(f"UE_train_{ep}", x, 0, mobility, serving_tower=towers[0])
        
        # Train episode
        reward = agent.train_episode(ue, towers, num_steps=50)
        
        if (ep + 1) % 20 == 0:
            avg_reward = np.mean(agent.rewards_history[-20:])
            print(f"Episode {ep+1:>3}/{num_episodes} | Avg Reward (last 20): {avg_reward:>7.2f}")
    
    print(f"\n✅ Training complete! Total episodes: {agent.episodes}")
    print(f"   Q-table size: {len(agent.q_table)} states")
    
    return agent


# ===========================================================================
# Demo
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RL-BASED HANDOVER OPTIMIZATION — DEMO")
    print("=" * 70)
    
    # Train agent
    agent = train_rl_agent(num_episodes=200)
    
    # Save trained agent to results directory
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    agent_path = os.path.join(results_dir, 'rl_handover_agent.pkl')
    agent.save(agent_path)
    
    print("\n" + "=" * 70)
    print("  ✅ RL agent training complete!")
    print("=" * 70)
    print("\n💡 The agent has learned:")
    print("   • When to handover (signal + SINR context)")
    print("   • When to stay (avoid unnecessary switching)")
    print("   • How to avoid ping-pong (penalty learned)")
    print("=" * 70)
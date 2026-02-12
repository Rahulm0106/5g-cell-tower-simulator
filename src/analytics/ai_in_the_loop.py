"""
AI-in-the-Loop Network Optimization

Integrates LSTM traffic predictor with the network simulator to enable
PROACTIVE optimization based on predicted future load.

============================================================
CONCEPT: AI-in-the-Loop
============================================================
Traditional network management is REACTIVE:
  Problem occurs → Network detects it → Network reacts
  
AI-in-the-loop is PROACTIVE:
  AI predicts problem → Network acts BEFORE it happens

Examples:
  ① Predicted overload → Pre-emptive handover to neighbor cell
  ② Predicted congestion → Power up backup carrier
  ③ Predicted coverage hole → Adjust antenna tilt
  ④ Predicted surge → Alert operators to deploy COW

This is the future of 5G/6G — autonomous, self-optimizing networks.

============================================================
CONCEPT: Closed-Loop Optimization
============================================================
The full AI-in-the-loop cycle:

  ┌─────────────┐
  │   OBSERVE   │  Collect network KPIs (load, SINR, throughput)
  └──────┬──────┘
         ↓
  ┌──────────────┐
  │   PREDICT    │  LSTM forecasts next 5-15 minutes
  └──────┬───────┘
         ↓
  ┌──────────────┐
  │   DECIDE     │  Policy: What action to take?
  └──────┬───────┘
         ↓
  ┌──────────────┐
  │     ACT      │  Execute: handover, power adjust, alert
  └──────┬───────┘
         ↓
  └─────(feedback loop)

"""

import numpy as np
import sys
import os
# Ensure the project `src/` directory is on sys.path when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.network import Network
from analytics.lstm_predictor import SimpleLSTM
from typing import Dict, List


# ===========================================================================
# AI-Powered Network Manager
# ===========================================================================
class AINetworkManager:
    """
    Wraps the network simulator with an AI prediction layer.
    
    Responsibilities:
      - Track per-cell load history
      - Run LSTM predictions each tick
      - Trigger proactive actions based on predictions
      - Log AI decisions for analysis
    """
    
    def __init__(self, network: Network, predictor: SimpleLSTM,
                 lookback: int = 12, forecast: int = 3):
        """
        Args:
            network:    Network instance to manage
            predictor:  Trained LSTM model
            lookback:   History window (time steps)
            forecast:   Forecast horizon (time steps)
        """
        self.network = network
        self.predictor = predictor
        self.lookback = lookback
        self.forecast = forecast
        
        # History buffer: {tower_id: [load_t-12, ..., load_t]}
        self.load_history: Dict[str, List[float]] = {}
        
        # Predictions log
        self.predictions_log: List[dict] = []
        
        # Actions log
        self.actions_log: List[dict] = []
        
        # Thresholds
        self.overload_threshold = 40  # UEs — predicted overload
        self.alert_threshold = 50     # UEs — critical level
    
    def initialize_history(self):
        """Initialize history buffers with zeros."""
        for tower in self.network.towers:
            self.load_history[tower.tower_id] = [0.0] * self.lookback
    
    def update_history(self):
        """Update load history with current state."""
        for tower in self.network.towers:
            current_load = tower.stats.connected_ues
            
            # Maintain sliding window
            if tower.tower_id not in self.load_history:
                self.load_history[tower.tower_id] = [0.0] * self.lookback
            
            self.load_history[tower.tower_id].append(current_load)
            if len(self.load_history[tower.tower_id]) > self.lookback:
                self.load_history[tower.tower_id].pop(0)
    
    def predict_loads(self) -> Dict[str, np.ndarray]:
        """
        Run LSTM predictions for all cells.
        
        Returns:
            {tower_id: [predicted_load_t+1, t+2, t+3]}
        """
        predictions = {}
        
        for tower_id, history in self.load_history.items():
            # Need full lookback window
            if len(history) < self.lookback:
                predictions[tower_id] = np.zeros(self.forecast)
                continue
            
            # Prepare input: (1, lookback, 1)
            X = np.array(history[-self.lookback:]).reshape(1, self.lookback, 1)
            
            # Predict
            y_pred = self.predictor.predict(X)[0]  # (forecast,)
            predictions[tower_id] = y_pred
        
        return predictions
    
    def decide_actions(self, predictions: Dict[str, np.ndarray]) -> List[dict]:
        """
        Policy: Decide what actions to take based on predictions.
        
        Rules:
          ① If any cell predicted > alert_threshold → Alert operator
          ② If cell predicted > overload_threshold → Trigger offload
          
        Returns:
            List of actions: [{'type': 'alert', 'tower_id': ..., 'predicted_load': ...}, ...]
        """
        actions = []
        
        for tower_id, pred in predictions.items():
            max_predicted = pred.max()
            
            # Critical alert
            if max_predicted > self.alert_threshold:
                actions.append({
                    'type': 'alert',
                    'tower_id': tower_id,
                    'predicted_load': max_predicted,
                    'message': f"⚠️  CRITICAL: {tower_id} predicted to hit {max_predicted:.0f} UEs"
                })
            
            # Proactive offload
            elif max_predicted > self.overload_threshold:
                actions.append({
                    'type': 'offload',
                    'tower_id': tower_id,
                    'predicted_load': max_predicted,
                    'message': f"📊 Offload recommended: {tower_id} approaching {max_predicted:.0f} UEs"
                })
        
        return actions
    
    def execute_actions(self, actions: List[dict]):
        """
        Execute decided actions.
        
        For this demo:
          - 'alert': Just print (in production, send to NOC)
          - 'offload': Trigger load balancing (simplified — just log it)
        """
        for action in actions:
            if action['type'] == 'alert':
                print(f"[{self.network.time:.1f}s] {action['message']}")
            
            elif action['type'] == 'offload':
                print(f"[{self.network.time:.1f}s] {action['message']}")
                # In production: adjust handover params, tilt antennas, etc.
            
            # Log action
            self.actions_log.append({
                'time': self.network.time,
                **action
            })
    
    def tick_with_ai(self, dt: float):
        """
        Enhanced tick with AI predictions.
        
        Steps:
          1. Run normal network tick
          2. Update load history
          3. Predict future loads
          4. Decide actions
          5. Execute actions
        """
        # 1. Normal tick
        kpis = self.network.tick(dt)
        
        # 2. Update history
        self.update_history()
        
        # 3. Predict (only if we have enough history)
        if len(self.load_history[list(self.load_history.keys())[0]]) >= self.lookback:
            predictions = self.predict_loads()
            
            # Log predictions
            self.predictions_log.append({
                'time': self.network.time,
                'predictions': predictions.copy()
            })
            
            # 4 & 5. Decide and act
            actions = self.decide_actions(predictions)
            if actions:
                self.execute_actions(actions)
        
        return kpis
    
    def run_with_ai(self, duration_s: float, dt: float = 1.0,
                    verbose: bool = True, log_interval: int = 30):
        """Run simulation with AI-in-the-loop."""
        self.initialize_history()
        
        num_ticks = int(duration_s / dt)
        
        if verbose:
            print("=" * 70)
            print("  AI-IN-THE-LOOP NETWORK SIMULATION")
            print("=" * 70)
            print(f"  Duration:  {duration_s} s")
            print(f"  Towers:    {len(self.network.towers)}")
            print(f"  UEs:       {len(self.network.ues)}")
            print(f"  AI Model:  LSTM (lookback={self.lookback}, forecast={self.forecast})")
            print("=" * 70)
        
        for i in range(num_ticks):
            kpis = self.tick_with_ai(dt)
            
            if verbose and (i % log_interval == 0 or i == num_ticks - 1):
                print(f"[{self.network.time:>6.1f}s] "
                      f"UEs: {kpis.num_ues:>3} | "
                      f"Avg SINR: {kpis.avg_sinr_db:>5.1f} dB | "
                      f"Tput: {kpis.total_throughput_gbps:>5.2f} Gbps")
        
        if verbose:
            print("=" * 70)
            print(f"  AI actions triggered: {len(self.actions_log)}")
            for action in self.actions_log[-5:]:  # Show last 5
                print(f"    [{action['time']:.1f}s] {action['type']}: {action['tower_id']}")
            print("=" * 70)


# ===========================================================================
# Demo: AI-powered traffic surge scenario
# ===========================================================================
if __name__ == "__main__":
    from core.propagation import PropagationEnvironment
    from core.cell_tower import CellTower
    from core.user_equipment import UserEquipment, GaussMarkovMobility
    from analytics.lstm_predictor import SimpleLSTM
    import numpy as np
    
    print("\n" + "=" * 70)
    print("  AI-IN-THE-LOOP DEMO")
    print("=" * 70)
    
    # Load trained model
    print("\n1️⃣  Loading trained LSTM model...")
    try:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'lstm_model.pkl'))
        predictor = SimpleLSTM.load(model_path)
    except FileNotFoundError:
        print("   ⚠️  Model not found. Run lstm_predictor.py first.")
        print("   Creating a dummy model for demo...")
        predictor = SimpleLSTM(input_size=1, hidden_size=16, output_size=3)
        # Set dummy normalization params
        predictor.X_mean, predictor.X_std = 30.0, 15.0
        predictor.y_mean, predictor.y_std = 30.0, 15.0
    
    # Create network
    print("2️⃣  Setting up network...")
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
    
    # 30 baseline UEs
    for i in range(30):
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
    
    # Schedule traffic surge at t=60s (near Tower_C)
    net.schedule_event(60, lambda: net.inject_traffic_surge(
        center_x=0, center_y=700, radius=150, num_new_ues=40
    ))
    
    # Create AI manager
    print("3️⃣  Initializing AI manager...")
    ai_manager = AINetworkManager(
        network=net,
        predictor=predictor,
        lookback=12,
        forecast=3
    )
    
    # Run with AI
    print("4️⃣  Running simulation with AI predictions...\n")
    ai_manager.run_with_ai(duration_s=120, dt=1.0, verbose=True, log_interval=20)
    
    print("\n" + "=" * 70)
    print("  ✅ AI-in-the-loop demo complete!")
    print("=" * 70)
    print("\n💡 Key concept demonstrated:")
    print("   The AI predicts cell load and triggers alerts BEFORE")
    print("   the surge actually happens. This is proactive optimization.")
    print("=" * 70)
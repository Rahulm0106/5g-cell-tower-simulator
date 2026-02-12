"""
Traffic Pattern Generation for 5G Networks

============================================================
CONCEPT: Why Traffic Prediction Matters
============================================================
Mobile networks have PREDICTABLE load patterns:

  Daily cycle:
    - 3 AM:  Low (people sleeping) → ~10% capacity
    - 8 AM:  Morning rush (commute) → ~60% capacity
    - 12 PM: Lunch spike → ~70% capacity
    - 6 PM:  Evening rush → ~80% capacity
    - 9 PM:  Evening plateau → ~60% capacity
    
  Weekly cycle:
    - Mon-Fri: Business districts busy, residential quiet
    - Sat-Sun: Opposite pattern

If you can PREDICT load 5-10 minutes ahead, you can:
  ① Pre-emptively handover UEs before a cell overloads
  ② Power up backup cells before the peak
  ③ Alert operators to impending congestion
  ④ Optimize spectrum allocation (assign more bandwidth to hot cells)

This is "AI-in-the-loop" — the AI predicts, the network reacts.

============================================================
CONCEPT: Time-Series Forecasting
============================================================
Forecasting = predicting future values from past observations.

For network traffic:
  Input:   Past 30 minutes of cell load [t-30, t-29, ..., t-1, t]
  Output:  Next 5 minutes [t+1, t+2, t+3, t+4, t+5]

Common approaches:
  ① ARIMA:      Classical stats, assumes stationarity
  ② Prophet:    Facebook's tool, handles seasonality well
  ③ LSTM/GRU:   Deep learning, captures complex patterns
  ④ Transformer: State-of-the-art, but overkill for this

We'll use LSTM — it's the industry standard for network traffic.

============================================================
CONCEPT: LSTM (Long Short-Term Memory)
============================================================
LSTM is a type of Recurrent Neural Network (RNN) designed to
remember patterns over long time sequences.

Structure:
  - Input gate:   Controls what new info to add
  - Forget gate:  Controls what old info to discard
  - Output gate:  Controls what to output
  - Cell state:   Long-term memory

Why it works for traffic:
  - Remembers daily patterns (24-hour cycle)
  - Handles weekly patterns (7-day cycle)
  - Learns rush hour spikes
  - Robust to noise (random fluctuations)

We'll train on synthetic data, then deploy in the simulator.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os


# ===========================================================================
# Traffic Pattern Components
# ===========================================================================
@dataclass
class TrafficProfile:
    """
    Defines a traffic pattern for one cell tower.
    
    Combines multiple cyclic components:
      - Daily cycle (24 hours)
      - Weekly cycle (7 days)
      - Random noise
      - Special events (optional)
    """
    cell_id: str
    base_load: float = 20.0           # Baseline UE count
    daily_amplitude: float = 15.0      # Daily variation magnitude
    weekly_amplitude: float = 5.0      # Weekly variation magnitude
    noise_std: float = 3.0             # Random noise std dev
    rush_hour_boost: float = 10.0     # Extra load during rush hours


def generate_daily_pattern(hours: np.ndarray, amplitude: float = 15.0) -> np.ndarray:
    """
    Generate a realistic 24-hour traffic pattern.
    
    Pattern characteristics:
      - Low at night (3-6 AM)
      - Morning rush (7-9 AM)
      - Lunch spike (12-1 PM)
      - Evening rush (5-7 PM)
      - Evening plateau (7-10 PM)
    
    Uses a combination of sinusoids to create realistic shape.
    
    Args:
        hours: Array of hours [0, 24)
        amplitude: Peak-to-trough amplitude
        
    Returns:
        Traffic multiplier (centered at 0)
    """
    # Main daily cycle (peak at 6 PM = 18:00)
    main_cycle = amplitude * np.sin(2 * np.pi * (hours - 6) / 24)
    
    # Morning rush boost (7-9 AM)
    morning_rush = 0.3 * amplitude * np.exp(-((hours - 8)**2) / 4)
    
    # Lunch spike (12-1 PM)
    lunch_spike = 0.2 * amplitude * np.exp(-((hours - 12.5)**2) / 2)
    
    return main_cycle + morning_rush + lunch_spike


def generate_weekly_pattern(day_of_week: np.ndarray, amplitude: float = 5.0) -> np.ndarray:
    """
    Generate a 7-day weekly pattern.
    
    Pattern:
      - Mon-Thu: High (work days)
      - Fri:     Peak (end of week)
      - Sat-Sun: Lower (weekend)
    
    Args:
        day_of_week: Array of day indices [0=Mon, 6=Sun]
        amplitude: Peak-to-trough amplitude
        
    Returns:
        Traffic multiplier
    """
    # Simple cosine with peak on Friday
    weekly = amplitude * np.cos(2 * np.pi * (day_of_week - 4) / 7)
    
    # Weekend dip
    weekend_mask = (day_of_week >= 5)  # Sat, Sun
    weekly[weekend_mask] -= amplitude * 0.5
    
    return weekly


def generate_traffic_timeseries(profile: TrafficProfile,
                                duration_hours: int = 168,  # 1 week
                                sample_rate_min: int = 5     # Sample every 5 min
                               ) -> pd.DataFrame:
    """
    Generate a complete synthetic traffic time series.
    
    Combines:
      1. Base load
      2. Daily pattern
      3. Weekly pattern
      4. Random noise
      
    Returns:
        DataFrame with columns: [timestamp, hour, day_of_week, ue_count]
    """
    # Time axis (in minutes)
    num_samples = duration_hours * 60 // sample_rate_min
    timestamps = pd.date_range(start='2024-01-01', periods=num_samples, freq=f'{sample_rate_min}min')
    
    # Extract hour and day
    hours = timestamps.hour + timestamps.minute / 60.0
    day_of_week = timestamps.dayofweek  # 0=Mon, 6=Sun
    
    # Build traffic pattern
    daily = generate_daily_pattern(hours.values, profile.daily_amplitude)
    weekly = generate_weekly_pattern(day_of_week.values, profile.weekly_amplitude)
    noise = np.random.normal(0, profile.noise_std, size=num_samples)
    
    # Combine
    ue_count = profile.base_load + daily + weekly + noise
    
    # Clip to non-negative
    ue_count = np.maximum(ue_count, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'hour': hours,
        'day_of_week': day_of_week,
        'ue_count': ue_count
    })
    
    return df


# ===========================================================================
# Multi-Cell Traffic Generation
# ===========================================================================
def generate_network_traffic(num_cells: int = 3,
                             duration_hours: int = 168,
                             sample_rate_min: int = 5,
                             cell_types: List[str] = None) -> pd.DataFrame:
    """
    Generate traffic for a multi-cell network.
    
    Different cell types have different profiles:
      - Urban business:  High daily variation, weekday peaks
      - Residential:     Evening peaks, weekend activity
      - Highway:         Constant moderate load
      
    Args:
        num_cells:        Number of cells to generate
        duration_hours:   Duration in hours
        sample_rate_min:  Sampling interval in minutes
        cell_types:       List of cell types (optional)
        
    Returns:
        DataFrame with columns: [timestamp, cell_id, ue_count]
    """
    if cell_types is None:
        # Default: mix of business and residential
        cell_types = ['business', 'residential', 'highway'] * (num_cells // 3 + 1)
        cell_types = cell_types[:num_cells]
    
    all_data = []
    
    for i, cell_type in enumerate(cell_types):
        cell_id = f"Cell_{i:02d}"
        
        # Define profile based on cell type
        if cell_type == 'business':
            profile = TrafficProfile(
                cell_id=cell_id,
                base_load=30.0,
                daily_amplitude=20.0,
                weekly_amplitude=8.0,
                noise_std=4.0
            )
        elif cell_type == 'residential':
            profile = TrafficProfile(
                cell_id=cell_id,
                base_load=25.0,
                daily_amplitude=15.0,
                weekly_amplitude=3.0,
                noise_std=3.0
            )
        else:  # highway
            profile = TrafficProfile(
                cell_id=cell_id,
                base_load=15.0,
                daily_amplitude=8.0,
                weekly_amplitude=2.0,
                noise_std=2.0
            )
        
        # Generate time series
        df = generate_traffic_timeseries(profile, duration_hours, sample_rate_min)
        df['cell_id'] = cell_id
        df['cell_type'] = cell_type
        
        all_data.append(df)
    
    # Combine all cells
    combined = pd.concat(all_data, ignore_index=True)
    
    return combined


# ===========================================================================
# Dataset Preparation for ML
# ===========================================================================
def create_sequences(data: np.ndarray,
                     lookback: int = 12,    # 12 * 5min = 60 min history
                     forecast: int = 3       # 3 * 5min = 15 min ahead
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences for LSTM training.
    
    Sliding window approach:
      Input:  [t-lookback, ..., t-1, t]     (lookback points)
      Output: [t+1, ..., t+forecast]         (forecast points)
      
    Args:
        data:      1D array of traffic values
        lookback:  Number of past time steps to use
        forecast:  Number of future time steps to predict
        
    Returns:
        (X, y) where:
          X.shape = (num_samples, lookback, 1)
          y.shape = (num_samples, forecast)
    """
    X, y = [], []
    
    for i in range(lookback, len(data) - forecast):
        # Input: past lookback steps
        X.append(data[i - lookback:i])
        
        # Output: next forecast steps
        y.append(data[i:i + forecast])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for LSTM: (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y


def prepare_train_test_split(df: pd.DataFrame,
                             cell_id: str,
                             train_frac: float = 0.8,
                             lookback: int = 12,
                             forecast: int = 3
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare train/test sets for one cell.
    
    Args:
        df:          Traffic DataFrame
        cell_id:     Which cell to extract
        train_frac:  Fraction for training (rest is test)
        lookback:    History window
        forecast:    Prediction window
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Extract this cell's data
    cell_data = df[df['cell_id'] == cell_id]['ue_count'].values
    
    # Create sequences
    X, y = create_sequences(cell_data, lookback, forecast)
    
    # Split
    split_idx = int(len(X) * train_frac)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test


# ===========================================================================
# Visualization
# ===========================================================================
def plot_traffic_patterns(df: pd.DataFrame, save_path: str = None):
    """
    Visualize generated traffic patterns.
    
    Creates 2-panel figure:
      - Top: Time series for all cells
      - Bottom: Average daily profile
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # ── Panel 1: Full time series ──────────────────────────────────
    for cell_id in df['cell_id'].unique():
        cell_df = df[df['cell_id'] == cell_id]
        ax1.plot(cell_df['timestamp'], cell_df['ue_count'], 
                label=f"{cell_id} ({cell_df['cell_type'].iloc[0]})",
                alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('UE Count')
    ax1.set_title('Synthetic Traffic Patterns — Multi-Cell Network (1 Week)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ── Panel 2: Average daily profile ─────────────────────────────
    for cell_id in df['cell_id'].unique():
        cell_df = df[df['cell_id'] == cell_id]
        
        # Group by hour and average
        hourly_avg = cell_df.groupby('hour')['ue_count'].mean()
        
        ax2.plot(hourly_avg.index, hourly_avg.values,
                label=cell_id, linewidth=2.5, marker='o', markersize=4)
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average UE Count')
    ax2.set_title('Average Daily Traffic Profile')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 3))
    
    # Annotate rush hours
    ax2.axvspan(7, 9, alpha=0.1, color='orange', label='Morning Rush')
    ax2.axvspan(17, 19, alpha=0.1, color='red', label='Evening Rush')
    
    plt.tight_layout()
    
    if save_path:
        # If save_path is just a filename, save to results/plots/
        if not os.path.dirname(save_path):
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'plots'))
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()


# ===========================================================================
# Demo
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  TRAFFIC PATTERN GENERATION — DEMO")
    print("=" * 70)
    
    # Generate 1 week of traffic for 3 cells
    print("\nGenerating synthetic traffic data...")
    df = generate_network_traffic(
        num_cells=3,
        duration_hours=168,  # 1 week
        sample_rate_min=5,
        cell_types=['business', 'residential', 'highway']
    )
    
    print(f"  Generated {len(df)} samples")
    print(f"  Cells: {df['cell_id'].unique().tolist()}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Show statistics
    print("\n📊 Traffic Statistics:")
    print(df.groupby('cell_id')['ue_count'].describe()[['mean', 'std', 'min', 'max']])
    
    # Visualize
    print("\nGenerating visualization...")
    plot_traffic_patterns(df, save_path='traffic_patterns.png')
    
    # Prepare ML dataset for one cell
    print("\n🔬 Preparing ML dataset for Cell_00...")
    X_train, y_train, X_test, y_test = prepare_train_test_split(
        df, cell_id='Cell_00', 
        train_frac=0.8,
        lookback=12,   # 1 hour history
        forecast=3     # 15 min forecast
    )
    
    print(f"  X_train shape: {X_train.shape}  (samples, time_steps, features)")
    print(f"  y_train shape: {y_train.shape}  (samples, forecast_horizon)")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  y_test shape:  {y_test.shape}")
    
    print("\n✅ Traffic generation complete!")
    print("=" * 70)
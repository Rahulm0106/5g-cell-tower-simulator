"""
User Equipment (UE) Module

============================================================
CONCEPT: What is a UE?
============================================================
In 3GPP terminology, any end-user device is a "User Equipment":
  - Smartphones, tablets, laptops
  - IoT sensors, vehicles (V2X)
  - Fixed wireless terminals

The UE continuously:
  1. Measures signal from its SERVING cell (the one it's connected to)
  2. Measures signals from NEIGHBOR cells (for handover decisions)
  3. Reports measurements back to the network
  4. Decides (with network help) when to HANDOVER

============================================================
CONCEPT: Mobility Models — How do UEs move?
============================================================
In simulation we need to model HOW devices move through space.
Three common models (simplest → most realistic):

  ① Random Waypoint
     - Pick a random destination
     - Walk toward it at some speed
     - On arrival, pick a new random destination
     - Simple but produces unrealistic clustering

  ② Random Direction
     - Pick a random direction
     - Walk in that direction for a random time
     - Then pick a new direction
     - Smoother paths than Random Waypoint

  ③ Gauss-Markov (we use this one)
     - Position AND velocity are correlated over time
     - Next velocity = weighted average of current velocity + random noise
     - Produces smooth, realistic-looking trajectories
     - Good model for pedestrians and vehicles

     Formula:
       v(t+1) = α * v(t) + (1-α) * v_mean + σ * √(1-α²) * N(0,1)
       where:
         α     = memory factor (0 = random walk, 1 = straight line)
         v_mean = average speed
         σ     = speed standard deviation
         N(0,1)= standard normal random variable

============================================================
CONCEPT: SINR — The Number That Actually Matters
============================================================
SINR = Signal-to-Interference-plus-Noise Ratio

This is THE metric that determines your actual data speed.

                    Signal Power (from serving cell)
    SINR  =  ─────────────────────────────────────────────
              Interference (other cells) + Noise (thermal)

In dB:
    SINR_dB = Signal_dBm - 10*log10(10^(I_dBm/10) + 10^(N_dBm/10))

Why interference matters:
  - When you're between two towers, BOTH transmit to you
  - The one you're NOT connected to is INTERFERENCE
  - More towers nearby = more interference = lower SINR
  - This is why network planning is so critical!

SINR → Throughput mapping (Shannon):
    Throughput = B * log2(1 + SINR_linear)
    where B = bandwidth

    SINR (dB)  →  Approximate Throughput (100 MHz BW)
    -5  dB     →  ~30 Mbps
     0  dB     →  ~100 Mbps
     5  dB     →  ~200 Mbps
    10  dB     →  ~350 Mbps
    20  dB     →  ~700 Mbps
    30  dB     →  ~1000 Mbps (1 Gbps)

============================================================
CONCEPT: Handover — Switching Towers While Moving
============================================================
When you walk between two cell towers, your phone must switch.
This is called HANDOVER (or HANDOFF in older terminology).

Why it's tricky:
  - Too early → you disconnect from a still-good tower
  - Too late  → signal drops, data interrupts
  - Too fast  → "ping-pong" between towers (wastes resources)

3GPP uses A3 event for handover decision:
    Handover when:  RSRP_neighbor > RSRP_serving + A3_offset + hysteresis

    A3_offset:   Bias how much better the neighbor must be (e.g. 3 dB)
    Hysteresis:  Must stay above threshold for TIME_TO_TRIGGER ms
                 before actually handing over (prevents ping-pong)

    Time-to-Trigger (TTT): typically 0–640 ms
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.cell_tower import CellTower, compute_noise_power_dbm


# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
DEFAULT_BANDWIDTH_HZ  = 100e6       # 100 MHz — common 5G NR bandwidth
DEFAULT_A3_OFFSET_DB  = 3.0         # neighbor must be 3 dB stronger
DEFAULT_HYSTERESIS_MS = 200         # must hold for 200 ms before HO
DEFAULT_TTT_MS        = 200         # time-to-trigger = hysteresis here


# ---------------------------------------------------------------------------
# Mobility models
# ---------------------------------------------------------------------------
class GaussMarkovMobility:
    """
    Gauss-Markov mobility model.

    Produces smooth, correlated trajectories — much more realistic than
    pure random-walk.

    State per UE:
        (x, y)      — current position
        (vx, vy)    — current velocity

    Update each tick:
        vx_new = α*vx + (1-α)*v_mean_x + σ*√(1-α²)*randn()
        vy_new = α*vy + (1-α)*v_mean_y + σ*√(1-α²)*randn()
        x_new  = x + vx_new * dt
        y_new  = y + vy_new * dt
    """

    def __init__(self,
                 alpha: float = 0.7,          # memory factor
                 mean_speed: float = 3.0,     # m/s (walking ~3, driving ~15)
                 speed_std: float = 1.0,      # speed variation
                 mean_direction_deg: float = 90.0,  # average heading
                 bounds: Optional[tuple] = None):   # (xmin,xmax,ymin,ymax)
        self.alpha      = alpha
        self.mean_speed = mean_speed
        self.speed_std  = speed_std
        self.bounds     = bounds  # reflection if UE leaves area

        # Decompose mean velocity into x/y components
        dir_rad = np.radians(mean_direction_deg)
        self.v_mean_x = mean_speed * np.sin(dir_rad)   # East component
        self.v_mean_y = mean_speed * np.cos(dir_rad)   # North component

    def update(self, x: float, y: float,
               vx: float, vy: float, dt: float) -> tuple:
        """
        One time-step update.

        Returns:
            (new_x, new_y, new_vx, new_vy)
        """
        noise_scale = self.speed_std * np.sqrt(1 - self.alpha**2)

        # Update velocities
        new_vx = (self.alpha * vx +
                  (1 - self.alpha) * self.v_mean_x +
                  noise_scale * np.random.randn())
        new_vy = (self.alpha * vy +
                  (1 - self.alpha) * self.v_mean_y +
                  noise_scale * np.random.randn())

        # Update positions
        new_x = x + new_vx * dt
        new_y = y + new_vy * dt

        # Reflect off boundaries if defined
        if self.bounds:
            xmin, xmax, ymin, ymax = self.bounds
            if new_x < xmin:
                new_x  = xmin + (xmin - new_x)
                new_vx = -new_vx
            elif new_x > xmax:
                new_x  = xmax - (new_x - xmax)
                new_vx = -new_vx
            if new_y < ymin:
                new_y  = ymin + (ymin - new_y)
                new_vy = -new_vy
            elif new_y > ymax:
                new_y  = ymax - (new_y - ymax)
                new_vy = -new_vy

        return new_x, new_y, new_vx, new_vy


# ---------------------------------------------------------------------------
# UE signal measurement snapshot
# ---------------------------------------------------------------------------
@dataclass
class CellMeasurement:
    """What the UE 'sees' from one cell tower at one instant."""
    tower_id: str
    rsrp_dbm: float
    is_serving: bool = False    # True if this is the currently connected cell


# ---------------------------------------------------------------------------
# UserEquipment
# ---------------------------------------------------------------------------
class UserEquipment:
    """
    One mobile device moving through the network.

    Each tick the UE:
      1. Moves (Gauss-Markov)
      2. Measures RSRP from every tower in range
      3. Calculates SINR against its serving cell
      4. Checks handover conditions (A3 event)

    Args:
        ue_id:          Unique identifier
        x, y:           Starting position (metres)
        mobility:       GaussMarkovMobility instance
        serving_tower:  The CellTower it starts connected to
    """

    def __init__(self,
                 ue_id: str,
                 x: float,
                 y: float,
                 mobility: GaussMarkovMobility,
                 serving_tower: Optional[CellTower] = None):
        self.ue_id = ue_id

        # Position & velocity state
        self.x  = x
        self.y  = y
        self.vx = 0.0
        self.vy = 0.0

        # Mobility engine
        self.mobility = mobility

        # Network state
        self.serving_tower: Optional[CellTower] = serving_tower
        if serving_tower:
            serving_tower.connect_ue(ue_id)

        # Handover timer state
        self._ho_candidate: Optional[str] = None   # tower_id of HO target
        self._ho_timer_ms: float = 0.0             # accumulated ms
        self._ho_cooldown_ms: float = 0.0          # cooldown remaining after a HO

        # History (for plotting later)
        self.trajectory: List[tuple] = [(x, y)]
        self.sinr_history: List[float] = []
        self.rsrp_history: List[float] = []
        self.handover_log: List[tuple] = []  # (time_s, from_id, to_id)

        # Per-tick measurement cache (populated at start of each tick)
        self._tick_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def move(self, dt: float) -> None:
        """Advance position by dt seconds using mobility model."""
        self.x, self.y, self.vx, self.vy = self.mobility.update(
            self.x, self.y, self.vx, self.vy, dt
        )
        self.trajectory.append((self.x, self.y))

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Measurement (single stable pass per tick)
    # ------------------------------------------------------------------
    def _measure_all_cached(self, towers: List[CellTower]) -> Dict[str, float]:
        """
        Measure RSRP from every tower ONCE and cache for the whole tick.

        Why cache?
            The 3GPP propagation model randomly decides LoS vs NLoS each
            time it is called.  If we call it separately for SINR, for
            handover, etc., the same tower can flip between LoS and NLoS
            mid-tick, causing 20+ dB jumps and spurious handovers.

            Real radios do NOT do this — the channel changes on the order
            of milliseconds (fast fading) but the LoS/NLoS state is
            stable for tens of seconds (slow / shadow fading).

            Caching one measurement per tick per tower keeps the
            simulation self-consistent.

        Returns:
            {tower_id: rsrp_dbm}
        """
        cache: Dict[str, float] = {}
        for t in towers:
            cache[t.tower_id] = t.calculate_rsrp(self.x, self.y)
        return cache

    def measure_all_towers(self, towers: List[CellTower]) -> List[CellMeasurement]:
        """Public wrapper that uses the last cached measurements."""
        measurements = []
        for t in towers:
            rsrp = self._tick_cache.get(t.tower_id, -999.0)
            is_serving = (self.serving_tower is not None and
                          t.tower_id == self.serving_tower.tower_id)
            measurements.append(CellMeasurement(
                tower_id=t.tower_id, rsrp_dbm=rsrp, is_serving=is_serving
            ))
        return measurements

    # ------------------------------------------------------------------
    # SINR calculation  (uses tick cache)
    # ------------------------------------------------------------------
    def calculate_sinr(self, towers: List[CellTower],
                       bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ) -> float:
        """
        Calculate SINR using cached RSRP values from this tick.

        SINR = S / (I + N)
        """
        if self.serving_tower is None:
            return -999.0

        signal_dbm = self._tick_cache.get(self.serving_tower.tower_id, -999.0)
        signal_w   = 10 ** (signal_dbm / 10.0) * 1e-3

        interference_w = 0.0
        for t in towers:
            if t.tower_id == self.serving_tower.tower_id:
                continue
            interferer_dbm = self._tick_cache.get(t.tower_id, -999.0)
            interference_w += 10 ** (interferer_dbm / 10.0) * 1e-3

        noise_dbm = compute_noise_power_dbm(bandwidth_hz)
        noise_w   = 10 ** (noise_dbm / 10.0) * 1e-3

        sinr_linear = signal_w / (interference_w + noise_w)
        sinr_db     = 10 * np.log10(sinr_linear) if sinr_linear > 0 else -999.0
        return sinr_db

    # ------------------------------------------------------------------
    # Throughput estimate (Shannon)
    # ------------------------------------------------------------------
    @staticmethod
    def sinr_to_throughput_mbps(sinr_db: float,
                                bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ) -> float:
        """
        Shannon capacity estimate.

        C = B * log2(1 + SINR_linear)   [bits/s]

        This is a theoretical upper bound. Real throughput is typically
        60-80% of this due to overhead, coding, retransmissions.

        We apply a 0.7 efficiency factor as a rough correction.
        """
        sinr_linear = 10 ** (sinr_db / 10.0)
        capacity_bps = bandwidth_hz * np.log2(1 + sinr_linear)
        # Apply 70% efficiency for overhead
        throughput_mbps = (capacity_bps * 0.7) / 1e6
        return throughput_mbps

    # ------------------------------------------------------------------
    # Handover logic (3GPP A3 event)
    # ------------------------------------------------------------------
    def check_handover(self,
                       towers: List[CellTower],
                       dt_ms: float,
                       a3_offset_db: float  = DEFAULT_A3_OFFSET_DB,
                       ttt_ms: float        = DEFAULT_TTT_MS,
                       cooldown_ms: float   = 5000.0) -> bool:
        """
        Evaluate A3 handover event with anti-ping-pong protection.
        Uses cached RSRP values from _tick_cache for consistency.

        Two-layer defence against ping-pong:
          ① Time-to-Trigger (TTT): A3 condition must hold for TTT ms
                                   continuously before HO executes.
          ② Post-HO Cooldown:      After any handover, block ALL new
                                   handover attempts for cooldown_ms.

        Returns:
            True if a handover was executed this tick.
        """
        if self.serving_tower is None:
            return False

        # ── Cooldown check ─────────────────────────────────────────
        if self._ho_cooldown_ms > 0:
            self._ho_cooldown_ms -= dt_ms
            if self._ho_cooldown_ms < 0:
                self._ho_cooldown_ms = 0.0
            return False

        serving_rsrp = self._tick_cache.get(self.serving_tower.tower_id, -999.0)

        # Find best neighbor from cache
        best_neighbor      = None
        best_neighbor_rsrp = -999.0

        for t in towers:
            if t.tower_id == self.serving_tower.tower_id:
                continue
            rsrp = self._tick_cache.get(t.tower_id, -999.0)
            if rsrp > best_neighbor_rsrp:
                best_neighbor      = t
                best_neighbor_rsrp = rsrp

        if best_neighbor is None:
            return False

        # ── A3 condition ───────────────────────────────────────────
        a3_triggered = (best_neighbor_rsrp > serving_rsrp + a3_offset_db)

        if a3_triggered:
            if self._ho_candidate == best_neighbor.tower_id:
                self._ho_timer_ms += dt_ms
            else:
                self._ho_candidate = best_neighbor.tower_id
                self._ho_timer_ms  = dt_ms

            if self._ho_timer_ms >= ttt_ms:
                self._execute_handover(best_neighbor)
                self._ho_cooldown_ms = cooldown_ms
                return True
        else:
            self._ho_candidate = None
            self._ho_timer_ms  = 0.0

        return False

    def _execute_handover(self, new_tower: CellTower) -> None:
        """Perform the actual cell switch."""
        old_id = self.serving_tower.tower_id if self.serving_tower else "None"
        self.serving_tower.disconnect_ue(self.ue_id)
        new_tower.connect_ue(self.ue_id)
        self.serving_tower = new_tower

        # Reset HO state
        self._ho_candidate = None
        self._ho_timer_ms  = 0.0

        self.handover_log.append((len(self.trajectory), old_id, new_tower.tower_id))

    # ------------------------------------------------------------------
    # Full tick (move + measure + SINR + handover)
    # ------------------------------------------------------------------
    def tick(self, towers: List[CellTower], dt: float = 1.0) -> dict:
        """
        Execute one simulation tick for this UE.

        Order matters:
          1. Move          — update position
          2. Cache         — ONE stable RSRP reading per tower
          3. Auto-connect  — pick strongest if not yet served
          4. SINR          — uses cache
          5. Handover      — uses cache
          6. Log           — record KPIs

        Args:
            towers: All towers in the network
            dt:     Time step in seconds
        Returns:
            dict with current KPIs
        """
        # 1. Move
        self.move(dt)

        # 2. Cache — single measurement pass, stable for this tick
        self._tick_cache = self._measure_all_cached(towers)

        # 3. Auto-connect if not yet connected (pick strongest)
        if self.serving_tower is None:
            best_id  = max(self._tick_cache, key=self._tick_cache.get)
            for t in towers:
                if t.tower_id == best_id:
                    self.serving_tower = t
                    t.connect_ue(self.ue_id)
                    break

        # 4. SINR  (reads cache internally)
        sinr = self.calculate_sinr(towers)
        self.sinr_history.append(sinr)

        # 5. RSRP of serving cell (from cache)
        rsrp = self._tick_cache.get(
            self.serving_tower.tower_id, -999.0
        ) if self.serving_tower else -999.0
        self.rsrp_history.append(rsrp)

        # 6. Handover check  (dt is in seconds; convert to ms)
        self.check_handover(towers, dt_ms=dt * 1000.0, cooldown_ms=5000.0)

        # 7. Throughput
        throughput = self.sinr_to_throughput_mbps(sinr)

        return {
            "ue_id": self.ue_id,
            "x": self.x, "y": self.y,
            "serving": self.serving_tower.tower_id if self.serving_tower else None,
            "rsrp_dbm": rsrp,
            "sinr_db": sinr,
            "throughput_mbps": throughput
        }

    def __repr__(self):
        srv = self.serving_tower.tower_id if self.serving_tower else "None"
        return f"UE(id={self.ue_id}, pos=({self.x:.0f},{self.y:.0f}), serving={srv})"


# ---------------------------------------------------------------------------
# Demo: single UE walking between two towers
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from core.propagation import PropagationEnvironment

    print("=" * 70)
    print("  USER EQUIPMENT + SINR + HANDOVER — CONCEPT DEMO")
    print("=" * 70)

    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )

    # Two towers, 1000 m apart on the x-axis
    tower_A = CellTower.create_standard_3sector("Tower_A", x=-500, y=0, environment=env)
    tower_B = CellTower.create_standard_3sector("Tower_B", x= 500, y=0, environment=env)
    towers  = [tower_A, tower_B]

    # UE starts near Tower A, walks East toward Tower B
    mobility = GaussMarkovMobility(
        alpha=0.8,
        mean_speed=5.0,                # 5 m/s ≈ jogging
        speed_std=1.0,
        mean_direction_deg=90.0,       # 90° = due East
        bounds=(-600, 600, -300, 300)
    )

    ue = UserEquipment(
        ue_id="UE_01",
        x=-450, y=0,                   # Start near Tower A
        mobility=mobility,
        serving_tower=tower_A
    )

    print(f"\n📱 UE starts at ({ue.x:.0f}, {ue.y:.0f}), connected to {tower_A.tower_id}")
    print(f"📡 Tower A at (-500, 0)   |   Tower B at (500, 0)")
    print(f"   UE walks EAST →  (expect handover near midpoint)\n")
    print("-" * 70)
    print(f"{'Step':<6} {'Position':<18} {'Serving':<10} {'RSRP':<10} {'SINR':<10} {'Tput Mbps':<12} {'HO?'}")
    print("-" * 70)

    dt = 1.0   # 1-second ticks
    for step in range(120):
        result = ue.tick(towers, dt=dt)

        # Print every 10 steps
        if step % 10 == 0:
            ho_mark = ""
            if ue.handover_log and ue.handover_log[-1][0] > step - 10:
                ho_mark = "⚡ HO!"
            print(f"{step:<6} ({result['x']:>6.0f},{result['y']:>5.0f})   "
                  f"{result['serving']:<10} "
                  f"{result['rsrp_dbm']:<10.1f} "
                  f"{result['sinr_db']:<10.1f} "
                  f"{result['throughput_mbps']:<12.1f} "
                  f"{ho_mark}")

    print("-" * 70)
    print(f"\n📊 Handover log:")
    if ue.handover_log:
        for tick_idx, from_id, to_id in ue.handover_log:
            print(f"   Tick {tick_idx:>4}: {from_id} → {to_id}")
    else:
        print("   (no handovers recorded)")

    print(f"\n💡 Key observations:")
    print(f"   • SINR is high when close to serving tower (little interference)")
    print(f"   • SINR drops near the MIDPOINT (interference from both towers)")
    print(f"   • Handover fires when neighbor becomes {DEFAULT_A3_OFFSET_DB} dB stronger")
    print(f"   • After HO the 'old' tower becomes the interferer")
    print("=" * 70)
    
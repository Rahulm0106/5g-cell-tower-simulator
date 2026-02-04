"""
Cell Tower (gNodeB) Module

============================================================
CONCEPT: What is a gNodeB?
============================================================
In 5G, the base station is called a gNodeB (generation Node B).
It is the physical tower + equipment that:
  1. Transmits signals DOWN to user devices (Downlink / DL)
  2. Receives signals UP from user devices (Uplink / UL)
  3. Manages radio resources (who gets what spectrum, when)

CONCEPT: Sectors
============================================================
Real cell towers don't radiate signal in a perfect circle.
They use DIRECTIONAL antennas split into SECTORS:

        Sector A (North)
            ▲
            |  120°
     -------●-------
    /   S.C  |  S.B  \
   (  West   |  East  )
    \       |       /
     -------+-------
            |
        Sector B (South-East)

- A typical macro tower has 3 sectors (each 120°)
- Each sector acts like an independent cell
- This TRIPLES the capacity of one tower location!
- Antenna gain is higher in the main direction

CONCEPT: Antenna Gain
============================================================
Antennas are not equal. Gain (dBi) measures how much better
an antenna radiates in its main direction vs a perfect sphere:

  Omnidirectional (0 dBi):   Radiates equally in all directions
  Directional (15-21 dBi):   Focuses energy in one direction
  
  Higher gain = stronger signal in main beam direction
  BUT = weaker signal outside the beam (trade-off!)

CONCEPT: Thermal Noise
============================================================
Every receiver has background noise from random electron movement.
  Noise Power = k * T * B
  where:
    k = Boltzmann constant (1.38e-23 J/K)
    T = Temperature in Kelvin (typically 290K = 17°C)
    B = Bandwidth in Hz

  For 5G with 100 MHz bandwidth:
    Noise = -174 dBm/Hz + 10*log10(100e6) = -94 dBm
    
  This is the FLOOR — received signal must be above this to work.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import sys
import os

# Support both direct execution and module import
try:
    from .propagation import (
        PropagationEnvironment,
        ThreeGPP_38_901_UMa,
        calculate_received_power,
        dbm_to_watts,
        watts_to_dbm
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.propagation import (
        PropagationEnvironment,
        ThreeGPP_38_901_UMa,
        calculate_received_power,
        dbm_to_watts,
        watts_to_dbm
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOLTZMANN_CONSTANT = 1.38e-23   # J/K
NOISE_TEMPERATURE  = 290.0      # Kelvin (standard reference)
NOISE_FIGURE_DB    = 7.0        # dB — typical receiver noise figure


def compute_noise_power_dbm(bandwidth_hz: float,
                            noise_figure_db: float = NOISE_FIGURE_DB) -> float:
    """
    Calculate thermal noise power at the receiver.

    Formula (all in dB):
        N = -174 + 10*log10(B) + NF

        -174 dBm/Hz  →  thermal noise spectral density at 290 K
        10*log10(B)  →  bandwidth contribution
        NF           →  receiver noise figure (how much noise the receiver adds)

    Args:
        bandwidth_hz:   Allocated bandwidth in Hz
        noise_figure_db: Receiver noise figure in dB
    Returns:
        Noise power in dBm
    """
    # -174 dBm/Hz is the standard thermal noise floor
    noise_spectral_density = -174.0                          # dBm/Hz
    bandwidth_db = 10 * np.log10(bandwidth_hz)     # convert Hz → dB
    noise_power = noise_spectral_density + bandwidth_db + noise_figure_db
    return noise_power


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SectorConfig:
    """
    Configuration for one 120° sector of a cell tower.

    Each sector is essentially an independent transmitter with its own
    directional antenna pointing in a specific direction.
    """
    sector_id: str                  # e.g. "A", "B", "C"
    azimuth_deg: float              # Center direction (0°=North, clockwise)
    beamwidth_deg: float = 120.0    # Typical 3-sector site
    tx_power_dbm: float   = 43.0   # 20 W per sector (typical macro)
    antenna_gain_dbi: float = 18.0  # Directional gain in main beam
    antenna_height_m: float = 25.0  # Tower height


@dataclass
class CellTowerStats:
    """Live counters updated every simulation tick."""
    connected_ues: int = 0
    total_throughput_mbps: float = 0.0
    avg_sinr_db: float = 0.0
    avg_rsrp_dbm: float = 0.0
    handovers_out: int = 0          # UEs that left this cell
    handovers_in: int = 0           # UEs that joined this cell


# ---------------------------------------------------------------------------
# CellTower
# ---------------------------------------------------------------------------
class CellTower:
    """
    Represents one physical cell tower site with multiple sectors.

    Responsibilities:
      - Know its own position and configuration
      - Compute received power at any 2-D point (per sector)
      - Track which UEs are currently connected
      - Report per-sector and per-site KPIs

    Usage:
        tower = CellTower(tower_id="T1", x=0, y=0, sectors=[...])
        rsrp  = tower.calculate_rsrp(ue_x=300, ue_y=400)  # dBm
    """

    def __init__(self,
                 tower_id: str,
                 x: float,
                 y: float,
                 sectors: List[SectorConfig],
                 environment: PropagationEnvironment):
        """
        Args:
            tower_id:    Unique identifier (e.g. "Tower_01")
            x, y:        Position in metres (origin = top-left or centre)
            sectors:     List of SectorConfig (usually 3 for a macro site)
            environment: Propagation environment settings
        """
        self.tower_id   = tower_id
        self.x          = x
        self.y          = y
        self.sectors    = {s.sector_id: s for s in sectors}
        self.environment = environment

        # Propagation model (shared; environment already carries freq/heights)
        self.prop_model = ThreeGPP_38_901_UMa(environment)

        # Live state
        self.connected_ue_ids: List[str] = []
        self.stats = CellTowerStats()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _distance_to(self, ue_x: float, ue_y: float) -> float:
        """Euclidean 2-D distance from tower to a point."""
        return np.sqrt((self.x - ue_x)**2 + (self.y - ue_y)**2)

    def _angle_to(self, ue_x: float, ue_y: float) -> float:
        """
        Bearing from tower to point, in degrees (0 = North, clockwise).

        Uses atan2 on (dx, dy) — note the argument order swap vs math
        convention because we want 0° = North.
        """
        dx = ue_x - self.x
        dy = ue_y - self.y
        # atan2 gives angle from positive-x axis; convert to compass bearing
        angle_rad = np.arctan2(dx, dy)          # North-referenced
        angle_deg = np.degrees(angle_rad) % 360 # Wrap to [0, 360)
        return angle_deg

    # ------------------------------------------------------------------
    # Which sector serves a given point?
    # ------------------------------------------------------------------
    def _get_serving_sector(self, ue_x: float, ue_y: float) -> Optional[SectorConfig]:
        """
        Determine which sector's beam covers the UE location.

        Logic:
          1. Calculate bearing from tower to UE
          2. Find the sector whose azimuth centre is closest
          3. Check the UE falls within that sector's beamwidth

        Returns None if somehow no sector covers the angle
        (shouldn't happen with a full 360° site, but safe).
        """
        bearing = self._angle_to(ue_x, ue_y)

        best_sector = None
        best_diff  = 999.0

        for sec in self.sectors.values():
            # Angular difference, wrapped to [-180, 180]
            diff = (bearing - sec.azimuth_deg + 180) % 360 - 180
            abs_diff = abs(diff)

            if abs_diff <= sec.beamwidth_deg / 2.0:
                if abs_diff < best_diff:
                    best_diff  = abs_diff
                    best_sector = sec

        return best_sector

    # ------------------------------------------------------------------
    # Core signal calculation
    # ------------------------------------------------------------------
    def calculate_rsrp(self, ue_x: float, ue_y: float) -> float:
        """
        Calculate RSRP (Reference Signal Received Power) at a UE location.

        Steps:
          1. Find which sector is serving this location
          2. Compute 2-D distance
          3. Ask propagation model for path loss
          4. Apply link budget: RSRP = Tx_power + Tx_gain - Path_loss

        Returns:
            RSRP in dBm.  Returns -999 if no sector covers the point.
        """
        sector = self._get_serving_sector(ue_x, ue_y)
        if sector is None:
            return -999.0   # effectively no signal

        distance = self._distance_to(ue_x, ue_y)
        if distance < 1.0:
            distance = 1.0  # avoid log(0)

        path_loss = self.prop_model.calculate_path_loss(distance)

        rsrp = calculate_received_power(
            tx_power_dbm  = sector.tx_power_dbm,
            path_loss_db  = path_loss,
            tx_gain_dbi   = sector.antenna_gain_dbi,
            rx_gain_dbi   = 0.0   # phone antenna ≈ isotropic
        )
        return rsrp

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect_ue(self, ue_id: str) -> None:
        if ue_id not in self.connected_ue_ids:
            self.connected_ue_ids.append(ue_id)
            self.stats.connected_ues += 1
            self.stats.handovers_in += 1

    def disconnect_ue(self, ue_id: str) -> None:
        if ue_id in self.connected_ue_ids:
            self.connected_ue_ids.remove(ue_id)
            self.stats.connected_ues -= 1
            self.stats.handovers_out += 1

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------
    @staticmethod
    def create_standard_3sector(tower_id: str,
                                x: float,
                                y: float,
                                environment: PropagationEnvironment,
                                tx_power_dbm: float = 43.0) -> 'CellTower':
        """
        Convenience factory: create a typical 3-sector macro tower.

        Sectors point North (0°), South-East (120°), South-West (240°).
        """
        sectors = [
            SectorConfig(sector_id="A", azimuth_deg=0.0,   tx_power_dbm=tx_power_dbm),
            SectorConfig(sector_id="B", azimuth_deg=120.0, tx_power_dbm=tx_power_dbm),
            SectorConfig(sector_id="C", azimuth_deg=240.0, tx_power_dbm=tx_power_dbm),
        ]
        return CellTower(tower_id, x, y, sectors, environment)

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"CellTower(id={self.tower_id}, pos=({self.x:.0f},{self.y:.0f}), "
                f"sectors={list(self.sectors.keys())}, "
                f"connected_ues={self.stats.connected_ues})")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  CELL TOWER (gNodeB) — CONCEPT DEMO")
    print("=" * 70)

    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0
    )

    # Create a standard 3-sector tower at the origin
    tower = CellTower.create_standard_3sector("Tower_01", x=0, y=0, environment=env)
    print(f"\n📡 Created: {tower}")
    print(f"   Sectors: {[(s.sector_id, s.azimuth_deg) for s in tower.sectors.values()]}")

    # ── Test points in each sector ──────────────────────────────────────
    test_points = [
        ("North  (Sector A)",  0,    500),
        ("SE     (Sector B)",  433,  -250),   # 120° at 500 m
        ("SW     (Sector C)", -433,  -250),   # 240° at 500 m
        ("Close  (50 m N)",    0,    50),
        ("Far    (1500 m N)",  0,    1500),
    ]

    print("\n" + "-" * 70)
    print(f"{'Location':<25} {'Sector':<8} {'Distance (m)':<15} {'RSRP (dBm)':<12} {'Quality'}")
    print("-" * 70)

    for label, ux, uy in test_points:
        rsrp    = tower.calculate_rsrp(ux, uy)
        dist    = tower._distance_to(ux, uy)
        sector  = tower._get_serving_sector(ux, uy)
        sec_id  = sector.sector_id if sector else "—"

        if rsrp > -80:   quality = "★★★★★ Excellent"
        elif rsrp > -90: quality = "★★★★☆ Good"
        elif rsrp > -100:quality = "★★★☆☆ Fair"
        else:            quality = "★★☆☆☆ Poor"

        print(f"{label:<25} {sec_id:<8} {dist:<15.1f} {rsrp:<12.2f} {quality}")

    # ── Noise floor demo ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  NOISE FLOOR CALCULATION")
    print("=" * 70)
    bandwidths = [20e6, 50e6, 100e6]   # 20 / 50 / 100 MHz
    for bw in bandwidths:
        noise = compute_noise_power_dbm(bw)
        print(f"  Bandwidth = {bw/1e6:>6.0f} MHz  →  Noise floor = {noise:.1f} dBm")

    print("\n  💡 The receiver MUST see a signal above this noise floor to decode data.")
    print("     SINR = (Signal) / (Interference + Noise)  — next concept!")
    print("=" * 70)
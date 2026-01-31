"""
Radio Propagation Models for 5G Cell Tower Simulation

This module implements various path loss models that predict how radio signals
degrade as they travel from cell towers to user devices.

Key Concept: PATH LOSS
-----------------------
Path loss is the reduction in power density of a radio wave as it propagates.
Measured in decibels (dB), it represents how much the signal weakens.

Formula: Received_Power (dBm) = Transmitted_Power (dBm) - Path_Loss (dB)

Why this matters:
- Determines coverage area of a cell tower
- Affects data rates (higher signal = higher throughput)
- Critical for network planning and optimization
"""

import numpy as np
from typing import Tuple, Literal
from dataclasses import dataclass


@dataclass
class PropagationEnvironment:
    """
    Configuration for the propagation environment.
    
    Different environments have different signal behavior:
    - Urban: Lots of buildings, high path loss
    - Suburban: Mix of buildings and open space
    - Rural: Mostly open, lower path loss
    """
    environment_type: Literal['urban', 'suburban', 'rural']
    base_station_height: float = 25.0  # meters (typical: 20-50m)
    mobile_height: float = 1.5          # meters (phone at chest height)
    carrier_frequency: float = 3500.0    # MHz (common 5G mid-band)
    
    
class PropagationModel:
    """Base class for all propagation models."""
    
    def __init__(self, environment: PropagationEnvironment):
        self.env = environment
        
    def calculate_path_loss(self, distance_m: float) -> float:
        """
        Calculate path loss in dB for a given distance.
        
        Args:
            distance_m: Distance between transmitter and receiver in meters
            
        Returns:
            Path loss in dB (higher = more signal loss)
        """
        raise NotImplementedError("Subclasses must implement this method")


class FreeSpacePathLoss(PropagationModel):
    """
    Free Space Path Loss (FSPL) Model
    
    Assumption: Perfect vacuum, no obstacles, direct line of sight
    Reality: Only accurate in ideal conditions (rarely true!)
    
    Use case: Theoretical baseline, satellite communications
    
    Formula: FSPL(dB) = 20*log10(d) + 20*log10(f) + 32.45
    where:
        d = distance in km
        f = frequency in MHz
        32.45 = constant derived from speed of light
        
    Physical interpretation:
    - Signal spreads out spherically
    - Power density decreases with distance squared (inverse square law)
    - Higher frequencies have shorter wavelengths → more loss
    """
    
    def calculate_path_loss(self, distance_m: float) -> float:
        if distance_m < 1.0:  # Minimum distance to avoid log(0)
            distance_m = 1.0
            
        distance_km = distance_m / 1000.0
        frequency_mhz = self.env.carrier_frequency
        
        # FSPL formula
        path_loss = (20 * np.log10(distance_km) + 
                     20 * np.log10(frequency_mhz) + 
                     32.45)
        
        return path_loss


class OkumuraHataModel(PropagationModel):
    """
    Okumura-Hata Model (1968/1980)
    
    Assumption: Urban/suburban environments, based on measurements in Tokyo
    Validity: 150-1500 MHz, distances 1-20 km
    
    Use case: Traditional cellular (2G, 3G, 4G in lower bands)
    
    Formula (Urban):
    PL = 69.55 + 26.16*log10(f) - 13.82*log10(hb) - a(hm) + 
         (44.9 - 6.55*log10(hb))*log10(d)
    
    where:
        f = frequency (MHz)
        hb = base station height (m)
        hm = mobile height (m)
        d = distance (km)
        a(hm) = mobile antenna correction factor
        
    Key insight: Antenna height matters! Higher towers cover more area.
    """
    
    def _mobile_antenna_correction(self) -> float:
        """
        Calculate correction factor based on mobile antenna height.
        Different formulas for different environment types.
        """
        hm = self.env.mobile_height
        f = self.env.carrier_frequency
        
        if self.env.environment_type == 'urban':
            # For large cities
            if f >= 400:
                return 3.2 * (np.log10(11.75 * hm))**2 - 4.97
            else:
                return (1.1 * np.log10(f) - 0.7) * hm - (1.56 * np.log10(f) - 0.8)
        else:
            # For suburban/rural
            return (1.1 * np.log10(f) - 0.7) * hm - (1.56 * np.log10(f) - 0.8)
    
    def calculate_path_loss(self, distance_m: float) -> float:
        if distance_m < 100:  # Model not accurate for very short distances
            distance_m = 100
            
        distance_km = distance_m / 1000.0
        f = self.env.carrier_frequency
        hb = self.env.base_station_height
        
        a_hm = self._mobile_antenna_correction()
        
        # Base urban formula
        path_loss = (69.55 + 
                     26.16 * np.log10(f) - 
                     13.82 * np.log10(hb) - 
                     a_hm + 
                     (44.9 - 6.55 * np.log10(hb)) * np.log10(distance_km))
        
        # Corrections for suburban and rural
        if self.env.environment_type == 'suburban':
            path_loss = path_loss - 2 * (np.log10(f / 28))**2 - 5.4
        elif self.env.environment_type == 'rural':
            path_loss = path_loss - 4.78 * (np.log10(f))**2 + 18.33 * np.log10(f) - 40.94
            
        return path_loss


class ThreeGPP_38_901_UMa(PropagationModel):
    """
    3GPP 38.901 Urban Macro (UMa) Model
    
    Standard: 3GPP TR 38.901 V17.0.0 (2022-03)
    Assumption: 5G/6G urban macro-cell deployments
    Validity: 0.5-100 GHz, distances up to 5 km
    
    Use case: Official 5G network planning model
    
    Key features:
    - Separate formulas for Line of Sight (LoS) and Non-Line of Sight (NLoS)
    - Probabilistic LoS based on distance
    - Includes breakpoint distance (signal behavior changes)
    
    Why LoS matters:
    - LoS: Direct path, lower loss
    - NLoS: Blocked by buildings, higher loss, more realistic in cities
    
    Formula (simplified LoS):
    PL = 28.0 + 22*log10(d) + 20*log10(f)  [d < breakpoint]
    PL = 28.0 + 40*log10(d) + 20*log10(f) - 9*log10(breakpoint^2 + (hBS-hUT)^2)  [d >= breakpoint]
    
    Formula (NLoS):
    PL = max(PL_LoS, PL_NLoS')
    where PL_NLoS' has additional terms for diffraction and scattering
    """
    
    def __init__(self, environment: PropagationEnvironment):
        super().__init__(environment)
        # 3GPP parameters
        self.h_BS = environment.base_station_height  # Base station height
        self.h_UT = environment.mobile_height         # User terminal height
        self.h_E = 1.0  # Effective environment height (average building height)
        
    def _calculate_breakpoint_distance(self) -> float:
        """
        Breakpoint distance: where propagation mechanism changes
        
        Before breakpoint: Ground reflection is dominant
        After breakpoint: Ground reflection and diffraction both matter
        
        Formula: d_BP = 4 * h'_BS * h'_UT * f_c / c
        where:
            h'_BS = hBS - hE (effective BS height)
            h'_UT = hUT - hE (effective UT height)
            f_c = carrier frequency (Hz)
            c = speed of light
        """
        h_prime_BS = self.h_BS - self.h_E
        h_prime_UT = self.h_UT - self.h_E
        f_hz = self.env.carrier_frequency * 1e6  # MHz to Hz
        c = 3e8  # speed of light in m/s
        
        d_BP = 4 * h_prime_BS * h_prime_UT * f_hz / c
        return d_BP
    
    def _los_probability(self, distance_2d: float) -> float:
        """
        Probability of Line of Sight based on distance.
        
        In cities: closer = more likely to have direct path
        
        3GPP model:
        P_LoS = min(18/d, 1) * (1 - exp(-d/63)) + exp(-d/63)
        
        Interpretation:
        - Very close (< 18m): Almost always LoS
        - Medium distance: Probability decreases
        - Far (> 1000m): Very low LoS probability
        """
        if distance_2d <= 18:
            return 1.0
        else:
            p_los = (18 / distance_2d + 
                     np.exp(-distance_2d / 63) * (1 - 18 / distance_2d))
            return p_los
    
    def _calculate_los_path_loss(self, distance_3d: float, distance_2d: float) -> float:
        """
        Line of Sight path loss calculation.
        
        Uses two-slope model with breakpoint.
        """
        f_ghz = self.env.carrier_frequency / 1000.0  # MHz to GHz
        d_BP = self._calculate_breakpoint_distance()
        
        if distance_2d < 10:  # Minimum distance
            distance_2d = 10
            distance_3d = 10
        
        if distance_2d <= d_BP:
            # Before breakpoint: free space-like propagation
            PL = 28.0 + 22 * np.log10(distance_3d) + 20 * np.log10(f_ghz)
        else:
            # After breakpoint: ground reflection becomes important
            PL = (28.0 + 40 * np.log10(distance_3d) + 20 * np.log10(f_ghz) - 
                  9 * np.log10((d_BP)**2 + (self.h_BS - self.h_UT)**2))
        
        return PL
    
    def _calculate_nlos_path_loss(self, distance_3d: float, distance_2d: float) -> float:
        """
        Non-Line of Sight path loss calculation.
        
        Higher loss due to:
        - Diffraction around buildings
        - Scattering from rough surfaces
        - Multiple reflections
        """
        f_ghz = self.env.carrier_frequency / 1000.0
        
        if distance_2d < 10:
            distance_2d = 10
            distance_3d = 10
        
        # NLoS formula from 3GPP 38.901
        PL_nlos_prime = (13.54 + 39.08 * np.log10(distance_3d) + 
                         20 * np.log10(f_ghz) - 
                         0.6 * (self.h_UT - 1.5))
        
        # NLoS path loss is always at least as high as LoS
        PL_los = self._calculate_los_path_loss(distance_3d, distance_2d)
        PL_nlos = max(PL_los, PL_nlos_prime)
        
        return PL_nlos
    
    def calculate_path_loss(self, distance_m: float, 
                           force_los: bool = None) -> float:
        """
        Calculate path loss with probabilistic LoS/NLoS.
        
        Args:
            distance_m: 2D distance in meters
            force_los: If None, use probabilistic LoS. If True/False, force LoS/NLoS.
            
        Returns:
            Path loss in dB
        """
        # Calculate 3D distance (accounting for height difference)
        distance_2d = distance_m
        distance_3d = np.sqrt(distance_2d**2 + (self.h_BS - self.h_UT)**2)
        
        if force_los is None:
            # Probabilistic approach
            p_los = self._los_probability(distance_2d)
            is_los = np.random.random() < p_los
        else:
            is_los = force_los
        
        if is_los:
            return self._calculate_los_path_loss(distance_3d, distance_2d)
        else:
            return self._calculate_nlos_path_loss(distance_3d, distance_2d)


def calculate_received_power(tx_power_dbm: float, 
                             path_loss_db: float,
                             tx_gain_dbi: float = 18.0,
                             rx_gain_dbi: float = 0.0) -> float:
    """
    Calculate received signal power.
    
    This is the fundamental link budget equation in wireless communications.
    
    Formula: P_rx = P_tx + G_tx + G_rx - PL
    
    where (all in dB/dBm):
        P_rx = received power (dBm)
        P_tx = transmitted power (dBm)
        G_tx = transmit antenna gain (dBi)
        G_rx = receive antenna gain (dBi)
        PL = path loss (dB)
        
    Typical values:
        P_tx: 40-46 dBm (10-40 Watts) for macro cell
        G_tx: 15-21 dBi (directional antennas)
        G_rx: 0 dBi (omnidirectional phone antenna)
        
    Args:
        tx_power_dbm: Transmit power in dBm
        path_loss_db: Path loss in dB
        tx_gain_dbi: Transmit antenna gain in dBi
        rx_gain_dbi: Receive antenna gain in dBi
        
    Returns:
        Received power in dBm
    """
    received_power = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss_db
    return received_power


def dbm_to_watts(power_dbm: float) -> float:
    """Convert power from dBm to Watts."""
    return 10 ** ((power_dbm - 30) / 10)


def watts_to_dbm(power_watts: float) -> float:
    """Convert power from Watts to dBm."""
    return 10 * np.log10(power_watts) + 30


# Example usage and validation
if __name__ == "__main__":
    print("=" * 70)
    print("5G RADIO PROPAGATION MODELS DEMONSTRATION")
    print("=" * 70)
    
    # Set up environment
    env = PropagationEnvironment(
        environment_type='urban',
        base_station_height=25.0,
        mobile_height=1.5,
        carrier_frequency=3500.0  # 3.5 GHz (common 5G band)
    )
    
    # Test distances
    distances = [50, 100, 200, 500, 1000, 2000]  # meters
    
    print(f"\nEnvironment: {env.environment_type}")
    print(f"Frequency: {env.carrier_frequency} MHz")
    print(f"Base Station Height: {env.base_station_height} m")
    print(f"Mobile Height: {env.mobile_height} m")
    print("\n" + "-" * 70)
    
    # Initialize models
    fspl = FreeSpacePathLoss(env)
    okumura = OkumuraHataModel(env)
    threeGpp = ThreeGPP_38_901_UMa(env)
    
    print(f"\n{'Distance (m)':<15} {'FSPL (dB)':<15} {'Okumura (dB)':<15} {'3GPP LoS (dB)':<15} {'3GPP NLoS (dB)':<15}")
    print("-" * 70)
    
    for d in distances:
        fspl_loss = fspl.calculate_path_loss(d)
        okumura_loss = okumura.calculate_path_loss(d)
        threeGpp_los = threeGpp.calculate_path_loss(d, force_los=True)
        threeGpp_nlos = threeGpp.calculate_path_loss(d, force_los=False)
        
        print(f"{d:<15} {fspl_loss:<15.2f} {okumura_loss:<15.2f} {threeGpp_los:<15.2f} {threeGpp_nlos:<15.2f}")
    
    # Demonstrate received power calculation
    print("\n" + "=" * 70)
    print("RECEIVED POWER CALCULATION EXAMPLE")
    print("=" * 70)
    
    tx_power = 43  # dBm (20 Watts) - typical macro cell
    distance = 500  # meters
    
    path_loss = threeGpp.calculate_path_loss(distance, force_los=True)
    rx_power = calculate_received_power(tx_power, path_loss)
    
    print(f"\nScenario: User at {distance}m from tower")
    print(f"Transmit Power: {tx_power} dBm ({dbm_to_watts(tx_power):.2f} W)")
    print(f"Path Loss: {path_loss:.2f} dB")
    print(f"Received Power: {rx_power:.2f} dBm ({dbm_to_watts(rx_power)*1e3:.6f} mW)")
    
    # Signal quality interpretation
    if rx_power > -80:
        quality = "EXCELLENT"
    elif rx_power > -90:
        quality = "GOOD"
    elif rx_power > -100:
        quality = "FAIR"
    else:
        quality = "POOR"
    
    print(f"Signal Quality: {quality}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("- FSPL is optimistic (assumes no obstacles)")
    print("- Okumura-Hata adds realistic urban losses")
    print("- 3GPP NLoS has highest path loss (most realistic in cities)")
    print("- Received power decreases rapidly with distance")
    print("=" * 70)
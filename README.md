# 5G Cell Tower Simulator

## 🎯 Project Goals
By the end of this project, you will understand:
1. **5G RAN (Radio Access Network) Architecture**
2. **Radio Propagation Models** (how signals travel and degrade)
3. **Key Performance Indicators (KPIs)** in wireless networks
4. **User Equipment (UE) behavior** and mobility
5. **Network optimization concepts**
6. **Time-series prediction** for traffic forecasting

---

## 📚 Part 1: 5G Fundamentals

### What is a Cell Tower?
A cell tower (base station or gNodeB in 5G) is the infrastructure that:
- **Transmits** radio signals to user devices (downlink)
- **Receives** signals from user devices (uplink)
- **Manages** radio resources (frequency, time slots, power)
- **Handles** handovers when users move between cells

### Key 5G Concepts

#### 1. **Frequency Bands**
5G operates in different frequency ranges:
- **Sub-6 GHz** (e.g., 3.5 GHz): Better coverage, moderate speed
- **mmWave** (e.g., 28 GHz, 39 GHz): Very high speed, limited coverage

**Trade-off**: Higher frequency = more bandwidth (speed) but worse propagation (coverage)

#### 2. **Signal Propagation**
When radio waves travel from tower to device, they:
- **Attenuate** (lose power) over distance
- **Reflect** off buildings and surfaces
- **Diffract** around obstacles
- Experience **interference** from other signals

#### 3. **Path Loss Models**
Mathematical models that predict signal strength based on distance:

**Free Space Path Loss (FSPL)** - Ideal conditions (no obstacles):
```
PL(dB) = 20*log10(d) + 20*log10(f) + 32.45
where:
  d = distance in km
  f = frequency in MHz
```

**Okumura-Hata Model** - Urban environments:
```
PL(dB) = 69.55 + 26.16*log10(f) - 13.82*log10(hb) - a(hm) + (44.9 - 6.55*log10(hb))*log10(d)
where:
  f = frequency (MHz)
  hb = base station antenna height (m)
  hm = mobile antenna height (m)
  d = distance (km)
```

**3GPP 38.901 Model** - 5G standard model (we'll implement this):
```
Different formulas for:
- Line of Sight (LoS)
- Non-Line of Sight (NLoS)
- Indoor scenarios
```

#### 4. **Key Performance Indicators (KPIs)**

**RSRP (Reference Signal Received Power)**:
- Measured power of reference signals from the cell
- Range: typically -140 dBm (very weak) to -44 dBm (very strong)
- **Good signal**: > -80 dBm
- **Fair signal**: -80 to -100 dBm
- **Poor signal**: < -100 dBm

**SINR (Signal-to-Interference-plus-Noise Ratio)**:
```
SINR = Signal_Power / (Interference_Power + Noise_Power)
```
- Measured in dB
- **Excellent**: > 20 dB
- **Good**: 13-20 dB
- **Fair**: 0-13 dB
- **Poor**: < 0 dB

**Throughput (Data Rate)**:
Using Shannon-Hartley theorem:
```
Throughput = Bandwidth * log2(1 + SINR)
```
- Measured in Mbps or Gbps
- Depends on: SINR, available bandwidth, modulation scheme

**Latency**:
- End-to-end delay for data transmission
- 5G target: < 1ms for URLLC
- Typical: 10-20ms

#### 5. **Handover**
When a user moves between cells:
- **Measurement**: UE measures signal strength from neighbor cells
- **Decision**: Network decides when to handover (based on RSRP, SINR)
- **Execution**: Connection switches to new cell
- **Types**:
  - Intra-frequency handover (same frequency band)
  - Inter-frequency handover (different bands)
  - Inter-RAT handover (e.g., 5G to 4G)

---

## 🔧 Implementation Deep Dives

### 1. gNodeB & Sectors
A real cell tower splits its antenna into 3 sectors (each 120°). Each sector is an independent transmitter with a directional antenna. This triples capacity at one physical location. The implementation uses sector selection based on bearing angle to properly model this realistic behavior.

### 2. Gauss-Markov Mobility
Pure random walk produces jerky, unrealistic paths. Gauss-Markov adds velocity memory — the next velocity is a weighted blend of current velocity, a mean drift, and noise. The alpha parameter controls how much the UE "remembers" its current heading:
- **alpha = 0**: Pure random walk (very jerky)
- **alpha = 1**: Straight line (no turning)
- **alpha = 0.75**: Smooth, realistic pedestrian/vehicle paths

With alpha=0.75, the model generates movement that closely resembles real-world user equipment behavior.

### 3. SINR — The Real Performance Metric
**RSRP** (raw signal strength) tells you how loud one tower is. **SINR** tells you how well you can hear that tower over all the other noise:

```
SINR = Signal / (Interference + Noise)
```

**Key insight from the simulation**: SINR drops sharply at cell edges where two towers overlap. At those points your serving tower's signal is strong, but so is the interferer — the ratio collapses. This is why handovers exist.

### 4. Shannon Throughput
The theoretical upper bound on data rate is given by the Shannon-Hartley theorem:

```
Throughput = Bandwidth × log₂(1 + SINR)
```

The implementation applies a 0.7 efficiency factor for real-world overhead (headers, retransmissions, guard intervals). Notice how throughput tracks SINR almost perfectly — SINR is the single most important number in wireless.

### 5. Handover & Ping-Pong (and how we fixed it)

#### The A3 Event
3GPP's standard handover trigger:
```
HO when: RSRP_neighbor > RSRP_serving + A3_offset
```
The A3_offset (3 dB here) means the neighbor must be meaningfully better, not just marginally.

#### Bug 1: Stochastic LoS Flipping
**Problem**: The 3GPP propagation model randomly decides LoS vs NLoS each time it's called. Two calls for the same tower at the same position could return values 20 dB apart. This caused RSRP to jump wildly within a single tick, triggering and then immediately cancelling handovers.

**Fix**: We added a per-tick measurement cache (`_tick_cache`). All RSRP readings for one tick come from a single call per tower. SINR, handover logic, and KPI logging all read from this cache. This models reality correctly — the LoS/NLoS state (shadow fading) changes on the order of seconds, not milliseconds.

#### Bug 2: Ping-Pong Handovers
**Problem**: Even with TTT (Time-to-Trigger), the UE was bouncing A→B→A in consecutive ticks because the cooldown wasn't being applied correctly (dt was passed in seconds but compared against milliseconds).

**Fix**: Two-layer anti-ping-pong strategy:
1. **TTT (200 ms)**: A3 condition must hold continuously for 200 ms before HO fires
2. **Post-HO Cooldown (5 s)**: After any handover, ALL new HO attempts are blocked for 5 seconds (standard 3GPP practice)

#### What the Remaining Handovers Mean
The simulation still shows several HOs. Some of these (spaced ~6 s apart) are the UE spending time right at a cell edge where the two towers trade dominance as the UE drifts. This is realistic — in real networks, a UE lingering at a cell boundary does experience frequent handovers. Production networks handle this with larger A3 offsets and longer cooldowns tuned to the specific deployment.

---

## 🏗️ Project Architecture

```
5g-cell-tower-sim/
├── src/
│   ├── core/
│   │   ├── cell_tower.py          # Cell tower class with transmission logic
│   │   ├── user_equipment.py      # UE (phone/device) with mobility
│   │   ├── propagation.py         # Path loss models (FSPL, Okumura, 3GPP)
│   │   └── network.py             # RAN network manager
│   ├── simulation/
│   │   ├── engine.py              # Main simulation loop
│   │   └── scenarios.py           # Pre-defined test scenarios
│   ├── analytics/
│   │   ├── metrics.py             # KPI calculation
│   │   └── ml_predictor.py        # Traffic prediction (LSTM/Prophet)
│   └── visualization/
│       ├── dashboard.py           # Streamlit dashboard
│       └── plots.py               # Plotting utilities
├── tests/
│   └── test_propagation.py        # Unit tests
├── data/
│   └── scenarios/                 # Saved scenario configurations
├── requirements.txt
└── README.md
```

---

## 📖 Learning Path

### Phase 1: Core Radio Fundamentals (Today - Day 1-2)
1. ✅ Understand propagation models
2. ✅ Implement path loss calculations
3. ✅ Calculate received power (RSRP)
4. ✅ Understand SINR calculation

### Phase 2: Network Components (Day 3-4)
1. Build CellTower class
2. Build UserEquipment class
3. Implement mobility models
4. Calculate handover decisions

### Phase 3: Simulation Engine (Day 5-7)
1. Time-stepped simulation loop
2. Multi-cell interference modeling
3. KPI tracking and logging
4. Scenario management

### Phase 4: Visualization (Day 8-10)
1. Coverage heatmaps
2. Real-time KPI dashboards
3. User trajectory visualization
4. Interactive parameter tuning

### Phase 5: ML Integration (Day 11-14)
1. Traffic pattern generation
2. Time-series forecasting (LSTM)
3. Predictive load balancing
4. Anomaly detection

---

## 🎓 Key Learning Objectives

By building this simulator, you'll learn:

**Radio Engineering**:
- How radio waves propagate in different environments
- Why cell planning is crucial
- Trade-offs in coverage vs capacity

**Network Performance**:
- What metrics matter and why
- How interference affects performance
- Optimization strategies

**System Modeling**:
- Discrete event simulation
- State management in complex systems
- Performance vs accuracy trade-offs

**Data Science**:
- Time-series analysis
- Predictive modeling
- Real-time analytics

**Software Engineering**:
- Object-oriented design for physical systems
- Modular architecture
- Testing and validation

# Distributed Transportation System Simulation Platform

A modular framework for developing and testing traffic algorithms in SUMO simulation environment, featuring genetic algorithm optimization and continuous vehicle generation.

## Quick Start

### System Requirements
- SUMO with `SUMO_HOME` environment variable configured
- Python 3.6+
- Dependencies: `numpy`, `matplotlib`

### Run Simulations

```bash
# Crossroad intersection
python crossroad_runner.py

# Roundabout scenario
python roundabout_runner.py

# Genetic algorithm training
python ga_crossroad_runner.py --mode train --generations 10

# Test vehicle generation
python test_sumo_flows.py --steps 1500
```

## Key Features

- **Modular Algorithm Framework**: Abstract base classes for Consensus, Networking, Driving Strategy, and Scheduling
- **🆕 Real-Time Visualization**: Rich visualization for consensus and communication processes
- **Genetic Algorithm Optimization**: Automated strategy training with 18-state space encoding
- **SUMO Flow-Based Generation**: Continuous vehicle generation with lane-specific routing
- **Multiple Scenarios**: Crossroad intersection and roundabout configurations
- **Real-Time Control**: 30-meter decision zone with acceleration-based control

## Documentation

📚 **[Complete Guide](COMPLETE_GUIDE.md)** - Comprehensive documentation covering all features

🎨 **[Visualization Guide](VISUALIZATION_GUIDE.md)** - How to integrate your algorithm with the visualization system

Quick links:
- [Getting Started](COMPLETE_GUIDE.md#getting-started)
- [Simulation Scenarios](COMPLETE_GUIDE.md#simulation-scenarios)
- [Developer Guide](COMPLETE_GUIDE.md#developer-guide)
- [Genetic Algorithm](COMPLETE_GUIDE.md#genetic-algorithm-system)
- [SUMO Flow Configuration](COMPLETE_GUIDE.md#sumo-flow-configuration)
- [Visualization API](VISUALIZATION_GUIDE.md#api-参考)

## Project Structure

```
├── algorithms.py              # Algorithm base classes
├── visualization_interface.py # 🆕 Visualization API for algorithm developers
├── crossroad_runner.py        # Crossroad simulation
├── roundabout_runner.py       # Roundabout simulation
├── ga_traffic_strategy.py     # Genetic algorithm core
├── ga_crossroad_runner.py     # GA-based simulation
├── visualize_strategy.py      # Strategy visualization
├── test_sumo_flows.py         # Flow generation test
├── crossroad.rou.xml          # Crossroad routes & flows
├── roundabout.rou.xml         # Roundabout routes & flows
├── COMPLETE_GUIDE.md          # Complete documentation
├── VISUALIZATION_GUIDE.md     # 🆕 Visualization integration guide
├── DEVELOPER_GUIDE.md         # Developer reference
└── vehicle_documentation.md   # Vehicle parameters
```

## Scenarios

### Crossroad
- 4-way intersection with traffic lights
- 3 lanes per approach (right turn, straight, left turn)
- Continuous vehicle generation via SUMO flows
- 30m decision zone for strategy control

### Roundabout
- Circular intersection with coordinated signals
- 3 vehicle types (Cautious, Normal, Aggressive)
- Non-uniform traffic distribution
- Green wave signal coordination

## Genetic Algorithm

Optimizes vehicle control strategies through evolution:
- **State Space**: 18 states (3 distances × 3 speeds × 2 light states)
- **Actions**: Accelerate, Maintain, Decelerate
- **Fitness**: Travel time + waiting time + collision penalty
- **Operations**: Tournament selection, crossover, mutation

## Vehicle Generation

Uses SUMO native flows for optimal performance:
- 12 concurrent flows (4 directions × 3 lanes)
- Default: 120 vehicles/hour per lane
- Lane-specific routing (right turn, straight, left turn)
- Continuous generation with automatic overflow handling

## Development

### Implement Custom Algorithm with Visualization 🆕

```python
from visualization_interface import (
    VisualizableConsensus,
    VehicleState,
    MessageType,
    ConsensusPhase
)
import traci

class MyConsensus(VisualizableConsensus):
    def update(self, step: int):
        vehicles = traci.vehicle.getIDList()
        
        if len(vehicles) >= 2:
            # Send message visualization (particles flying between vehicles)
            self.emit_message(vehicles[0], vehicles[1], MessageType.PREPARE)
            
            # Update vehicle state (colored ring around vehicle)
            self.emit_state_change(vehicles[0], VehicleState.PREPARING)
            
            # Update progress bar
            self.emit_progress(ConsensusPhase.PREPARE, 1, 4)
```

For detailed visualization API, see **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)**

### Implement Custom Driving Strategy

```python
from algorithms import DrivingStrategy

class MyStrategy(DrivingStrategy):
    def update(self, step: int):
        vehicles = traci.vehicle.getIDList()
        for veh in vehicles:
            speed = traci.vehicle.getSpeed(veh)
            # Your control logic
```

### Adjust Traffic Density

Edit `crossroad.rou.xml`:
```xml
<flow ... vehsPerHour="120" ... />  <!-- Adjust this value -->
```

### Train GA Strategy

```bash
python ga_crossroad_runner.py --mode train --generations 20 --population 30
```

## Testing

```bash
# Check flow configuration
python test_sumo_flows.py --check-config

# Run flow test
python test_sumo_flows.py --steps 1500

# Visualize strategy
python visualize_strategy.py
```

## References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [TraCI API](https://sumo.dlr.de/docs/TraCI.html)
- Holland, J. H. (1992). Adaptation in Natural and Artificial Systems

---

**For detailed information, see [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)**

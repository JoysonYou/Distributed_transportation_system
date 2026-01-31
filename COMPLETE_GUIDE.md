# Distributed Transportation System Simulation Platform - Complete Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Project Overview](#project-overview)
3. [Getting Started](#getting-started)
4. [Simulation Scenarios](#simulation-scenarios)
5. [Developer Guide](#developer-guide)
6. [Genetic Algorithm System](#genetic-algorithm-system)
7. [SUMO Flow Configuration](#sumo-flow-configuration)
8. [Vehicle Parameters](#vehicle-parameters)
9. [Customization Guide](#customization-guide)

---

## System Requirements

Before running this project, ensure your system meets the following requirements:

1. **SUMO**: SUMO must be installed, and the `SUMO_HOME` environment variable must be set correctly.
   - Download from [SUMO Official Website](https://sumo.dlr.de/)
2. **Python**: Python 3.6 or higher
3. **Python Dependencies**:
   ```bash
   pip install numpy matplotlib
   ```
4. **TraCI Library**: Usually comes with SUMO, located in `SUMO_HOME/tools` directory

---

## Project Overview

This project provides a modular framework for developing and testing traffic algorithms in SUMO simulation environment. It supports two main scenarios: crossroad intersections and roundabouts.

### Key Features
- Modular algorithm framework with abstract base classes
- Genetic algorithm for traffic strategy optimization
- SUMO flow-based continuous vehicle generation
- Real-time vehicle control and monitoring
- Multiple vehicle types with different driving behaviors

---

## Getting Started

### Basic Simulations

#### 1. Run Crossroad Simulation
```bash
python crossroad_runner.py
```
Launches a SUMO GUI window displaying a traffic light-controlled intersection.

#### 2. Run Roundabout Simulation
```bash
python roundabout_runner.py
```
Launches a roundabout scenario with vehicles exhibiting different driving styles.

#### 3. Run Genetic Algorithm Training
```bash
python ga_crossroad_runner.py --mode train --generations 10
```
Trains an optimized traffic strategy using genetic algorithm.

### Quick Tests

#### Test SUMO Flow Generation
```bash
python test_sumo_flows.py --steps 1500
```
Verify continuous vehicle generation configuration.

#### Test Genetic Algorithm Demo
```bash
python quick_demo.py
```
Interactive demonstration of genetic algorithm optimization.

---

## Simulation Scenarios

### Crossroad Intersection

#### Network Configuration
- **Network File**: `crossroad.net.xml`
- **Routes**: `crossroad.rou.xml`
- **4 approaches**: West, East, North, South
- **3 lanes per approach**: 
  - Lane 0 (Right): Right turn only
  - Lane 1 (Middle): Straight only
  - Lane 2 (Left): Left turn only

#### Vehicle Generation
Vehicles are generated using SUMO flows with continuous generation:
- 12 concurrent flows (4 directions × 3 lanes)
- Default rate: 120 vehicles/hour per lane
- Lane-specific routing automatically enforced

#### Control Strategy
- **Decision Zone**: 30 meters before intersection
- **Control Inputs**: Vehicle state (speed, acceleration, distance) and traffic light state
- **Control Outputs**: Acceleration commands (Accelerate, Maintain, Decelerate)

#### Vehicle Type
- **Connected Vehicle (connected_car)**
  - Color: Yellow
  - V2X communication capability
  - Maximum speed: 20 m/s
  - Controlled by strategy algorithm in decision zone

### Roundabout Scenario

#### Network Configuration
- **Network File**: `roundabout.net.xml`
- **Signal Control**: `roundabout.tll.xml` with "Green Wave" logic
- **Traffic Flow**: Non-uniform distribution
  - North: High traffic (1 vehicle/45s)
  - East: Medium traffic (1 vehicle/90s)
  - South: Low traffic (1 vehicle/120s)
  - West: Very low traffic (1 vehicle/180s)

#### Vehicle Types

**Cautious Car (Blue)**
- Max Acceleration: 2.2 m/s²
- Max Deceleration: 5.0 m/s²
- Max Speed: 12.0 m/s
- Driver Imperfection: 0.9 (very cautious)
- Time Headway: 1.8 s (larger gap)
- Probability: 20%

**Normal Car (Green)**
- Max Acceleration: 2.8 m/s²
- Max Deceleration: 4.5 m/s²
- Max Speed: 13.89 m/s
- Driver Imperfection: 0.5 (average)
- Time Headway: 1.2 s (standard)
- Probability: 60%

**Aggressive Car (Red)**
- Max Acceleration: 3.5 m/s²
- Max Deceleration: 4.0 m/s²
- Max Speed: 15.0 m/s
- Driver Imperfection: 0.2 (aggressive)
- Time Headway: 1.0 s (smaller gap)
- Probability: 20%

---

## Developer Guide

### Core Concepts

The `algorithms.py` file defines four Abstract Base Classes (ABCs) as interfaces:

1. **ConsensusAlgorithm**: Decision-making and coordination among vehicles
2. **NetworkingProtocol**: V2V or V2I information exchange
3. **DrivingStrategy**: Individual vehicle driving behavior
4. **SchedulingAlgorithm**: Traffic light or flow scheduling

### Implementing Custom Algorithms

#### Step 1: Create Your Algorithm Class

```python
import traci
from algorithms import DrivingStrategy

class MyCustomStrategy(DrivingStrategy):
    def __init__(self):
        print("Custom strategy initialized")
        self.initialized_vehicles = set()
    
    def update(self, step: int):
        # Your algorithm logic here
        vehicle_ids = traci.vehicle.getIDList()
        for veh_id in vehicle_ids:
            # Process each vehicle
            speed = traci.vehicle.getSpeed(veh_id)
            # Apply control logic
            pass
```

#### Step 2: Common TraCI Commands

```python
# Get vehicle information
vehicle_ids = traci.vehicle.getIDList()
speed = traci.vehicle.getSpeed(veh_id)
position = traci.vehicle.getPosition(veh_id)
lane = traci.vehicle.getLaneID(veh_id)

# Control vehicle
traci.vehicle.setSpeed(veh_id, new_speed)
traci.vehicle.setColor(veh_id, (R, G, B, A))

# Traffic light control
phase = traci.trafficlight.getPhase(tls_id)
traci.trafficlight.setPhase(tls_id, new_phase)
```

#### Step 3: Integrate into Runner

```python
# In crossroad_runner.py or ga_crossroad_runner.py
import algorithms

# Instantiate your algorithm
my_strategy = algorithms.MyCustomStrategy()
my_scheduling = algorithms.MyTrafficLightScheduling(junction_id="J0")

# Pass to simulation manager
manager = SimulationManager(
    sumoCmd,
    SIMULATION_STEPS,
    driving_strategy=my_strategy,
    scheduling_algo=my_scheduling
)
manager.run()
```

### Crossroad Strategy Interface

#### Decision Zone Control

Vehicles within 30 meters of intersection are controlled by `compute_strategy` method:

```python
def compute_strategy(self, step: int, vehicle_data: dict, traffic_light_data: dict) -> dict:
    """
    Args:
        step: Current simulation step
        vehicle_data: Dict containing vehicle information
            - id: Vehicle ID
            - dist_to_junction: Distance to stop line (m)
            - speed: Current speed (m/s)
            - acceleration: Current acceleration (m/s²)
            - route: Assigned route ID
        traffic_light_data: Traffic light information
            - phase: Current phase index
            - state: Signal state string
    
    Returns:
        commands: Dict mapping vehicle_id to acceleration command
    """
    commands = {}
    for veh_id, info in vehicle_data.items():
        # Example logic
        if "r" in traffic_light_data['state'] and info['dist_to_junction'] < 10:
            commands[veh_id] = self.ACCEL_NEG  # Brake
        else:
            commands[veh_id] = self.ACCEL_ZERO  # Maintain
    return commands
```

#### Acceleration Commands
- `ACCEL_NEG = -2.0`: Decelerate
- `ACCEL_ZERO = 0.0`: Maintain speed
- `ACCEL_POS = 2.0`: Accelerate

---

## Genetic Algorithm System

### Overview

Uses Genetic Algorithm to optimize vehicle control strategies for intersection crossing.

### Problem Modeling

**Representation**:
- **Individual**: Complete traffic strategy
- **Chromosome**: 18-gene encoding (one per state)
- **Gene**: Acceleration decision (0=decel, 1=maintain, 2=accel)
- **Fitness**: Performance metric (travel time + waiting time + collisions)

**State Space** (18 states total):
- Distance to intersection: 3 ranges [0-10m, 10-20m, 20-30m]
- Vehicle speed: 3 ranges [0-5, 5-15, 15-20 m/s]
- Traffic light: 2 states (red, green)

**Action Space** (3 actions):
| Action | Acceleration | Description |
|--------|-------------|-------------|
| 0 | -2.0 m/s² | Decelerate |
| 1 | 0.0 m/s² | Maintain |
| 2 | +2.0 m/s² | Accelerate |

### Fitness Function

```
Fitness = Avg_Travel_Time + 2 × Avg_Waiting_Time + 100 × Collision_Count
```
Lower fitness is better.

### Genetic Operations

**Selection**:
- Elite preservation: Keep best 2 individuals
- Tournament selection: Select from random groups of 3

**Crossover**:
- Single-point crossover
- Swap chromosome segments between parents

**Mutation**:
- Random gene change with 15% probability
- Maintains population diversity

### Usage

#### Train Strategy
```bash
# Default training (10 generations, 20 population)
python ga_crossroad_runner.py --mode train

# Custom parameters
python ga_crossroad_runner.py --mode train --generations 20 --population 30
```

Output files:
- `best_ga_strategy.pkl`: Best strategy
- `ga_history.pkl`: Evolution history
- `ga_evolution.png`: Evolution curve

#### Test Strategy
```bash
# With GUI
python ga_crossroad_runner.py --mode test

# Without GUI
python ga_crossroad_runner.py --mode test --no-gui
```

#### Combined Training and Testing
```bash
python ga_crossroad_runner.py --mode both --generations 15
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| --mode | Run mode: train/test/both | train |
| --generations | Number of generations | 10 |
| --population | Population size | 20 |
| --strategy-file | Strategy file path | best_ga_strategy.pkl |
| --no-gui | Run without GUI | False |

### Using Trained Strategy

```python
from ga_traffic_strategy import GeneticTrafficStrategy

# Load strategy
strategy = GeneticTrafficStrategy.load("best_ga_strategy.pkl")

# Use strategy
distance = 15.0  # meters
speed = 10.0     # m/s
is_red = True    # red light

# Get acceleration command
accel = strategy.get_action(distance, speed, is_red)
print(f"Recommended acceleration: {accel} m/s^2")
```

### Advanced Features

**Custom Fitness Function**:
```python
def evaluate(self, strategy):
    # Custom fitness calculation
    fitness = (avg_travel_time * 1.0 +
               avg_waiting_time * 3.0 +
               collision_count * 200.0 +
               fuel_consumption * 0.5)
    return fitness
```

**Multi-objective Optimization**:
```python
def pareto_selection(population, fitness_scores):
    # Implement Pareto front selection
    pass
```

**Adaptive Parameters**:
```python
def adaptive_mutation_rate(generation, max_generations):
    return 0.3 * (1 - generation / max_generations) + 0.05
```

---

## SUMO Flow Configuration

### Overview

Vehicle generation is handled directly by SUMO through flow definitions in `crossroad.rou.xml`. This is the recommended method for:
- High performance (native SUMO implementation)
- Simple configuration (XML-based)
- Realistic traffic patterns (built-in randomization)

### Flow Definition

```xml
<flow id="flow_W_E" type="connected_car" route="W_E" 
      begin="0" end="1e9" vehsPerHour="120" 
      departLane="1" departSpeed="max" />
```

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| id | Unique flow identifier | flow_W_E |
| type | Vehicle type | connected_car |
| route | Route ID | W_E |
| begin | Start time (seconds) | 0 |
| end | End time (seconds) | 1e9 (infinite) |
| vehsPerHour | Generation rate | 120 |
| departLane | Specific lane (0/1/2) | 1 |
| departSpeed | Initial speed | max |

### Current Setup

**12 concurrent flows**:
- 4 directions (W, E, N, S)
- 3 lanes per direction
- 120 vehicles/hour per lane
- Total: 1440 vehicles/hour

### Lane Assignment

```
West Approach:
  Lane 0: W→S (Right turn)
  Lane 1: W→E (Straight)
  Lane 2: W→N (Left turn)

East Approach:
  Lane 0: E→N (Right turn)
  Lane 1: E→W (Straight)
  Lane 2: E→S (Left turn)

North Approach:
  Lane 0: N→W (Right turn)
  Lane 1: N→S (Straight)
  Lane 2: N→E (Left turn)

South Approach:
  Lane 0: S→E (Right turn)
  Lane 1: S→N (Straight)
  Lane 2: S→W (Left turn)
```

### Adjusting Traffic Density

Edit `vehsPerHour` in `crossroad.rou.xml`:

```xml
<!-- Light traffic -->
<flow ... vehsPerHour="60" ... />

<!-- Normal traffic (current) -->
<flow ... vehsPerHour="120" ... />

<!-- Heavy traffic -->
<flow ... vehsPerHour="240" ... />

<!-- Very heavy traffic -->
<flow ... vehsPerHour="360" ... />
```

### Key Features

1. **Continuous Generation**
   - Vehicles generate from begin to end time
   - No time limits (end=1e9)
   - Runs as long as simulation runs

2. **Simultaneous Start**
   - All flows begin at time 0
   - Creates concurrent traffic pressure
   - Tests strategy under realistic load

3. **Automatic Randomization**
   - SUMO randomizes exact generation times
   - Maintains average rate
   - Creates realistic traffic patterns

4. **Overflow Handling**
   - SUMO delays generation if lane full
   - No vehicle loss
   - Natural queue formation

### Advanced Configuration

**Different Rates Per Direction**:
```xml
<flow id="flow_W_E" ... vehsPerHour="240" />  <!-- Heavy -->
<flow id="flow_E_W" ... vehsPerHour="60" />   <!-- Light -->
```

**Time-Varying Traffic**:
```xml
<!-- Morning rush -->
<flow id="flow_W_E_morning" route="W_E" begin="0" end="3600" vehsPerHour="300" />

<!-- Off-peak -->
<flow id="flow_W_E_offpeak" route="W_E" begin="3600" end="7200" vehsPerHour="60" />
```

**Random Departure**:
```xml
<flow ... departPos="random" departSpeed="random" />
```

### Verification

```bash
# Check configuration
python test_sumo_flows.py --check-config

# Run test
python test_sumo_flows.py --steps 1500

# Monitor in GUI
python crossroad_runner.py
```

---

## Vehicle Parameters

### Crossroad Vehicles

**Connected Car (Yellow)**
- Type ID: `connected_car`
- Acceleration: 3.0 m/s²
- Deceleration: 4.5 m/s²
- Max Speed: 20 m/s
- Driver Imperfection (sigma): 0 (perfect)
- Min Gap: 2.5 m
- Length: 5 m
- Car Following Model: Krauss
- Lane Change Model: LC2013

### Roundabout Vehicles

**Vehicle Distribution**:
- Cautious: 20%
- Normal: 60%
- Aggressive: 20%

**Detailed Parameters**:

| Parameter | Cautious | Normal | Aggressive |
|-----------|----------|--------|------------|
| Color | Blue | Green | Red |
| Accel (m/s²) | 2.2 | 2.8 | 3.5 |
| Decel (m/s²) | 5.0 | 4.5 | 4.0 |
| Max Speed (m/s) | 12.0 | 13.89 | 15.0 |
| Sigma | 0.9 | 0.5 | 0.2 |
| Time Headway (s) | 1.8 | 1.2 | 1.0 |
| Min Gap (m) | 2.5 | 2.0 | 1.5 |

### Lane Behavior

**Pre-defined Routes**:
- Routes define complete path from start to destination
- Vehicles select appropriate lane based on route
- No manual lane selection needed

**Lane Assignment**:
- Lane 0 (Right): Right turn vehicles
- Lane 1 (Middle): Straight vehicles
- Lane 2 (Left): Left turn vehicles

**Lane Changing Model (LC2013)**:
- Vehicles automatically merge to correct lane
- Efficient lane selection before intersection
- Based on assigned route and turn direction

---

## Customization Guide

### Modify Traffic Flow

1. Open `crossroad.rou.xml`
2. Adjust `vehsPerHour` in flow definitions
3. Changes take effect on next simulation run

```xml
<flow id="flow_W_E" ... vehsPerHour="180" ... />
```

### Modify Vehicle Parameters

1. Open `crossroad.rou.xml` or `roundabout.rou.xml`
2. Edit `<vType>` parameters
3. Adjust acceleration, deceleration, maxSpeed, etc.

```xml
<vType id="connected_car" accel="3.5" decel="5.0" maxSpeed="25" />
```

### Implement New Algorithms

1. Open `algorithms.py`
2. Inherit from appropriate abstract base class
3. Implement `update(self, step)` method
4. Instantiate in runner script

```python
class MyNewAlgorithm(DrivingStrategy):
    def update(self, step: int):
        # Your logic here
        pass
```

### Customize Strategy

1. Locate `compute_strategy` in `algorithms.py`
2. Implement your control logic
3. Use provided vehicle and traffic light data
4. Return acceleration commands

```python
def compute_strategy(self, step, vehicle_data, traffic_light_data):
    commands = {}
    for veh_id, info in vehicle_data.items():
        # Your strategy logic
        commands[veh_id] = self.ACCEL_ZERO
    return commands
```

### Create Custom Scenarios

1. Use NETEDIT to design network
2. Define routes in `.rou.xml`
3. Configure flows for vehicle generation
4. Create runner script following existing examples

---

## Troubleshooting

### SUMO Errors

**Check environment**:
```bash
# Windows
echo %SUMO_HOME%

# Linux/Mac
echo $SUMO_HOME
```

**Verify files**:
```bash
ls crossroad.sumocfg crossroad.net.xml crossroad.rou.xml
```

### Performance Issues

**Too many vehicles**:
- Reduce `vehsPerHour` in flow definitions
- Decrease generation interval

**Simulation too slow**:
- Use `sumo` instead of `sumo-gui`
- Reduce `max_steps` in training
- Decrease population size in GA

### Strategy Problems

**Poor performance**:
- Increase training generations
- Adjust mutation rate
- Optimize fitness function weights

**No convergence**:
- Increase population size
- Adjust selection parameters
- Check fitness function design

---

## Reference

### Documentation
- [SUMO Official Documentation](https://sumo.dlr.de/docs/)
- [TraCI API Reference](https://sumo.dlr.de/docs/TraCI.html)
- [Vehicle Type Parameters](https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html)

### Algorithms
- Holland, J. H. (1992). Adaptation in Natural and Artificial Systems
- Sánchez, J. et al. (2008). "Applying genetic algorithms to traffic signal control"

### Project Files
- `algorithms.py`: Algorithm base classes and implementations
- `ga_traffic_strategy.py`: Genetic algorithm core
- `crossroad_runner.py`: Main simulation runner
- `ga_crossroad_runner.py`: GA-based simulation
- `test_sumo_flows.py`: Flow generation test

---

**Last Updated**: 2025-12-02
**Project Version**: 2.0


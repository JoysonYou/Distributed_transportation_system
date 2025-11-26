# Developer Guide: Implementing and Integrating Custom Algorithms in the Distributed Transportation System Simulation Platform

## Introduction

This project provides a modular framework that allows developers to easily implement and integrate custom traffic algorithms, including Consensus, Networking protocols, Driving Strategies, and Scheduling. This document will guide you on how to use these interfaces to develop your algorithms and seamlessly integrate them into the SUMO simulation.

## Core Concepts

The `algorithms.py` file defines four Abstract Base Classes (ABCs), which serve as the "interfaces" for your algorithms:

-   **`ConsensusAlgorithm`**: Used for implementing decision-making and coordination among groups of vehicles.
-   **`NetworkingProtocol`**: Used for simulating information exchange between vehicles (V2V) or between vehicles and infrastructure (V2I).
-   **`DrivingStrategy`**: Used for defining the driving behavior of individual vehicles (e.g., speed adjustment, lane-changing decisions).
-   **`SchedulingAlgorithm`**: Used for implementing scheduling logic for traffic lights or other traffic flows.

Each abstract base class contains an abstract method `update(self, step: int)`. Your custom algorithm class must inherit from these base classes and implement this `update` method. The `update` method will be called at every time step of the SUMO simulation, where the `step` parameter represents the current simulation step.

## Steps to Implement Custom Consensus and Communication Algorithms

### 1. Create Your Algorithm File

You can add your new class directly in the `algorithms.py` file, or for better organization, create a new Python file (e.g., `my_custom_algorithms.py`) and define your algorithm class there. If you create a new file, make sure to import it correctly in `runner.py`.

### 2. Inherit from Abstract Base Class and Implement `update` Method

Your custom algorithm class must inherit from the corresponding abstract base class and implement its `update` method. The `update` method is the core of your algorithm logic.

**Example: A Simple Consensus Algorithm**

```python
# Assuming this is in algorithms.py or my_custom_algorithms.py

import traci
from abc import ABC, abstractmethod # Import if it's a new file
import random # Used in the example

# Assuming ConsensusAlgorithm and NetworkingProtocol are already defined in algorithms.py
# If it's a new file, you need to import them from algorithms
# from algorithms import ConsensusAlgorithm, NetworkingProtocol

class MyNewConsensusAlgorithm(ConsensusAlgorithm):
    """
    A custom consensus algorithm example.
    Goal: Make all vehicles try to converge to the average speed.
    """
    def __init__(self):
        # Initialize any parameters or state required by your algorithm
        print("[MyNewConsensusAlgorithm] Initialization complete.")
        self.last_consensus_step = -1

    def update(self, step: int):
        # Implement your consensus logic here
        # This method is called at every simulation step

        # Example: Perform consensus calculation every certain number of steps
        if step % 50 == 0 and step != self.last_consensus_step:
            self.last_consensus_step = step
            print(f"[MyNewConsensusAlgorithm] Simulation Step {step}: Starting consensus process...")

            vehicle_ids = traci.vehicle.getIDList()
            if not vehicle_ids:
                # print("  No vehicles in current simulation.")
                return

            # 1. Get current speed of all vehicles
            current_speeds = {veh_id: traci.vehicle.getSpeed(veh_id) for veh_id in vehicle_ids}

            # 2. Calculate average speed (Consensus Target)
            avg_speed = sum(current_speeds.values()) / len(current_speeds)
            print(f"  Average speed of all vehicles: {avg_speed:.2f} m/s")

            # 3. Make each vehicle try to adjust speed to approach average speed
            for veh_id, speed in current_speeds.items():
                # Simple speed adjustment strategy: Adjust a small fraction towards average speed
                # Real consensus algorithms would be more complex, potentially involving communication, voting, leader election, etc.
                adjustment_factor = 0.05 # Adjust 5% of the difference each time
                target_speed = speed + (avg_speed - speed) * adjustment_factor

                # Limit speed within reasonable range (e.g., not exceeding vehicle's max speed)
                max_veh_speed = traci.vehicle.getMaxSpeed(veh_id)
                if target_speed > max_veh_speed:
                    target_speed = max_veh_speed
                elif target_speed < 0: # Speed cannot be negative
                    target_speed = 0

                traci.vehicle.setSpeed(veh_id, target_speed)
                # print(f"  Vehicle {veh_id}: Speed adjusted from {speed:.2f} to {target_speed:.2f}")
            print(f"[MyNewConsensusAlgorithm] Simulation Step {step}: Consensus adjustment complete.")
```

### 3. Use TraCI Interface to Get and Control Simulation Data

In your `update` method, you can use the `traci` library to interact with the SUMO simulation. `traci` provides a rich API to retrieve information about vehicles, traffic lights, networks, etc., and to control their behavior.

**Common `traci` Command Examples:**

*   **Get all vehicle IDs**: `vehicle_ids = traci.vehicle.getIDList()`
*   **Get vehicle speed**: `speed = traci.vehicle.getSpeed(veh_id)`
*   **Set vehicle speed**: `traci.vehicle.setSpeed(veh_id, new_speed)`
*   **Set vehicle color**: `traci.vehicle.setColor(veh_id, (R, G, B, A))`
*   **Get traffic light phase**: `current_phase = traci.trafficlight.getPhase(tls_id)`
*   **Set traffic light phase**: `traci.trafficlight.setPhase(tls_id, new_phase_index)`

For more `traci` APIs, please refer to the SUMO TraCI Documentation.

### 5. Customizing Vehicle Generation (Crossroad Scenario)

The Crossroad scenario now uses a `VehicleGenerator` class in `crossroad_runner.py` to deterministically generate vehicles with specific types and routes. This replaces the random flow generation in the `.rou.xml` file.

**How to Customize:**

1.  Open `crossroad_runner.py`.
2.  Locate the `VehicleGenerator` class.
3.  Modify the `self.generation_sequence` list in the `__init__` method. This list defines the sequence of (Vehicle Type, Route ID) pairs that will be generated.
4.  You can also adjust `self.step_interval` to change the frequency of vehicle generation.

```python
self.generation_sequence = [
# Developer Guide: Implementing and Integrating Custom Algorithms in the Distributed Transportation System Simulation Platform

## Introduction

This project provides a modular framework that allows developers to easily implement and integrate custom traffic algorithms, including Consensus, Networking protocols, Driving Strategies, and Scheduling. This document will guide you on how to use these interfaces to develop your algorithms and seamlessly integrate them into the SUMO simulation.

## Core Concepts

The `algorithms.py` file defines four Abstract Base Classes (ABCs), which serve as the "interfaces" for your algorithms:

-   **`ConsensusAlgorithm`**: Used for implementing decision-making and coordination among groups of vehicles.
-   **`NetworkingProtocol`**: Used for simulating information exchange between vehicles (V2V) or between vehicles and infrastructure (V2I).
-   **`DrivingStrategy`**: Used for defining the driving behavior of individual vehicles (e.g., speed adjustment, lane-changing decisions).
-   **`SchedulingAlgorithm`**: Used for implementing scheduling logic for traffic lights or other traffic flows.

Each abstract base class contains an abstract method `update(self, step: int)`. Your custom algorithm class must inherit from these base classes and implement this `update` method. The `update` method will be called at every time step of the SUMO simulation, where the `step` parameter represents the current simulation step.

## Steps to Implement Custom Consensus and Communication Algorithms

### 1. Create Your Algorithm File

You can add your new class directly in the `algorithms.py` file, or for better organization, create a new Python file (e.g., `my_custom_algorithms.py`) and define your algorithm class there. If you create a new file, make sure to import it correctly in `runner.py`.

### 2. Inherit from Abstract Base Class and Implement `update` Method

Your custom algorithm class must inherit from the corresponding abstract base class and implement its `update` method. The `update` method is the core of your algorithm logic.

**Example: A Simple Consensus Algorithm**

```python
# Assuming this is in algorithms.py or my_custom_algorithms.py

import traci
from abc import ABC, abstractmethod # Import if it's a new file
import random # Used in the example

# Assuming ConsensusAlgorithm and NetworkingProtocol are already defined in algorithms.py
# If it's a new file, you need to import them from algorithms
# from algorithms import ConsensusAlgorithm, NetworkingProtocol

class MyNewConsensusAlgorithm(ConsensusAlgorithm):
    """
    A custom consensus algorithm example.
    Goal: Make all vehicles try to converge to the average speed.
    """
    def __init__(self):
        # Initialize any parameters or state required by your algorithm
        print("[MyNewConsensusAlgorithm] Initialization complete.")
        self.last_consensus_step = -1

    def update(self, step: int):
        # Implement your consensus logic here
        # This method is called at every simulation step

        # Example: Perform consensus calculation every certain number of steps
        if step % 50 == 0 and step != self.last_consensus_step:
            self.last_consensus_step = step
            print(f"[MyNewConsensusAlgorithm] Simulation Step {step}: Starting consensus process...")

            vehicle_ids = traci.vehicle.getIDList()
            if not vehicle_ids:
                # print("  No vehicles in current simulation.")
                return

            # 1. Get current speed of all vehicles
            current_speeds = {veh_id: traci.vehicle.getSpeed(veh_id) for veh_id in vehicle_ids}

            # 2. Calculate average speed (Consensus Target)
            avg_speed = sum(current_speeds.values()) / len(current_speeds)
            print(f"  Average speed of all vehicles: {avg_speed:.2f} m/s")

            # 3. Make each vehicle try to adjust speed to approach average speed
            for veh_id, speed in current_speeds.items():
                # Simple speed adjustment strategy: Adjust a small fraction towards average speed
                # Real consensus algorithms would be more complex, potentially involving communication, voting, leader election, etc.
                adjustment_factor = 0.05 # Adjust 5% of the difference each time
                target_speed = speed + (avg_speed - speed) * adjustment_factor

                # Limit speed within reasonable range (e.g., not exceeding vehicle's max speed)
                max_veh_speed = traci.vehicle.getMaxSpeed(veh_id)
                if target_speed > max_veh_speed:
                    target_speed = max_veh_speed
                elif target_speed < 0: # Speed cannot be negative
                    target_speed = 0

                traci.vehicle.setSpeed(veh_id, target_speed)
                # print(f"  Vehicle {veh_id}: Speed adjusted from {speed:.2f} to {target_speed:.2f}")
            print(f"[MyNewConsensusAlgorithm] Simulation Step {step}: Consensus adjustment complete.")
```

### 3. Use TraCI Interface to Get and Control Simulation Data

In your `update` method, you can use the `traci` library to interact with the SUMO simulation. `traci` provides a rich API to retrieve information about vehicles, traffic lights, networks, etc., and to control their behavior.

**Common `traci` Command Examples:**

*   **Get all vehicle IDs**: `vehicle_ids = traci.vehicle.getIDList()`
*   **Get vehicle speed**: `speed = traci.vehicle.getSpeed(veh_id)`
*   **Set vehicle speed**: `traci.vehicle.setSpeed(veh_id, new_speed)`
*   **Set vehicle color**: `traci.vehicle.setColor(veh_id, (R, G, B, A))`
*   **Get traffic light phase**: `current_phase = traci.trafficlight.getPhase(tls_id)`
*   **Set traffic light phase**: `traci.trafficlight.setPhase(tls_id, new_phase_index)`

For more `traci` APIs, please refer to the SUMO TraCI Documentation.

### 5. Customizing Vehicle Generation (Crossroad Scenario)

The Crossroad scenario now uses a `VehicleGenerator` class in `crossroad_runner.py` to deterministically generate vehicles with specific types and routes. This replaces the random flow generation in the `.rou.xml` file.

**How to Customize:**

11. Open `crossroad_runner.py`.
12. Locate the `VehicleGenerator` class.
13. Modify the `self.generation_sequence` list in the `__init__` method. This list defines the sequence of (Vehicle Type, Route ID) pairs that will be generated.
14. You can also adjust `self.step_interval` to change the frequency of vehicle generation.

```python
self.generation_sequence = [
    ("type_straight", "W_E"), # West to East, Straight
    ("type_left", "N_E"),     # North to East, Left Turn
    # ... add your custom sequence here
]
```

### 6. Implementing Crossroad Strategy

The Crossroad scenario now features a dedicated interface for implementing vehicle control strategies, specifically for vehicles approaching the intersection.

**Key Concepts:**

*   **Decision Zone**: A 30-meter zone before the intersection stop line. Vehicles entering this zone are automatically switched to manual control mode.
*   **Strategy Interface**: The `compute_strategy` method in `MyDrivingStrategy` (in `algorithms.py`) is the entry point for your custom logic.
*   **Sampling Rate**: The strategy is executed every 0.2 seconds (2 simulation steps).

**How to Implement:**

1.  Open `algorithms.py` and locate the `MyDrivingStrategy` class.
2.  Find the `compute_strategy` method.
3.  Implement your logic using the provided arguments:
    *   `step`: Current simulation step.
    *   `vehicle_data`: A dictionary containing real-time data for all vehicles in the decision zone.
        *   `id`: Vehicle ID.
        *   `dist_to_junction`: Distance to the stop line (m).
        *   `speed`: Current speed (m/s).
        *   `acceleration`: Current acceleration (m/s^2).
        *   `route`: Assigned route ID.
    *   `traffic_light_data`: A dictionary containing the current state of the intersection's traffic lights.
        *   `phase`: Current phase index.
        *   `state`: Current signal state string (e.g., "GrGr").

**Output:**

Your method must return a dictionary where:
*   **Key**: Vehicle ID.
*   **Value**: Acceleration command (float).
    *   `self.ACCEL_POS` (2.0): Accelerate.
    *   `self.ACCEL_ZERO` (0.0): Maintain speed.
    *   `self.ACCEL_NEG` (-2.0): Decelerate.

**Example Logic:**

```python
def compute_strategy(self, step, vehicle_data, traffic_light_data):
    commands = {}
    for veh_id, info in vehicle_data.items():
        # Simple Red Light Stop Logic
        if "r" in traffic_light_data['state'] and info['dist_to_junction'] < 10:
             commands[veh_id] = self.ACCEL_NEG # Brake
        else:
             commands[veh_id] = self.ACCEL_ZERO # Cruise
    return commands
```

### 4. Integrate Your Algorithm into the Simulation Runner

Once you have implemented your custom algorithm class, you need to modify `crossroad_runner.py` or `roundabout_runner.py` to instantiate and use them.

**Steps to Modify `runner.py`:**

1.  **Import your algorithm class**:
    Add the import statement at the top of the `runner.py` file.
    ```python
    import algorithms
    ```

2.  **Instantiate your algorithm**:
    In the `if __name__ == "__main__":` block, create an instance of your algorithm class.

    ```python
    my_custom_consensus = algorithms.MyNewConsensusAlgorithm()
    my_custom_networking = algorithms.MyNetworking()
    ```

3.  **Pass the algorithm instance to `SimulationManager`**:
    Modify the instantiation of `SimulationManager` to pass your algorithm instance as an argument.

    ```python
    manager = SimulationManager(
        sumoCmd,
        SIMULATION_STEPS,
        consensus_algo=my_custom_consensus,
        networking_proto=my_custom_networking
    )
    ```
# Vehicle Documentation

This document provides a detailed explanation of the vehicle types and their behavior in the SUMO simulation.


## Crossroad Simulation

## Vehicle Types

There are two types of vehicles in this simulation:

*   **Autonomous Car (Blue)**: Represents an autonomous or self-driving vehicle.
*   **Standard Car (Red)**: Represents a standard vehicle driven by a human.

---

## Vehicle Parameters

The base physical and behavioral parameters for each vehicle type are defined in `test_1.rou.xml`.

### Blue Car (autonomous_car)

*   **Type**: Autonomous Vehicle
*   **Color**: Blue
*   **Max Acceleration (`accel`)**: 3.5 m/s²
*   **Max Deceleration (`decel`)**: 5.5 m/s²
*   **Max Speed (`maxSpeed`)**: 25 m/s (90 km/h)
*   **Car Following Model**: `IDM` (Intelligent Driver Model) - A sophisticated model that provides smooth and efficient following behavior.
*   **Driver Imperfection (`sigma`)**: 0.2 - A low value indicating a near-perfect driver with quick and consistent reactions.

### Red Car (standard_car)

*   **Type**: Standard Human-Driven Vehicle
*   **Color**: Red
*   **Max Acceleration (`accel`)**: 2.5 m/s² (Slower than the autonomous car)
*   **Max Deceleration (`decel`)**: 4.5 m/s² (Slower than the autonomous car)
*   **Max Speed (`maxSpeed`)**: 20 m/s (72 km/h) (Slower than the autonomous car)
*   **Car Following Model**: `Krauss` - A simpler model often used to simulate human driving.
*   **Driver Imperfection (`sigma`)**: 0.5 - A higher value representing a less perfect human driver with more variability in reaction time.

---

## Driving Behavior

The driving behavior is a combination of the base parameters and real-time modifications from `algorithms.py`.

### Blue Car (Autonomous)

The blue autonomous cars are not subject to any real-time modifications in the `MyDrivingStrategy`. They operate based on their pre-defined "perfect" parameters, resulting in consistent and efficient driving.

### Red Car (Human-Driven)

The red standard cars are actively influenced by the `MyDrivingStrategy` in `algorithms.py` to simulate human-like imperfections:

*   **Speed Variation**: To mimic a human driver's fluctuating attention or intent, the car's desired speed is periodically multiplied by a random factor between 0.9 and 1.1. This causes the car to occasionally drive slightly faster or slower than the "optimal" speed.

---

## Turning and Routing

A vehicle's path and turning decisions are governed by two main components:

### 1. Pre-defined Routes

*   **What it is**: The overall path a vehicle will take from its start to its destination.
*   **How it works**: In `test_1.rou.xml`, `<route>` elements define specific paths (e.g., from West to North). Each vehicle is assigned a route when it is created. When approaching an intersection, the vehicle already knows which direction to turn based on its assigned route.

### 2. Lane Changing Model

*   **What it is**: The logic a vehicle uses to decide when to change lanes on a multi-lane road.
*   **How it works**: Both vehicle types use the `LC2013` model. This model allows vehicles to change lanes for two main reasons:
    1.  **Strategic**: To overtake a slower vehicle and achieve a higher speed.
    2.  **Cooperative/Tactical**: To get into the correct lane for an upcoming turn that is dictated by its pre-defined route.



## Roundabout Simulation

In the roundabout simulation, there are three types of vehicles, each with distinct driving behaviors to simulate a more realistic traffic flow. The vehicle types are defined in `roundabout.rou.xml`.

### Cautious Car (Blue)

*   **Type**: Cautious Driver
*   **Color**: Blue
*   **Max Acceleration (`accel`)**: 2.2 m/s²
*   **Max Deceleration (`decel`)**: 5.0 m/s²
*   **Max Speed (`maxSpeed`)**: 12.0 m/s (43.2 km/h)
*   **Car Following Model**: `IDM` (Intelligent Driver Model)
*   **Driver Imperfection (`sigma`)**: 0.9 - A high value indicating a very cautious and hesitant driver.
*   **Desired Time Headway (`tau`)**: 1.8 s - A larger headway, meaning the driver keeps a greater distance from the car in front.
*   **Minimum Gap (`minGap`)**: 2.5 m - The minimum distance kept from the car in front.

### Normal Car (Green)

*   **Type**: Normal Driver
*   **Color**: Green
*   **Max Acceleration (`accel`)**: 2.8 m/s²
*   **Max Deceleration (`decel`)**: 4.5 m/s²
*   **Max Speed (`maxSpeed`)**: 13.89 m/s (50 km/h)
*   **Car Following Model**: `IDM` (Intelligent Driver Model)
*   **Driver Imperfection (`sigma`)**: 0.5 - An average value representing a typical driver.
*   **Desired Time Headway (`tau`)**: 1.2 s - A standard time headway.
*   **Minimum Gap (`minGap`)**: 2.0 m - A standard minimum distance.

### Aggressive Car (Red)

*   **Type**: Aggressive Driver
*   **Color**: Red
*   **Max Acceleration (`accel`)**: 3.5 m/s²
*   **Max Deceleration (`decel`)**: 4.0 m/s²
*   **Max Speed (`maxSpeed`)**: 15.0 m/s (54 km/h)
*   **Car Following Model**: `IDM` (Intelligent Driver Model)
*   **Driver Imperfection (`sigma`)**: 0.2 - A low value indicating a very confident and aggressive driver.
*   **Desired Time Headway (`tau`)**: 1.0 s - A smaller headway, meaning the driver follows cars more closely.
*   **Minimum Gap (`minGap`)**: 1.5 m - A smaller minimum distance.

### Vehicle Distribution

Instead of fixed flows for each vehicle type, the roundabout simulation uses a `vTypeDistribution` with the following probabilities:

*   **Cautious Car**: 20%
*   **Normal Car**: 60%
*   **Aggressive Car**: 20%

This creates a more varied and realistic mix of traffic in the simulation.
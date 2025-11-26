### 1. Pre-defined Routes

*   **What it is**: The overall path a vehicle will take from its start to its destination.
*   **How it works**: In `crossroad.rou.xml`, `<route>` elements define specific paths (e.g., from West to North). Each vehicle is assigned a route when it is created. When approaching an intersection, the vehicle already knows which direction to turn based on its assigned route.

### 2. Lane Changing Model

*   **What it is**: The logic a vehicle uses to decide when to change lanes on a multi-lane road.
*   **How it works**: The intersection is configured with **dedicated lanes** for each movement:
    *   **Lane 0 (Rightmost)**: Dedicated for **Right Turns**.
    *   **Lane 1 (Middle)**: Dedicated for **Straight** traffic.
    *   **Lane 2 (Leftmost)**: Dedicated for **Left Turns**.
    
    Vehicles will automatically select the appropriate lane based on their assigned route and turn direction well before reaching the intersection. The `LC2013` model ensures they merge into the correct lane efficiently.

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

### Traffic Flow Configuration

The simulation is configured with non-uniform traffic flows to simulate realistic traffic conditions:

*   **North Input**: High traffic volume (1 vehicle every 45s).
*   **East Input**: Medium traffic volume (1 vehicle every 90s).
*   **South Input**: Low traffic volume (1 vehicle every 120s).
*   **West Input**: Very low traffic volume (1 vehicle every 180s).

This creates a more varied and realistic mix of traffic in the simulation.
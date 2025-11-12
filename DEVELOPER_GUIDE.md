# 开发者指南：在分布式交通系统仿真平台中实现和集成自定义算法

## 引言

本项目提供了一个模块化的框架，允许开发者轻松地实现和集成自定义的交通算法，包括车辆间的共识（Consensus）、通信协议（Networking）、驾驶策略（Driving Strategy）和交通调度（Scheduling）。本文档将指导你如何利用这些接口来开发你的算法，并将其无缝集成到SUMO仿真中。

## 核心概念

`algorithms.py` 文件定义了四个抽象基类（Abstract Base Class, ABC），它们是你的算法的“接口”：

-   **`ConsensusAlgorithm`**: 用于实现车辆群体间的决策和协调。
-   **`NetworkingProtocol`**: 用于模拟车辆之间（V2V）或车辆与基础设施之间（V2I）的信息交换。
-   **`DrivingStrategy`**: 用于定义单个车辆的驾驶行为（如速度调整、变道决策）。
-   **`SchedulingAlgorithm`**: 用于实现交通信号灯或其他交通流的调度逻辑。

每个抽象基类都包含一个抽象方法 `update(self, step: int)`。你的自定义算法类必须继承这些基类，并实现这个 `update` 方法。`update` 方法会在SUMO仿真的每个时间步长被调用，`step` 参数表示当前的仿真步数。

## 实现自定义共识和通信算法的步骤

### 1. 创建你的算法文件

你可以在 `algorithms.py` 文件中直接添加你的新类，或者为了更好的组织，创建一个新的Python文件（例如 `my_custom_algorithms.py`），并在其中定义你的算法类。如果创建新文件，请确保在 `runner.py` 中正确导入。

### 2. 继承抽象基类并实现 `update` 方法

你的自定义算法类必须继承自相应的抽象基类，并实现其 `update` 方法。`update` 方法是你的算法逻辑的核心。

**示例：一个简单的共识算法**

```python
# 假设这是在 algorithms.py 或 my_custom_algorithms.py 中

import traci
from abc import ABC, abstractmethod # 如果是新文件，需要导入
import random # 示例中可能用到

# 假设 ConsensusAlgorithm 和 NetworkingProtocol 已经定义在 algorithms.py 中
# 如果是新文件，需要从 algorithms 导入它们
# from algorithms import ConsensusAlgorithm, NetworkingProtocol

class MyNewConsensusAlgorithm(ConsensusAlgorithm):
    """
    一个自定义的共识算法示例。
    目标：让所有车辆尝试向平均速度靠拢。
    """
    def __init__(self):
        # 初始化你的算法所需的任何参数或状态
        print("[MyNewConsensusAlgorithm] 初始化完成。")
        self.last_consensus_step = -1

    def update(self, step: int):
        # 在这里实现你的共识逻辑
        # 这个方法会在每个仿真步长被调用

        # 示例：每隔一定步数执行一次共识计算
        if step % 50 == 0 and step != self.last_consensus_step:
            self.last_consensus_step = step
            print(f"[MyNewConsensusAlgorithm] 仿真步长 {step}: 启动共识过程...")

            vehicle_ids = traci.vehicle.getIDList()
            if not vehicle_ids:
                # print("  当前仿真中没有车辆。")
                return

            # 1. 获取所有车辆的当前速度
            current_speeds = {veh_id: traci.vehicle.getSpeed(veh_id) for veh_id in vehicle_ids}

            # 2. 计算平均速度 (共识目标)
            avg_speed = sum(current_speeds.values()) / len(current_speeds)
            print(f"  当前所有车辆的平均速度: {avg_speed:.2f} m/s")

            # 3. 让每辆车尝试调整速度以接近平均速度
            for veh_id, speed in current_speeds.items():
                # 简单的速度调整策略：向平均速度调整一小部分
                # 实际共识算法会更复杂，可能涉及通信、投票、领导者选举等
                adjustment_factor = 0.05 # 每次调整5%的差值
                target_speed = speed + (avg_speed - speed) * adjustment_factor

                # 限制速度在合理范围内 (例如，不超过车辆的最大速度)
                max_veh_speed = traci.vehicle.getMaxSpeed(veh_id)
                if target_speed > max_veh_speed:
                    target_speed = max_veh_speed
                elif target_speed < 0: # 速度不能为负
                    target_speed = 0

                traci.vehicle.setSpeed(veh_id, target_speed)
                # print(f"  车辆 {veh_id}: 速度从 {speed:.2f} 调整到 {target_speed:.2f}")
            print(f"[MyNewConsensusAlgorithm] 仿真步长 {step}: 共识调整完成。")
```

### 3. 使用 TraCI 接口获取和控制仿真数据

在你的 `update` 方法中，你可以使用 `traci` 库来与SUMO仿真进行交互。`traci` 提供了丰富的API来获取车辆、交通灯、路网等信息，并可以控制它们的行为。

**常用 `traci` 命令示例:**

*   **获取所有车辆ID**: `vehicle_ids = traci.vehicle.getIDList()`
*   **获取车辆速度**: `speed = traci.vehicle.getSpeed(veh_id)`
*   **设置车辆速度**: `traci.vehicle.setSpeed(veh_id, new_speed)`
*   **设置车辆颜色**: `traci.vehicle.setColor(veh_id, (R, G, B, A))`
*   **获取交通灯相位**: `current_phase = traci.trafficlight.getPhase(tls_id)`
*   **设置交通灯相位**: `traci.trafficlight.setPhase(tls_id, new_phase_index)`

更多 `traci` API 请参考 SUMO TraCI 文档。

### 4. 将你的算法集成到仿真运行器中

一旦你实现了你的自定义算法类，你需要修改 `crossroad_runner.py` 或 `roundabout_runner.py` 来实例化并使用它们。

**修改 `runner.py` 文件的步骤:**

1.  **导入你的算法类**:
    在 `runner.py` 文件的顶部，添加导入语句。
    ```python
    import algorithms
    ```

2.  **实例化你的算法**:
    在 `if __name__ == "__main__":` 块中，创建你的算法类的实例。

    ```python
    my_custom_consensus = algorithms.MyNewConsensusAlgorithm()
    my_custom_networking = algorithms.MyNetworking()
    ```

3.  **将算法实例传递给 `SimulationManager`**:
    修改 `SimulationManager` 的实例化，将你的算法实例作为参数传递进去。

    ```python
    manager = SimulationManager(
        sumoCmd,
        SIMULATION_STEPS,
        consensus_algo=my_custom_consensus,
        networking_proto=my_custom_networking
    )
    ```
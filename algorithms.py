from abc import ABC, abstractmethod
import traci
import random

class ConsensusAlgorithm(ABC):
    """
    共识算法的抽象基类。
    用于实现车辆间的协调和决策。
    """
    @abstractmethod
    def update(self, step: int):
        """在每个仿真步长调用此方法。"""
        pass

class NetworkingProtocol(ABC):
    """
    网络协议的抽象基类。
    用于模拟车辆之间或车辆与基础设施之间的通信。
    """
    @abstractmethod
    def update(self, step: int):
        """在每个仿真步长调用此方法。"""
        pass

class DrivingStrategy(ABC):
    """
    驾驶策略的抽象基类。
    用于定义单个车辆的驾驶行为。
    """
    @abstractmethod
    def update(self, step: int):
        """在每个仿真步长调用此方法。"""
        pass

class SchedulingAlgorithm(ABC):
    """
    调度算法的抽象基类。
    用于实现交通信号灯调度或车辆路径规划。
    """
    @abstractmethod
    def update(self, step: int):
        """在每个仿真步长调用此方法。"""
        pass

# --- 算法实现示例 ---
# 您可以在下面创建自己的算法实现，并替换runner.py中的占位符。

class MyConsensus(ConsensusAlgorithm):
    """一个简单的共识算法示例。"""
    def update(self, step: int):
        if step % 10 == 0:
            # print(f"[Consensus] Step {step}: Re-evaluating consensus...")
            pass

class MyNetworking(NetworkingProtocol):
    """一个简单的网络协议示例。"""
    def update(self, step: int):
        # vehicle_ids = traci.vehicle.getIDList()
        # for veh_id in vehicle_ids:
        #     # 模拟发送和接收消息
        #     pass
        pass

import random

class MyDrivingStrategy(DrivingStrategy):
    """
    一个驾驶策略示例，用于演示如何根据车辆类型改变其行为和外观。
    """
    def __init__(self):
        self.initialized_vehicles = set()

    def update(self, step: int):
        """
        在每个仿真步长调用此方法。
        - 为新出现的车辆根据其类型设置颜色。
        - 对标准汽车的速度进行微小扰动，模拟人类驾驶行为。
        """
        vehicle_ids = traci.vehicle.getIDList()
        for veh_id in vehicle_ids:
            # 当车辆首次出现时，根据类型设置颜色
            if veh_id not in self.initialized_vehicles:
                veh_type = traci.vehicle.getTypeID(veh_id)
                if veh_type == "autonomous_car":
                    traci.vehicle.setColor(veh_id, (0, 0, 255, 255))  # 蓝色
                elif veh_type == "standard_car":
                    traci.vehicle.setColor(veh_id, (255, 0, 0, 255))  # 红色
                self.initialized_vehicles.add(veh_id)

            # 对标准汽车的速度进行微小扰动
            if traci.vehicle.getTypeID(veh_id) == "standard_car":
                # 使用setSpeedFactor来调整速度，避免车辆卡在0速
                # 我们只在新车辆出现时或以一定概率调整，以减少计算开销
                if veh_id not in self.initialized_vehicles or random.random() < 0.1:
                    random_factor = random.uniform(0.9, 1.1)
                    traci.vehicle.setSpeedFactor(veh_id, random_factor)

class MyTrafficLightScheduling(SchedulingAlgorithm):
    """一个简单的交通灯调度示例。"""
    def __init__(self, junction_id: str):
        self.junction_id = junction_id
        self.phase_index = 0
        self._phases = None

    @property
    def phases(self):
        if self._phases is None:
            # Lazy load phases once traci is connected
            try:
                self._phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.junction_id)[0].phases
            except traci.TraCIException:
                # Traci not connected yet
                pass
        return self._phases

    def update(self, step: int):
        if self.phases is None:
            return
            
        # 每30步切换一次交通灯相位
        if step > 0 and step % 30 == 0:
            self.phase_index = (self.phase_index + 1) % len(self.phases)
            traci.trafficlight.setPhase(self.junction_id, self.phase_index)
            # print(f"[Scheduling] Step {step}: Switched TLS '{self.junction_id}' to phase {self.phase_index}")
        pass

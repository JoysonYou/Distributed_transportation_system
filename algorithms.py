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
    驾驶策略。
    定义了三种固定的加速度挡位供策略选择。
    """
    # 加速度挡位 (m/s^2)
    ACCEL_POS = 2.0   # 加速
    ACCEL_ZERO = 0.0  # 匀速
    ACCEL_NEG = -2.0  # 减速
    
    # 速度限制 (m/s)
    MAX_SPEED = 20.0
    MIN_SPEED = 0.0

    def __init__(self):
        self.initialized_vehicles = set()

    def update(self, step: int):
        """
        在每个仿真步长调用此方法。
        """
        # 1. 采样间隔检查 (0.2s = 2 steps)
        if step % 2 != 0:
            return

        # 2. 获取红绿灯信息
        junction_id = "J0"
        try:
            tl_state = traci.trafficlight.getRedYellowGreenState(junction_id)
            tl_phase = traci.trafficlight.getPhase(junction_id)
        except traci.TraCIException:
            tl_state = ""
            tl_phase = -1
        
        tl_info = {
            "id": junction_id,
            "state": tl_state,
            "phase": tl_phase
        }

        # 3. 获取车辆信息并分类
        vehicle_ids = traci.vehicle.getIDList()
        decision_vehicles = {}
        
        for veh_id in vehicle_ids:
            if veh_id not in self.initialized_vehicles:
                traci.vehicle.setColor(veh_id, (255, 255, 0, 255)) # 黄色
                self.initialized_vehicles.add(veh_id)

            try:
                lane_id = traci.vehicle.getLaneID(veh_id)
                edge_id = traci.vehicle.getRoadID(veh_id)
                
                # 只处理进入路口的车辆 (edge以_in结尾)
                if edge_id.endswith("_in"):
                    lane_len = traci.lane.getLength(lane_id)
                    lane_pos = traci.vehicle.getLanePosition(veh_id)
                    dist_to_junction = lane_len - lane_pos
                    
                    if dist_to_junction <= 30.0:
                        # --- 进入决策区域 (<= 30m) ---
                        # 收集信息传给策略
                        veh_info = {
                            "id": veh_id,
                            "speed": traci.vehicle.getSpeed(veh_id),
                            "acceleration": traci.vehicle.getAcceleration(veh_id),
                            "route": traci.vehicle.getRouteID(veh_id),
                            "dist_to_junction": dist_to_junction,
                            "lane_id": lane_id
                        }
                        decision_vehicles[veh_id] = veh_info
                        
                        # 设置为手动控制模式 (SpeedMode 0), 允许完全控制加速度
                        traci.vehicle.setSpeedMode(veh_id, 0)
                    else:
                        # --- 决策区域外 (> 30m) ---
                        # 保持最大速度 (恢复默认SpeedMode或设置为最大速度)
                        # 这里我们使用默认的CarFollowing模型，但请求最大速度
                        traci.vehicle.setSpeedMode(veh_id, 31) # 恢复默认行为
                        traci.vehicle.setSpeed(veh_id, self.MAX_SPEED)
                else:
                    # --- 已经在路口内或离开路口 ---
                    # 恢复默认行为
                    traci.vehicle.setSpeedMode(veh_id, 31)

            except traci.TraCIException:
                continue

        # 4. 调用策略接口并应用控制
        if decision_vehicles:
            commands = self.compute_strategy(step, decision_vehicles, tl_info)
            
            for veh_id, accel_cmd in commands.items():
                if veh_id in decision_vehicles:
                    # 应用加速度控制，持续时间为采样间隔 (0.2s)
                    traci.vehicle.setAcceleration(veh_id, accel_cmd, duration=0.2)

    def compute_strategy(self, step: int, vehicle_data: dict, traffic_light_data: dict) -> dict:
        """
        策略算法接口。
        根据车辆状态和红绿灯信息，计算每辆车的加速度控制指令。
        
        Args:
            step: 当前仿真步数
            vehicle_data: 包含决策区域内车辆信息的字典。
            traffic_light_data: 包含红绿灯信息的字典。
                                
        Returns:
            commands: 包含每辆车加速度指令的字典。
        """
        commands = {}
        
        # --- 实时数据打印 (用于验证) ---
        print(f"\n[Strategy] Step {step} | TL Phase: {traffic_light_data['phase']} | State: {traffic_light_data['state']}")
        print(f"{'Vehicle ID':<15} | {'Dist (m)':<10} | {'Speed (m/s)':<12} | {'Accel (m/s^2)':<15} | {'Command':<10}")
        print("-" * 70)

        # TODO: 其他开发者将在此处实现具体的策略逻辑
        # 目前作为占位符，所有车辆保持匀速
        for veh_id, info in vehicle_data.items():
            # 简单的示例逻辑：如果红灯且距离近，减速；否则匀速
            # 注意：这只是为了演示数据流，并非真正的策略
            command = self.ACCEL_ZERO
            
            commands[veh_id] = command
            
            # 打印每辆车的数据和决策
            print(f"{veh_id:<15} | {info['dist_to_junction']:<10.2f} | {info['speed']:<12.2f} | {info['acceleration']:<15.2f} | {command:<10}")
            
        return commands

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

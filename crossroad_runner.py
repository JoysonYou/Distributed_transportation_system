import os
import sys
import traci

import algorithms # 导入算法模块
# --- SUMO环境配置 ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("请设置环境变量 'SUMO_HOME'")

class VehicleGenerator:
    """
    固定路线车辆生成器。
    用于替代随机生成的车流，按照固定的顺序和时间间隔生成车辆。
    每个车型（直行、左转、右转）都有其固定的行驶路线模式。
    """
    def __init__(self, step_interval=40):
        self.step_interval = step_interval  # 生成间隔 (步数)
        self.vehicle_count = 0
        
        # 定义生成序列: (车型ID, 路线ID)
        # 确保覆盖所有方向和所有行为
        self.generation_sequence = [
            # 组1: 主要是向东行驶
            ("type_straight", "W_E"), # 西->东 (直行)
            ("type_left", "N_E"),     # 北->东 (左转)
            ("type_right", "S_E"),    # 南->东 (右转)
            
            # 组2: 主要是向西行驶
            ("type_straight", "E_W"), # 东->西 (直行)
            ("type_left", "S_W"),     # 南->西 (左转)
            ("type_right", "N_W"),    # 北->西 (右转)
            
            # 组3: 主要是向南行驶
            ("type_straight", "N_S"), # 北->南 (直行)
            ("type_left", "E_S"),     # 东->南 (左转)
            ("type_right", "W_S"),    # 西->南 (右转)
            
            # 组4: 主要是向北行驶
            ("type_straight", "S_N"), # 南->北 (直行)
            ("type_left", "W_N"),     # 西->北 (左转)
            ("type_right", "E_N"),    # 东->北 (右转)
        ]
        self.sequence_index = 0

    def update(self, step):
        """在每个仿真步长调用，检查是否需要生成车辆。"""
        if step % self.step_interval == 0:
            type_id, route_id = self.generation_sequence[self.sequence_index]
            veh_id = f"fixed_veh_{self.vehicle_count}"
            
            try:
                traci.vehicle.add(veh_id, route_id, typeID=type_id)
                # print(f"Generated {type_id} vehicle {veh_id} on route {route_id} at step {step}")
            except traci.TraCIException as e:
                print(f"Error adding vehicle {veh_id}: {e}")

            self.sequence_index = (self.sequence_index + 1) % len(self.generation_sequence)
            self.vehicle_count += 1

class SimulationManager:
    """
    仿真管理器，用于设置和运行SUMO仿真。
    """
    def __init__(self, sumo_cmd, max_steps,
                 driving_strategy=None,
                 scheduling_algo=None,
                 consensus_algo=None,
                 networking_proto=None,
                 vehicle_generator=None):
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.driving_strategy = driving_strategy
        self.scheduling_algo = scheduling_algo
        self.consensus_algo = consensus_algo
        self.networking_proto = networking_proto
        self.vehicle_generator = vehicle_generator

    def run(self):
        """执行TraCI控制循环"""
        traci.start(self.sumo_cmd)
        step = 0
        while step < self.max_steps:
            traci.simulationStep()

            # 调用各种算法的update方法
            if self.driving_strategy:
                self.driving_strategy.update(step)
            if self.scheduling_algo:
                self.scheduling_algo.update(step)
            if self.networking_proto:
                self.networking_proto.update(step)
            if self.consensus_algo:
                self.consensus_algo.update(step)
            if self.vehicle_generator:
                self.vehicle_generator.update(step)

            step += 1
        try:
            while step < self.max_steps:
                traci.simulationStep()
                # ... (此处省略了算法调用，与上面相同) ...
                step += 1
        except traci.exceptions.TraCIException as e:
            print(f"TraCI error during simulation: {e}")
        finally:
            traci.close()

if __name__ == "__main__":
    # --- 仿真参数 ---
    sumoBinary = "sumo-gui"  # 使用 "sumo-gui" 或 "sumo"
    sumoCmd = [sumoBinary, "-c", "crossroad.sumocfg"]
    SIMULATION_STEPS = 5000

    # --- 实例化算法 ---
    # 根据crossroad.net.xml，交通灯ID是"J0"
    my_driving_strategy = algorithms.MyDrivingStrategy()
    my_traffic_light_scheduling = algorithms.MyTrafficLightScheduling(junction_id="J0")
    my_consensus = algorithms.MyConsensus()
    my_networking = algorithms.MyNetworking()
    
    # 实例化固定路线生成器
    GENERATION_INTERVAL = 40 # 设置生成车辆的时间间隔（步数）
    my_vehicle_generator = VehicleGenerator(step_interval=GENERATION_INTERVAL)

    # --- 运行仿真 ---
    print("正在启动十字路口仿真...")
    manager = SimulationManager(
        sumoCmd,
        SIMULATION_STEPS,
        driving_strategy=my_driving_strategy,
        scheduling_algo=my_traffic_light_scheduling,
        consensus_algo=my_consensus,
        networking_proto=my_networking,
        vehicle_generator=my_vehicle_generator
    )
    manager.run()
    print("十字路口仿真结束。")

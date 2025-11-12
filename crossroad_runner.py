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

class SimulationManager:
    """
    仿真管理器，用于设置和运行SUMO仿真。
    """
    def __init__(self, sumo_cmd, max_steps,
                 driving_strategy=None,
                 scheduling_algo=None,
                 consensus_algo=None,
                 networking_proto=None):
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.driving_strategy = driving_strategy
        self.scheduling_algo = scheduling_algo
        self.consensus_algo = consensus_algo
        self.networking_proto = networking_proto

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

    # --- 运行仿真 ---
    print("正在启动十字路口仿真...")
    manager = SimulationManager(
        sumoCmd,
        SIMULATION_STEPS,
        driving_strategy=my_driving_strategy,
        scheduling_algo=my_traffic_light_scheduling,
        consensus_algo=my_consensus,
        networking_proto=my_networking
    )
    manager.run()
    print("十字路口仿真结束。")

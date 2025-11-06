import os
import sys
import traci

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
    def __init__(self, sumo_cmd, max_steps, port):
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.port = port

    def run(self):
        """执行TraCI控制循环"""
        traci.start(self.sumo_cmd, port=self.port)
        step = 0
        while step < self.max_steps:
            traci.simulationStep()
            step += 1
        traci.close()

if __name__ == "__main__":
    # --- 仿真参数 ---
    sumoBinary = "sumo-gui"  # 使用 "sumo-gui" 或 "sumo"
    sumoCmd = [sumoBinary, "-c", "roundabout.sumocfg"]
    SIMULATION_STEPS = 5000
    PORT = 8814

    # --- 运行仿真 ---
    print("正在启动环岛仿真...")
    manager = SimulationManager(sumoCmd, SIMULATION_STEPS, PORT)
    manager.run()
    print("环岛仿真结束。")

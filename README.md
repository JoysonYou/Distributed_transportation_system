# 分布式交通系统SUMO仿真平台

这是一个基于 [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) 和 Python 的交通仿真项目。该平台旨在模拟和分析不同交通场景下的车辆行为，并为开发、测试和评估新的交通管理算法（如共识算法、驾驶策略、调度算法等）提供一个可扩展的基础框架。

## 主要特性

- **多场景支持**: 内置了两种典型的城市交通场景：
  - **十字路口 (Crossroad)**: 一个由交通信号灯控制的标准四向十字路口。
  - **环岛 (Roundabout)**: 一个无信号灯的四向环岛，车辆需根据优先规则通行。
- **多样化的车辆行为**:
  - 在十字路口场景中，模拟了 **自动驾驶车辆** 和 **人类驾驶车辆** 的混合交通流。
  - 在环岛场景中，根据驾驶风格将车辆分为 **谨慎型**、**普通型** 和 **激进型**，使仿真更加真实。
- **Python控制**: 通过SUMO的TraCI (Traffic Control Interface) API，使用Python脚本来启动、控制和监控仿真过程。
- **模块化算法设计**: 在 `algorithms.py` 文件中定义了算法的抽象基类，方便研究人员插入和测试自己的交通协调和车辆控制算法。

## 环境要求

在运行此项目之前，请确保你的系统满足以下要求：

1.  **SUMO**: 必须安装SUMO，并且需要正确设置 `SUMO_HOME` 环境变量。
    -   可以从 SUMO官方网站 下载。
2.  **Python**: Python 3.x 版本。
3.  **Traci库**: 该库通常随SUMO一起安装。如果未安装，它位于 `SUMO_HOME/tools` 目录下。项目脚本会自动将其路径添加到sys.path。

## 如何运行仿真

你可以通过运行项目根目录下的Python脚本来启动相应的仿真场景。

### 1. 运行十字路口仿真

打开终端，进入项目根目录，然后执行以下命令：

```bash
python crossroad_runner.py
```

这会启动一个带GUI的SUMO仿真窗口，展示一个交通信号灯控制的十字路口。

### 2. 运行环岛仿真

同样，在项目根目录下执行以下命令：

```bash
python roundabout_runner.py
```

这会启动一个环岛场景的仿真，你可以观察不同驾驶风格的车辆如何交互通行。

## 场景详解

### 十字路口 (Crossroad)

- **路网**: `crossroad.net.xml`
- **车流定义**: `crossroad.rou.xml`
- **车辆类型**:
  - **自动驾驶车辆 (autonomous_car)**: 蓝色，具有更优的加速度和反应能力 (`sigma=0.2`)。
  - **人类驾驶车辆 (standard_car)**: 红色，驾驶参数模拟普通人类司机 (`sigma=0.5`)，并通过 `algorithms.py` 中的策略引入了轻微的速度扰动。

### 环岛 (Roundabout)

- **路网**: `roundabout.net.xml`
- **车流定义**: `roundabout.rou.xml`
- **车辆类型**: 通过 `vTypeDistribution` 以不同概率生成三种类型的车辆：
  - **谨慎型车辆 (cautious_car)**: 蓝色 (20%概率)，驾驶保守，保持较大车距。
  - **普通型车辆 (normal_car)**: 绿色 (60%概率)，行为符合一般驾驶习惯。
  - **激进型车辆 (aggressive_car)**: 红色 (20%概率)，驾驶风格更具侵略性，车距更小，速度更快。

## 如何扩展和自定义

本项目被设计为易于扩展。

- **修改车流**: 打开任一场景的 `.rou.xml` 文件 (如 `crossroad.rou.xml`)，你可以修改 `<flow>` 标签中的 `vehsPerHour` (每小时车辆数) 或 `period` (生成周期) 属性来调整交通密度。
- **修改车辆参数**: 在 `.rou.xml` 文件中，你可以修改 `<vType>` 标签下的参数（如 `accel`, `decel`, `maxSpeed`）来自定义车辆的物理性能。
- **实现新算法**: 打开 `algorithms.py`，继承相应的抽象基类（如 `DrivingStrategy`, `SchedulingAlgorithm`）并实现你自己的 `update` 方法。然后在对应的 `runner.py` 文件中实例化并调用你的新算法。

# 分布式交通系统SUMO仿真平台

这是一个基于 [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) 和 Python 的交通仿真项目。该平台旨在模拟和分析不同交通场景下的车辆行为，并为开发、测试和评估新的交通管理算法（如共识算法、驾驶策略、调度算法等）提供一个可扩展的基础框架。

## 主要特性

- **多场景支持**: 内置了两种典型的城市交通场景：
  - **十字路口 (Crossroad)**: 一个由交通信号灯控制的标准四向十字路口。
  - **环岛 (Roundabout)**: 一个由 **交通信号灯协调控制** 的四向环岛，具有优化的圆形几何形状和非均匀的交通流。
- **多样化的车辆行为**:
  - 在十字路口场景中，模拟了 **自动驾驶车辆** 和 **人类驾驶车辆** 的混合交通流。
  - 在环岛场景中，根据驾驶风格将车辆分为 **谨慎型**、**普通型** 和 **激进型**，并设置了 **不同方向的差异化车流量**。
- **Python控制**: 通过SUMO的TraCI (Traffic Control Interface) API，使用Python脚本来启动、控制和监控仿真过程。
- **模块化算法设计**: 在 `algorithms.py` 文件中定义了算法的抽象基类，方便研究人员插入和测试自己的交通协调和车辆控制算法。

## 环境要求

在运行此项目之前，请确保你的系统满足以下要求：

1.  **SUMO**: 必须安装SUMO，并且需要正确设置 `SUMO_HOME` 环境变量。
    -   可以从 SUMO官方网站 下载。
2.  **Python**: Python 3.x 版本。
3.  **Traci库**: 该库通常随SUMO一起安装。如果未安装，它位于 `SUMO_HOME/tools` 目录下。项目脚本会自动将其路径添加到sys.path。

## 如何运行仿真

你可以通过运行项目根目录下的Python脚本来启动相应的仿真场景。

### 1. 运行十字路口仿真

打开终端，进入项目根目录，然后执行以下命令：

```bash
python crossroad_runner.py
```

这会启动一个带GUI的SUMO仿真窗口，展示一个交通信号灯控制的十字路口。

### 2. 运行环岛仿真

同样，在项目根目录下执行以下命令：

```bash
python roundabout_runner.py
```

这会启动一个环岛场景的仿真，你可以观察不同驾驶风格的车辆如何交互通行。

## 场景详解

### 十字路口 (Crossroad)

- **路网**: `crossroad.net.xml`
- **车流定义**: `crossroad.rou.xml`
- **车辆类型**:
  - **自动驾驶车辆 (autonomous_car)**: 蓝色，具有更优的加速度和反应能力 (`sigma=0.2`)。
  - **人类驾驶车辆 (standard_car)**: 红色，驾驶参数模拟普通人类司机 (`sigma=0.5`)，并通过 `algorithms.py` 中的策略引入了轻微的速度扰动。

### 环岛 (Roundabout)

- **路网**: `roundabout.net.xml` (由 `roundabout.netccfg` 生成，包含 `roundabout.edg.xml` 中的圆形几何定义)
- **信号控制**: `roundabout.tll.xml` (定义了协调的红绿灯相位，包含“绿波”通行逻辑)
- **车流定义**: `roundabout.rou.xml` (定义了北/东/南/西四个方向分别为 高/中/低/极低 的差异化交通密度)
- **车辆类型**: 通过 `vTypeDistribution` 以不同概率生成三种类型的车辆：
  - **谨慎型车辆 (cautious_car)**: 蓝色 (20%概率)，驾驶保守，保持较大车距。
  - **普通型车辆 (normal_car)**: 绿色 (60%概率)，行为符合一般驾驶习惯。
  - **激进型车辆 (aggressive_car)**: 红色 (20%概率)，驾驶风格更具侵略性，车距更小，速度更快。

## 如何扩展和自定义

本项目被设计为易于扩展。

- **修改车流**: 打开任一场景的 `.rou.xml` 文件 (如 `crossroad.rou.xml`)，你可以修改 `<flow>` 标签中的 `vehsPerHour` (每小时车辆数) 或 `period` (生成周期) 属性来调整交通密度。
- **修改车辆参数**: 在 `.rou.xml` 文件中，你可以修改 `<vType>` 标签下的参数（如 `accel`, `decel`, `maxSpeed`）来自定义车辆的物理性能。
- **实现新算法**: 打开 `algorithms.py`，继承相应的抽象基类（如 `DrivingStrategy`, `SchedulingAlgorithm`）并实现你自己的 `update` 方法。然后在对应的 `runner.py` 文件中实例化并调用你的新算法。

## 项目文件结构

```
.
├── .gitignore               # Git忽略文件配置
├── algorithms.py            # 交通算法的定义与实现
├── crossroad.edg.xml        # 十字路口场景：边文件
├── crossroad.net.xml        # 十字路口场景：路网文件
├── crossroad.nod.xml        # 十字路口场景：节点文件
├── crossroad.rou.xml        # 十字路口场景：车流和车辆类型定义
├── crossroad.sumocfg        # 十字路口场景：SUMO配置文件
├── crossroad_runner.py      # 十字路口场景：Python启动脚本
├── roundabout.edg.xml       # 环岛场景：边文件 (包含圆形几何形状定义)
├── roundabout.net.xml       # 环岛场景：路网文件
├── roundabout.netccfg       # 环岛场景：Netconvert配置文件
├── roundabout.nod.xml       # 环岛场景：节点文件
├── roundabout.rou.xml       # 环岛场景：车流和车辆类型定义
├── roundabout.sumocfg       # 环岛场景：SUMO配置文件
├── roundabout.tll.xml       # 环岛场景：交通信号灯逻辑定义
├── roundabout_runner.py     # 环岛场景：Python启动脚本
└── vehicle_documentation.md # 车辆参数的详细文档
```

## 开发者贡献

如果你希望为本项目贡献代码，例如实现自己的共识或通信算法，请查阅我们的 **[开发者指南](DEVELOPER_GUIDE.md)**。
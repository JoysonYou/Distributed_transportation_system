"""
遗传算法交通策略优化器 - 全面改进版本
用于优化十字路口车辆通行策略，解决碰撞问题并提高收敛性
"""
import numpy as np
import random
import pickle
import os
from typing import List, Dict, Tuple

class GeneticTrafficStrategy:
    """
    使用遗传算法优化的交通策略 - 改进版
    
    策略编码方式：
    - 更精细的状态空间划分
    - 连续的加速度输出（使用实数编码）
    - 增加安全距离考虑
    
    状态特征（共48种状态）：
    1. 距离区间：[0-5m], [5-10m], [10-15m], [15-20m], [20-30m]  (5个区间)
    2. 速度区间：[0-3m/s], [3-8m/s], [8-13m/s], [13-20m/s]  (4个区间)
    3. 红绿灯状态：红灯/绿灯  (2种状态)
    4. 速度状态：过快/正常  (2种状态，用于安全控制)
    
    总共：5 * 4 * 2 = 40 种基础状态
    
    染色体：实数编码，每个基因值在[-3.0, 2.0]之间
           红灯时更倾向于负加速度，绿灯时更灵活
    """
    
    # 加速度范围
    MAX_ACCEL = 2.0  # 最大加速度
    MAX_DECEL = -4.0  # 最大减速度（紧急制动）
    SAFE_DECEL = -2.5  # 安全减速度
    COMFORT_DECEL = -1.5  # 舒适减速度
    
    # 安全参数
    MIN_SAFE_DISTANCE = 3.0  # 最小安全距离 (m)
    SAFE_TIME_HEADWAY = 1.5  # 安全时间间隔 (s)
    
    def __init__(self, chromosome: List[float] = None):
        """
        初始化策略
        
        Args:
            chromosome: 染色体（策略参数），长度为40的列表，每个值为实数
        """
        self.num_genes = 40  # 5距离 × 4速度 × 2灯光
        
        if chromosome is None:
            # 智能初始化：根据状态特征设置合理的初始值
            self.chromosome = self._initialize_smart_chromosome()
        else:
            self.chromosome = list(chromosome)
    
    def _initialize_smart_chromosome(self) -> List[float]:
        """
        智能初始化染色体，使其从合理的起点开始
        红灯时策略全部偏向减速，确保安全
        """
        chromosome = []
        
        for i in range(self.num_genes):
            # 解析状态索引
            light_idx = i // 20  # 0=绿灯, 1=红灯
            remaining = i % 20
            dist_idx = remaining // 4  # 0-4: 五个距离区间
            speed_idx = remaining % 4  # 0-3: 四个速度区间
            
            # 根据状态设置初始偏向
            if light_idx == 1:  # 红灯 - 全部偏向减速
                if dist_idx == 0:  # 0-5m：紧急制动
                    base_accel = random.uniform(-4.0, -3.0)
                elif dist_idx == 1:  # 5-10m：强力减速
                    if speed_idx >= 2:  # 速度较快
                        base_accel = random.uniform(-3.5, -2.5)
                    else:
                        base_accel = random.uniform(-3.0, -2.0)
                elif dist_idx == 2:  # 10-15m：安全减速
                    if speed_idx >= 2:  # 速度较快
                        base_accel = random.uniform(-3.0, -1.5)
                    else:
                        base_accel = random.uniform(-2.5, -1.0)
                elif dist_idx == 3:  # 15-20m：预防减速
                    if speed_idx >= 3:  # 速度很快
                        base_accel = random.uniform(-2.5, -1.0)
                    else:
                        base_accel = random.uniform(-2.0, -0.5)
                else:  # 20-30m：轻度减速
                    if speed_idx >= 3:  # 速度很快
                        base_accel = random.uniform(-2.0, -0.5)
                    else:
                        base_accel = random.uniform(-1.5, 0.0)
            else:  # 绿灯
                if dist_idx <= 1:  # 距离很近
                    if speed_idx <= 1:  # 速度较慢：加速通过
                        base_accel = random.uniform(0.5, 2.0)
                    else:  # 速度较快：维持或轻微减速
                        base_accel = random.uniform(-0.5, 1.0)
                else:  # 距离较远
                    if speed_idx <= 1:  # 速度较慢：积极加速
                        base_accel = random.uniform(1.0, 2.0)
                    else:  # 速度较快：维持
                        base_accel = random.uniform(-0.5, 1.5)
            
            chromosome.append(base_accel)
        
        return chromosome
    
    def _get_state_index(self, distance: float, speed: float, is_red_light: bool) -> int:
        """
        根据车辆状态获取状态索引（更精细的划分）
        
        Args:
            distance: 距离路口的距离 (m)
            speed: 当前速度 (m/s)
            is_red_light: 是否是红灯
            
        Returns:
            状态索引 (0-39)
        """
        # 距离区间 (0-4) - 5个区间
        if distance <= 5:
            dist_idx = 0
        elif distance <= 10:
            dist_idx = 1
        elif distance <= 15:
            dist_idx = 2
        elif distance <= 20:
            dist_idx = 3
        else:
            dist_idx = 4
        
        # 速度区间 (0-3) - 4个区间
        if speed <= 3:
            speed_idx = 0
        elif speed <= 8:
            speed_idx = 1
        elif speed <= 13:
            speed_idx = 2
        else:
            speed_idx = 3
        
        # 红绿灯状态 (0-1)
        light_idx = 1 if is_red_light else 0
        
        # 计算总索引：light * 20 + dist * 4 + speed
        state_idx = light_idx * 20 + dist_idx * 4 + speed_idx
        return state_idx
    
    def get_action(self, distance: float, speed: float, is_red_light: bool) -> float:
        """
        根据当前状态获取加速度决策
        安全约束在驾驶策略层实现，这里只提供基础策略
        
        Args:
            distance: 距离路口的距离 (m)
            speed: 当前速度 (m/s)
            is_red_light: 是否是红灯
            
        Returns:
            加速度指令 (m/s^2)
        """
        state_idx = self._get_state_index(distance, speed, is_red_light)
        base_accel = self.chromosome[state_idx]
        
        # 红灯时的基本约束（不过度限制，让驾驶策略层处理安全）
        if is_red_light:
            # 红灯近距离时偏向减速
            if distance < 10.0 and speed > 3.0:
                # 计算安全停车所需距离
                required_stop_distance = (speed * speed) / (2 * abs(self.SAFE_DECEL)) + self.MIN_SAFE_DISTANCE
                
                if distance < required_stop_distance:
                    # 需要减速
                    available_distance = max(distance - self.MIN_SAFE_DISTANCE, 0.5)
                    required_decel = -(speed * speed) / (2 * available_distance)
                    required_decel = max(required_decel, self.MAX_DECEL)
                    base_accel = min(base_accel, required_decel)
        
        # 限制加速度范围
        accel = np.clip(base_accel, self.MAX_DECEL, self.MAX_ACCEL)
        
        return accel
    
    def mutate(self, mutation_rate: float = 0.15, mutation_strength: float = 0.5):
        """
        改进的变异操作：使用高斯变异和自适应强度
        红灯状态的基因始终偏向负值
        
        Args:
            mutation_rate: 变异概率
            mutation_strength: 变异强度（标准差）
        """
        for i in range(len(self.chromosome)):
            if random.random() < mutation_rate:
                # 解析状态
                light_idx = i // 20
                remaining = i % 20
                dist_idx = remaining // 4
                
                # 使用高斯分布进行微调
                mutation_type = random.random()
                
                if mutation_type < 0.7:  # 70%概率：高斯变异（小幅调整）
                    delta = np.random.normal(0, mutation_strength)
                    self.chromosome[i] += delta
                elif mutation_type < 0.9:  # 20%概率：较大调整
                    delta = np.random.normal(0, mutation_strength * 2)
                    self.chromosome[i] += delta
                else:  # 10%概率：完全重置（探索新区域）
                    # 根据状态智能重置
                    if light_idx == 1:  # 红灯，必须偏向减速
                        if dist_idx <= 1:  # 近距离
                            self.chromosome[i] = random.uniform(-4.0, -2.0)
                        elif dist_idx <= 2:  # 中距离
                            self.chromosome[i] = random.uniform(-3.0, -1.0)
                        else:  # 远距离
                            self.chromosome[i] = random.uniform(-2.0, 0.0)
                    else:  # 绿灯，更灵活
                        self.chromosome[i] = random.uniform(-1.0, 2.0)
                
                # 限制范围
                self.chromosome[i] = np.clip(self.chromosome[i], self.MAX_DECEL, self.MAX_ACCEL)
                
                # 红灯时额外限制：确保不会变成正值
                if light_idx == 1:  # 红灯
                    if dist_idx <= 2:  # 近中距离
                        self.chromosome[i] = min(self.chromosome[i], -0.5)
                    else:  # 远距离
                        self.chromosome[i] = min(self.chromosome[i], 0.0)
    
    @staticmethod
    def crossover(parent1: 'GeneticTrafficStrategy', 
                  parent2: 'GeneticTrafficStrategy',
                  method: str = 'uniform') -> Tuple['GeneticTrafficStrategy', 'GeneticTrafficStrategy']:
        """
        改进的交叉操作：支持多种交叉方式
        
        Args:
            parent1: 父代1
            parent2: 父代2
            method: 交叉方法 ('uniform', 'two_point', 'blend')
            
        Returns:
            两个子代
        """
        n = len(parent1.chromosome)
        
        if method == 'uniform':
            # 均匀交叉：每个基因随机选择父母之一
            child1_chromosome = []
            child2_chromosome = []
            for i in range(n):
                if random.random() < 0.5:
                    child1_chromosome.append(parent1.chromosome[i])
                    child2_chromosome.append(parent2.chromosome[i])
                else:
                    child1_chromosome.append(parent2.chromosome[i])
                    child2_chromosome.append(parent1.chromosome[i])
                    
        elif method == 'two_point':
            # 两点交叉
            point1 = random.randint(1, n - 2)
            point2 = random.randint(point1 + 1, n - 1)
            
            child1_chromosome = (parent1.chromosome[:point1] + 
                               parent2.chromosome[point1:point2] + 
                               parent1.chromosome[point2:])
            child2_chromosome = (parent2.chromosome[:point1] + 
                               parent1.chromosome[point1:point2] + 
                               parent2.chromosome[point2:])
                               
        else:  # 'blend'
            # 混合交叉：子代是父代的加权平均
            child1_chromosome = []
            child2_chromosome = []
            for i in range(n):
                alpha = random.uniform(0.3, 0.7)  # 混合比例
                child1_chromosome.append(alpha * parent1.chromosome[i] + (1 - alpha) * parent2.chromosome[i])
                child2_chromosome.append((1 - alpha) * parent1.chromosome[i] + alpha * parent2.chromosome[i])
        
        return GeneticTrafficStrategy(child1_chromosome), GeneticTrafficStrategy(child2_chromosome)
    
    def save(self, filename: str):
        """保存策略到文件"""
        with open(filename, 'wb') as f:
            pickle.dump(self.chromosome, f)
        print(f"Strategy saved to {filename}")
    
    @staticmethod
    def load(filename: str) -> 'GeneticTrafficStrategy':
        """从文件加载策略"""
        with open(filename, 'rb') as f:
            chromosome = pickle.load(f)
        print(f"Strategy loaded from {filename}")
        return GeneticTrafficStrategy(chromosome)


class GeneticAlgorithmOptimizer:
    """
    遗传算法优化器 - 改进版
    用于演化交通策略，增加自适应机制和多样性保护
    """
    
    def __init__(self, 
                 population_size: int = 30,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 elite_size: int = 3,
                 adaptive: bool = True):
        """
        初始化遗传算法优化器
        
        Args:
            population_size: 种群大小（增加到30以提高多样性）
            mutation_rate: 初始变异率
            crossover_rate: 交叉率
            elite_size: 精英保留数量
            adaptive: 是否使用自适应参数
        """
        self.population_size = population_size
        self.initial_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.adaptive = adaptive
        
        # 初始化种群
        self.population: List[GeneticTrafficStrategy] = []
        self.fitness_scores: List[float] = []
        self.best_strategy: GeneticTrafficStrategy = None
        self.best_fitness: float = float('inf')
        
        # 演化历史
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'std_fitness': [],
            'generation': [],
            'mutation_rate': []
        }
        
        # 收敛检测
        self.stagnation_counter = 0
        self.last_best_fitness = float('inf')
    
    def initialize_population(self):
        """初始化种群"""
        self.population = [GeneticTrafficStrategy() for _ in range(self.population_size)]
        print(f"Population initialized with size: {self.population_size}")
    
    def evaluate_fitness(self, strategy: GeneticTrafficStrategy, 
                        fitness_evaluator) -> float:
        """
        评估单个策略的适应度
        
        Args:
            strategy: 要评估的策略
            fitness_evaluator: 适应度评估函数（运行仿真）
            
        Returns:
            适应度分数（越小越好）
        """
        return fitness_evaluator(strategy)
    
    def evaluate_population(self, fitness_evaluator):
        """
        评估整个种群的适应度
        
        Args:
            fitness_evaluator: 适应度评估函数
        """
        self.fitness_scores = []
        for i, strategy in enumerate(self.population):
            fitness = self.evaluate_fitness(strategy, fitness_evaluator)
            self.fitness_scores.append(fitness)
            
            # 更新最佳策略
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_strategy = GeneticTrafficStrategy(strategy.chromosome)
                print(f"  Better strategy found. Fitness: {fitness:.2f}")
        
        print(f"  Average fitness: {np.mean(self.fitness_scores):.2f}")
        print(f"  Best fitness: {min(self.fitness_scores):.2f}")
    
    def selection(self) -> List[GeneticTrafficStrategy]:
        """
        改进的选择操作：结合精英保留、锦标赛选择和轮盘赌选择
        
        Returns:
            选中的父代列表
        """
        selected = []
        
        # 1. 精英保留：直接保留最优的个体
        elite_indices = np.argsort(self.fitness_scores)[:self.elite_size]
        for idx in elite_indices:
            selected.append(GeneticTrafficStrategy(self.population[idx].chromosome))
        
        # 2. 锦标赛选择（60%）：通过竞争选出
        tournament_size = 4  # 增加竞争强度
        tournament_count = int((self.population_size - self.elite_size) * 0.6)
        
        for _ in range(tournament_count):
            tournament = random.sample(range(self.population_size), tournament_size)
            winner_idx = min(tournament, key=lambda i: self.fitness_scores[i])
            selected.append(GeneticTrafficStrategy(self.population[winner_idx].chromosome))
        
        # 3. 适应度比例选择（40%）：给较差个体一些机会，保持多样性
        # 将适应度转换为选择概率（越小越好，所以取倒数）
        fitness_array = np.array(self.fitness_scores)
        # 避免除零和负数问题
        min_fitness = fitness_array.min()
        if min_fitness <= 0:
            fitness_array = fitness_array - min_fitness + 1
        
        # 使用倒数作为权重
        weights = 1.0 / (fitness_array + 1e-6)
        weights = weights / weights.sum()
        
        remaining_count = self.population_size - len(selected)
        if remaining_count > 0:
            indices = np.random.choice(
                range(self.population_size), 
                size=remaining_count, 
                replace=True, 
                p=weights
            )
            for idx in indices:
                selected.append(GeneticTrafficStrategy(self.population[idx].chromosome))
        
        return selected
    
    def update_adaptive_parameters(self, generation: int):
        """
        自适应调整遗传算法参数
        
        Args:
            generation: 当前代数
        """
        if not self.adaptive:
            return
        
        # 检测停滞
        improvement = self.last_best_fitness - self.best_fitness
        if improvement < 0.1:  # 改进很小
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.last_best_fitness = self.best_fitness
        
        # 自适应变异率
        if self.stagnation_counter > 3:
            # 停滞时增加变异率，探索新区域
            self.mutation_rate = min(0.4, self.mutation_rate * 1.2)
            print(f"  检测到停滞，增加变异率至 {self.mutation_rate:.3f}")
        else:
            # 恢复到初始值
            self.mutation_rate = max(self.initial_mutation_rate, self.mutation_rate * 0.95)
        
        # 多样性保护：如果种群过于相似，引入新个体
        if len(self.fitness_scores) > 0:
            fitness_std = np.std(self.fitness_scores)
            if fitness_std < 1.0 and self.stagnation_counter > 5:
                print("  种群多样性过低，注入新个体...")
                # 替换最差的30%个体
                n_replace = max(3, self.population_size // 3)
                worst_indices = np.argsort(self.fitness_scores)[-n_replace:]
                for idx in worst_indices:
                    self.population[idx] = GeneticTrafficStrategy()
                self.stagnation_counter = 0
    
    def evolve(self, fitness_evaluator, generations: int = 15):
        """
        执行遗传算法演化（改进版，增加多样性和自适应机制）
        
        Args:
            fitness_evaluator: 适应度评估函数
            generations: 演化代数
        """
        print("\n" + "="*70)
        print("启动改进型遗传算法优化")
        print("="*70)
        print(f"种群大小: {self.population_size}")
        print(f"初始变异率: {self.mutation_rate}")
        print(f"交叉率: {self.crossover_rate}")
        print(f"精英保留: {self.elite_size}")
        
        # 初始化种群（如果还没有初始化）
        if not self.population:
            self.initialize_population()
        
        for generation in range(generations):
            print(f"\n{'='*70}")
            print(f"第 {generation + 1}/{generations} 代")
            print(f"{'='*70}")
            
            # 1. 评估适应度
            print("评估种群适应度...")
            self.evaluate_population(fitness_evaluator)
            
            # 记录详细历史
            self.history['generation'].append(generation + 1)
            self.history['best_fitness'].append(min(self.fitness_scores))
            self.history['avg_fitness'].append(np.mean(self.fitness_scores))
            self.history['worst_fitness'].append(max(self.fitness_scores))
            self.history['std_fitness'].append(np.std(self.fitness_scores))
            self.history['mutation_rate'].append(self.mutation_rate)
            
            # 打印统计信息
            print(f"  最佳适应度: {min(self.fitness_scores):.2f}")
            print(f"  平均适应度: {np.mean(self.fitness_scores):.2f}")
            print(f"  最差适应度: {max(self.fitness_scores):.2f}")
            print(f"  标准差: {np.std(self.fitness_scores):.2f}")
            print(f"  当前变异率: {self.mutation_rate:.3f}")
            
            # 2. 自适应参数调整
            self.update_adaptive_parameters(generation)
            
            # 3. 选择
            parents = self.selection()
            
            # 4. 交叉和变异，生成新种群
            new_population = []
            
            # 保留精英（不进行交叉变异）
            for i in range(self.elite_size):
                new_population.append(GeneticTrafficStrategy(parents[i].chromosome))
            
            # 生成子代
            crossover_methods = ['uniform', 'two_point', 'blend']
            while len(new_population) < self.population_size:
                # 选择两个不同的父代
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                while parent2 is parent1 and len(parents) > 1:
                    parent2 = random.choice(parents)
                
                # 交叉（使用随机选择的交叉方法）
                if random.random() < self.crossover_rate:
                    method = random.choice(crossover_methods)
                    child1, child2 = GeneticTrafficStrategy.crossover(parent1, parent2, method)
                else:
                    child1 = GeneticTrafficStrategy(parent1.chromosome)
                    child2 = GeneticTrafficStrategy(parent2.chromosome)
                
                # 变异（使用自适应变异率和强度）
                mutation_strength = 0.3 + 0.2 * (generation / generations)  # 随代数增加探索
                child1.mutate(self.mutation_rate, mutation_strength)
                child2.mutate(self.mutation_rate, mutation_strength)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population
        
        print("\n" + "="*70)
        print("遗传算法优化完成！")
        print(f"最佳适应度: {self.best_fitness:.2f}")
        print(f"总改进幅度: {self.history['avg_fitness'][0] - self.best_fitness:.2f}")
        print("="*70)
    
    def get_best_strategy(self) -> GeneticTrafficStrategy:
        """获取最佳策略"""
        return self.best_strategy
    
    def save_history(self, filename: str = "ga_history.pkl"):
        """保存演化历史"""
        with open(filename, 'wb') as f:
            pickle.dump(self.history, f)
        print(f"Evolution history saved to {filename}")
    
    def plot_history(self, save_path: str = "ga_evolution.png"):
        """
        绘制详细的演化曲线（改进版，显示更多信息）
        
        Args:
            save_path: 图片保存路径
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 子图1：适应度演化
            ax1.plot(self.history['generation'], self.history['best_fitness'], 
                    'b-', label='最佳适应度', linewidth=2.5, marker='o', markersize=4)
            ax1.plot(self.history['generation'], self.history['avg_fitness'], 
                    'r--', label='平均适应度', linewidth=2, marker='s', markersize=3)
            
            # 添加标准差阴影区域
            if 'std_fitness' in self.history:
                avg = np.array(self.history['avg_fitness'])
                std = np.array(self.history['std_fitness'])
                generations = np.array(self.history['generation'])
                ax1.fill_between(generations, avg - std, avg + std, alpha=0.2, color='red')
            
            ax1.set_xlabel('代数', fontsize=12)
            ax1.set_ylabel('适应度分数（越小越好）', fontsize=12)
            ax1.set_title('遗传算法演化曲线 - 适应度变化', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11, loc='upper right')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # 添加改进百分比注释
            if len(self.history['best_fitness']) > 0:
                initial_fitness = self.history['avg_fitness'][0]
                final_fitness = self.history['best_fitness'][-1]
                improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100
                ax1.text(0.02, 0.98, f'改进: {improvement:.1f}%', 
                        transform=ax1.transAxes, fontsize=11,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 子图2：变异率变化和多样性
            if 'mutation_rate' in self.history and len(self.history['mutation_rate']) > 0:
                ax2_twin = ax2.twinx()
                
                # 变异率
                ax2.plot(self.history['generation'], self.history['mutation_rate'], 
                        'g-', label='变异率', linewidth=2, marker='^', markersize=3)
                ax2.set_xlabel('代数', fontsize=12)
                ax2.set_ylabel('变异率', fontsize=12, color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                
                # 多样性（用标准差表示）
                if 'std_fitness' in self.history:
                    ax2_twin.plot(self.history['generation'], self.history['std_fitness'], 
                                'purple', linestyle='--', label='适应度标准差', linewidth=2, marker='d', markersize=3)
                    ax2_twin.set_ylabel('适应度标准差（多样性）', fontsize=12, color='purple')
                    ax2_twin.tick_params(axis='y', labelcolor='purple')
                
                ax2.set_title('自适应参数和种群多样性变化', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, linestyle='--')
                
                # 合并图例
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"演化曲线已保存至 {save_path}")
            plt.close()
        except ImportError:
            print("警告: matplotlib 未安装，无法绘制演化曲线")


if __name__ == "__main__":
    # 测试代码
    print("测试改进的遗传算法交通策略...")
    
    # 创建一个策略
    strategy = GeneticTrafficStrategy()
    print(f"染色体长度: {len(strategy.chromosome)}")
    print(f"染色体示例: {strategy.chromosome[:10]}...")
    
    # 测试状态-动作映射
    test_cases = [
        (5.0, 3.0, True),    # 近距离, 低速, 红灯
        (5.0, 15.0, True),   # 近距离, 高速, 红灯（应该急刹车）
        (15.0, 10.0, True),  # 中等距离, 中速, 红灯
        (25.0, 18.0, False), # 远距离, 高速, 绿灯
        (10.0, 5.0, False),  # 中近距离, 低速, 绿灯（可以加速）
    ]
    
    print("\n测试策略决策:")
    print(f"{'距离(m)':<10} {'速度(m/s)':<12} {'灯光':<8} -> {'加速度(m/s^2)':<15} {'说明':<20}")
    print("-" * 75)
    for dist, speed, is_red in test_cases:
        action = strategy.get_action(dist, speed, is_red)
        light_str = "红灯" if is_red else "绿灯"
        
        # 判断决策类型
        if action < -2:
            decision = "紧急制动"
        elif action < -0.5:
            decision = "减速"
        elif action < 0.5:
            decision = "维持速度"
        else:
            decision = "加速"
        
        print(f"{dist:<10.1f} {speed:<12.1f} {light_str:<8} -> {action:<15.2f} {decision:<20}")
    
    # 测试交叉和变异
    print("\n测试遗传操作...")
    strategy2 = GeneticTrafficStrategy()
    
    # 测试不同的交叉方法
    for method in ['uniform', 'two_point', 'blend']:
        child1, child2 = GeneticTrafficStrategy.crossover(strategy, strategy2, method)
        print(f"{method} 交叉后子代1: {child1.chromosome[:5]}...")
    
    # 测试变异
    child1_copy = GeneticTrafficStrategy(child1.chromosome)
    child1.mutate(mutation_rate=0.3, mutation_strength=0.5)
    
    # 统计变异的基因数量
    mutations = sum(1 for i in range(len(child1.chromosome)) 
                   if abs(child1.chromosome[i] - child1_copy.chromosome[i]) > 0.01)
    print(f"\n变异后改变了 {mutations}/{len(child1.chromosome)} 个基因")
    
    print("\n测试完成！")


# Projet "robotique" IA&Jeux 2025
#
# Binome:
#  Prénom Nom No_étudiant/e : Antoine Lecomte 21103457
#  Prénom Nom No_étudiant/e : Yuxiang Zhang 21202829

from robot import *
import math
import random
import csv

nb_robots = 0
debug = False

class Robot_player(Robot):
    team_name = "Yutoine"
    robot_id = -1
    memory = 0
    iteration = 0

    # Genetic Algorithm parameters
    param = []
    bestParam = []
    best_score = -float('inf')
    current_score = 0
    bestTrial = -1
    total_score = 0

    evaluations = 200
    it_per_evaluation = 300
    subtrial_per_evaluation = 3
    trial = 0
    subtrial = 0

    # Coverage tracking
    visited_cells = set()
    cell_size = 0.5
    log_sum_of_translation = 0
    log_sum_of_rotation = 0

    def __init__(self, x_0, y_0, theta_0, name="n/a", team="n/a"):
        global nb_robots
        self.robot_id = nb_robots
        nb_robots += 1

        # Initialize GA parameters
        self.param = [random.uniform(-1, 1) for _ in range(8)]
        self.bestParam = self.param.copy()
        
        # Initialize position tracking
        self.x_0 = x_0
        self.y_0 = y_0
        self.theta_0 = theta_0
        self.visited_cells = set()
        self.total_score = 0
        
        super().__init__(x_0, y_0, theta_0, name="Robot "+str(self.robot_id), team=self.team_name)

    def reset(self):
        """Reset robot position and tracking metrics"""
        self.theta = random.uniform(0, 2 * math.pi)
        self.log_sum_of_translation = 0
        self.log_sum_of_rotation = 0
        super().reset()

    def update_coverage(self, x, y):
        """Track visited cells for coverage calculation"""
        cell_x = int(x / self.cell_size)
        cell_y = int(y / self.cell_size)
        self.visited_cells.add((cell_x, cell_y))

    def calculate_coverage_score(self):
        """Calculate normalized coverage score"""
        max_cells = (20 / self.cell_size) * (20 / self.cell_size)
        return len(self.visited_cells) / max_cells

    def mutate(self, params):
        child = params.copy()
        i = random.randint(0, len(child) - 1)
        original_value = child[i]
        new_value = random.choice([-1, 0, 1])
        while new_value == original_value:
            new_value = random.choice([-1, 0, 1])
        child[i] = new_value
        return child

    def step(self, sensors, sensor_view=None, sensor_robot=None, sensor_team=None):
        if sensor_view is None:
            sensor_view = [0] * 8

        # Update coverage tracking
        self.update_coverage(self.x, self.y)

        if not hasattr(self, 'replay_mode'):
            self.replay_mode = False
        
        # GA evaluation logic
        if not self.replay_mode and self.iteration % self.it_per_evaluation == 0 and self.iteration > 0:
            coverage_score = self.calculate_coverage_score()
            efficiency_score = self.log_sum_of_translation * (1 - abs(self.log_sum_of_rotation/self.it_per_evaluation))
            total_score = 0.7 * coverage_score + 0.3 * efficiency_score
            
            self.total_score += total_score
            self.subtrial += 1

            if self.subtrial == self.subtrial_per_evaluation:
                if self.total_score > self.best_score:
                    self.best_score = self.total_score
                    self.bestParam = self.param.copy()
                
                candidates = [self.mutate(self.bestParam) for _ in range(5)] + [self.bestParam]
                self.param = max(candidates, key=lambda p: self.evaluate_params(p))
                
                with open('ga_results.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.trial, self.total_score, self.best_score] + self.bestParam)
                
                self.trial += 1
                self.total_score = 0
                self.subtrial = 0
                
                if self.trial >= self.evaluations:
                    print(f"Optimization complete. Best score: {self.best_score}")
                    print(f"Best params: {self.bestParam}")
                    self.replay_mode = True
                    self.param = self.bestParam.copy()
                
                self.reset()
                return 0, 0, True

        # Sensor processing with correct indices
        sensor_to_wall = [1.0 if sensor_view[i] != 1 else sensors[i] for i in range(8)]
        sensor_to_robot = [1.0 if sensor_view[i] != 2 else sensors[i] for i in range(8)]

        # Layer 1: Wall avoidance (GA-optimized)
        if any(sensor_to_wall[i] != 1.0 for i in range(8)):
            translation = self.param[0] * sensor_to_wall[sensor_front]  # Index 0
            
            # Using all 8 sensors for wall avoidance with GA parameters
            rotation = (
                self.param[0] * sensor_to_wall[sensor_front] +        # 0
                self.param[1] * sensor_to_wall[sensor_front_left] +   # 1
                self.param[2] * sensor_to_wall[sensor_left] +         # 2
                self.param[3] * sensor_to_wall[sensor_rear_left] +    # 3
                self.param[4] * sensor_to_wall[sensor_rear] +         # 4
                self.param[5] * sensor_to_wall[sensor_rear_right] +   # 5
                self.param[6] * sensor_to_wall[sensor_right] +        # 6
                self.param[7] * sensor_to_wall[sensor_front_right]    # 7
            )
            translation = max(-1,min(translation,1))
            rotation = max(-1, min(rotation, 1))
            return self.normalize_output(translation, rotation)

        # Layer 2: Robot interaction
        if any(sensor_to_robot[i] != 1.0 for i in range(8)):
            if any(sensor_team[i] == "Yutoine" for i in range(8) if sensor_team[i] is not None):
                # Avoid teammates
                translation = sensor_to_robot[sensor_front]
                rotation = (
                    sensor_to_robot[sensor_front_left] * -1.0 +   
                    sensor_to_robot[sensor_left] * -1.0 +        
                    sensor_to_robot[sensor_rear_left] * -1.0 +    
                    sensor_to_robot[sensor_front_right] * 1.0 +   
                    sensor_to_robot[sensor_right] * 1.0 +        
                    sensor_to_robot[sensor_rear_right] * 1.0   
                )
                translation = max(-1,min(translation,1))
                rotation = max(-1, min(rotation, 1))
                return self.normalize_output(translation, rotation)
            else:
                # Chase enemies
                translation = sensor_to_robot[sensor_front]
                rotation = (
                    sensor_to_robot[sensor_front_right] * 1.0 +  
                    sensor_to_robot[sensor_right] * 1.0 +         
                    sensor_to_robot[sensor_rear_right] * 1.0 +    
                    sensor_to_robot[sensor_front_left] * -1.0 +   
                    sensor_to_robot[sensor_left] * -1.0 +         
                    sensor_to_robot[sensor_rear_left] * -1.0      
                )
                translation = max(-1,min(translation,1))
                rotation = max(-1, min(rotation, 1))
                return self.normalize_output(translation, rotation)

        # Layer 3: Default behavior - straight forward
        return self.normalize_output(1, 0)  # Full speed forward, no rotation

    def normalize_output(self, translation, rotation):
        """Clip and log movement values"""
        translation = max(-1, min(1, translation))
        rotation = max(-1, min(1, rotation))
        
        self.log_sum_of_translation += translation
        self.log_sum_of_rotation += rotation
        self.iteration += 1
        return translation, rotation, False

    def evaluate_params(self, params):
        """Evaluate parameters based on coverage and efficiency"""
        coverage = sum(abs(p) for p in params[:5])
        interaction = sum(abs(p) for p in params[5:])
        return coverage * 0.6 + interaction * 0.4

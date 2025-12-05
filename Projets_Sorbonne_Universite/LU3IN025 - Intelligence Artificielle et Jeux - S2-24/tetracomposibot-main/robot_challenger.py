# Projet "robotique" IA&Jeux 2025
#
# Binôme :
#  Prénom Nom No_étudiant/e : Antoine Lecomte 21103457
#  Prénom Nom No_étudiant/e : Yuxiang Zhang 21202829

from robot import *
import math
import random

nb_robots = 0
debug = False

class Robot_player(Robot):
    team_name = "Yutoine"
    robot_id = -1
    memory = 0  # nombre d'itérations où le robot reste dans la même case (blocage)
    iteration = 0

    # Paramètres de l’algorithme génétique (seulement pour le robot 1)
    param = []
    bestParam = []
    best_score = -float('inf')
    current_score = 0
    bestTrial = -1
    total_score = 0

    evaluations = 10000
    it_per_evaluation = 5000
    subtrial_per_evaluation = 1
    trial = 0
    subtrial = 0

    # Suivi de couverture (seulement pour le robot 1)
    visited_cells = set()
    cell_size = 0.5
    log_sum_of_translation = 0
    log_sum_of_rotation = 0
    last_position = (0, 0)

    def __init__(self, x_0, y_0, theta_0, name="n/a", team="n/a"):
        global nb_robots
        self.robot_id = nb_robots
        nb_robots += 1

        # Initialise l’algorithme génétique uniquement pour le robot 1
        if self.robot_id == 1:
            self.param = [
                random.choice([-1, 0, 1]),
                random.choice([-1, 0, 1]),
                random.choice([-1, 0, 1]),
                random.choice([-1, 0, 1]),
                random.choice([-1, 0, 1]),
                random.choice([-1, 0, 1]),
                random.choice([-1, 0, 1]),
                random.choice([-1, 0, 1])
            ]
            self.bestParam = self.param.copy()
            self.visited_cells = set()

        self.x_0 = x_0
        self.y_0 = y_0
        self.theta_0 = theta_0
        self.last_position = (x_0, y_0)
        self.total_score = 0

        super().__init__(x_0, y_0, theta_0, name="Robot "+str(self.robot_id), team=self.team_name)

    def reset(self):
        self.theta = random.uniform(0, 2 * math.pi)
        if self.robot_id == 1:
            self.log_sum_of_translation = 0
            self.log_sum_of_rotation = 0
            self.last_position = (self.x, self.y)
        super().reset()

    # ===== Fonctions GA (seulement pour le robot 1) =====
    def update_coverage(self, x, y):
        if self.robot_id == 1:
            cell_x = int(x / self.cell_size)
            cell_y = int(y / self.cell_size)
            self.visited_cells.add((cell_x, cell_y))

    def calculate_coverage_score(self):
        if self.robot_id == 1:
            max_cells = (20 / self.cell_size) * (20 / self.cell_size)
            return len(self.visited_cells) / max_cells
        return 0

    def mutate(self, params, mutation_rate=0.5):
        child = params.copy()
        for i in range(len(child)):
            if random.random() < mutation_rate:
                original = child[i]
                child[i] = random.choice([-1, -0.5, 0, 0.5, 1, original*1.2])
                child[i] = max(-1, min(1, child[i]))
        return child

    def evaluate_params(self, params):
        if self.robot_id == 1:
            couverture = self.calculate_coverage_score()
            efficacite = 1 - abs(self.log_sum_of_rotation/self.it_per_evaluation)
            return couverture * 0.9 + efficacite * 0.1
        return 0

    def selection(self, population):
        evaluated = [(p, self.evaluate_params(p)) for p in population]
        evaluated.sort(key=lambda x: -x[1])
        return [x[0] for x in evaluated[:5]]

    # ===== Fonction principale =====
    def step(self, sensors, sensor_view=None, sensor_robot=None, sensor_team=None):
        if sensor_view is None:
            sensor_view = [0] * 8

        # Seulement pour le robot 1 : suivi couverture et mouvements
        if self.robot_id == 1:
            self.update_coverage(self.x, self.y)
            actual_move = math.sqrt((self.x - self.last_position[0])**2 +
                                   (self.y - self.last_position[1])**2)
            self.log_sum_of_translation += actual_move
            self.last_position = (self.x, self.y)

        # ===== Logique GA (robot 1 uniquement) =====
        if self.robot_id == 1 and not hasattr(self, 'replay_mode'):
            self.replay_mode = False

        if self.robot_id == 1 and not self.replay_mode and self.iteration % self.it_per_evaluation == 0 and self.iteration > 0:
            total_score = self.evaluate_params(self.param)
            self.total_score += total_score
            self.subtrial += 1

            if self.subtrial == self.subtrial_per_evaluation:
                if self.total_score > self.best_score:
                    self.best_score = self.total_score
                    self.bestParam = self.param.copy()

                current_mutation_rate = 0.5 - 0.4*(self.trial/self.evaluations)
                children = [self.mutate(self.bestParam, current_mutation_rate) for _ in range(20)]
                next_gen = self.selection(children + [self.bestParam]*5)

                self.param = random.choice(next_gen)
                self.bestParam = next_gen[0]

                if debug:
                    print(f"Gen {self.trial}: Score={self.best_score:.2f} Params={self.bestParam}")

                self.trial += 1
                self.total_score = 0
                self.subtrial = 0

                if self.trial >= self.evaluations:
                    self.replay_mode = True
                    self.param = self.bestParam.copy()

                self.reset()
                return 0, 0, True

        # ===== Comportements partagés (couches) =====
        sensor_to_wall = [1.0 if sensor_view[i] != 1 else sensors[i] for i in range(8)]
        sensor_to_robot = [1.0 if sensor_view[i] != 2 else sensors[i] for i in range(8)]

        # si bloqué depuis 5 itérations dans la même case
        if self.robot_id != 1 & self.memory >= 5:
            translation = -sensors[sensor_front]  # reculer
            rotation = 0
            self.memory = 0  # réinitialiser la mémoire
            return self.normalize_output(translation, rotation)

        # Couche 1 : évitement des murs
        if any(sensor_to_wall[i] != 1.0 for i in range(8)):
            translation = sensor_to_wall[sensor_front]
            rotation = (
                sensor_to_wall[sensor_front_right] * (-1) +
                sensor_to_wall[sensor_right] * (-1) +
                sensor_to_wall[sensor_rear_right] * (-1) +
                sensor_to_wall[sensor_front_left] * (1) +
                sensor_to_wall[sensor_left] * (1) +
                sensor_to_wall[sensor_rear_left] * (1) +
                sensor_to_wall[sensor_front] * random.randint(-1,1) +
                random.randint(-2,2)
            )
            return self.normalize_output(translation, rotation)

        # Couche 2 : interaction entre robots
        if any(sensor_to_robot[i] != 1.0 for i in range(8)):
            if any(sensor_team[i] == "Yutoine" for i in range(8) if sensor_team[i] is not None):
                # Éviter les coéquipiers
                translation = sensor_to_robot[sensor_front]
                rotation = (
                    sensor_to_robot[sensor_front_right] * (-1) +
                    sensor_to_robot[sensor_right] * (-1) +
                    sensor_to_robot[sensor_rear_right] * (-1) +
                    sensor_to_robot[sensor_front_left] * (1) +
                    sensor_to_robot[sensor_left] * (1) +
                    sensor_to_robot[sensor_rear_left] * (1) +
                    sensor_to_robot[sensor_front] * random.randint(-1,1) +
                    random.randint(-2,2)
                )
            else:
                # Pourchasser les ennemis
                translation = sensor_to_robot[sensor_front]
                rotation = (
                    sensor_to_robot[sensor_front_right] * (1) +
                    sensor_to_robot[sensor_right] * (1) +
                    sensor_to_robot[sensor_rear_right] * (1) +
                    sensor_to_robot[sensor_front_left] * (-1) +
                    sensor_to_robot[sensor_left] * (-1) +
                    sensor_to_robot[sensor_rear_left] * (-1)
                )
            return self.normalize_output(translation, rotation)

        # Couche 3 : comportement par défaut
        if self.robot_id == 1 and hasattr(self, 'replay_mode') and self.replay_mode:
            # Robot 1 en mode replay utilise les paramètres GA optimisés
            translation = math.tanh(
                self.param[0] +
                self.param[1] * sensors[sensor_front_left] +
                self.param[2] * sensors[sensor_front] +
                self.param[3] * sensors[sensor_front_right]
            )
            rotation = math.tanh(
                self.param[4] +
                self.param[5] * sensors[sensor_front_left] +
                self.param[6] * sensors[sensor_front] +
                self.param[7] * sensors[sensor_front_right]
            )
        else:
            # Tous les autres robots avancent simplement tout droit
            translation = 1
            rotation = 0

        return self.normalize_output(translation, rotation)

    def normalize_output(self, translation, rotation):
        translation = max(-1, min(1, translation))
        rotation = max(-1, min(1, rotation))

        if self.robot_id == 1:
            self.log_sum_of_rotation += abs(rotation)

        # mise à jour de la mémoire
        if self.x == self.last_position[0] and self.y == self.last_position[1]:
            self.memory += 1
        else:
            self.memory = 0

        self.iteration += 1
        return translation, rotation, False

from robot import * 
import math
import random
import csv

nb_robots = 0
debug = False

class Robot_player(Robot):

    team_name = "RandomSearch2"
    robot_id = -1
    iteration = 0

    param = []
    bestParam = []
    best_score = -float('inf')
    current_score = 0
    bestTrial = -1

    evaluations = 500
    it_per_evaluation = 400
    subtrial_per_evaluation = 3
    trial = 0
    subtrial = 0

    x_0 = 0
    y_0 = 0
    theta_0 = 0

    replay_mode = False

    def __init__(self, x_0, y_0, theta_0, name="n/a", team="n/a", evaluations=500, it_per_evaluation=400):
        global nb_robots
        self.robot_id = nb_robots
        nb_robots += 1

        self.x_0 = x_0
        self.y_0 = y_0
        self.theta_0 = theta_0

        self.param = [random.randint(-1, 1) for _ in range(8)]
        self.evaluations = evaluations
        self.it_per_evaluation = it_per_evaluation

        self.total_score = 0
        self.subtrial = 0

        super().__init__(x_0, y_0, theta_0, name=name, team=team)

    def reset(self):
        self.theta = random.uniform(0, 2 * math.pi)
        super().reset()

    def step(self, sensors, sensor_view=None, sensor_robot=None, sensor_team=None):
        if self.replay_mode and self.iteration % 1000 == 0:
            print("\n[REPLAY MODE] Resetting position at step", self.iteration)
            self.reset()
            return 0, 0, True

        if not self.replay_mode and self.iteration % self.it_per_evaluation == 0:
            if self.iteration > 0:
                effective_translation = self.log_sum_of_translation
                average_rotation = self.log_sum_of_rotation / self.it_per_evaluation
                trial_score = effective_translation * (1 - abs(average_rotation))
                self.total_score += trial_score

                print(f"\n[Trial {self.trial} - Subtrial {self.subtrial}/{self.subtrial_per_evaluation}] Score: {trial_score}")
                print("\tTranslation =", effective_translation)
                print("\tRotation moyenne =", average_rotation)

            self.subtrial += 1

            if self.subtrial == self.subtrial_per_evaluation:
                print(f"\n>>> Trial {self.trial}/{self.evaluations} completed: Total score = {self.total_score}")
                print(f"    Params: {self.param}")

                if self.total_score > self.best_score:
                    self.best_score = self.total_score
                    self.bestParam = self.param.copy()
                    self.bestTrial = self.trial
                    print(">>> New best strategy found!")
                    print(">>> Best score =", self.best_score)

                mode = 'w' if self.trial == 0 else 'a'
                with open('random_search_results.csv', mode, newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.trial, self.total_score, self.best_score] + self.bestParam)

                self.trial += 1
                self.total_score = 0
                self.subtrial = 0

                if self.trial >= self.evaluations:
                    print("\n[INFO] All evaluations done. Entering replay mode.")
                    print(">>> FINAL BEST STRATEGY from trial", self.bestTrial)
                    print(">>> Parameters:", self.bestParam)
                    print(">>> Score:", self.best_score)
                    self.param = self.bestParam.copy()
                    self.replay_mode = True
                else:
                    self.param = [random.randint(-1, 1) for _ in range(8)]
                    print("\nTrying new strategy, trial", self.trial)

            self.reset()
            self.iteration += 1
            return 0, 0, True

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

        score = translation * (1 - abs(rotation))
        self.current_score += score

        if debug and self.iteration % 100 == 0:
            print("Robot", self.robot_id, "(team", self.team_name + ")", "at step", self.iteration)
            print("\tsensors =", sensors)
            print("\ttranslation =", translation, "; rotation =", rotation)

        self.iteration += 1
        return translation, rotation, False

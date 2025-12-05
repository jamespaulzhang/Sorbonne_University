from robot import * 
import math
import random

nb_robots = 0
debug = False

class Robot_player(Robot):

    team_name = "RandomSearch"
    robot_id = -1
    iteration = 0

    param = []
    bestParam = []
    best_score = -float('inf')
    current_score = 0
    bestTrial = -1

    evaluations = 500
    it_per_evaluation = 400
    trial = 0

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

        super().__init__(x_0, y_0, theta_0, name=name, team=team)

    def reset(self):
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
                self.current_score = effective_translation * (1 - abs(average_rotation))

                print("\nTrial", self.trial, "ended.")
                print("\tScore =", self.current_score)
                print("\tParam =", self.param)
                print("\tTranslation effective =", effective_translation)
                print("\tRotation moyenne =", average_rotation)
                print("\tDistance from origin =", math.sqrt((self.x - self.x_0)**2 + (self.y - self.y_0)**2))

                if self.current_score > self.best_score:
                    self.best_score = self.current_score
                    self.bestParam = self.param.copy()
                    self.bestTrial = self.trial
                    print(">>> New best strategy found at trial", self.trial)
                    print(">>> Best score =", self.best_score)

            self.trial += 1
            self.current_score = 0

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
            print("\tcurrent score =", self.current_score)

        self.iteration += 1
        return translation, rotation, False

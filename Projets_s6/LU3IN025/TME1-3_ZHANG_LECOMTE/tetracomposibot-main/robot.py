import random

robots_count = 1

sensor_front = 0
sensor_front_left = 1
sensor_left = 2
sensor_rear_left = 3
sensor_rear = 4
sensor_rear_right = 5
sensor_right = 6
sensor_front_right = 7

# Extended Robot class with name and team.
class Robot:
    name = "n/a"
    team = "n/a"
    log_sum_of_translation = 0
    log_sum_of_rotation = 0
    def __init__(self, x_0, y_0, theta_0, name="n/a", team="n/a"):
        global robots_count
        self.x0 = self.x = x_0
        self.y0 = self.y = y_0
        self.theta0 = self.theta = theta_0
        self.id = robots_count
        self.name = name
        self.team = team
        robots_count += 1
    def reset(self):
        self.x = self.x0
        self.y = self.y0
        self.theta = self.theta0
        self.log_sum_of_translation = 0
        self.log_sum_of_rotation = 0
    # Modified: step now takes extra arguments sensor_view and sensor_robot with defaults.
    def step(self, sensors, sensor_view=None, sensor_robot=None, sensor_team=None):
        translation, rotation = 0, 0
        ask_for_reset = False
        return translation, rotation, ask_for_reset


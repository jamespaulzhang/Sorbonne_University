# Nicolas
# 2025-03-24
#
# comportement par défaut
# 

from robot import * 

nb_robots = 0
debug = False 

class Robot_player(Robot):
    
    team_name = "Professor X"
    robot_id = -1
    iteration = 0

    def __init__(self, x_0, y_0, theta_0, name="n/a", team="n/a"):
        global nb_robots
        self.robot_id = nb_robots
        nb_robots+=1
        super().__init__(x_0, y_0, theta_0, name="Robot "+str(self.robot_id), team=self.team_name)

    def step(self, sensors, sensor_view=None, sensor_robot=None, sensor_team=None):
        translation = sensors[sensor_front]*0.1+0.2
        rotation = 0.2 * sensors[sensor_left] + 0.2 * sensors[sensor_front_left] - 0.2 * sensors[sensor_right] - 0.2 * sensors[sensor_front_right] + (random.random()-0.5)*1. #+ sensors[sensor_front] * 0.1
        if debug == True:
            if self.iteration % 100 == 0 and self.robot_id == 0:
                print ("Robot",self.robot_id,"(team "+str(self.team_name)+")","at step",self.iteration,":")
                print ("\tsensors (distance, max is 1.0)  =",sensors)
                print ("\ttype (0:empty, 1:wall, 2:robot) =",sensor_view)
                print ("\trobot's name (if relevant)      =",sensor_robot)
                print ("\trobot's team (if relevant)      =",sensor_team)
        self.iteration = self.iteration + 1
        return translation, rotation, False

'''
def step(robotId, sensors):

    translation = 1
    rotation = 0
    if sensors["sensor_front_left"]["distance"] < 1 or sensors["sensor_front"]["distance"] < 1:
        rotation = 0.5
    elif sensors["sensor_front_right"]["distance"] < 1:
        rotation = -0.5

    return translation, rotation, False
'''
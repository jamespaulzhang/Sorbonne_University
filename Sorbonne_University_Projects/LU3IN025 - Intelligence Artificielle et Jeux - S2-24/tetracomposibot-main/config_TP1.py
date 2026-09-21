# Configuration file.

import arenas

# general -- first three parameters can be overwritten with command-line arguments (cf. "python tetracomposibot.py --help")

display_mode = 0
arena = 1
position = False
max_iterations = 500

# affichage

display_welcome_message = False
verbose_minimal_progress = True  # display iterations
display_robot_stats = False
display_team_stats = False
display_tournament_results = False
display_time_stats = True

# initialization : create and place robots at initial positions (returns a list containing the robots)

import robot_braitenberg_avoider
import robot_braitenberg_loveWall
import robot_braitenberg_hateWall
import robot_braitenberg_loveBot
import robot_braitenberg_hateBot
import robot_subsomption

def initialize_robots(arena_size=-1, particle_box=-1):  # particle_box: size of the robot enclosed in a square
    # Place robots around the center of the arena
    center_x = arena_size // 2
    center_y = arena_size // 2
    offset = 12  # Distance from the center

    positions = [
        (center_x, center_y + offset),        # Top
        (center_x + offset, center_y),        # Right
        (center_x, center_y - offset),        # Bottom
        (center_x - offset, center_y),        # Left
        (center_x + offset, center_y + offset),   # Top-right
        (center_x - offset, center_y - offset),   # Bottom-left
    ]

    robots = []
    robots.append(robot_braitenberg_avoider.Robot_player(positions[0][0], positions[0][1], 0, name="Avoider", team="A"))
    robots.append(robot_braitenberg_loveWall.Robot_player(positions[1][0], positions[1][1], 0, name="LoveWall", team="B"))
    robots.append(robot_braitenberg_hateWall.Robot_player(positions[2][0], positions[2][1], 0, name="HateWall", team="C"))
    robots.append(robot_braitenberg_loveBot.Robot_player(positions[3][0], positions[3][1], 0, name="LoveBot", team="D"))
    robots.append(robot_braitenberg_hateBot.Robot_player(positions[4][0], positions[4][1], 0, name="HateBot", team="E"))
    robots.append(robot_subsomption.Robot_player(positions[5][0], positions[5][1], 0, name="Subsumption", team="S"))

    return robots

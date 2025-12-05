# Configuration file.

import arenas

# general -- first three parameters can be overwritten with command-line arguments (cf. "python tetracomposibot.py --help")

display_mode = 2  # Set display mode to 2 for faster simulation
arena = 1
position = False

# affichage

display_welcome_message = False
verbose_minimal_progress = False  # display iterations
display_robot_stats = False
display_team_stats = False
display_tournament_results = False
display_time_stats = True

# optimization

evaluations = 500
it_per_evaluation = 400
subtrial_per_evaluation = 3
max_iterations = evaluations * it_per_evaluation * subtrial_per_evaluation + 1

# initialization : create and place robots at initial positions (returns a list containing the robots)

import robot_randomsearch2
import genetic_algorithms

def initialize_robots(arena_size=-1, particle_box=-1, algorithm='genetic'):  #algorithm a modifie
    x_center = arena_size // 2 - particle_box / 2
    y_center = arena_size // 2 - particle_box / 2
    robots = []

    if algorithm == 'random':
        robots.append(robot_randomsearch2.Robot_player(x_center, y_center, 0, name="Random Robot", team="A", evaluations=evaluations, it_per_evaluation=it_per_evaluation))
    elif algorithm == 'genetic':
        robots.append(genetic_algorithms.Robot_player(x_center, y_center, 0, name="Genetic Robot", team="B", evaluations=evaluations, it_per_evaluation=it_per_evaluation))
    else:
        raise ValueError("Invalid algorithm type. Choose 'random' or 'genetic'.")

    return robots

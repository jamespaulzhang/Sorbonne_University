#!/usr/bin/env python3

#### #### #### #### #### #### #### #### 
# 
# !!!! !!!! !!!! !!!! !!!! !!!! !!!! !!!!
# !!!! !!!! !!!! !!!! !!!! !!!! !!!! !!!!
#
# !!!! READ BELOW !!!!
#
# * I STRONGLY SUGGEST YOU USE tetracompositor.py INSTEAD *
#
# The version in this file does *not* integrate numba acceleration
# Use *only* if numba is not accessible
# This version will be comparable for display_mode = 0 and 1
# ... but x16 slower in display_mode = 2
# This version will *not* be updated after 2025-03-29
#
# !!!! !!!! !!!! !!!! !!!! !!!! !!!! !!!!
# !!!! !!!! !!!! !!!! !!!! !!!! !!!! !!!!
# 
#### #### #### #### #### #### #### #### 

# Tetracomposibot -- **** NO-OPTIMIZATION VERSION ****
# A simple python simulator for playing with one... to many robots
# "With the Tetracomposibot it has now become possible to play and compose REAL simple robot demo"

# UE IA & JEUX - L3, SU
# LU3IN025 "introduction à l'IA", partie Robotique
# Projet: https://github.com/nekonaute/SU-LU3IN025-robots/blob/main/instructions_projet.md

# contact  : nicolas.bredeche@sorbonne-universite
# initiated: 2025-03-25
#
# History: 
# - 2025-03-24: new code, entirely Python w/ pygame, for LU3IN025
# - 2025-03-29: as of this date, the "official" version is numba-enabled. 
#       This version is kept as a fallback solution if numba is *not* available on your computer and cannot be installed.
#

from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from numba import njit 
import numpy as np
import math
import random
import sys
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import os
import argparse

import arenas


welcome_message = "Tetracomposibot, a simple robot simulator.\n\tSimulate single or multiple robots.\n\tWritten in Python w/ Pygame.\n\tContact: nb@su, 2025."

#############################
### 0. Experimental setup ###
#############################

max_iterations = 5001 # nb of iterations (default - might be modified later)

absolute_orientation = 0  # -90  # in degrees
translation_per_step = 1.0  # in [0,1]
max_translation_per_step = 1.0  # >0. pixel per move.
rotation_per_step = 0.0  # in [-1,1]
max_rotation_per_step = 3.0  # >0, in degrees

arena_size = 100
sensor_length = 10
nb_sensors = 8

display_fps = 10 # default

display_occupancy = True  # Set to True to display visited cells as a background layer.
display_occupancy_alpha = 0.8  # Transparency: from 0 (transparent) to 1 (opaque).
occupancy_robot_to_color = color_list = [
    "#005F91", "#0072B2", "#0091C2", "#00A7D4",  # Blue variations
    "#C65E00", "#E69F00", "#F0A733", "#F7BF66"   # Orange variations
	] # if more robots than in the list, use grey.
#['red', 'orange', 'green', 'blue','pink','violet']  

particle_box = 2  # bounding box (square)

particle = np.zeros((particle_box, particle_box), dtype=int)

particle_radius_real = particle_box / 2.0
center_particle_real = (particle_box / 2 - 0.5, particle_box / 2 - 0.5)

occupancy_scale = 4

####################
### 1. Structure ###
####################

def init():  # perform only once
    global particle, center_particle_real, particle_radius_real, robots_count
    for i in range(particle.shape[0]):
        for j in range(particle.shape[1]):
            if ((i - center_particle_real[0])**2 + (j - center_particle_real[1])**2) <= particle_radius_real**2:
                particle[i, j] = 2
            else:
                particle[i, j] = 3
    init_arena()
    init_trace()
    init_occupancy()
    init_occupancy_small()
    robots_count = 1

def init_arena():
    global arena
    arena = np.zeros((arena_size, arena_size), dtype=int)

def init_trace():
    global trace
    trace = np.zeros((arena_size, arena_size), dtype=int)

def init_occupancy():
    global occupancy
    occupancy = np.zeros((arena_size, arena_size), dtype=int)

def init_occupancy_small():
    global occupancy_small
    occupancy_small = np.zeros((arena_size // occupancy_scale, arena_size // occupancy_scale), dtype=int)

def clear_arena():
    global arena
    arena[arena != 1] = 0

def clear_trace():
    global trace
    trace[trace != 1] = 0

def clear_occupancy():
    global occupancy
    occupancy[:, :] = 0

def clear_occupancy_small():
    global occupancy_small
    occupancy_small[:, :] = 0

def environment_reset():
    clear_arena()
    clear_trace()
    clear_occupancy()
    clear_occupancy_small()

#@njit
def draw_line(x1, y1, x2, y2, color):
    global arena
    num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
    x_points = np.linspace(x1, x2, num_points, dtype=int)
    y_points = np.linspace(y1, y2, num_points, dtype=int)
    for i in range(len(x_points)):
        if 0 <= y_points[i] < arena.shape[0] and 0 <= x_points[i] < arena.shape[1]:
            if arena[y_points[i], x_points[i]] != 1:
                arena[y_points[i], x_points[i]] = color

# returns (normalized distance, sensor type, sensor_robot info)
#@njit
def cast_sensor(x1, y1, x2, y2, color=6):
    global arena, display_cast, occupancy, robot_by_id
    sensor_type = 0  # default: nothing
    sensor_robot_info = "n/a"
    sensor_team_info = "n/a"
    num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
    x_points = np.linspace(x1, x2, num_points, dtype=int)
    y_points = np.linspace(y1, y2, num_points, dtype=int)
    max_distance = math.sqrt((x_points[0]-x_points[-1])**2 + (y_points[0]-y_points[-1])**2) - particle_radius_real
    for i in range(len(x_points)):
        if 0 <= y_points[i] < arena.shape[0] and 0 <= x_points[i] < arena.shape[1]:
            cell_value = arena[y_points[i], x_points[i]]
            if cell_value in [1, 2, 4]:
                dist = (math.sqrt((x_points[0]-x_points[i])**2 + (y_points[0]-y_points[i])**2) - particle_radius_real) / max_distance
                if cell_value == 1:
                    sensor_type = 1  # wall
                    sensor_robot_info = "n/a"
                    sensor_team_info = "n/a"
                else:
                    sensor_type = 2  # robot
                    rid = occupancy[y_points[i], x_points[i]]
                    #print ("rid:",rid,"--",robot_by_id)
                    if rid != 0 and rid in robot_by_id:
                        sensor_robot_info = robot_by_id[rid].name
                        sensor_team_info = robot_by_id[rid].team
                    else:
                        sensor_robot_info = "n/a"
                        sensor_team_info = "n/a"
                return dist, sensor_type, sensor_robot_info, sensor_team_info
            elif display_cast:
                arena[y_points[i], x_points[i]] = color
    return 1.0, 0, "n/a", "n/a"

translation_per_step = 0.0
rotation_per_step = 0.0

############################
### 2. General functions ###
############################

def create_wall(x1, y1, x2, y2):
    arena[max(y1, 0):min(y2, arena_size), max(x1, 0):min(x2, arena_size)] = 1
    if display_trace:
        trace[max(y1, 0):min(y2, arena_size), max(x1, 0):min(x2, arena_size)] = 1

# get_sensors returns three lists.
def get_sensors(x, y, theta):
    global sensor_length, nb_sensors
    source_x, source_y = int(x+particle_radius_real-1), int(y+particle_radius_real-1)
    sensor_values = []
    sensor_view = []
    sensor_robot = []
    sensor_team = []
    for i in range(nb_sensors):
        offset_x = math.cos(math.radians(theta+i*360.0/nb_sensors)) * sensor_length
        offset_y = math.sin(math.radians(theta+i*360.0/nb_sensors)) * sensor_length
        target_x = int(source_x + offset_x + 0.5)
        target_y = int(source_y - offset_y + 0.5)
        dist, view, robot_info, team_info = cast_sensor(int(source_x), int(source_y), target_x, target_y, 6)
        sensor_values.append(dist)
        sensor_view.append(view)
        sensor_robot.append(robot_info)
        sensor_team.append(team_info)
    return sensor_values, sensor_view, sensor_robot, sensor_team

def clean_sensors(x, y, theta):
    global sensor_length, nb_sensors
    if not display_cast:
        return
    source_x, source_y = int(x+particle_radius_real-1), int(y+particle_radius_real-1)
    for i in range(nb_sensors):
        offset_x = math.cos(math.radians(theta+i*360.0/nb_sensors)) * sensor_length
        offset_y = math.sin(math.radians(theta+i*360.0/nb_sensors)) * sensor_length
        target_x = int(source_x + offset_x + 0.5)
        target_y = int(source_y - offset_y + 0.5)
        draw_line(int(source_x), int(source_y), target_x, target_y, 0)

def place_particle(x, y, theta, robot_id):
    global arena, particle, occupancy, occupancy_small
    x, y = int(x), int(y)
    arena_slice = arena[y:y+particle_box, x:x+particle_box]
    collision = np.logical_and(arena_slice != 0, particle == 2)
    if np.any(collision):
        collision_array = np.where(collision, 3, particle)
        return collision_array
    for i in range(particle_box):
        for j in range(particle_box):
            if arena[y+i, x+j] in [0, 4, 6]:
                arena[y+i, x+j] = 2
            occupancy[y+i, x+j] = robot_id
    trace[int(y + particle_box / 2.), int(x + particle_box / 2.)] = 2
    # Update occupancy and occupancy_small
    cell_y = int(y + particle_box / 2.)
    cell_x = int(x + particle_box / 2.)
    #occupancy[cell_y, cell_x] = robot_id
    small_y = cell_y // occupancy_scale
    small_x = cell_x // occupancy_scale
    occupancy_small[small_y, small_x] = robot_id
    offset_x = math.cos(math.radians(theta)) * particle_box / 2.0 * .999
    offset_y = math.sin(math.radians(theta)) * particle_box / 2.0 * .999
    x_head = int(x + particle_box / 2. + offset_x)
    y_head = int(y + particle_box / 2. - offset_y)
    arena[y_head, x_head] = 4
    return None

def erase_particle(x, y):
    global arena, particle
    x, y = int(x), int(y)
    for i in range(particle_box):
        for j in range(particle_box):
            if arena[y+i, x+j] in [2, 4]:
                arena[y+i, x+j] = 0
            occupancy[y+i, x+j] = 0

def draw(arena, double_size=False):
    if double_size:
        plt.figure(figsize=(12.8, 9.6))
    cmap = colors.ListedColormap(['white', 'black', 'grey', 'red', 'orange', 'green'])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(arena, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                 format=plt.FuncFormatter(lambda x, _: {
                     0.5: '0 - nothing', 
                     1.5: '1 - obstacle', 
                     2.5: '2 - robot', 
                     3.5: '3 - collision', 
                     4.5: '4 - robot front', 
                     5.5: '5 - sensor ray'
                 }.get(x, f'{x:.1f}')))
    plt.show()

def show_arena():
    draw(arena, double_size=False)

def show_trace():
    draw(trace, double_size=False)

####################################
### 3. PARTICLE UPDATE functions ###
####################################

#@njit
def update_particle_position(x, y, theta, translation_per_step, rotation_per_step, iteration, noiseFlag=False):
    new_orientation = theta + rotation_per_step
    if noiseFlag:
        noise = 10 * math.sin(2 * math.pi * iteration / 5)
        new_orientation += noise
    orientation_radians = math.radians(new_orientation)
    new_x = x + translation_per_step * math.cos(orientation_radians)
    new_y = y - translation_per_step * math.sin(orientation_radians)
    return new_x, new_y, new_orientation

def update_particle_dynamics(x, y, absolute_orientation, translation_per_step, rotation_per_step, iteration, collision_array):
    return absolute_orientation

#######################
### 4. Main loop    ###
#######################

# Global dictionary to map robot ids to robot instances
robot_by_id = {}

def simulate(my_robots):
    global x_particle, y_particle, absolute_orientation, trace, occupancy, occupancy_small, display_fps, robot_by_id, display_screen, display_frameskip
    environment_reset()
    # Building the robot lookup dictionary
    robot_by_id = {}
    for robot in my_robots:
        robot_by_id[robot.id] = robot
    # Placing initial robots
    for k in range(len(my_robots)):
        collision_array = place_particle(my_robots[k].x0, my_robots[k].y0, my_robots[k].theta0, my_robots[k].id)
        if collision_array is not None:
            print("[ERROR] initial location triggers collision.")
            sys.exit()
    # Initializing pygame (if display is enabled).
    if display_screen:
        pygame.init()
        scale = 5  # scale factor: each arena cell becomes a 5x5 pixel square
        screen = pygame.display.set_mode((arena_size * scale, arena_size * scale))
    iteration = 0
    while iteration < max_iterations:
        if verbose_minimal_progress and iteration % gap_between_display_minimal_progress == 0:
            print("### iteration", iteration, "/", max_iterations, "###")
        count_ask_for_reset = 0
        for k in range(len(my_robots)):
            erase_particle(my_robots[k].x, my_robots[k].y)
            clean_sensors(my_robots[k].x, my_robots[k].y, my_robots[k].theta)
            sensor_values, sensor_view, sensor_robot, sensor_team = get_sensors(my_robots[k].x, my_robots[k].y, my_robots[k].theta)
            translation_per_step, rotation_per_step, ask_for_reset = my_robots[k].step(sensor_values, sensor_view, sensor_robot, sensor_team)
            if ask_for_reset == True:
                count_ask_for_reset = count_ask_for_reset + 1
            my_robots[k].log_sum_of_rotation += abs(max(-1., min(+1., rotation_per_step)))
            backup_x_particle, backup_y_particle = my_robots[k].x, my_robots[k].y
            my_robots[k].x, my_robots[k].y, my_robots[k].theta = update_particle_position(
                my_robots[k].x, my_robots[k].y, my_robots[k].theta,
                max(-1., min(+1., translation_per_step)) * max_translation_per_step,
                max(-1., min(+1., rotation_per_step)) * max_rotation_per_step,
                iteration, noiseFlag=False)
            sensor_values, sensor_view, sensor_robot, sensor_team = get_sensors(my_robots[k].x, my_robots[k].y, my_robots[k].theta)
            collision_array = place_particle(my_robots[k].x, my_robots[k].y, my_robots[k].theta, my_robots[k].id)
            if collision_array is not None:
                if verbose_debug:
                    print("Collision detected!")
                if display_collision:
                    print(collision_array)
                clean_sensors(my_robots[k].x, my_robots[k].y, my_robots[k].theta)
                my_robots[k].x, my_robots[k].y = backup_x_particle, backup_y_particle
                sensor_values, sensor_view, sensor_robot, sensor_team = get_sensors(my_robots[k].x, my_robots[k].y, my_robots[k].theta)
                if display_collision:
                    draw(collision_array)
                my_robots[k].theta = update_particle_dynamics(
                    my_robots[k].x, my_robots[k].y, my_robots[k].theta,
                    max(-1., min(+1., translation_per_step)) * max_translation_per_step,
                    max(-1., min(+1., rotation_per_step)) * max_rotation_per_step,
                    iteration, collision_array)
                place_particle(my_robots[k].x, my_robots[k].y, my_robots[k].theta, my_robots[k].id)
                if verbose_debug:
                    print("Backtrack")
            if my_robots[k].x != backup_x_particle or my_robots[k].y != backup_y_particle:
                my_robots[k].log_sum_of_translation += math.sqrt((my_robots[k].x - backup_x_particle)**2 + (my_robots[k].y - backup_y_particle)**2)
        if display_screen:
            if display_frameskip == 0 or iteration % (display_frameskip+1) == True:		    
                pygame_draw_arena(arena, screen, scale)
                pygame.time.delay(int(1.0/display_fps*1000))
        if verbose_debug:
            print(arena)
        if count_ask_for_reset > 0: # at least one robot asked for reset.
            environment_reset()
            for k in range(len(my_robots)):
                my_robots[k].reset()
                collision_array = place_particle(my_robots[k].x0, my_robots[k].y0, my_robots[k].theta0, my_robots[k].id)
            #print ("[EVENT] 1+ robot(s) asked for reset.")
        iteration += 1
    if display_screen:
        pygame.quit()
    log_pixels_visited = np.sum(trace == 2)
    log_total_sum_of_translation, log_total_sum_of_rotation = 0, 0
    for k in range(len(my_robots)):
        log_total_sum_of_translation += my_robots[k].log_sum_of_translation
        log_total_sum_of_rotation += my_robots[k].log_sum_of_rotation
    retValues = {
        "translations": log_total_sum_of_translation / max_iterations,
        "rotations": log_total_sum_of_rotation / max_iterations,
        "coverage": log_pixels_visited / (arena_size**2)
    }
    return retValues

def blend_with_white(color_str, blend_factor):
    # blend_factor in [0,1]: 0 returns original color, 1 returns white.
    c = pygame.Color(color_str)
    r = int((1 - blend_factor) * c.r + blend_factor * 255)
    g = int((1 - blend_factor) * c.g + blend_factor * 255)
    b = int((1 - blend_factor) * c.b + blend_factor * 255)
    return (r, g, b)


def pygame_draw_arena(arena, screen, scale):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    base_surf = pygame.Surface((arena_size * scale, arena_size * scale), pygame.SRCALPHA)
    base_surf.fill((255, 255, 255))
    
    if display_occupancy:
        cell_draw_size = scale * occupancy_scale
        for y in range(occupancy_small.shape[0]):
            for x in range(occupancy_small.shape[1]):
                rid = occupancy_small[y, x]
                if rid != 0:
                    if rid <= len(occupancy_robot_to_color):
                        color_str = occupancy_robot_to_color[rid - 1]
                    else:
                        color_str = 'grey'
                    col = blend_with_white(color_str, display_occupancy_alpha)
                    #col = pygame.Color(color_str)
                    #col.a = int(display_occupancy_alpha * 255)
                    rect = pygame.Rect(x * cell_draw_size, y * cell_draw_size, cell_draw_size, cell_draw_size)
                    base_surf.fill(col, rect)
    
    # Draw arena cells where arena != 0 (walls, robots, sensor rays, etc)
    color_map = {
        1: (0, 0, 0),        # wall
        2: (128, 128, 128),  # robot body
        3: (255, 0, 0),      # collision
        4: (255, 165, 0),    # robot front
        5: (0, 255, 0),      # sensor ray
        6: (0, 255, 0)       # sensor ray
    }
    for y in range(arena.shape[0]):
        for x in range(arena.shape[1]):
            if arena[y, x] != 0:
                rect = pygame.Rect(x * scale, y * scale, scale, scale)
                base_surf.fill(color_map.get(arena[y, x], (0, 0, 0)), rect)
    
    screen.blit(base_surf, (0, 0))
    pygame.display.flip()


#### #### ####


def display_stats(retValues):
    global display_robot_stats, display_team_stats, display_tournament_results, occupancy

    if display_robot_stats == True:
        print()
        print("Statistiques:")
        print("\t translations      :", retValues['translations'])
        print("\t rotations         :", retValues['rotations'])
        print("\t couverture        :", retValues['coverage'])
        print()

    robot_ids, visited_cells = np.unique(occupancy_small, return_counts=True)
    if display_team_stats == True:
        for value, count in zip(robot_ids, visited_cells):
            if value != 0:
                print(f"  Robot #{value} (\"{robot_by_id[value].name} of team {robot_by_id[value].team}\") captured {count} cells")

    if display_tournament_results == True:
        # per-team counts and result
        team_scores = {}
        for value, count in zip(robot_ids, visited_cells):
            if value != 0:
                team = robot_by_id[value].team if value in robot_by_id else "unknown"
                team_scores[team] = team_scores.get(team, 0) + count

        final_result = ""
        for team, score in team_scores.items():
            #print(f"Team {team} captured {score} cells")
            final_result += "[ " + team + " => " + str(score) + " ]"

        max_score = max(team_scores.values())
        winners = [team for team, score in team_scores.items() if score == max_score]

        if len(winners) == 1:
            #print(f"{winners[0]} wins!")
            final_result += " => " + winners[0] + " wins!"
        else:
            #print("Tie!")
            final_result += " => tie!"

        print (final_result)

#### #### ####

def build_arena(arena_description):
    global arena, arena_size
    nrows = len(arena_description)
    ncols = len(arena_description[0])
    for i in range(nrows):
        for j in range(ncols):
            if arena_description[i][j] == 1:
                x1 = int(j * arena_size / ncols)
                x2 = int((j + 1) * arena_size / ncols)
                y1 = int(i * arena_size / nrows)
                y2 = int((i + 1) * arena_size / nrows)
                create_wall(x1, y1, x2, y2)

# ############################## #
# ############################## #
# ### MAIN PROGRAM         ##### #
# ############################## #
# ############################## #

parser = argparse.ArgumentParser(
    description=welcome_message
)
parser.add_argument(
    "config_file", nargs="?", type=str, default="config",
    help="python configuration filename (eg.: config, or config.py)"
)
parser.add_argument(
    "arena", nargs="?", type=int, default=-1,
    help="arena (default: 0)"
)
parser.add_argument(
    "position", nargs="?", type=str, default="n/a",
    help="starting position (False: default, True: inverse)"
)
parser.add_argument(
    "display_mode", nargs="?", type=int, default=-1,
    help="speed (0: normal ; 1: fast ; 2: fastest, no visual)"
)
parser.add_argument(
    "max_iterations", nargs="?", type=int, default=-1,
    help="number of simulation iterations"
)
args = parser.parse_args()

### #### ####
### General data
### #### ####

config = ""
if args.config_file.endswith('.py'):
    config = __import__(args.config_file[:-3])
else:
    config = __import__(args.config_file)

if args.arena != -1:
    config.arena = args.arena
if args.position != "n/a":
    if args.position == 'True': 
        config.position = True
    elif args.position == 'False':
        config.position = False
    else: 
        print ("Unkown \"position\" argument. Ignored.")
if args.display_mode != -1:
    config.display_mode = args.display_mode
if args.max_iterations != -1:
    config.max_iterations = args.display_mode

arena_size = 100
max_translation_per_step = 1.0
max_rotation_per_step = 10.0
max_iterations = config.max_iterations

display_mode = config.display_mode # 0: 'real-time' w/ display -- 1: fast w/ display -- 2: fastest, no display

display_frameskip = 0

if display_mode == 2:
    display_screen = False
    display_fps = 100000000
    display_frameskip = 100000000
elif display_mode == 1:
    display_screen = True
    display_fps = 100000000
    display_frameskip = 60
elif display_mode == 0:
    display_screen = True
    display_fps = 60
    display_frameskip = 0
else:
    print ("display_mode =",display_mode,"is not implemented. Stopping.")
    sys.exit()

display_collision = False
display_trace = False
display_cast = False

verbose_minimal_progress = config.verbose_minimal_progress # display iterations
gap_between_display_minimal_progress = int(max_iterations / 5)  # display iterations (freq)
verbose_debug = False
display_welcome_message = config.display_welcome_message
display_robot_stats = config.display_robot_stats
display_team_stats = config.display_team_stats
display_tournament_results = config.display_tournament_results
display_time_stats = config.display_time_stats

### #### ####
### Main
### #### ####

if display_welcome_message == True:
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n",welcome_message,"\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

if display_time_stats == True:
    print("[START]", date.today(), datetime.now().strftime("%H:%M:%S"), "GMT")

init()
build_arena(config.arenas.get_arena(config.arena))
robots = config.initialize_robots(arena_size,particle_box)
retValues = simulate(robots)

display_stats(retValues) 

if display_time_stats == True:
    print("[STOP] ", date.today(), datetime.now().strftime("%H:%M:%S"), "GMT")
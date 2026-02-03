#!/usr/bin/python3

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

from __future__ import print_function


import carla
import argparse
import logging
import sys
import time

from mc_utils_screenless import clock_sleep, get_ego_vehicle, setup_world, close_world



WEATHERS = {
    "sunny": carla.WeatherParameters(
        cloudiness=0.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_altitude_angle=75.0,
        sun_azimuth_angle=0.0,
        fog_density=0.0,
        fog_distance=100.0
    ),
    "overcast": carla.WeatherParameters(
        cloudiness=100.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=20.0,
        sun_altitude_angle=40.0,
        sun_azimuth_angle=60.0,
        fog_density=5.0,
        fog_distance=50.0
    ),
    "rainstorm": carla.WeatherParameters(
        cloudiness=100.0,
        precipitation=90.0,
        precipitation_deposits=70.0,
        wind_intensity=60.0,
        sun_altitude_angle=10.0,
        sun_azimuth_angle=180.0,
        fog_density=20.0,
        fog_distance=20.0
    ),
    "denseFog": carla.WeatherParameters(
        cloudiness=50.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_altitude_angle=20.0,
        sun_azimuth_angle=90.0,
        fog_density=80.0,
        fog_distance=5.0
    ),
    "coldWinterMorning": carla.WeatherParameters(
        cloudiness=60.0,
        precipitation=0.0,
        precipitation_deposits=5.0,
        wind_intensity=30.0,
        sun_altitude_angle=5.0,
        sun_azimuth_angle=45.0,
        fog_density=30.0,
        fog_distance=25.0
    ),
    "sunsetGoldenHour": carla.WeatherParameters(
        cloudiness=25.0,               # algunas nubes altas que dan textura al cielo
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        fog_density=8.0,               # niebla ligera para suavizar la escena
        fog_distance=30.0,
        sun_altitude_angle=8.0,        # sol muy bajo -> efecto de atardecer
        sun_azimuth_angle=200.0        # ajusta para que la luz venga del lado deseado
    ),
    "clearDaySoft": carla.WeatherParameters(
        cloudiness=15.0,               # ligero de nubes, no completamente despejado
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=8.0,            # brisa un poco marcada
        sun_altitude_angle=55.0,       # sol alto pero no como el mediodía puro
        sun_azimuth_angle=10.0,
        fog_density=3.0,               # muy suave para dar atmósfera
        fog_distance=80.0
    )
}


def init_game(args):

    original_settings = None

    ### Create the client
    print(f"Creating client to connect to {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    ### Create traffic manager on desired port
    print(f"Connecting to Traffic Manager at {args.tm_port}...")
    traffic_manager = client.get_trafficmanager(args.tm_port)
    print(f"Connected to Traffic Manager at {args.tm_port}")

    ### Get the world object 
    sim_world = client.get_world()
    if args.map is not None:
        print("Loading map: ", args.map) # MOD
        client.load_world(args.map)
        # MODS =====
        try:
            print("Setting weather...")
            sim_world.set_weather(WEATHERS[args.weather])
            print("Weather set")

        except Exception as e:
            print("Exception while loading map: ", e)
            # Not fatal; continue if layers can't be unloaded
            pass
        # MODS =====
        time.sleep(10)

    if args.sync:

    ### Configure simulation in synchronous mode
        original_settings = sim_world.get_settings()
        settings = sim_world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)

    if args.mode == 'auto' and not sim_world.get_settings().synchronous_mode:
        print("WARNING: You are currently in asynchronous mode and could "
              "experience some issues with the traffic simulation")

    print('Setting up world...')
    setup_world(sim_world, args)
    print('World setup complete.')

    ### It's recomended to do first simulation tick outside the simulation loop
    ### to make effective the new settings before entering the loop

    if args.sync:
        sim_world.tick()
    else:
        sim_world.wait_for_tick()

    ## wait for ego vehicle 
    ego = get_ego_vehicle()
    while ego is None:
        clock_sleep(60)
        ego = get_ego_vehicle()

    return client, sim_world, original_settings

def iter_game(args, client, sim_world):
    if args.sync:
        sim_world.tick()

    return None

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    original_settings = None
    client = None
    try:
        client, sim_world, original_settings = init_game(args)

        while True:
            iter_game(args, client, sim_world)
            clock_sleep(60)
    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        close_world(client)


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-t', '--tm_port',
        metavar='T',
        default=8000,
        type=int,
        help='TCP TrafficManager port to listen to (default: 8000)')
    argparser.add_argument(
        '-m', '--mode',
        choices=['manual', 'auto', 'ai'],
        default='manual',
        type=str,
        help='choose initial driving mode of the vehicle (default: manual)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    
    argparser.add_argument(
        '--wait_for_hero',
        action='store_true',
        help='wait for a hero vehicle or create a new one')
    
    argparser.add_argument(
        '--map',
        metavar='TOWN',
        type=str,
        default='Town01',
        help='load a new town (default: Town01)')
    
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        default=False,
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
"""my_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from vehicle import Driver

import random

# create the Robot instance.


# get the time step of the current world.
tesla = Driver()
timestep = int(tesla.getBasicTimeStep())

# fl = car.getDevice('wheelFrontLeft')

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
coef = 1
dir = False
while tesla.step() != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    if dir:
        coef += 0.001 * random.randint(-20, 200)
    else:
        coef -= 0.001 * random.randint(-20, 200)
    if abs(coef) >= 1:
        dir = not dir
        coef = 1
        
    angle = coef * 1.2
    if abs(angle) > 0.8:
        angle = 0.8 * angle/abs(angle)
    
    tesla.setSteeringAngle(angle)
    tesla.setBrakeIntensity(0.0)
    tesla.setGear(1)
    tesla.setThrottle(1)

    #print(angle)
    pass

# Enter here exit cleanup code.

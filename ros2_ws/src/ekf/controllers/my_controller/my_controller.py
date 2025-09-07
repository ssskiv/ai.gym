"""my_controller controller."""

from vehicle import Driver

import random

tesla = Driver()
timestep = int(tesla.getBasicTimeStep())

compass = tesla.getDevice('compass')
compass.enable(timestep)

coef = 1
dir = False
angle = 0

Kp = 1
Ki = 0
Kd = 0

I = 0
last_e = 0

def computePID(target, real, dt):
    global I, last_e
    e = target - real
    I = I + e * dt
    D = (e - last_e) / dt
    last_e = e
    return e * Kp + I * Ki + D * Kd

while tesla.step() != -1:
    
    real_dir = -compass.getValues()[0]
    # print(real_dir)
    
    tesla.setSteeringAngle(computePID(0, real_dir, timestep))
    
    
    tesla.setBrakeIntensity(0.0)
    tesla.setGear(1)
    tesla.setThrottle(1)
    
    pass


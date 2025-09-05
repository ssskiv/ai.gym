"""supervisor_controller controller."""

from controller import Supervisor


sv = Supervisor()
timestep = int(sv.getBasicTimeStep())


### DEFINE NODES BEGIN ###
car = sv.getFromDef('CAR')
gps = sv.getDevice('gps') # GPS to get real speed data

# input data for NN
accel = sv.getDevice('accelerometer')
gyro = sv.getDevice('gyro')

### DEFINE NODES END ###

### ENABLE SENSORS BEGIN ###
gps.enable(timestep)
accel.enable(timestep)
gyro.enable(timestep)
### ENABLE SENSORS END ###

while sv.step(timestep) != -1:



    ### SENSOR READ BEGIN ###
    position = car.getPosition()
    speed_vector = gps.getSpeedVector()
    
    ### SENSOR READ END ###
    
    
    
    
    ### STDIO BEGIN ###
    
    #print(speed_vector)
    print(position)
    
    ### STDIO END ###
    
    
    
    pass

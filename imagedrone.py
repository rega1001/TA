# https://www.playsheep.de/drone/tutorials.html
# import cv2
import time
from pyardrone import ARDrone, at
drone = ARDrone()
drone.navdata_ready.wait() # wait until NavData is ready
while not drone.state.fly_mask:
    drone.takeoff()

drone.send(at.CONFIG('general:navdata_demo', True))
drone.send(at.CONFIG('video:video_channel', 2))
drone.video_ready.wait()
print('av data ready')

while drone.state.fly_mask:
    image = drone.frame
    # cv2.imshow('beeld1', image)

time.sleep(5) # hover for a while

while drone.state.fly_mask:
    drone.land()


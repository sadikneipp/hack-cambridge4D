import camera
import sys
import os
import time

path = sys.argv[1]
now = str(int(time.time()))

try:
    os.makedirs(str(path))
except FileExistsError:
    print('appending to existing folder!')
webcam = camera.init_camera()
i= 0
while True:
    camera.save_ss(webcam, path + '/' + now + '_'  + str(i) + '.jpg')
    i += 1


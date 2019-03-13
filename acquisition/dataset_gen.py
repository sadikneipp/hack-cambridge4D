import camera
import sys
import os
import time

PREPATH = 'dataset_arm/'
path = sys.argv[1]
n_pics = int(sys.argv[2])
now = str(int(time.time()))

try:
    os.makedirs(PREPATH + str(path))
except FileExistsError:
    print('appending to existing folder!')
webcam = camera.init_camera()
for i in range(n_pics):
    camera.save_ss(webcam, PREPATH + path + '/' + now + '_'  + str(i) + '.jpg')

print(f'currently {len(os.listdir(PREPATH + path ))} files on path')
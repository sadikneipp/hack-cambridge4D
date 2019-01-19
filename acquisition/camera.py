import pygame.camera
import pygame.image
import sys
import os

CAM_NO = 2

def init_camera():
    pygame.camera.init()
    cameras = pygame.camera.list_cameras()
    print( "Using camera %s ..." % cameras[CAM_NO])
    webcam = pygame.camera.Camera(cameras[CAM_NO])
    webcam.start()
    return webcam

def snap_ss(webcam):
    img = webcam.get_image()
    return img

def save_ss(webcam, path):

    print('Screenshot saved to ' + path )
    return  pygame.image.save(snap_ss(webcam), path)

if __name__ == '__main__.py':
    path = sys.argv[1]
    os.makedirs(str(path))
    webcam  = init_camera()
    i = 0

    while True:
        save_ss(webcam, path + '/' + str(i) + '.jpg')
        i += 1
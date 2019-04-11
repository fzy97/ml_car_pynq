import pygame
from pygame.locals import *
import socket
import time


MOTION_STOP = b'S'
MOTION_FORWARD  = b'F'
MOTION_REVERSE  = b'B'
MOTION_LEFT     = b'L'
MOTION_RIGHT    = b'R'
MOTION_REVERSE_LEFT  = b'H'
MOTION_REVERSE_RIGHT = b'J'
MOTION_FORWARD_LEFT  = b'G'
MOTION_FORWARD_RIGHT = b'I'

SPEED_LEFT_FAST  = b'5'
SPEED_RIGHT_FAST  = b'5'
SPEED_LEFT_SLOW = b'1'
SPEED_RIGHT_SLOW = b'1'

class RCTest(object):

    def __init__(self):
        pygame.init()
        window_size = Rect(0,0,100,100)#设置窗口的大小
        screen = pygame.display.set_mode(window_size.size)#设置窗口模式
        # initialize TCP connection
        self.address = ('192.168.1.103', 8484)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(self.address)
        # begin control
        self.send_inst = True
        self.steer()

    def steer(self):

        while self.send_inst:
            time.sleep(0.1)
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    key_input = pygame.key.get_pressed()

                    # complex orders
                    if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                        print("Forward Right")
                        self.s.send(MOTION_FORWARD_RIGHT)

                    elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                        print("Forward Left")
                        self.s.send(MOTION_FORWARD_LEFT)

                    elif key_input[pygame.K_DOWN] and key_input[pygame.K_RIGHT]:
                        print("Reverse Right")
                        self.s.send(MOTION_REVERSE_RIGHT)

                    elif key_input[pygame.K_DOWN] and key_input[pygame.K_LEFT]:
                        print("Reverse Left")
                        self.s.send(MOTION_REVERSE_LEFT)

                    # simple orders
                    elif key_input[pygame.K_UP]:
                        print("Forward")
                        self.s.send(MOTION_FORWARD)

                    elif key_input[pygame.K_DOWN]:
                        print("Reverse")
                        self.s.send(MOTION_REVERSE)

                    elif key_input[pygame.K_RIGHT]:
                        print("Right")
                        self.s.send(MOTION_RIGHT)

                    elif key_input[pygame.K_LEFT]:
                        print("Left")
                        self.s.send(MOTION_LEFT)

                    elif event.key == pygame.K_SPACE:
                        print('Stop')
                        self.s.send(MOTION_STOP)
                        time.sleep(0.01)

                    # exit
                    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print('Exit')
                        self.send_inst = False
                        self.s.send(SPEED_LEFT_FAST)
                        time.sleep(0.01)
                        self.s.send(SPEED_RIGHT_FAST)
                        time.sleep(0.01)
                        self.s.send(MOTION_STOP)
                        time.sleep(0.01)
                        self.s.close()
                        break

                # No KEY Pressing
                elif event.type == pygame.KEYUP:
                    self.s.send(SPEED_LEFT_FAST)
                    time.sleep(0.01)
                    self.s.send(SPEED_RIGHT_FAST)
                    time.sleep(0.01)
                    self.s.send(MOTION_STOP)
                    time.sleep(0.01)

if __name__ == '__main__':
    RCTest()

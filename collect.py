import numpy as np
import cv2
import pygame
from pygame.locals import *
import socket
import time
import os
from urllib.request import urlopen
import re

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

class CollectTrainingData(object):
    
    def __init__(self):

        self.send_inst = True

        # create labels
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1

        self.temp_label = np.zeros((1, 4), 'float')


        self.address = ('192.168.1.103', 8484)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(self.address)

        pygame.init()
        window_size = Rect(0,0,100,100)#设置窗口的大小
        screen = pygame.display.set_mode(window_size.size)#设置窗口模式
        self.collect_image()

    def collect_image(self):

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print('Start collecting images...')
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 4), 'float')

        # mjpg-streamer URL
        url = 'http://192.168.1.101:1010/?action=stream'
        stream = urlopen(url)
            
        # Read the boundary message and discard
        stream.readline()

        sz = 0
        rdbuffer = None

        clen_re = re.compile(b'Content-Length: (\d+)\\r\\n')

        # stream video frames one by one
        try:
            stream_bytes = ' '
            frame = 1
            while self.send_inst:
                stream.readline()                    # content type
    
                try:                                 # content length
                    m = clen_re.match(stream.readline()) 
                    clen = int(m.group(1))
                except:
                    break
                
                stream.readline()                    # timestamp
                stream.readline()                    # empty line
                
                # Reallocate buffer if necessary
                if clen > sz:
                    sz = clen*2
                    rdbuffer = bytearray(sz)
                    rdview = memoryview(rdbuffer)
                
                # Read frame into the preallocated buffer
                stream.readinto(rdview[:clen])
                
                stream.readline() # endline
                stream.readline() # boundary
                    
                # This line will need to be different when using OpenCV 2.x
                image = cv2.imdecode(np.frombuffer(rdbuffer, count=clen, dtype=np.byte), flags=cv2.IMREAD_GRAYSCALE)
                height, width = image.shape[:2]

                # shrink image
                size = (int(320), int(240))
                shrink = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                
                # select lower half of the image
                roi = shrink[120:320,:]
                
                # save streamed images
                cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame), image)
                print(image.shape)
                #cv2.imshow('roi_image', roi)
                cv2.imshow('image', image)
                cv2.imshow('image_gray', roi)
                # reshape the roi image into one row array
                temp_array = roi.reshape(1, 38400).astype(np.float32)
                
                frame += 1
                total_frame += 1

                # get input from human driver
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        key_input = pygame.key.get_pressed()

                        # complex orders
                        if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
                            print("Forward Right")
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[1]))
                            saved_frame += 1
                            self.s.send(MOTION_FORWARD_RIGHT)

                        elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
                            print("Forward Left")
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[0]))
                            saved_frame += 1
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
                            saved_frame += 1
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[2]))
                            self.s.send(MOTION_FORWARD)

                        elif key_input[pygame.K_DOWN]:
                            print("Reverse")
                            saved_frame += 1
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[3]))
                            self.s.send(MOTION_REVERSE)
                        
                        elif key_input[pygame.K_RIGHT]:
                            print("Right")
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[1]))
                            saved_frame += 1
                            self.s.send(MOTION_RIGHT)

                        elif key_input[pygame.K_LEFT]:
                            print("Left")
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, self.k[0]))
                            saved_frame += 1
                            self.s.send(MOTION_LEFT)

                        elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                            print('exit')
                            self.send_inst = False
                            self.s.send(MOTION_STOP)
                            break
                                
                    elif event.type == pygame.KEYUP:
                        self.s.send(MOTION_STOP)

            # save training images and labels
            train = image_array[1:, :]
            train_labels = label_array[1:, :]

            # save training data as a numpy file
            file_name = str(int(time.time()))
            directory = "training_data"
            if not os.path.exists(directory):
                os.makedirs(directory)
            try:    
                np.savez(directory + '/' + file_name + '.npz', train=train, train_labels=train_labels)
            except IOError as e:
                print(e)

            e2 = cv2.getTickCount()
            # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print('Streaming duration:', time0)

            print((train.shape))
            print((train_labels.shape))
            print('Total frame:', total_frame)
            print('Saved frame:', saved_frame)
            print('Dropped frame', total_frame - saved_frame)

        finally:
            self.s.close()

if __name__ == '__main__':
    CollectTrainingData()

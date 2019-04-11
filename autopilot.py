import threading
import cv2
import numpy as np
import math
import socket
import re
from urllib.request import urlopen
from sys import exit
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

class ObjectDetection(object):

    def __init__(self):
        self.red_light = False
        self.green_light = False
        self.yellow_light = False
        self.stop_sign = False

    def detect(self, cascade_classifier, gray_image, image):

        # y camera coordinate of the target point 'P'
        v = 0

        # minimum value to proceed traffic light state validation
        threshold = 150     
        
        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
            v = y_pos + height - 5
            # stop sign
            if width/height == 1:
                cv2.putText(image, 'STOP', (x_pos, y_pos-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if width > 190:
                    self.stop_sign = True
                    print("STOP Sign")

            # traffic lights
            else:
                self.stop_sign = False
                roi = gray_image[y_pos+10:y_pos + height-10, x_pos+10:x_pos + width-10]
                mask = cv2.GaussianBlur(roi, (25, 25), 0)
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
                
                # check if light is on
                if maxVal - minVal > threshold:
                    cv2.circle(roi, maxLoc, 5, (255, 0, 0), 2)
                    
                    # Red light
                    if 1.0/8*(height-30) < maxLoc[1] < 4.0/8*(height-30):
                        cv2.putText(image, 'Red', (x_pos+5, y_pos-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if width > 70:
                            self.red_light = True
                            print("Red Light")
                    
                    # Green light
                    elif 5.5/8*(height-30) < maxLoc[1] < height-30:
                        cv2.putText(image, 'Green', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if width > 70:
                            self.green_light = True
                            print("Green Light")
                else:
                    self.red_light = False
                    self.green_light = False
        return v



class RCControl(object):

    def __init__(self):
        self.address = ('192.168.1.103', 8484)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(self.address)
        self.s.send(SPEED_LEFT_FAST)
        time.sleep(0.01)
        self.s.send(SPEED_RIGHT_FAST)
        time.sleep(0.01)
    def steer(self, prediction):
        if prediction == 2:
            self.s.send(MOTION_FORWARD)
            print("Forward")
        elif prediction == 0:
            self.s.send(MOTION_LEFT)
            print("Left")
        elif prediction == 1:
            self.s.send(MOTION_RIGHT)
            print("Right")
        else:
            self.stop()

    def stop(self):
        self.s.send(SPEED_LEFT_FAST)
        time.sleep(0.01)
        self.s.send(SPEED_RIGHT_FAST)
        time.sleep(0.01)
        self.s.send(MOTION_STOP)
        print("Stop")

class VideoStream_Thread(threading.Thread):
    def __init__(self):
        super(VideoStream_Thread, self).__init__()
        self.url = 'http://192.168.1.103:8080/?action=stream'
        self.stream = urlopen(self.url)
        # Read the boundary message and discard
        self.stream.readline()
        self.sz = 0
        self.rdbuffer = None
        self.clen_re = re.compile(b'Content-Length: (\d+)\\r\\n')
        self.stop_cascade = cv2.CascadeClassifier('cascade_xml/stop_sign.xml')
        self.light_cascade = cv2.CascadeClassifier('cascade_xml/traffic_light.xml')
        self.rc_car = RCControl()
        self.obj_detection = ObjectDetection()
        self.model = cv2.ml.ANN_MLP_load('mlp_xml/mlp.xml')
        print("model loaded")

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

    def run(self):

        stop_flag = False
        stop_sign_active = True

        # stream video frames one by one
        try:
            while True:
                self.stream.readline()                    # content type
                try:                                 # content length
                    m = self.clen_re.match(self.stream.readline()) 
                    clen = int(m.group(1))
                except:
                    break
                self.stream.readline()                    # timestamp
                self.stream.readline()                    # empty line
                
                # Reallocate buffer if necessary
                if clen > self.sz:
                    self.sz = clen*2
                    self.rdbuffer = bytearray(self.sz)
                    rdview = memoryview(self.rdbuffer)
                
                # Read frame into the preallocated buffer
                self.stream.readinto(rdview[:clen])
                
                self.stream.readline() # endline
                self.stream.readline() # boundary
                
                # This line will need to be different when using OpenCV 2.x
                image = cv2.imdecode(np.frombuffer(self.rdbuffer, count=clen, dtype=np.byte), flags=cv2.IMREAD_COLOR)
                gray = cv2.imdecode(np.frombuffer(self.rdbuffer, count=clen, dtype=np.byte), flags=cv2.IMREAD_GRAYSCALE)
                
                # shrink image
                size = (int(320), int(240))
                shrink = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
                # lower half of the image
                half_gray = shrink[120:240, :]

                # object detection
                v_param1 = self.obj_detection.detect(self.stop_cascade, gray, image)
                v_param2 = self.obj_detection.detect(self.light_cascade, gray, image)
                cv2.imshow("image", image)

                # cv2.imshow('image', half_gray)
                cv2.imshow('ANN_image', half_gray)




                # reshape image
                image_array = half_gray.reshape(1, 38400).astype(np.float32)
                
                # neural network makes prediction
                prediction = self.predict(image_array)
                # print(prediction)

                # else:
                if self.obj_detection.stop_sign == True or self.obj_detection.red_light == True:
                    self.rc_car.stop()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        time.sleep(0.2)
                        break
                    continue

                self.rc_car.steer(prediction)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.rc_car.stop()
                    time.sleep(0.2)
                    break

            cv2.destroyAllWindows()

        finally:
            self.rc_car.s.close()
            print("Connection closed on thread 1")


class ThreadServer(object):
    video_thread = VideoStream_Thread()
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()

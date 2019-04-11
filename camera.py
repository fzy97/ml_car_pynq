import re
from urllib.request import urlopen
import cv2
import numpy as np
# mjpg-streamer URL
url = 'http://localhost:8080/?action=stream'
stream = urlopen(url)
    
# Read the boundary message and discard
stream.readline()

sz = 0
rdbuffer = None

clen_re = re.compile(b'Content-Length: (\d+)\\r\\n')

# Read each frame
# TODO: This is hardcoded to mjpg-streamer's behavior
while True:
      
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
    img = cv2.imdecode(np.frombuffer(rdbuffer, count=clen, dtype=np.byte), flags=cv2.IMREAD_COLOR)
    # Show Image
    cv2.imshow('Image', img)
    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        exit(0)

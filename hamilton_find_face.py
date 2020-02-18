%pylab inline 
import face_recognition
import cv2
import matplotlib.patches as patches
from IPython.display import clear_output
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt

# loading video
video_capture = cv2.VideoCapture("/hamilton_clip.mp4")

frame_count = 0

while video_capture.isOpened():    
    # a single frame of video
    ret, frame = video_capture.read()

    if not ret:
        video_capture.release()
        break
        
    frame_count += 1
    if frame_count % 15 == 0:    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # display video frame
        title("Input Stream")
        plt.imshow(frame)        

        # find all the faces
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # mark faces on frame with blue dots
        for face_location in face_locations:        
            plt.plot(face_location[1], face_location[0], 'bo')
            plt.plot(face_location[1], face_location[2], 'bo')
            plt.plot(face_location[3], face_location[2], 'bo')
            plt.plot(face_location[3], face_location[0], 'bo')

        plt.show() 
        clear_output(wait=True)
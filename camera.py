import cv2
import numpy as np
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        #image2=image+50
        
        # image2 = stylize(self.sess,self.target,self.content,self.weight,image,self.a)
        # # print(str(self.a)+"--------------------------------------------")
        # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        # fin = np.hstack((image, image2))
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        return image

import cv2
import numpy as np
from pupil_tracker import PupilTracker
from collections import deque, Counter
import time
import matplotlib.pyplot as plt
import argparse
import sys

class EyeOfSauron:
    '''
    User behavior analysis module
    '''
    def __init__(self, display=0):
        # stores coordinates of detected face in the last 60 frames
        self.face_detections_in_time_window = deque([], maxlen=60)
        
        # stores centroids of detected pupils in the last 60 frames
        self.pupil_detections_in_time_window = deque([], maxlen=60)
        
        # stores istory of predictions on user behavior
        self.history = []

        # pupil detection instance
        self.tracker = PupilTracker(display=display)

        # viode capture.
        self.cam = cv2.VideoCapture(0)

    def storeObservations(self, pupils, faces):
        '''
        store faces and pupils in a double ended queue
        '''

        self.face_detections_in_time_window.append(faces)
        self.pupil_detections_in_time_window.append(pupils)
    
    def statistics(self, ):
        '''
        count faces and pupils detected in a single frame
        '''
        
        faces_detected = 0
        pupils_detected = 0
        tiredness_detected = 0
        for x in self.face_detections_in_time_window:
            if x != None:
                faces_detected += 1
        for x in self.pupil_detections_in_time_window:
            if x != None:
                if sum(x[0]) > -4000:
                    pupils_detected += 1
                else:
                    tiredness_detected += 1
        return faces_detected, pupils_detected, tiredness_detected

    def estimateMotion(self, ):
        '''
        determine gaze as pupil centroid displacement.
        '''

        left_displacement = 0
        right_displacement = 0
        valid_pupils = list(filter(lambda x: x is not None ,self.pupil_detections_in_time_window))
        left_pupils = [x[0][:2] for x in valid_pupils if -1000 not in x[0][:2]]
        right_pupils = [x[0][2:] for x in valid_pupils if -1000 not in x[0][2:]]
        for i in range(1, len(left_pupils)):
            t2 = left_pupils[i]
            t1 = left_pupils[i-1]
            displacement = np.sqrt(np.sum(np.square(np.array(t2) - np.array(t1))))
            left_displacement += abs(displacement)
        for i in range(1, len(right_pupils)):
            t2 = right_pupils[i]
            t1 = right_pupils[i-1]
            displacement = np.sqrt(np.sum(np.square(np.array(t2) - np.array(t1))))
            right_displacement += abs(displacement)
        mean_displacement = (left_displacement+right_displacement)/(len(left_pupils)+len(right_pupils))
        return mean_displacement

    def analysis(self, ):
        '''
        perform analysis based on the statistics copmuted.
        if no faces are detected, it is classified as 'no show', i.e absent subject.
        if detected faces in the past 60 frames is less than 50%, it is classified user moving out of bounds.
        if pupils were not detected more than they were in the last 60 frames, it is classified as the subject being tired.
        if the calculated mean pupil displcement is within a specified range, it is classfied as the user being in focus, else distracted.
        '''
        
        fd, pd, td = self.statistics()
        # print (fd, pd, td)
        if fd == 0:
            self.history.append("away")
        elif fd < 50:
            self.history.append("moving")
        else:
            if td > pd:
                self.history.append("tired")
            else:
                displacement = self.estimateMotion()
                # print(displacement)
                if displacement > 20:
                    self.history.append("distracted")
                else:
                    self.history.append("focused")

    def record(self, ):
        '''
        putting everything together and starting the recording.
        '''
        
        while 1:
            _, frame = self.cam.read()
            pupils, faces = self.tracker.test(frame)
            # print(pupils)
            self.storeObservations(pupils, faces)
            if len(self.pupil_detections_in_time_window) >= 60: 
                self.analysis()
                print(self.history[-1])
                cv2.putText(frame, self.history[-1], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            if self.tracker.display:
                try:
                    leftx, lefty, rightx, righty = pupils[0]
                    cv2.rectangle(frame, (leftx-3, lefty-3), (leftx+3, lefty+3), (0,0,255), 2)
                    cv2.rectangle(frame, (rightx-3, righty-3), (rightx+3, righty+3), (0,0,255), 2)
                except:
                    pass
            cv2.imshow("image", frame)
            cv2.waitKey(100)
            # print(pupils)
            # print("\\\\")
            # print(faces)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="gaze tracking",
        description="detects user pupils to track user focus",
        epilog="use the display argument to render the tracking from video feed"
    )
    parser.add_argument('-d', '--display', default=0, required=True, help="renders the tracking")
    args = parser.parse_args()
    # print(args.display)
    # print(args.display)
    barad_dur = EyeOfSauron(args.display)
    # start = time.time()
    try:
        barad_dur.record()
    except KeyboardInterrupt:
        counts = Counter(barad_dur.history)
        f, ax = plt.subplots()
        ax.pie(list(counts.values()), labels=list(counts.keys()))

        plt.show()
        # pass


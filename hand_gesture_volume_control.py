import cv2
import mediapipe as mp
import time
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img,
                              cv2.COLOR_BGR2RGB)  # bcoz this class only accepts RGB images, so converting it into RGB images
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLns in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLns, self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lm = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, ln in enumerate(myHand.landmark):

                h, w, c = img.shape
                cx, cy = int(ln.x * w), int(ln.y * h)

                lm.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

        return lm


def main():
    wCam, hCam = 1080, 720
    cap = cv2.VideoCapture(0)

    cap.set(3, wCam)
    cap.set(4, hCam)

    ptime = 0
    ctime = 0

    detector = handDetector()
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # volume.GetMute()
    # volume.GetMasterVolumeLevel()
    VolRange = volume.GetVolumeRange()

    minVol = VolRange[0]
    maxVol = VolRange[1]
    vol = 0
    volBar = 400
    volPer = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm = detector.findPosition(img)
        if len(lm) != 0:
            # print(lm[4], lm[8])

            x1, y1 = lm[4][1], lm[4][2]
            x2, y2 = lm[8][1], lm[8][2]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            length = math.hypot(x2 - x1, y2 - y1)
            # print(length)

            vol = np.interp(length, [50, 300], [minVol, maxVol])
            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])

            print(int(length), vol)
            volume.SetMasterVolumeLevel(vol, None)

            if length <= 50:
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)  # VOlUME BAR
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)  # VOLUME BAR
            cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # PERCENT

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
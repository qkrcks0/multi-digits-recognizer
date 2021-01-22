import sys
import numpy as np
import cv2

def onMouse(event, x, y, flags, _):
    global oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN:
        oldx, oldy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        oldx, oldy = -1, -1

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(recognizer, (oldx, oldy), (x, y), (255, 255, 255), 20, cv2.LINE_AA)
            oldx, oldy = x, y
            cv2.imshow('recognizer', recognizer)

def norm_digit(img):
    # 무게중심
    m = cv2.moments(img)
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']

    h, w = img.shape[:2]
    aff = np.array([[1, 0, w/2 - cx], [0, 1, h/2 - cy]], dtype=np.float32)
    
    dst = cv2.warpAffine(img, aff, (0, 0)) # 무게중심을 중앙으로 이동
    return dst

net = cv2.dnn.readNet("mnist_cnn.pb")

if net.empty():
    print("Network load failed")
    sys.exit()

recognizer = np.zeros((800,800), np.uint8)
cv2.imshow("recognizer", recognizer)
cv2.setMouseCallback("recognizer", onMouse)

while True:
    c = cv2.waitKey()

    if c == 27: # esc
        break
    elif c == ord(' '):
        
        cnt, _, stats, _ = cv2.connectedComponentsWithStats(recognizer)
        dst = recognizer.copy()
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        stats = sorted(stats, key=lambda x : x[0])

        for i in range(1, cnt):
            (x,y,w,h,s) = stats[i]
            cv2.rectangle(dst, (x-50, y-50), (x+w+50, y+h+50), (0, 0, 255))
            
            crop = dst[y-40:y+h+40, x-40:x+w+40]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            blob = cv2.dnn.blobFromImage(norm_digit(crop), 1/255., (28,28))
            net.setInput(blob)
            prob = net.forward()

            _, maxVal, _, maxLoc = cv2.minMaxLoc(prob)
            digit = maxLoc[0]
            print(f"{digit}", end="")

        print()
        recognizer.fill(0)
        cv2.imshow("recognizer", recognizer)

cv2.destroyAllWindows()

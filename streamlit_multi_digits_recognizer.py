import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas

def norm_digit(img):
    # 무게중심
    m = cv2.moments(img)
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']

    h, w = img.shape[:2]
    aff = np.array([[1, 0, w/2 - cx], [0, 1, h/2 - cy]], dtype=np.float32)
    
    dst = cv2.warpAffine(img, aff, (0, 0)) # 무게중심을 중앙으로 이동
    return dst

@st.cache(allow_output_mutation=True)
def load_net():
    return cv2.dnn.readNet("mnist_cnn.pb")

net = load_net()

st.write("# Multi Digits Recognizer")

CANVAS_SIZE = 320

col1, col2 = st.beta_columns(2)

with col1:
    canvas = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas"
    )

if canvas.image_data is not None:
    img = canvas.image_data.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cnt, _, stats, _ = cv2.connectedComponentsWithStats(img)
    dst = img.copy()
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    stats = sorted(stats, key=lambda x : x[0])

    nums=[]

    for i in range(1, cnt):
        (x,y,w,h,s) = stats[i]
        cv2.rectangle(dst, (x-30, y-30), (x+w+30, y+h+30), (255, 0, 0))

        crop = dst[y-30:y+h+30, x-30:x+w+30]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        blob = cv2.dnn.blobFromImage(norm_digit(crop), 1/255., (28,28))
        net.setInput(blob)
        prob = net.forward()

        st.bar_chart(prob[0])
        
        _, maxVal, _, maxLoc = cv2.minMaxLoc(prob)
        digit = maxLoc[0]
        nums.append(digit)

    col2.image(dst)
    st.write(f"## Result: {str(nums[0:])}")


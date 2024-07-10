import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import matplotlib.image as mpimg    
from moviepy.editor import VideoFileClip
import math 


# Fame masking & region of interest:
def interested_region(img, vertices):
    if len(img.shape) > 2:
        mask_color_ignore = (255,) * img.shape[2]
    else:
        mask_color_ignore = 255

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, mask_color_ignore)
    return cv2.bitwise_and(img, mask)

# Pixels to a line
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines_drawn(line_img, lines)
    return line_img

# Create two lines in each frame after Hough transform
def lines_drawn(img, lines, color=[255,0,0], thickness=6):
    global first_frame
    global cache

    slope_l, slope_r = [], []
    lane_l, lane_r = [], []

    α = 0.2 

    if lines is None:
        print('No lines detected')
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + 1e-5)  # Added small value to avoid division by zero

            if slope > 0.4:
                slope_l.append(slope)
                lane_l.append(line)
            elif slope < -0.4:
                slope_r.append(slope)
                lane_r.append(line)

    if len(lane_l) == 0 or len(lane_r) == 0:
        print('No lane detected')
        return

    slope_mean_l = np.mean(slope_l)
    slope_mean_r = np.mean(slope_r)

    mean_l = np.mean(np.array(lane_l), axis=0)
    mean_r = np.mean(np.array(lane_r), axis=0)

    y1_l = img.shape[0]
    y2_l = int(y1_l - (slope_mean_l * (x2 - x1)))
    x1_l = int((y1_l - mean_l[0][1]) / slope_mean_l + mean_l[0][0])
    x2_l = int((y2_l - mean_l[0][1]) / slope_mean_l + mean_l[0][0])
    
    y1_r = img.shape[0]
    y2_r = int(y1_r - (slope_mean_r * (x2 - x1)))
    x1_r = int((y1_r - mean_r[0][1]) / slope_mean_r + mean_r[0][0])
    x2_r = int((y2_r - mean_r[0][1]) / slope_mean_r + mean_r[0][0])
    
    if x1_l > x1_r:
        x1_l, x1_r = x1_r, x1_l
        y1_l, y1_r = y1_r, y1_l
        y2_l, y2_r = y2_r, y2_l

    present_frame = np.array([x1_l, y1_l, x2_l, y2_l, x1_r, y1_r, x2_r, y2_r], dtype="float32")
    
    if first_frame == 1:
        next_frame = present_frame
        first_frame = 0
    else:
        next_frame = (1 - α) * cache + α * present_frame

    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), color, thickness)
    
    cache = next_frame

# Process each frame to detect lane:
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    global first_frame
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 200], dtype='uint8')
    upper_yellow = np.array([30, 255, 255], dtype='uint8')

    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    gauss_gray = cv2.GaussianBlur(mask_yw_image, (5, 5), 0)
    canny_edges = cv2.Canny(gauss_gray, 50, 150)

    imshape = image.shape
    lower_left = [imshape[1] / 9, imshape[0]]
    lower_right = [imshape[1] - imshape[1] / 9, imshape[0]]
    top_left = [imshape[1] / 2 - imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
    top_right = [imshape[1] / 2 + imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]

    roi_image = interested_region(canny_edges, vertices)
    theta = np.pi / 180
    line_image = hough_lines(roi_image, 4, theta, 30, 100, 180)
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    
    return result


first_frame = 1
white_output = '/Users/chanathippaka/Downloads/4434242-uhd_2160_3840_24fps.mp4'
clip1 = VideoFileClip("/Users/chanathippaka/Downloads/4434242-uhd_2160_3840_24fps.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

# Code for Lane Line Detection Project GUI
import tkinter as tk
from tkinter import Button, LEFT, RIGHT, BOTTOM
from PIL import Image, ImageTk

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global last_frame2
last_frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
global cap2
cap1 = cv2.VideoCapture("/Users/chanathippaka/Downloads/4434242-uhd_2160_3840_24fps.mp4")
cap2 = cv2.VideoCapture("/Users/chanathippaka/Downloads/4434242-uhd_2160_3840_24fps.mp4")

def show_vid():
    global last_frame1
    if not cap1.isOpened():
        print("Can't open the camera")
        return

    flag1, frame1 = cap1.read()
    if flag1:
        frame1 = cv2.resize(frame1, (400, 500))
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)
    else:
        print('Error reading from camera 1')

def show_vid2():
    global last_frame2
    if not cap2.isOpened():
        print("Can't open the camera 2")
        return

    flag2, frame2 = cap2.read()
    if flag2:
        frame2 = cv2.resize(frame2, (400, 500))
        last_frame2 = frame2.copy()
        pic2 = cv2.cvtColor(last_frame2, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(pic2)
        imgtk2 = ImageTk.PhotoImage(image=img2)
        lmain2.imgtk = imgtk2
        lmain2.configure(image=imgtk2)
        lmain2.after(10, show_vid2)
    else:
        print('Error reading from camera 2')

if __name__ == '__main__':
    root = tk.Tk()
    lmain = tk.Label(master=root)
    lmain2 = tk.Label(master=root)
    
    lmain.pack(side=LEFT)
    lmain2.pack(side=RIGHT)
    root.title('Lane Line Detection')
    root.geometry("900x700+100+10")
    exitbutton = Button(root, text='Quit', fg='red', command=root.quit).pack(side=BOTTOM)
    show_vid()
    show_vid2()
    root.mainloop()
    cap1.release()
    cap2.release()
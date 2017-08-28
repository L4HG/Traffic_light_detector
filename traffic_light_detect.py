from os import listdir
from os.path import isfile, join
from collections import deque
import math
import argparse

import numpy as np
import cv2

#определяем границы для красного и зеленого цвета

#красный цвет представляет из себя две области в пространстве HSV
lower_red = np.array([0, 70, 80], dtype = "uint8")
upper_red = np.array([19, 255, 255], dtype = "uint8")

lower_violet = np.array([160, 85, 110], dtype = "uint8")
upper_violet = np.array([180, 255, 255], dtype = "uint8")

#с зеленым все проще - он в центре диапазона
lower_green = np.array([53, 90, 90], dtype = "uint8")
upper_green = np.array([91, 255, 255], dtype = "uint8")

#своя функция для рассчета дистанции - "велосипед", но дает выйгрышь в скорости
def fastest_calc_dist(p1,p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 +
                     (p2[1] - p1[1]) ** 2 )

def find_traffic_light(video_file):
    cap = cv2.VideoCapture(video_file)

    frame_len = 7
    semaphors_array = deque(maxlen=frame_len)
    true_semaphors = []

    offset = 10
    x_mult_offset = 30
    frame_num = 0
    red_frame_array = deque(maxlen=2*frame_len)
    green_frame_array = deque(maxlen=2*frame_len)
    prev_frame = deque(maxlen=2*frame_len)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if frame is None:
            if len(true_semaphors) == 0:
                print(file_name, -1)
            break
        sem_frame = frame_num
        frame_num = frame_num + 1
        frame_h,frame_w, _ = frame.shape

        crop_h, crop_w = int(0.7 * frame_h), int(0.8 * frame_w)
        frame_crop = frame[0:crop_h,int((frame_w-crop_w)/2):int((frame_w+crop_w)/2)]

        blurred = cv2.GaussianBlur(frame_crop, (7, 7), 0.5)
        converted = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        red_mask = cv2.inRange(converted, lower_red, upper_red) + cv2.inRange(converted, lower_violet, upper_violet)
        green_mask = cv2.inRange(converted, lower_green, upper_green)

        if len(prev_frame) < 1:
            prev_frame.appendleft(blurred)

        frameDelta = cv2.absdiff(prev_frame[len(prev_frame)-1], blurred)
        prev_frame.appendleft(blurred)

        diff_red_mask = np.zeros(crop_h*crop_w, dtype = "uint8").reshape(crop_h,crop_w)
        if len(red_frame_array) > 1 :
            diff_red_mask = red_frame_array[len(red_frame_array) - 1] - red_mask
        red_frame_array.appendleft(red_mask)

        diff_green_mask = np.zeros(crop_h*crop_w, dtype = "uint8").reshape(crop_h,crop_w)
        if len(green_frame_array) > 1 :
            diff_green_mask = green_mask - green_frame_array[len(green_frame_array) - 1]
        green_frame_array.appendleft(green_mask)

        diff_red_mask = cv2.erode(diff_red_mask, None, iterations=1)
        diff_red_mask = cv2.dilate(diff_red_mask, None, iterations=3)
        ret, diff_red_mask = cv2.threshold(diff_red_mask,127,250,cv2.THRESH_BINARY)

        diff_green_mask = cv2.erode(diff_green_mask, None, iterations=1)
        diff_green_mask = cv2.dilate(diff_green_mask, None, iterations=3)
        ret, diff_green_mask = cv2.threshold(diff_green_mask,127,250,cv2.THRESH_BINARY)

        cnts_red = cv2.findContours(diff_red_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]


        cnts_green = cv2.findContours(diff_green_mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        area = 0
        red_circle_array = []
        for red_contour in cnts_red:
            red_circle = cv2.minEnclosingCircle(red_contour)
            ((red_contour_x, red_contour_y), red_contour_radius) = red_circle
            min_radius = 2 + (red_contour_x/crop_h) * 3
            max_radius = 35 + (red_contour_x/crop_h) * 30
            if red_contour_radius > min_radius and red_contour_radius < max_radius:
                new_circle = (red_circle, area)
                red_circle_array.append(new_circle)

        green_circle_array = []
        for green_contour in cnts_green:
            green_circle = cv2.minEnclosingCircle(green_contour)
            ((green_contour_x, green_contour_y), green_contour_radius) = green_circle
            min_radius = 2 + (green_contour_x/crop_h) * 3
            max_radius = 35 + (green_contour_y/crop_h) * 30
            if green_contour_radius > min_radius and green_contour_radius < max_radius:
                new_circle = (green_circle, area)
                green_circle_array.append(new_circle)

        semaphors = []
        for red_circle in red_circle_array:
            ((red_x, red_y), red_contour_radius) = red_circle[0]
            red_x = int(red_x)
            red_y = int(red_y)
            top = 0
            left = 0
            right = frameDelta.shape[1]
            bottom = frameDelta.shape[0]
            circle_offset = offset + red_contour_radius
            if top < int(red_y - circle_offset):
                top = int(red_y - circle_offset)
            if left < int(red_x - (x_mult_offset*circle_offset)):
                left = int(red_x - (x_mult_offset*circle_offset))
            if bottom > int(red_y + circle_offset):
                bottom = int(red_y + circle_offset)
            if right > int(red_x + (x_mult_offset*circle_offset)):
                right = int(red_x + (x_mult_offset*circle_offset))
            red_crop = frameDelta[top:bottom,left:right,:]

            total_move_green_red = np.sum(red_crop) - (380*red_contour_radius*red_contour_radius)
            total_move = total_move_green_red / (red_crop.shape[0]*red_crop.shape[1])
            for green_circle in green_circle_array:

                dif_x = red_circle[0][0][0] - green_circle[0][0][0]
                dif_y = red_circle[0][0][1] - green_circle[0][0][1]
                dif_r = red_circle[0][1]/green_circle[0][1]
                radius_red = red_circle[0][1]
                max_dist_y = -5*(radius_red+green_circle[0][1])/2
                if max_dist_y < -170:
                    max_dist_y = -170
                if dif_r > 0.4 and dif_r < 2.5:
                    if dif_x < (radius_red/2) and dif_x > (-radius_red/2):
                        if dif_y > max_dist_y and dif_y < (-(radius_red+green_circle[0][1])/4) and dif_y < -7:
                            if total_move < 33:
                                new_semaphor = (red_circle[0], green_circle[0],total_move)
                                semaphors.append(new_semaphor)

        for semaphor in semaphors:
            true_semaphor = 0
            frame_delta = 1
            for last_semaphors in semaphors_array:
                frame_delta = frame_delta + 1
                for last_semaphor in last_semaphors:
                    distance_red = fastest_calc_dist(semaphor[0][0],last_semaphor[0][0])
                    distance_green = fastest_calc_dist(semaphor[1][0],last_semaphor[1][0])
                    if distance_red < (semaphor[0][1]*0.5) and distance_green < (semaphor[1][1]*0.5):
                        true_semaphor = true_semaphor + 1
                        if (frame_num - frame_delta) < sem_frame:
                            sem_frame = frame_num - frame_delta
                        break
            if true_semaphor > 1:
                true_semaphors.append(semaphor)
                break

        semaphors_array.appendleft(semaphors)

        for semaphor in semaphors:
            cv2.circle(frame_crop, (int(semaphor[0][0][0]), int(semaphor[0][0][1])), int(20),
                                (255, 0, 255), 2)

        for red_circle in red_circle_array:
            cv2.circle(frame_crop, (int(red_circle[0][0][0]), int(red_circle[0][0][1])), int(red_circle[0][1]),
                                (0, 0, 255), 2)

        for green_circle in green_circle_array:
            cv2.circle(frame_crop, (int(green_circle[0][0][0]), int(green_circle[0][0][1])), int(green_circle[0][1]),
                                (0, 255, 0), 2)

        for semaphor in true_semaphors:
            cv2.circle(frame_crop, (int(semaphor[0][0][0]), int(semaphor[0][0][1])), int(semaphor[0][1]),
                                (255, 0, 0), 2)
            cv2.circle(frame_crop, (int(semaphor[1][0][0]), int(semaphor[1][0][1])), int(semaphor[1][1]),
                                (255, 255, 0), 2)

        cv2.imshow("images", frame_crop)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(true_semaphors) > 0:
            print(file_name, sem_frame)
            break

    cap.release()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=False, help="path to videos directory")
    args = vars(ap.parse_args())

    #путь по умолчанию к видеофайлам
    mypath = '/Users/kirillovchinnikov/Downloads/testset/'

    if args['folder'] is not None:
        mypath = args['folder']
    print(mypath)
    #берем все файлы с разрешением .avi и начинаем работу с ними
    file_list = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.split('.')[-1] == 'avi']
    for file_name in file_list:
        find_traffic_light(mypath + file_name)


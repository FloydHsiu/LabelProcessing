from os.path import isdir, isfile, join
import cv2 as cv
import json
import sys
import numpy as np
import matplotlib.pyplot as plt


def draw_bbox_center(img, bbox):
    img = np.copy(img)
    x, y = int(bbox[0]), int(bbox[1])
    width, height = bbox[2], bbox[3]
    h_w, h_h = int(0.5*width), int(0.5*height)
    lt = (x - h_w, y - h_h)
    rb = (x + h_w, y + h_h)
    cv.rectangle(img, lt, rb, (0, 255, 0), 2)
    return img


def draw_bbox_lefttop(img, bbox):
    img = np.copy(img)
    x, y = int(bbox[0]), int(bbox[1])
    width, height = int(bbox[2]), int(bbox[3])
    lt = (x, y)
    rb = (x+width, y+height)
    cv.rectangle(img, lt, rb, (0, 255, 0), 2)
    return img


def draw_bbox_angle(img, bbox):
    img = np.copy(img)
    x, y = bbox[0], bbox[1]
    width, height = bbox[2], bbox[3]
    angle = bbox[4]
    h_w, h_h = 0.5*width, 0.5*height
    lt = np.array([x - h_w, y - h_h, 1])
    lb = np.array([x - h_w, y + h_h, 1])
    rt = np.array([x + h_w, y - h_h, 1])
    rb = np.array([x + h_w, y + h_h, 1])
    rotation_matrix = cv.getRotationMatrix2D(
        (x, y), -1*np.rad2deg(angle)+90, 1)
    lt = np.matmul(rotation_matrix, lt.reshape((3, 1))).reshape(2)
    lb = np.matmul(rotation_matrix, lb.reshape((3, 1))).reshape(2)
    rt = np.matmul(rotation_matrix, rt.reshape((3, 1))).reshape(2)
    rb = np.matmul(rotation_matrix, rb.reshape((3, 1))).reshape(2)
    lt = (int(lt[0]), int(lt[1]))
    rt = (int(rt[0]), int(rt[1]))
    lb = (int(lb[0]), int(lb[1]))
    rb = (int(rb[0]), int(rb[1]))
    img = np.copy(img)
    cv.line(img, lb, lt, (0, 255, 0), 2)
    cv.line(img, lt, rt, (0, 255, 0), 2)
    cv.line(img, rt, rb, (0, 255, 0), 2)
    cv.line(img, rb, lb, (0, 255, 0), 2)
    return img


def draw_seg(img, seg):
    lb = (int(seg[0][0]), int(seg[0][1]))
    lt = (int(seg[1][0]), int(seg[1][1]))
    rt = (int(seg[2][0]), int(seg[2][1]))
    rb = (int(seg[3][0]), int(seg[3][1]))
    img = np.copy(img)
    cv.line(img, lb, lt, (0, 255, 0), 2)
    cv.line(img, lt, rt, (0, 255, 0), 2)
    cv.line(img, rt, rb, (0, 255, 0), 2)
    cv.line(img, rb, lb, (0, 255, 0), 2)
    return img


def draw_parallelogram(img, parallelogram):
    width = parallelogram[2]
    height = parallelogram[3]
    slope = np.tan(parallelogram[4])
    lt = (int(parallelogram[0]), int(parallelogram[1]))
    rt = (int(lt[0]+width), int(lt[1]))
    b_left = lt[1] - slope*lt[0]
    lb_y = lt[1]+height
    lb = (int((lb_y-b_left)/slope), int(lb_y))
    rb = (int(lb[0]+width), lb[1])
    img = np.copy(img)
    cv.line(img, lb, lt, (0, 255, 0), 2)
    cv.line(img, lt, rt, (0, 255, 0), 2)
    cv.line(img, rt, rb, (0, 255, 0), 2)
    cv.line(img, rb, lb, (0, 255, 0), 2)
    return img


def draw_parallelogram_p(img, parallelogram_p):
    lb = (int(parallelogram_p[0][0]), int(parallelogram_p[0][1]))
    lt = (int(parallelogram_p[1][0]), int(parallelogram_p[1][1]))
    rt = (int(parallelogram_p[2][0]), int(parallelogram_p[2][1]))
    rb = (int(parallelogram_p[3][0]), int(parallelogram_p[3][1]))
    img = np.copy(img)
    cv.line(img, lb, lt, (0, 255, 0), 2)
    cv.line(img, lt, rt, (0, 255, 0), 2)
    cv.line(img, rt, rb, (0, 255, 0), 2)
    cv.line(img, rb, lb, (0, 255, 0), 2)
    return img


def tuple_int(point):
    tmp = tuple()
    for i in point:
        tmp = tmp + (int(i),)
    return tmp

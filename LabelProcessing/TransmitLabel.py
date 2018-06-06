from os.path import isdir, join, isfile
from os import listdir, mkdir
import json
import sys
import tqdm
import numpy as np

def seg2bbox_lefttop(seg):
    # transmit a segmentation label into bounding box label
    # input: 
    #    seg: list([xi, yi])
    # output:
    #    bbox_lefttop: [x, y, w, h]
    if not(len(seg) >= 1):
        return False
    x_left, x_right = seg[0][0], seg[0][0]
    y_top, y_bottom = seg[0][1], seg[0][1]
    for i in range(1, len(seg)):
        x, y = seg[i][0], seg[i][1]
        # compare x boundary
        if x < x_left:
            x_left = x
        elif x > x_right:
            x_right = x
        else:
            pass
        # compare y boundary
        if y < y_top:
            y_top = y
        elif y > y_bottom:
            y_bottom = y
        else:
            pass
    return [x_left, y_top, x_right - x_left, y_bottom - y_top]

def seg2bbox_center(seg):
    # transmit a segmentation label into bounding box label
    # input: 
    #    seg: list([xi, yi])
    # output:
    #    bbox_center: [x, y, w, h]
    if not(len(seg) >= 1):
        return False
    x_left, x_right = seg[0][0], seg[0][0]
    y_top, y_bottom = seg[0][1], seg[0][1]
    for i in range(1, len(seg)):
        x, y = seg[i][0], seg[i][1]
        # compare x boundary
        if x < x_left:
            x_left = x
        elif x > x_right:
            x_right = x
        else:
            pass
        # compare y boundary
        if y < y_top:
            y_top = y
        elif y > y_bottom:
            y_bottom = y
        else:
            pass
        width = x_right - x_left
        height = y_bottom - y_top
    return [x_left + width/2, y_top + height/2, width, height]

def seg2parallelogram(seg):
    # assume the segmentation is as the type below
    #    ----------
    #   2        3
    #  /         / (top-bottom are horizontal, and we need to calculate slope of left-right)
    # 1        4
    # ----------
    # input:
    #   seg: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # output:
    #   parallelogram: [x, y, w, h, angle]
    if len(seg) is not 4:
        return False
    #calculate left-right slope
    angle_left = np.arctan2(seg[0][1] - seg[1][1], seg[0][0] - seg[1][0])
    angle_right = np.arctan2(seg[3][1] - seg[2][1], seg[3][0] - seg[2][0])
    slope = np.tan((angle_left + angle_right)/2)
    #avoid degree(90) --> if deg > 89.99: call seg2bbox and append(90)
    if np.abs(slope) > 5730.0:
        bbox = seg2bbox_center(seg)
        return bbox.append(90)
    # calculate four line of slope passing for seg point
    # ax + b = y --> b = y - ax (b larger-> left side, b more less-> right side)
    b1 = seg[0][1] - slope * seg[0][0]
    b2 = seg[1][1] - slope * seg[1][0]
    b3 = seg[2][1] - slope * seg[2][0]
    b4 = seg[3][1] - slope * seg[3][0]
    b_left, b_right = b1, b1
    for bi in [b2, b3, b4]:
        if bi < b_left:
            b_left = bi
        elif bi > b_right:
            b_right = bi
        else:
            pass
    # calculate top-bottom
    y_top, y_bottom = seg[0][1], seg[0][1]
    for i in range(1, 4):
        if seg[i][1] > y_bottom:
            y_bottom = seg[i][1]
        elif seg[i][1] < y_top:
            y_top = seg[i][1]
        else:
            pass
    # calculate left-top, right-top point
    p_lt = [(y_top - b_left)/slope, y_top]
    p_rt = [(y_top - b_right)/slope, y_top]
    w = np.abs(p_lt[0] - p_rt[0])
    h = np.abs(y_bottom - y_top)
    angle = (angle_left + angle_right)/2
    return [p_lt[0], p_lt[1], w, h, angle]

def seg2parallelogram_point(seg):
    # assume the parallelogram is as the type below
    #    ----------
    #   2        3
    #  /         / (top-bottom are horizontal, and we need to calculate slope of left-right)
    # 1        4
    # ----------
    # input:
    #   seg: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # output:
    #   parallelogram_point: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]
    if len(seg) is not 4:
        return False
    #calculate left-right slope
    angle_left = np.arctan2(seg[0][1] - seg[1][1], seg[0][0] - seg[1][0])
    angle_right = np.arctan2(seg[3][1] - seg[2][1], seg[3][0] - seg[2][0])
    slope = np.tan((angle_left + angle_right)/2)
    #avoid degree(90) --> if deg > 89.99: call seg2bbox and append(90)
    if np.abs(slope) > 5730.0:
        bbox = seg2bbox_center(seg)
        return bbox.append(90)
    # calculate four line of slope passing for seg point
    # ax + b = y --> b = y - ax (b larger-> left side, b more less-> right side)
    b1 = seg[0][1] - slope * seg[0][0]
    b2 = seg[1][1] - slope * seg[1][0]
    b3 = seg[2][1] - slope * seg[2][0]
    b4 = seg[3][1] - slope * seg[3][0]
    b_left, b_right = b1, b1
    for bi in [b2, b3, b4]:
        if bi < b_left:
            b_left = bi
        elif bi > b_right:
            b_right = bi
        else:
            pass
    # calculate top-bottom
    y_top, y_bottom = seg[0][1], seg[0][1]
    for i in range(1, 4):
        if seg[i][1] > y_bottom:
            y_bottom = seg[i][1]
        elif seg[i][1] < y_top:
            y_top = seg[i][1]
        else:
            pass
    # calculate left-bottom, right-bottom, left-top, right-top point
    p_lb = [(y_bottom - b_left)/slope, y_bottom]
    p_rb = [(y_bottom - b_right)/slope, y_bottom]
    p_lt = [(y_top - b_left)/slope, y_top]
    p_rt = [(y_top - b_right)/slope, y_top]
    return [p_lb, p_lt, p_rt, p_rb]

def seg2bbox_angle(seg):
    # assume the parallelogram is as the type below
    #    ----------
    #   2        3
    #  /         / (top-bottom are horizontal, and we need to calculate slope of left-right)
    # 1        4
    # ----------
    # input:
    #   seg: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # output:
    #   bbox_angle: [x_mid, y_mid, w, h, angle]
    if len(seg) is not 4:
        return False
    parallelogram = seg2parallelogram(seg)
    lt = (parallelogram[0], parallelogram[1])
    width, height, angle = parallelogram[2], parallelogram[3], parallelogram[4]
    a_left = a_right = np.tan(angle)
    a_top = a_bottom = -1 / a_left
    b_left = lt[1] - a_left*lt[0]
    rt = (lt[0]+width, lt[1])
    b_right = rt[1] - a_right*(rt[0])
    #cal left_bottom
    lb_y = lt[1] + height
    lb = ((lb_y -b_left)/a_left, lb_y)
    #cal b_top, b_bottom
    b_top = lt[1] - a_top*lt[0]
    b_bottom = lb[1] - a_bottom*lb[0]
    #cal bbox_angle 4 point
    bbox_lt = lt
    bbox_lb = lb
    bbox_rt_x = -1*(b_top - b_right) / (a_top - a_right)
    bbox_rb_x = -1*(b_bottom - b_right) / (a_bottom - a_right)
    bbox_rt = (bbox_rt_x, a_right*bbox_rt_x+b_right)
    bbox_rb = (bbox_rb_x, a_right*bbox_rb_x+b_right)
    bbox_mid = ((bbox_lb[0]+bbox_rt[0])/2, (bbox_lb[1]+bbox_rt[1])/2)
    w = ((bbox_lb[0] - bbox_rb[0])**2 + (bbox_lb[1] - bbox_rb[1])**2)**0.5
    h = ((bbox_lb[0] - bbox_lt[0])**2 + (bbox_lb[1] - bbox_lt[1])**2)**0.5
    return [bbox_mid[0], bbox_mid[1], w, h, angle]

def seg2seg_expand(seg, origin, r_e, l_e):
    # expand label area
    # input:
    #   seg: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    #   origin: (h, w)
    #   r_e: percentage of right side expand
    #   l_e: percentage of left side expand
    # return:
    #   seg: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] or False
    if len(seg) is not 4:
        print(f'Error: input segmentation must be 4 points.')
        return False
    #calculate x's width of segmentation
    lb = seg[0][0]
    lt = seg[1][0]
    rt = seg[2][0]
    rb = seg[3][0]
    w = ((rb - lb) + (rt - lt))/ 2
    lb = seg[0][0] - w * l_e
    lt = seg[1][0] - w * l_e
    rt = seg[2][0] + w * r_e
    rb = seg[3][0] + w * r_e
    if lb<0 or lt<0 or rt>origin[1] or rb>origin[1]:
        return False
    return [lb, lt, rt, rb]

def resize2short(seg, origin, short):
    # resize label
    # input: 
    #   seg: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    #   origin: (h, w)
    #   short: int
    # return:
    #   seg: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    
    #choose short side and calculate scale
    if origin[0] >= origin[1]:
        scale = short / origin[1]
    else:
        scale = short / origin[0]
    seg_resized = []
    for s in seg:
        seg_resized.append([s[0]*scale, s[1]*scale])
    return seg_resized
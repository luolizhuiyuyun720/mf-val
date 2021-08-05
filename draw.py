import json
import os
import cv2
import sys

sys.path.append("/home/lynn/Work/utils/mf_draw")
# sys.path.append("/home/lynn/Work/utils/mf_draw/config")

from config.config import cfg
# import config.cfg as cfg
import numpy as np
import numba as nb
from numba import jit 

def load_json_file(json_file_path):
    with open(json_file_path, 'r') as fin:
        o = json.load(fin)
    return o

def DrawCanvas(bird_view, center_x, center_y, radius, factor):

    cv2.circle(bird_view, (center_x, center_y), radius[0], (255, 255, 0), 1)
    cv2.circle(bird_view, (center_x, center_y), radius[1], (255, 255, 0), 1)
    cv2.circle(bird_view, (center_x, center_y), radius[2], (255, 255, 0), 1)
    cv2.circle(bird_view, (center_x, center_y), radius[3], (255, 255, 0), 1)
    cv2.circle(bird_view, (center_x, center_y), radius[4], (255, 255, 0), 1)
    cv2.circle(bird_view, (center_x, center_y), radius[5], (255, 255, 0), 1)
    cv2.circle(bird_view, (center_x, center_y), radius[6], (255, 255, 0), 1)
    # cv2.rectangle(bird_view, (365, 390), (415, 410), (255, 255, 0), 1)
    cv2.putText(bird_view, "10m", (int(center_x + 10 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "20m", (int(center_x + 20 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "30m", (int(center_x + 30 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "40m", (int(center_x + 40 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "50m", (int(center_x + 50 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "60m", (int(center_x + 60 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "70m", (int(center_x + 70 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)

    cv2.putText(bird_view, "-10m", (int(center_x - 10 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "-20m", (int(center_x - 20 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "-30m", (int(center_x - 30 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "-40m", (int(center_x - 40 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "-50m", (int(center_x - 50 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "-60m", (int(center_x - 60 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "-70m", (int(center_x - 70 * factor / 0.2), center_y), cv2.FONT_HERSHEY_SIMPLEX, factor * 0.2 - 0.1, (0, 0, 0), 1, 2)

    return bird_view

def DrawAnnotation(bird_view):
    a = np.array([[[10, 10], [60, 10], [60, 20], [10, 20]]], dtype=np.int32)
    cv2.fillPoly(bird_view, a, (128, 0, 128))
    b = np.array([[[10, 20], [60, 20], [60, 30], [10, 30]]], dtype=np.int32)
    cv2.fillPoly(bird_view, b, (220, 20, 60))
    c = np.array([[[10, 30], [60, 30], [60, 40], [10, 40]]], dtype=np.int32)
    cv2.fillPoly(bird_view, c, (255, 165, 0))
    d = np.array([[[10, 40], [60, 40], [60, 50], [10, 50]]], dtype=np.int32)
    cv2.fillPoly(bird_view, d, (46, 139, 87))
    e = np.array([[[10, 50], [60, 50], [60, 60], [10, 60]]], dtype=np.int32)
    cv2.fillPoly(bird_view, e, (182, 194, 154))

    f = np.array([[[10, 65], [60, 65], [60, 75], [10, 75]]], dtype=np.int32)
    cv2.polylines(bird_view, f, 1, (255, 0, 0))
    g = np.array([[[10, 77], [60, 77], [60, 87], [10, 87]]], dtype=np.int32)
    cv2.polylines(bird_view, g, 1, (255, 140, 0))
    h = np.array([[[10, 89], [60, 89], [60, 99], [10, 99]]], dtype=np.int32)
    cv2.polylines(bird_view, h, 1, (0, 100, 0))

    cv2.putText(bird_view, "> 3.5m", (65, 17), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "2.0 ~ 3.5m", (65, 27), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "0.5 ~ 2.0m", (65, 37), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "-0.5 ~ 0.5m", (65, 47), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "< -0.5m", (65, 57), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "gt_box", (65, 72), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "gt_box ignore", (65, 84), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1, 2)
    cv2.putText(bird_view, "dt_box", (65, 97), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 0), 1, 2)

    return bird_view

@nb.jit(nopython=True)
def DrawPointcloud(birdview, lidar, x_min, x_max, y_min, y_max, voxel_x_size, voxel_y_size, factor=1):
    """
    将raw点云规整后安排在事先规划好大小的网格中，根据网格中每个点位是否有
    点来将该位置（相当于像素点）设置值（根据距离不同可以设置不同颜色），v2
    版本函数会讲gt框内点云按照高度等级绘制颜色.
    :param lidar: (N', 4)
    :param factor: 比例
    :return: (w, l, 3)图像
    """
    # birdview = np.zeros((cfg.INPUT_HEIGHT * factor, cfg.INPUT_WIDTH * factor, 3))
    # birdview[:, :, :] = [255, 255, 255]
    # numba加速比支持easydict，而cfg里面的X_MIN等参数是基于easy_dict构建的，所以这里面单独写出来。

    X_MIN = x_min
    X_MAX = x_max
    Y_MIN = y_min
    Y_MAX = y_max
    VOXEL_X_SIZE = voxel_x_size
    VOXEL_Y_SIZE = voxel_y_size

    if lidar is not None:
        filter = np.logical_and(lidar[:, 2] >= -5.0, lidar[:, 2] <= 5.0)
        lidar = lidar[filter]
        for kk in range(len(lidar)):
            if kk % 1 == 0:
                x, y = lidar[kk][0:2]
                z = lidar[kk][2]
                # print(z)
                # y, x = point[0], point[2]
                # y = -1 * y
                # 将x，y变成整数，并且以(-40,-40)为坐标原点
                if X_MIN < x < X_MAX and Y_MIN < y < Y_MAX:
                    x, y = int((x - X_MIN) / VOXEL_X_SIZE *factor), int(
                        (y - Y_MIN) / VOXEL_Y_SIZE *factor)
                    # print("{}--{}".format(x,y))
                    # z = 0.5 * np.sqrt(np.sum([(x - 200)**2, (y - 200)**2]))
                    # print(z)
                    if z <= -0.5:
                        birdview[y, x, :] = [182, 194, 154]  # 蓝色
                    elif z > -0.5 and z <= 0.5:
                        birdview[y, x, :] = [46, 139, 87]  # 绿色
                    elif z > 0.5 and z <= 2.0:
                        birdview[y, x, :] = [255, 165, 0]  # 黄色
                    elif z > 2.0 and z <= 3.5:
                        birdview[y, x, :] = [220, 20, 60]  # 橙红
                    else:
                        birdview[y, x, :] = [128, 0, 128]  # 紫色

    return birdview

# {'image_idx': [bbox[x,y,z,w,h,l,th], scores, num_pt, velo[vx, vy], track_id]}
def Json2BBoxInfo(json_file):
    o = load_json_file(json_file)
    frame_count = len(o)

    json_dict = {}

    for k in range(frame_count):
        if o[k] is None:
            continue

        image_idx = o[k]['image_idx']
        new_dict = {}
        new_dict['location'] = o[k]['location']
        new_dict['dimensions'] = o[k]['dimensions']
        new_dict['rotation_y'] = o[k]['rotation_y']
        new_dict['name'] = o[k]['name']
        new_dict['index'] = o[k]['index']
        
        if 'scores' in o[k].keys():
            new_dict['scores'] = o[k]['scores']
        
        if 'score' in o[k].keys():
            new_dict['scores'] = o[k]['score']

        if 'num_pts' in o[k].keys():
            new_dict['num_pts'] = o[k]['num_pts']
        
        if 'velocity' in o[k].keys():
            new_dict['velocity'] = o[k]['velocity']

        if 'track_id' in o[k].keys():
            new_dict['track_id'] = o[k]['track_id']
        
        json_dict[image_idx] = new_dict

    return json_dict


def center2cornor_bbox3d(boxes):
    location = np.array(boxes['location'])
    dim = np.array(boxes['dimensions'])
    rotation_y = np.array(boxes['rotation_y'])

    box_count = location.shape[0]
    if box_count == 0:
        return None

    ret = np.zeros((box_count, 8, 3), dtype=np.float32)

    for k in range(box_count):
        translation = location[k][0:3]
        size = dim[k][0:3]
        yaw = rotation_y[k]
        # yaw = limit_period(yaw + np.pi/2)

        h, w, l = size[0], size[1], size[2] #TODO
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h]])
        
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
                          np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()

        ret[k] = box3d
    
    return ret

def limit_period(val, offset=0.5, period=2 * np.pi):
    return val - np.floor(val / period + offset) * period

def points_to_bird_view(x, y):
    """
    将雷达数据转换为birdview数据（整的）
    """
    # using the cfg.INPUT_XXX
    a = (x - cfg.X_MIN) / cfg.VOXEL_X_SIZE * cfg.BV_LOG_FACTOR
    b = (y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * cfg.BV_LOG_FACTOR
    a = np.clip(a, a_max=(cfg.X_MAX - cfg.X_MIN) / cfg.VOXEL_X_SIZE * cfg.BV_LOG_FACTOR, a_min=0)
    b = np.clip(b, a_max=(cfg.Y_MAX - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * cfg.BV_LOG_FACTOR, a_min=0)
    return a, b

def DrawBBoxInfo(img, bbox_info, color, velo_color, text_color, text_offset, thickness):
    box_corner = center2cornor_bbox3d(bbox_info)
    if box_corner is None:
        return img

    num_pts = None
    if 'num_pts' in bbox_info.keys():
        num_pts = bbox_info['num_pts']

    box_velo = None
    if 'velocity' in bbox_info.keys():
        box_velo = bbox_info['velocity']
    
    box_score = None
    if 'scores' in bbox_info.keys():
        box_score = bbox_info['scores']
    
    track_id = None
    if 'track_id' in bbox_info.keys():
        track_id = bbox_info['track_id']

    for k in range(len(box_corner)):
        box = box_corner[k]
        x0, y0 = points_to_bird_view(*box[0, 0:2])
        x1, y1 = points_to_bird_view(*box[1, 0:2])
        x2, y2 = points_to_bird_view(*box[2, 0:2])
        x3, y3 = points_to_bird_view(*box[3, 0:2])
        # gt_color=(255, 0, 255), thickness=1
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)), color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)), color, thickness, cv2.LINE_AA)   

        if box_velo is not None:
            vx, vy = box_velo[k][0], box_velo[k][1]
            if vx < -60 or vx > 60 or vy < -60 or vy > 60:
               pass 
            else:
                cv2.arrowedLine(img, (int(0.5*x0+0.5*x2), int(0.5*y0+0.5*y2)), 
                                (int(0.5*x0+0.5*x2 + vx*5), int(0.5*y0+0.5*y2-vy*5)), 
                                velo_color, 2, cv2.LINE_AA)

        str_score = ''
        if box_score is not None:
            str_score = bbox_info['name'][k] + ': ' + str(round(box_score[k], 2))

        str_pts = ''
        if num_pts is not None:
            str_pts = 'num_pt: ' + str(bbox_info['num_pts'][k])
        
        str_track_id = ''
        if track_id is not None:
            str_track_id = 'track_id: ' + str(track_id[k]) 

        str_text = str_track_id + ', ' +  str_score + ', ' + str_pts

        cv2.putText(img, str_text, (int(x1 + text_offset[0]), int(y1) + text_offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, text_color, 1, 2)
    return img

def DrawFrameImage(bin_file, gt_box_info, dt_box_info, output_file):
    # bin_file = "/media/lynn/ElementSELynn/WANGSHUO/bin/1616665116.3878.bin"
    raw_lidar = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
    
    birdview = np.zeros((cfg.INPUT_HEIGHT * cfg.BV_LOG_FACTOR, 
                        cfg.INPUT_WIDTH * cfg.BV_LOG_FACTOR, 3))
    birdview[:, :, :] = [255, 255, 255]

    bird_view = DrawPointcloud(birdview, raw_lidar, 
                               cfg.X_MIN, cfg.X_MAX, cfg.Y_MIN, cfg.Y_MAX,
                               cfg.VOXEL_X_SIZE, cfg.VOXEL_Y_SIZE, cfg.BV_LOG_FACTOR)
    bird_view = bird_view[::-1, :, :]

    img = bird_view.copy() # why not DrawAnnotation(bird_view)

    # TODO: draw bbox
    img = DrawBBoxInfo(img, gt_box_info, (255,0,0), (255,128,0), (0,0,0), [2,-5], 1)
    img = DrawBBoxInfo(img, dt_box_info, (65,105,255), (0, 0, 255), (0, 0, 0), [-2, 5], 1)

    bird_view = DrawAnnotation(img)

    bird_view = cv2.cvtColor(bird_view.astype(np.uint8), cv2.COLOR_BGR2RGB)

    bird_view = DrawCanvas(bird_view, cfg.CENTER_X, cfg.CENTER_Y, 
                           [cfg.RADIUS_1, cfg.RADIUS_2, cfg.RADIUS_3, cfg.RADIUS_4, cfg.RADIUS_5, cfg.RADIUS_6, cfg.RADIUS_7],
                           cfg.BV_LOG_FACTOR)

    print('writing image: ' + output_file)
    cv2.imwrite(output_file, bird_view)


if __name__ == '__main__':
    # cfg = Config.fromfile('./config.py')

    gt_file = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/August1/gt.json'
    # dt_file = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/August1/dt.json'
    dt_file = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/August1/cpp_dt2.json'
    bin_path = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/bin/'
    output_path = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/August1/bev_cpp2/'

    dt_json = Json2BBoxInfo(dt_file)
    gt_json = Json2BBoxInfo(gt_file)

    for key in dt_json.keys():
        if key not in gt_json:
            continue
        bin_file = bin_path + key + '.bin'
        if not os.path.exists(bin_file):
            continue
        output_file = output_path + key + '.png'
        DrawFrameImage(bin_file, gt_json[key], dt_json[key], output_file)

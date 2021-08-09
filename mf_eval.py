import json
import os
import cv2
import sys

sys.path.append("/home/lynn/Work/utils/mf_draw")
# sys.path.append("/home/lynn/Work/utils/mf_draw/config")

from config.config import cfg
# import config.cfg as cfg
import numpy as np


def center_distance(gt_box: list, pred_box: list) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box[:2]) - np.array(gt_box[:2]))

def riou_distance(gt_box: list, pred_box: list) -> float:
    """
    gt_box = [x,y,z, w, l, h, theta]
    dt_box = [x,y,z, w, l, h, theta]
    """
    r1 = ((gt_box[0], gt_box[1]), (gt_box[3], gt_box[4]), gt_box[6] * 360.0/np.pi)
    r2 = ((pred_box[0], pred_box[1]), (pred_box[3], pred_box[4]), pred_box[6] * 360.0/np.pi)
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    
    if int_pts is None:
        return np.nan
    else:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        inter_area = cv2.contourArea(order_pts)
        
        area1 = gt_box[3] * gt_box[4]
        area2 = pred_box[3] * pred_box[4]

        iou = inter_area * 1.0 / (area1 + area2 - inter_area)

        return 1.0 - iou


def velocity_l2(gt_box_vel:list, pred_box_vel:list) -> float:
    """
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(gt_box_vel) - np.array(pred_box_vel))

def yaw_diff(gt_box_yaw: float, eval_box_yaw: float, period: float = 2*np.pi) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_gt = np.arctan2(np.sin(gt_box_yaw), np.cos(gt_box_yaw))
    yaw_est = np.arctan2(np.sin(eval_box_yaw), np.cos(eval_box_yaw))

    return abs(angle_diff(yaw_gt, yaw_est, period))

def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff

def scale_iou(gt_dim: list, dt_dim: list) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(gt_dim)
    sr_size = np.array(dt_dim)
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_annotation + volume_result - intersection  # type: float
    iou = intersection / union

    return iou

def attr_acc(gt_box_name: str, dt_box_name: str) -> float:
    """
    Computes the classification accuracy for the attribute of this class (if any).
    If the GT class has no attributes or the annotation is missing attributes, we assign an accuracy of nan, which is
    ignored later on.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Attribute classification accuracy (0 or 1) or nan if GT annotation does not have any attributes.
    """
    if gt_box_name == '':
        # If the class does not have attributes or this particular sample is missing attributes, return nan, which is
        # ignored later. Note that about 0.4% of the sample_annotations have no attributes, although they should.
        acc = np.nan
    else:
        # Check that label is correct.
        acc = float(gt_box_name == gt_box_name)
    return acc

def cummean(x: np.array) -> np.array:
    """
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)

def calc_tp(match_data: dict, confidence: list, min_recall: float, metric_name: str) -> float:
    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = 0 # md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.

    confidence = sorted(confidence, reverse=True)
    non_zero = np.nonzero(confidence)[0]
    if len(non_zero) > 0:  # If there are no matches, all the confidence values will be zero.
        max_recall_ind = non_zero[-1]

    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.

def calc_ap(precision: list, min_recall: float, min_precision: float) -> float:
    
    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(precision)
    # prec = prec[::-1]
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)

def calc_my_ap(precision: list):
    prec = np.copy(precision)
    bins = [k for k in prec.tolist() if k > 0]
    return float(np.sum(prec)) / float(len(bins))


# dt_box are sorted as score
def accumulate(gt_box: dict,
               dt_box: list, 
               class_name: str,
               dist_th: float = 0.5):
    # gt_mask = [k for k in range(len(gt_box)) if gt_box[k]['name'] == class_name]

    gt_count = 0
    gt_mask = {}
    for key in gt_box.keys():
        gt_filter = [k for k in range(len(gt_box[key])) if gt_box[key][k]['name'] == class_name]
        if len(gt_filter) > 0:
            gt_mask[key] = gt_filter
            gt_count = gt_count + len(gt_filter)
            if class_name == 'Unknown':
                print(key+'.bin')


    # mask = [k if gt_box['name'][k] == class_name else 0 for k in range(len(gt_box))]
    if gt_count == 0:
        return None, None, None, None, None, None

    dt_mask = [k for k in range(len(dt_box)) if dt_box[k]['name'] == class_name]
   
    if len(dt_mask) == 0:
        return None, None, None, None, None, None

    pred_confs = [dt_box[k]['scores'] for k in dt_mask]
    sortind = [dt_mask[i] for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    match_bbox = {'image_idx': [],
                  'dt_idx': [],
                  'gt_idx': []}
 
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    tp = []
    fp = []
    conf = []
    gt_taken = {}
    for frame_id in gt_mask.keys():
        gt_taken[frame_id] = []
    
    for dt_idx in sortind:
        dt_loc = dt_box[dt_idx]['locations']

        dt_dim = dt_box[dt_idx]['dimensions'] # h, w, l
        dt_dim = [dt_dim[1], dt_dim[2], dt_dim[0]] # w, l, h
        
        dt_velo = [0.0, 0.0]
        if 'velocity' in dt_box[dt_idx].keys():
            dt_velo = dt_box[dt_idx]['velocity']

        dt_score = dt_box[dt_idx]['scores']
        dt_yaw = dt_box[dt_idx]['rotation_y']
        name = dt_box[dt_idx]['name']
        image_idx = dt_box[dt_idx]['image_idx']

        min_dist = np.inf
        match_gt_idx = None

        if image_idx not in gt_mask.keys():
            tp.append(0)
            fp.append(1)
            conf.append(dt_score)
            continue

        for gt_ind in range(len(gt_mask[image_idx])):
            gt_idx = gt_mask[image_idx][gt_ind]
            if gt_idx in gt_taken[image_idx]:
                continue

            gt_box_info = gt_box[image_idx][gt_idx]

            gt_loc = gt_box_info['locations']
            gt_dim = gt_box_info['dimensions'] # h, w, l
            gt_dim = [gt_dim[1], gt_dim[2], gt_dim[0]] # w, l, h
            gt_velo = gt_box_info['velocity']
            gt_yaw = gt_box_info['rotation_y']

            dist = center_distance(gt_loc, dt_loc)
            # dist = riou_distance([gt_loc[0], gt_loc[1], gt_loc[2], gt_dim[0], gt_dim[1], gt_dim[2], gt_yaw], 
            #                      [dt_loc[0], dt_loc[1], dt_loc[2], dt_dim[0], dt_dim[1], dt_dim[2], dt_yaw])

            if dist < dist_th:
                min_dist = dist
                match_gt_idx = gt_idx
        
        is_match = min_dist < dist_th

        if is_match:
            gt_taken[image_idx].append(match_gt_idx)

            gt_loc = gt_box[image_idx][match_gt_idx]['locations']
            gt_dim = gt_box[image_idx][match_gt_idx]['dimensions']
            gt_velo = gt_box[image_idx][match_gt_idx]['velocity']
            gt_yaw = gt_box[image_idx][match_gt_idx]['rotation_y']

            match_data['trans_err'].append(center_distance(gt_loc, dt_loc))
            match_data['vel_err'].append(velocity_l2(gt_velo, dt_velo))
            match_data['scale_err'].append(1-scale_iou(gt_dim, dt_dim))
            match_data['orient_err'].append(yaw_diff(gt_yaw, dt_yaw))
            match_data['attr_err'].append(1-attr_acc(class_name, name))
            match_data['conf'].append(dt_score)
            conf.append(dt_score)

            match_bbox['image_idx'].append(image_idx)
            match_bbox['gt_idx'].append(match_gt_idx)
            match_bbox['dt_idx'].append(dt_box[dt_idx]['index'])

            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
            conf.append(dt_score)
    
    if len(match_data['trans_err']) == 0:
        return None, None, None, None, None, None

    # Calculate my precision and recall.
    recall = np.sum(tp).astype(float) / float(gt_count) 
    precision = np.sum(tp).astype(float) / float(len(sortind))

    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    prec = tp / (fp + tp)
    rec = tp / float(gt_count)

    rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0) #PR曲线
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp
    PRC = np.vstack((rec, conf, prec))


    # ap = calc_my_ap(prec)
    ap = calc_ap(prec, 0.1, 0.1)
    # ap3 = calc_ap(prec, 0.1, 0.3)
    # ap5 = calc_ap(prec, 0.1, 0.5)
    # ap7 = calc_ap(prec, 0.1, 0.7)

    # 25, 50, 75, 80, 85, 90, 95, 99分位
    stat_data = { 'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': []} 

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            if key in stat_data.keys():
                stat_data[key] = sorted(match_data[key])
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]
    
    return match_data, stat_data, PRC.tolist(), precision, recall, ap

def load_json_file(json_file_path):
    with open(json_file_path, 'r') as fin:
        o = json.load(fin)
    return o

def Json2BBoxInfo(json_file):
    o = load_json_file(json_file)
    frame_count = len(o)

    json_list = []
    json_dict = {}

    for k in range(frame_count):
        if o[k] is None:
            continue

        det_count = len(o[k]['location'])

        if det_count == 0:
            continue

        image_idx = o[k]['image_idx']
        json_dict[image_idx] = []

        for m in range(det_count):
            bboxinfo = {}
            bboxinfo['image_idx'] = image_idx
            bboxinfo['locations'] = o[k]['location'][m]
            bboxinfo['dimensions'] = o[k]['dimensions'][m]
            bboxinfo['rotation_y'] = o[k]['rotation_y'][m]
            bboxinfo['name'] = o[k]['name'][m]
            bboxinfo['index'] = o[k]['index'][m]
            if 'scores' in o[k].keys():
                bboxinfo['scores'] = o[k]['scores'][m]
            elif 'score' in o[k].keys():
                bboxinfo['scores'] = o[k]['score'][m]

            if 'velocity' in o[k].keys():
                bboxinfo['velocity'] = o[k]['velocity'][m]
            json_list.append(bboxinfo)
            json_dict[image_idx].append(bboxinfo)

    return json_list, json_dict

if __name__ == '__main__':
    # cfg = Config.fromfile('./config.py')

    gt_file = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/August1/gt.json'
    dt_file = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/August1/cpp_dt_unknown.json'

    dt_list, dt_dict = Json2BBoxInfo(dt_file)
    gt_list, gt_dict = Json2BBoxInfo(gt_file)

    # class_name = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'pedestr', 'Traffic_cone', 'Unknown']
    class_name = ['Unknown']

    match_data_cls = {}
    stat_data_cls = {}

    write_path = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/August1/nus_stat/'

    quantile = [25, 50, 75, 80, 85, 90, 95, 99]
    metric = ['trans_err', 'scale_err', 'orient_err', 'vel_err']

    for cls_name in class_name:
        match_data, stat_data, PRC, precision, recall, ap = accumulate(gt_dict, dt_list, cls_name, 0.5)
        if match_data is None or stat_data is None:
            print('invalid class_name: ', cls_name)
            continue

        match_data_cls[cls_name] = match_data
        stat_data_cls[cls_name] = stat_data

        txt_file_name = write_path + '/./' + cls_name + '_quant.csv'
        txt_file = open(txt_file_name, 'w')

        print(cls_name, ':...................', ap)
        txt_file.write(cls_name + '\n')
        txt_file.write('precision,' + str(round(precision, 3)) + '\n')
        txt_file.write('recall,' + str(round(recall, 3)) + '\n')
        txt_file.write('AP,' + str(round(ap, 3)) + '\n')
        txt_file.write('percent')
        txt_file.write(',min')
        for q in quantile:
            txt_file.write(',' + str(q))
        txt_file.write(',max')
        txt_file.write('\n')

        for key in stat_data.keys():
            total = len(stat_data[key])
            quant_data = {}
            for level in quantile:
                divide = (int)(total * level / 100)
                value = stat_data[key][divide]
                quant_data[str(level)] = value
            
            txt_file.write(key)
            txt_file.write(',' + str(round(min(stat_data[key]), 3)))
            for v in quant_data.values():
                txt_file.write(',' + str(round(v, 3)))
            txt_file.write(',' + str(round(max(stat_data[key]), 3)))
            txt_file.write('\n')

            # print(key, quant_data, ', max: ', max(stat_data[key]), ', min: ', min(stat_data[key]))

        txt_file.close()
        
        csv_file_name = write_path + '/./' + cls_name + '.csv'
        csv_file = open(csv_file_name, 'w')
        interp_count = 101
        for k in range(interp_count):
            s = ','.join([str(round(match_data['trans_err'][k],3)),
                          str(round(match_data['scale_err'][k],3)),
                          str(round(match_data['orient_err'][k],3)),
                          str(round(match_data['vel_err'][k],3))])
            s = s + '\n'
            csv_file.writelines(s)
        csv_file.close()

        PRC_file_name = write_path + '/./' + cls_name + '_prec.csv'
        PRC_file = open(PRC_file_name, 'w')

        PRC_file.write(',recall, conf, prec' + '\n')
        for k in range(len(PRC[0])):
            s = ','.join([str(round(PRC[0][k],2)), str(round(PRC[1][k], 3)), str(round(PRC[2][k], 3))])
            s = s + '\n'
            PRC_file.writelines(s)

        PRC_file.close()




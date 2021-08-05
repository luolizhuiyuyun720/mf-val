import pickle
import json
import numpy as np

def read_pickle(pickle_filename, mode="rb"):
    with open(pickle_filename, mode) as fr:
        ret = pickle.load(fr)
    return ret

def limit_period(val, offset=0.5, period=2 * np.pi):
    return val - np.floor(val / period + offset) * period

def GetImageIdx(sample):
    # if 'image_idx' in sample.keys():
    #     return sample['image_idx']
    if 'image' in sample.keys():
        if 'image_idx' in sample['image'].keys():
            return sample['image']['image_idx'] 
    elif 'image' in sample.keys() and 'image_idx' in sample['image'].keys() and isinstance(sample['image']['image_idx'], str):
        return sample['image']['image_idx']
    elif 'metadata' in sample.keys():
         if 'image_idx' in sample['metadata'].keys():
            return sample['metadata']['image_idx']     
    elif 'image_idx' in sample.keys() and isinstance(sample['image_idx'], str):
        return sample['image_idx']
    return None

def GetBoxesPrimary(sample, min_num_pts=10):
    sample_anno = sample
    if 'annos' in sample.keys():
        sample_anno = sample['annos']

    loc = sample_anno['location']

    if len(loc) == 0:
        print("bbox num is zero")
        return [],[],[]

    dim = sample_anno['dimensions']
    # velo = sample['velocity']
    # track_id = gt_anno['track_id']

    r_y = (sample_anno['rotation_y'] + np.pi / 2)
    r_y_pi = limit_period(r_y)

    new_loc = np.stack((loc[:, 0], -1 * loc[:,1], loc[:,2]), axis=1) # xyz lidar
    new_dim = np.stack((dim[:, 2], dim[:,0], dim[:,1]), axis=1) # hwl

    gt_box = np.hstack((new_loc, new_dim, np.reshape(r_y_pi, [-1, 1])))
    gt_type = sample_anno['name']

    gt_pts = None
    if 'num_pts' in sample_anno.keys():
        gt_pts = sample_anno['num_pts']
        ignore_mask = sample_anno['num_pts'] >= min_num_pts
        return gt_box[ignore_mask], gt_type[ignore_mask], gt_pts[ignore_mask], #velo[ignore_mask] , track_id[ignore_mask]
    else:
        # gt_pts = [min_num_pts for k in range(min_num_pts)]
        return gt_box, gt_type, gt_pts

def GetBoxesVelo(sample):
    sample_anno = sample
    if 'annos' in sample.keys():
        sample_anno = sample['annos']

    if 'velocity' in sample_anno.keys():
        return sample_anno['velocity']
    return None

def GetBoxesTrackId(sample):
    sample_anno = sample
    if 'annos' in sample.keys():
        sample_anno = sample['annos']

    if 'track_id' in sample_anno.keys():
        return sample_anno['track_id']
    return None

def GetBoxesScores(sample):
    sample_anno = sample
    if 'annos' in sample.keys():
        sample_anno = sample['annos']

    if 'score' in sample_anno.keys():
        return sample_anno['score']

    if 'scores' in sample_anno.keys():
        return sample_anno['scores']

    return None

if __name__ == '__main__':

    yaw_opposite = False

    pkl_file_name = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/August1/eval_ws1/dt_annos.pkl'
    json_file_name = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/August1/dt.json'

    pkl_info = read_pickle(pkl_file_name)
    json_info = open(json_file_name, 'w')

    json_list = []

    for k in range(len(pkl_info)):
        if pkl_info is None:
            print("the " + str(k) + "th frame info is None")
            continue

        image_idx = GetImageIdx(pkl_info[k])
        if image_idx == None:
            print("the " + str(k) + "th image_idx is None")
            continue

        print("parsing frame: " + image_idx + "...")

        boxes, boxes_type, boxes_pts = GetBoxesPrimary(pkl_info[k], min_num_pts=10)

        velo = None
        track_id = None
        scores = None

        if len(boxes) > 0:
            velo = GetBoxesVelo(pkl_info[k])
            track_id = GetBoxesTrackId(pkl_info[k])
            scores = GetBoxesScores(pkl_info[k])
        else:
            continue

        json_dict = {}
        json_dict['image_idx'] = image_idx
        json_dict['location'] = boxes[:, 0:3].tolist()
        json_dict['dimensions'] = boxes[:, 3:6].tolist()
        json_dict['rotation_y'] = boxes[:, 6].tolist()
        json_dict['name'] = boxes_type.tolist()
        json_dict['index'] = [idx for idx in range(len(boxes_type))]

        if yaw_opposite:
            for k in range(len(json_dict['rotation_y'])):
                json_dict['rotation_y'][k] = -json_dict['rotation_y'][k]

        if boxes_pts is not None:
            if isinstance(boxes_pts, list):
                json_dict['num_pts'] = boxes_pts
            else:
                json_dict['num_pts'] = boxes_pts.tolist()

        if velo is not None:
            json_dict['velocity'] = velo.tolist()
        
        if track_id is not None:
            json_dict['track_id'] = track_id.tolist()
        
        if scores is not None:
            json_dict['scores'] = scores.tolist()

        json_list.append(json_dict)

    json.dump(json_list, json_info, indent=4)
#!/usr/bin/env python
# -*- coding:UTF-8 -*-

"""
VoxelNet config system.
"""
import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from easydict import EasyDict as edict
import math


__C = edict()
# Consumers can get config by:
# import config as cfg
cfg = __C

# for dataset dir
__C.DATA_DIR = 'fabu/split'
__C.ANCHORS_FILE = 'res/anchors.txt'
#__C.CALIB_DIR = 'kitti_data/training/calib'

# for gpu allocation
__C.GPU_AVAILABLE = '0,1,2,3'
__C.GPU_USE_COUNT = len(__C.GPU_AVAILABLE.split(','))
__C.GPU_MEMORY_FRACTION = 1

# set the log image scale factor
__C.BV_LOG_FACTOR = 3

# selected object
__C.DETECT_OBJ = 'Obstacle'  # Pedestrian/Cyclist
__C.Y_MIN = -70
__C.Y_MAX = 70
__C.X_MIN = -70
__C.X_MAX = 70
__C.VOXEL_X_SIZE = 0.2
__C.VOXEL_Y_SIZE = 0.2
__C.RADIUS_1 = int(10 * __C.BV_LOG_FACTOR / 0.2)
__C.RADIUS_2 = int(20 * __C.BV_LOG_FACTOR / 0.2)
__C.RADIUS_3 = int(30 * __C.BV_LOG_FACTOR / 0.2)
__C.RADIUS_4 = int(40 * __C.BV_LOG_FACTOR / 0.2)
__C.RADIUS_5 = int(50 * __C.BV_LOG_FACTOR / 0.2)
__C.RADIUS_6 = int(60 * __C.BV_LOG_FACTOR / 0.2)
__C.RADIUS_7 = int(70 * __C.BV_LOG_FACTOR / 0.2)
__C.VOXEL_POINT_COUNT = 35
__C.INPUT_WIDTH = int((__C.X_MAX - __C.X_MIN) / __C.VOXEL_X_SIZE)
__C.INPUT_HEIGHT = int((__C.Y_MAX - __C.Y_MIN) / __C.VOXEL_Y_SIZE)
__C.CENTER_X = int((-__C.X_MIN) / __C.VOXEL_X_SIZE)*__C.BV_LOG_FACTOR
__C.CENTER_Y = int((-__C.Y_MIN) / __C.VOXEL_Y_SIZE)*__C.BV_LOG_FACTOR



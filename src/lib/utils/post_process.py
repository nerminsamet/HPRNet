from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds


def get_pred_depth(depth):
  return depth


def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:271].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 266)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret


def landmark_post_process(dets, c, s, h, w):
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:51].reshape(-1, 2), c[i], s[i], (w, h))
    face_bbox = transform_preds(dets[i, :, 51:55].reshape(-1, 2), c[i], s[i], (w, h))
    face_pts = transform_preds(dets[i, :, 56:192].reshape(-1, 2), c[i], s[i], (w, h))
    lefthand_pts = transform_preds(dets[i, :, 194:236].reshape(-1, 2), c[i], s[i], (w, h))
    righthand_pts = transform_preds(dets[i, :, 237:279].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5],
       pts.reshape(-1, 46), face_bbox.reshape(-1, 4), dets[i, :, 55:56],
       face_pts.reshape(-1, 136), dets[i, :, 193:194], lefthand_pts.reshape(-1, 42),
       dets[i, :, 236:237], righthand_pts.reshape(-1, 42),
       dets[i, :, 193:194], dets[i, :, 236:237],], axis=1).astype(np.float32).tolist()

    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret
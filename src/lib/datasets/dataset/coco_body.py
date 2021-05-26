from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from src.tools.coco_body_eval.myeval_facebox import MYeval_facebox
from src.tools.coco_body_eval.myeval_face import MYeval_face
from src.tools.coco_body_eval.myeval_lefthand import MYeval_lefthand
from src.tools.coco_body_eval.myeval_righthand import MYeval_righthand
from src.tools.coco_body_eval.myeval_foot import MYeval_foot
from src.tools.coco_body_eval.myeval_wholebody import MYeval_wholebody
from src.tools.coco_body_eval.myeval_body import MYeval_body

import numpy as np
import json
import os
import warnings
import torch.utils.data as data

class COCOBODY(data.Dataset):

  num_classes = 1
  num_joints = 17
  num_face_center_joint = 1
  num_hand_center_joint = 2
  num_foot_center_joint = 2

  num_face_landmarks = 68
  num_hand_landmarks = 42
  num_foot_landmarks = 6

  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
              [11, 12], [13, 14], [15, 16], [17, 20], [18, 21], [19, 22], [24, 25]]
  face_flip_idx = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], [17, 26],
                   [18, 25], [19, 24], [20, 23], [21, 22], [31, 35], [32, 34], [36, 45], [37, 44],
                   [38, 43], [39, 42], [40, 47], [41, 46], [48, 54], [49, 53], [50, 52], [55, 59],
                   [56, 58], [60, 64], [61, 63], [65, 67]]

  hand_flip_idx = [[0, 21], [1, 22], [2, 23], [3, 24], [4, 25], [5, 26], [6, 27], [7, 28],
                   [8, 29], [9, 30], [10, 31],[11, 32],[12, 33], [13, 34], [14, 35], [15, 36],
                   [16, 37], [17, 38], [18, 39], [19, 40], [20, 41]]

  foot_flip_idx = [[0, 3], [1, 4], [2, 5]]



  def __init__(self, opt, split):
    super(COCOBODY, self).__init__()

    self.task = opt.task
    self.data_dir = os.path.join(opt.data_dir, opt.coco_dir)
    self.img_dir = os.path.join(self.data_dir, 'images', '{}2017'.format(split))
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'image_info_test-dev2017.json').format(split)
    else:
      self.annot_path = os.path.join(
        self.data_dir, 'annotations',
        'coco_wholebody_{}_v1.0.json').format(split)
    self.max_objs = 32
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    image_ids = self.coco.getImgIds()

    if split == 'train':
      self.images = []
      for img_id in image_ids:
        idxs = self.coco.getAnnIds(imgIds=[img_id])
        if len(idxs) > 0:
          self.images.append(img_id)
    else:
      self.images = image_ids
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = 1
        for dets in all_bboxes[image_id][cls_ind]:
          bbox = dets[:4]
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = dets[4]
          bbox_out  = list(map(self._to_float, bbox))
          face_score = dets[55]

          # face box from the face detection branch
          face_bbox = dets[51:55]
          face_bbox[2] -= face_bbox[0]
          face_bbox[3] -= face_bbox[1]
          face_bbox_out = list(map(self._to_float, face_bbox))

          body_keypoints = np.concatenate([
            np.array(dets[5:39], dtype=np.float32).reshape(-1, 2),
            np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
          body_keypoints  = list(map(self._to_float, body_keypoints))

          foot_keypoints = np.concatenate([
              np.array(dets[39:51], dtype=np.float32).reshape(-1, 2),
              np.ones((6, 1), dtype=np.float32)], axis=1).reshape(18).tolist()
          foot_kpts = list(map(self._to_float, foot_keypoints))

          face_keypoints = np.concatenate([
            np.array(dets[56:192], dtype=np.float32).reshape(-1, 2),
            np.ones((68, 1), dtype=np.float32)], axis=1).reshape(204).tolist()
          face_keypoints  = list(map(self._to_float, face_keypoints))

          lefthand_keypoints = np.concatenate([
            np.array(dets[193:235], dtype=np.float32).reshape(-1, 2),
            np.ones((21, 1), dtype=np.float32)], axis=1).reshape(63).tolist()
          lefthand_keypoints  = list(map(self._to_float, lefthand_keypoints))

          righthand_keypoints = np.concatenate([
            np.array(dets[236:278], dtype=np.float32).reshape(-1, 2),
            np.ones((21, 1), dtype=np.float32)], axis=1).reshape(63).tolist()
          righthand_keypoints  = list(map(self._to_float, righthand_keypoints))

          # face boxes from face keypoints
          # face_bbox = kpts2box(face_keypoints)
          # face_bbox_out = list(map(self._to_float, face_bbox))

          score = float("{:.2f}".format(score))
          face_score = float("{:.2f}".format(face_score))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": score,
              "face_box": face_bbox_out,
              "face_kpts":face_keypoints,
              "keypoints": body_keypoints,
              "lefthand_kpts": lefthand_keypoints,
              "righthand_kpts": righthand_keypoints,
              "foot_kpts": foot_kpts,
              "body_score": score,
              "face_score": score,
              "wholebody_score": score,
              "facebox_score": score * face_score
          }

          detections.append(detection)
    return detections

  def convert_eval_format_baseline(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = 1
        for dets in all_bboxes[image_id][cls_ind]:
          bbox = dets[:4]
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = dets[4]
          bbox_out  = list(map(self._to_float, bbox))

          face_bbox = dets[51:56]
          face_bbox[2] -= face_bbox[0]
          face_bbox[3] -= face_bbox[1]
          face_bbox_out = list(map(self._to_float, face_bbox))

          body_keypoints = np.concatenate([
            np.array(dets[5:39], dtype=np.float32).reshape(-1, 2),
            np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
          body_keypoints  = list(map(self._to_float, body_keypoints))

          foot_keypoints = np.concatenate([
              np.array(dets[39:51], dtype=np.float32).reshape(-1, 2),
              np.ones((6, 1), dtype=np.float32)], axis=1).reshape(18).tolist()
          foot_kpts = list(map(self._to_float, foot_keypoints))

          face_keypoints = np.concatenate([
            np.array(dets[51:187], dtype=np.float32).reshape(-1, 2),
            np.ones((68, 1), dtype=np.float32)], axis=1).reshape(204).tolist()
          face_keypoints  = list(map(self._to_float, face_keypoints))

          lefthand_keypoints = np.concatenate([
            np.array(dets[187:229], dtype=np.float32).reshape(-1, 2),
            np.ones((21, 1), dtype=np.float32)], axis=1).reshape(63).tolist()
          lefthand_keypoints  = list(map(self._to_float, lefthand_keypoints))

          righthand_keypoints = np.concatenate([
            np.array(dets[229:271], dtype=np.float32).reshape(-1, 2),
            np.ones((21, 1), dtype=np.float32)], axis=1).reshape(63).tolist()
          righthand_keypoints  = list(map(self._to_float, righthand_keypoints))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score)),
              "face_box": face_bbox_out,
              "face_kpts":face_keypoints,
              "keypoints": body_keypoints,
              "lefthand_kpts": lefthand_keypoints,
              "righthand_kpts": righthand_keypoints,
              "foot_kpts": foot_kpts,
          }
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):

    if self.task == 'multi_pose':
        json.dump(self.convert_eval_format_baseline(results),
                  open('{}/results.json'.format(save_dir), 'w'))
    else:
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

  def run_eval(self, results, save_dir):

    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))

    test_body(self.coco,coco_dets)
    test_foot(self.coco, coco_dets)
    test_face(self.coco, coco_dets)
    test_lefthand(self.coco, coco_dets)
    test_righthand(self.coco, coco_dets)
    test_wholebody(self.coco, coco_dets)

    if self.task == 'landmark':
        test_face_box(self.coco, coco_dets)


def kpts2box(kpts):
    xd = kpts[0::3]
    yd = kpts[1::3]

    x_min = min(xd)
    x_max = max(xd)
    y_min = min(yd)
    y_max = max(yd)
    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height]


def check_part_score(coco_dt, part):
    flag_no_part_score = False
    for k in coco_dt.anns.keys():
        if '{}_score'.format(part) not in coco_dt.anns[k]:
            flag_no_part_score = True
            coco_dt.anns[k]['{}_score'.format(part)] = coco_dt.anns[k]['score']
    if flag_no_part_score:
        warnings.warn("'{}_score' not found, use 'score' instead.".format(part))


def test_body(coco,coco_dt):
    print('body mAP ----------------------------------')
    coco_eval = MYeval_body(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0


def test_foot(coco,coco_dt):
    print('foot mAP ----------------------------------')
    check_part_score(coco_dt, 'foot')
    coco_eval = MYeval_foot(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0


def test_face(coco,coco_dt):
    print('face mAP ----------------------------------')
    check_part_score(coco_dt, 'face')
    coco_eval = MYeval_face(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0


def test_lefthand(coco,coco_dt):
    print('lefthand mAP ----------------------------------')
    check_part_score(coco_dt, 'lefthand')
    coco_eval = MYeval_lefthand(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0


def test_righthand(coco,coco_dt):
    print('righthand mAP ----------------------------------')
    check_part_score(coco_dt, 'righthand')
    coco_eval = MYeval_righthand(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0


def test_wholebody(coco,coco_dt):
    print('wholebody mAP ----------------------------------')
    check_part_score(coco_dt, 'wholebody')
    coco_eval = MYeval_wholebody(coco, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0


def test_face_box(coco, coco_dt):
    print('\n Face Box---------------------------------------')
    coco_eval = MYeval_facebox(coco, coco_dt, "face_box")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 0



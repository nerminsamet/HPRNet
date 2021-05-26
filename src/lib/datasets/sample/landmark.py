from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import math

FACE_IND = 23
LH_IND = 24
RH_IND = 25

class LandmarkDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _calculate_foot_bbox(self, foot_kpts):
        xd = foot_kpts[0::3]
        yd = foot_kpts[1::3]

        x_min = min(xd)
        x_max = max(xd)
        y_min = min(yd)
        y_max = max(yd)
        # width = x_max - x_min
        # height = y_max - y_min
        foot_bbox = np.array([x_min, y_min, x_max, y_max],
                 dtype=np.float32)

        return foot_bbox

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            if np.random.random() < self.opt.aug_rot:
                rf = self.opt.rotate
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_res, self.opt.input_res])
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_res = self.opt.output_res

        num_body_joints = self.num_joints + self.num_face_center_joint + \
                         self.num_hand_center_joint + self.num_foot_landmarks

        num_face_landmarks = self.num_face_landmarks
        num_hand_landmarks = self.num_hand_landmarks
        one_hand_landmarks = int(num_hand_landmarks/2)
        num_foot_landmarks = self.num_foot_landmarks
        # one_foot_landmarks = int(num_foot_landmarks / 2)

        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
        trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

        # person properties - center - center offset - wh
        hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        # person kps attributes
        hm_hp = np.zeros((num_body_joints, output_res, output_res), dtype=np.float32)
        kps = np.zeros((self.max_objs, num_body_joints * 2), dtype=np.float32)
        kps_mask = np.zeros((self.max_objs, num_body_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objs * num_body_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_body_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objs * num_body_joints), dtype=np.int64)

        # face landmarks and wh
        lms_face = np.zeros((self.max_objs, num_face_landmarks * 2), dtype=np.float32)
        lms_face_mask = np.zeros((self.max_objs, num_face_landmarks * 2), dtype=np.uint8)
        wh_face = np.zeros((self.max_objs, 2), dtype=np.float32)
        hp_ind_face_cnt = np.zeros((self.max_objs * self.num_face_center_joint), dtype=np.int64)
        hp_mask_face_cnt = np.zeros((self.max_objs * self.num_face_center_joint), dtype=np.int64)

        # hand landmarks and wh
        lms_hand = np.zeros((self.max_objs, num_hand_landmarks * 2), dtype=np.float32)
        lms_hand_mask = np.zeros((self.max_objs, num_hand_landmarks * 2), dtype=np.uint8)
        hp_ind_hand_cnt = np.zeros((self.max_objs * self.num_hand_center_joint), dtype=np.int64)
        hp_mask_hand_cnt = np.zeros((self.max_objs * self.num_hand_center_joint), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            ann = anns[k]

            cls_id = int(ann['category_id']) - 1
            bbox = self._coco_box_to_bbox(ann['bbox'])
            pts = np.array(ann['keypoints'], np.float32).reshape(self.num_joints, 3)

            face_bbox = self._coco_box_to_bbox(ann['face_box'])
            face_kpts = np.array(ann['face_kpts'], np.float32).reshape(num_face_landmarks, 3)
            ct_face = np.array([(face_bbox[0] + face_bbox[2]) / 2,
                                (face_bbox[1] + face_bbox[3]) / 2], dtype=np.float32)

            # lefthand_valid = False
            lefthand_bbox = self._coco_box_to_bbox(ann['lefthand_box'])
            lefthand_kpts = np.array(ann['lefthand_kpts'], np.float32).reshape(one_hand_landmarks, 3)
            ct_lefthand = np.array([(lefthand_bbox[0] + lefthand_bbox[2]) / 2,
                                (lefthand_bbox[1] + lefthand_bbox[3]) / 2], dtype=np.float32)

            # righthand_valid = False
            righthand_bbox = self._coco_box_to_bbox(ann['righthand_box'])
            righthand_kpts = np.array(ann['righthand_kpts'], np.float32).reshape(one_hand_landmarks, 3)
            ct_righthand = np.array([(righthand_bbox[0] + righthand_bbox[2]) / 2,
                                (righthand_bbox[1] + righthand_bbox[3]) / 2], dtype=np.float32)

            foot_kpts = np.array(ann['foot_kpts'], np.float32).reshape(num_foot_landmarks, 3)
            pts = np.vstack([pts, foot_kpts])

            lefthand_valid =  ann['lefthand_valid']
            righthand_valid = ann['righthand_valid']

            if ann['face_valid']:
                ct_face = np.hstack([ct_face, 1])
                pts = np.vstack([pts, ct_face])
            else:
                ct_face = np.hstack([ct_face, 0])
                pts = np.vstack([pts, ct_face])

            if lefthand_valid:
                ct_lefthand = np.hstack([ct_lefthand, 1])
                pts = np.vstack([pts, ct_lefthand])
            else:
                ct_lefthand = np.hstack([ct_lefthand, 0])
                pts = np.vstack([pts, ct_lefthand])

            if righthand_valid:
                ct_righthand = np.hstack([ct_righthand, 1])
                pts = np.vstack([pts, ct_righthand])
            else:
                ct_righthand = np.hstack([ct_righthand, 0])
                pts = np.vstack([pts, ct_righthand])

            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1

                face_bbox[[0, 2]] = width - face_bbox[[2, 0]] - 1
                face_kpts[:, 0] = width - face_kpts[:, 0] - 1

                lefthand_bbox[[0, 2]] = width - lefthand_bbox[[2, 0]] - 1
                lefthand_kpts[:, 0] = width - lefthand_kpts[:, 0] - 1

                righthand_bbox[[0, 2]] = width - righthand_bbox[[2, 0]] - 1
                righthand_kpts[:, 0] = width - righthand_kpts[:, 0] - 1

                for e in self.flip_idx:
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
                for e in self.face_flip_idx:
                    face_kpts[e[0]], face_kpts[e[1]] = face_kpts[e[1]].copy(), face_kpts[e[0]].copy()

                hand_kpts = np.concatenate((lefthand_kpts, righthand_kpts))
                for e in self.hand_flip_idx:
                    hand_kpts[e[0]], hand_kpts[e[1]] = hand_kpts[e[1]].copy(), hand_kpts[e[0]].copy()

                lefthand_kpts = hand_kpts[:one_hand_landmarks]
                righthand_kpts = hand_kpts[one_hand_landmarks:]

                lefthand_valid, righthand_valid = righthand_valid, lefthand_valid

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox = np.clip(bbox, 0, output_res - 1)

            face_bbox[:2] = affine_transform(face_bbox[:2], trans_output)
            face_bbox[2:] = affine_transform(face_bbox[2:], trans_output)
            face_bbox = np.clip(face_bbox, 0, output_res - 1)

            lefthand_bbox[:2] = affine_transform(lefthand_bbox[:2], trans_output)
            lefthand_bbox[2:] = affine_transform(lefthand_bbox[2:], trans_output)

            righthand_bbox[:2] = affine_transform(righthand_bbox[:2], trans_output)
            righthand_bbox[2:] = affine_transform(righthand_bbox[2:], trans_output)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            h_f, w_f = face_bbox[3] - face_bbox[1], face_bbox[2] - face_bbox[0]

            if (h > 0 and w > 0) or (rot != 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_res + ct_int[0]
                reg[k] = ct - ct_int

                reg_mask[k] = 1
                num_kpts = pts[:self.num_joints,2].sum()
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                    reg_mask[k] = 0

                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
                for j in range(num_body_joints):
                    if pts[j, 2] > 0:
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                                pts[j, 1] >= 0 and pts[j, 1] < output_res:
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                            kps_mask[k, j * 2: j * 2 + 2] = 1
                            pt_int = pts[j, :2].astype(np.int32)
                            hp_offset[k * num_body_joints + j] = pts[j, :2] - pt_int
                            hp_ind[k * num_body_joints + j] = pt_int[1] * output_res + pt_int[0]
                            hp_mask[k * num_body_joints + j] = 1
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)

                            if j == FACE_IND:
                                wh_face[k] = 1. * w_f, 1. * h_f
                                hp_ind_face_cnt[k] = pt_int[1] * output_res + pt_int[0]
                                hp_mask_face_cnt[k] = 1

                            if j == LH_IND:
                                hp_ind_hand_cnt[k] = pt_int[1] * output_res + pt_int[0]
                                hp_mask_hand_cnt[k] = 1

                            if j == RH_IND:
                                hp_ind_hand_cnt[k + self.max_objs] = pt_int[1] * output_res + pt_int[0]
                                hp_mask_hand_cnt[k + self.max_objs] = 1

                draw_gaussian(hm[cls_id], ct_int, radius)

                if ann['face_valid']:
                    for jj in range(self.num_face_landmarks):
                        if face_kpts[jj, 2] > 0:
                            face_kpts[jj, :2] = affine_transform(face_kpts[jj, :2], trans_output_rot)
                            if face_kpts[jj, 0] >= 0 and face_kpts[jj, 0] < output_res and  \
                                    face_kpts[jj, 1] >= 0 and face_kpts[jj, 1] < output_res:
                                ct_face_int = pts[FACE_IND, :2].astype(np.int32)
                                lms_face[k, jj * 2: jj * 2 + 2] = (face_kpts[jj, :2] - ct_face_int)
                                lms_face_mask[k, jj * 2: jj * 2 + 2] = 1

                if lefthand_valid:
                    for jj in range(one_hand_landmarks):
                        if lefthand_kpts[jj, 2] > 0:
                            lefthand_kpts[jj, :2] = affine_transform(lefthand_kpts[jj, :2], trans_output_rot)
                            if lefthand_kpts[jj, 0] >= 0 and lefthand_kpts[jj, 0] < output_res and  \
                                    lefthand_kpts[jj, 1] >= 0 and lefthand_kpts[jj, 1] < output_res:
                                ct_lefthand_int = pts[LH_IND, :2].astype(np.int32)
                                lms_hand[k, jj * 2: jj * 2 + 2] = (lefthand_kpts[jj, :2] - ct_lefthand_int)
                                lms_hand_mask[k, jj * 2: jj * 2 + 2] = 1

                if righthand_valid:
                    for jj in range(one_hand_landmarks):
                        if righthand_kpts[jj, 2] > 0:
                            righthand_kpts[jj, :2] = affine_transform(righthand_kpts[jj, :2], trans_output_rot)
                            if righthand_kpts[jj, 0] >= 0 and righthand_kpts[jj, 0] < output_res and  \
                                    righthand_kpts[jj, 1] >= 0 and righthand_kpts[jj, 1] < output_res:
                                ct_righthand_int = pts[RH_IND, :2].astype(np.int32)
                                ll = jj + one_hand_landmarks
                                lms_hand[k, ll * 2: ll * 2 + 2] = (righthand_kpts[jj, :2] - ct_righthand_int)
                                lms_hand_mask[k, ll * 2: ll * 2 + 2] = 1

                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1] +
                               [cls_id])
        if rot != 0:
            hm = hm * 0 + 0.9999
            reg_mask *= 0
            kps_mask *= 0

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        ret.update({'hps': kps, 'hps_mask': kps_mask, 'wh_face':wh_face})
        ret.update({'lms_face': lms_face, 'lms_face_mask': lms_face_mask})
        ret.update({'lms_hand': lms_hand, 'lms_hand_mask': lms_hand_mask})

        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.hm_hp:
            ret.update({'hm_hp': hm_hp})
        if self.opt.reg_hp_offset:
            ret.update({'hp_offset': hp_offset,
                        'hp_ind': hp_ind, 'hp_mask': hp_mask,
                        'hp_ind_face_cnt': hp_ind_face_cnt, 'hp_mask_face_cnt': hp_mask_face_cnt,
                        'hp_ind_hand_cnt': hp_ind_hand_cnt, 'hp_mask_hand_cnt': hp_mask_hand_cnt})

        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 40), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret

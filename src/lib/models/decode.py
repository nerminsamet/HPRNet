from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


'''
# Slow for large number of categories
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
'''
def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def multi_pose_decode(
    heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
  batch, cat, height, width = heat.size()
  num_joints = kps.shape[1] // 2
  # heat = torch.sigmoid(heat)
  # perform nms on heatmaps
  heat = _nms(heat)
  scores, inds, clses, ys, xs = _topk(heat, K=K)

  kps = _tranpose_and_gather_feat(kps, inds)
  kps = kps.view(batch, K, num_joints * 2)
  kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
  kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
  if reg is not None:
    reg = _tranpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
  else:
    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
  wh = _tranpose_and_gather_feat(wh, inds)
  wh = wh.view(batch, K, 2)
  clses  = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)

  bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                      ys - wh[..., 1:2] / 2,
                      xs + wh[..., 0:1] / 2, 
                      ys + wh[..., 1:2] / 2], dim=2)
  if hm_hp is not None:
      hm_hp = _nms(hm_hp)
      thresh = 0.1
      kps = kps.view(batch, K, num_joints, 2).permute(
          0, 2, 1, 3).contiguous() # b x J x K x 2
      reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
      hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
      if hp_offset is not None:
          hp_offset = _tranpose_and_gather_feat(
              hp_offset, hm_inds.view(batch, -1))
          hp_offset = hp_offset.view(batch, num_joints, K, 2)
          hm_xs = hm_xs + hp_offset[:, :, :, 0]
          hm_ys = hm_ys + hp_offset[:, :, :, 1]
      else:
          hm_xs = hm_xs + 0.5
          hm_ys = hm_ys + 0.5
        
      mask = (hm_score > thresh).float()
      hm_score = (1 - mask) * -1 + mask * hm_score
      hm_ys = (1 - mask) * (-10000) + mask * hm_ys
      hm_xs = (1 - mask) * (-10000) + mask * hm_xs
      hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
          2).expand(batch, num_joints, K, K, 2)
      dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
      min_dist, min_ind = dist.min(dim=3) # b x J x K
      hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
      min_dist = min_dist.unsqueeze(-1)
      min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
          batch, num_joints, K, 1, 2)
      hm_kps = hm_kps.gather(3, min_ind)
      hm_kps = hm_kps.view(batch, num_joints, K, 2)
      l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
             (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
             (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
      mask = (mask > 0).float().expand(batch, num_joints, K, 2)
      kps = (1 - mask) * hm_kps + mask * kps
      kps = kps.permute(0, 2, 1, 3).contiguous().view(
          batch, K, num_joints * 2)
  detections = torch.cat([bboxes, scores, kps, clses], dim=2)
    
  return detections


def landmark_decode(
    heat, wh, kps, wh_face=None, reg=None, hm_hp=None, hp_offset=None, K=32,
        face_lms= None, hand_lms=None, foot_lms=None):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2 - 3

    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps_orj  = kps.clone()

    face_kps = kps_orj[:, 46:48, :, :]
    face_hm_hp = hm_hp[:,23:24,:,:]

    lefthand_kps = kps_orj[:, 48:50, :, :]
    lefthand_hm_hp = hm_hp[:,24:25,:,:]

    righthand_kps = kps_orj[:,50:52, :, :]
    righthand_hm_hp = hm_hp[:,25:26,:,:]

    kps = kps[:, :46, :, :]
    hm_hp = hm_hp[:, :23, :, :]
    kps = _tranpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)

    face_bboxes, face_cnt, hm_face_scores, face_lms = \
        decode_single_part(face_kps, face_lms, inds, xs, ys, batch, K, bboxes,
                                                                       face_hm_hp, hp_offset, wh_face)

    lefthand_bboxes, lefthand_cnt, hm_lefthand_scores, lefthand_lms = \
        decode_single_part(lefthand_kps, hand_lms[:, :42, :, :], inds, xs, ys, batch, K, bboxes,
                                                                       lefthand_hm_hp, hp_offset)

    righthand_bboxes, righthand_cnt, hm_righthand_scores, righthand_lms = \
        decode_single_part(righthand_kps, hand_lms[:, 42:, :, :], inds, xs, ys, batch, K, bboxes,
                                                                       righthand_hm_hp, hp_offset)

    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:
            hp_offset = _tranpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, face_bboxes, hm_face_scores, face_lms, clses,
                            hm_lefthand_scores, lefthand_lms, hm_righthand_scores, righthand_lms,], dim=2)

    return detections


def decode_single_part(part_cnt, part_lms, inds, xs, ys, batch, K, bboxes, hm_hp_part_cnt=None, hp_offset_part_cnt=None,
                       wh_part =None):
    part_bboxes = None
    num_part_joints = part_cnt.shape[1] // 2
    num_part_landmarks = part_lms.shape[1] // 2
    part_cnt = _tranpose_and_gather_feat(part_cnt, inds)
    part_cnt = part_cnt.view(batch, K, num_part_joints * 2)
    part_cnt[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_part_joints)
    part_cnt[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_part_joints)

    if hm_hp_part_cnt is not None:
        hm_hp_part_cnt = _nms(hm_hp_part_cnt)
        thresh = 0.1
        part_cnt = part_cnt.view(batch, K, num_part_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        kps_part_cnt = part_cnt.unsqueeze(3).expand(batch, num_part_joints, K, K, 2)
        hm_part_cnt_score, hm_part_cnt_inds, hm_part_cnt_ys, hm_part_cnt_xs = _topk_channel(hm_hp_part_cnt,
                                                                                            K=K)  # b x J x K

        part_lms = _tranpose_and_gather_feat(part_lms, hm_part_cnt_inds.squeeze(dim=0))
        part_lms = part_lms.view(batch, K, num_part_landmarks * 2)
        part_lms[..., ::2] += hm_part_cnt_xs.view(batch, K, 1).expand(batch, K, num_part_landmarks)
        part_lms[..., 1::2] += hm_part_cnt_ys.view(batch, K, 1).expand(batch, K, num_part_landmarks)
        part_lms = part_lms.view(batch, K, num_part_landmarks, 2).permute(
            0, 2, 1, 3).contiguous()

        if hp_offset_part_cnt is not None:
            hp_offset_part_cnt = _tranpose_and_gather_feat(
                hp_offset_part_cnt, hm_part_cnt_inds.view(batch, -1))
            hp_offset_part_cnt = hp_offset_part_cnt.view(batch, num_part_joints, K, 2)
            hm_part_cnt_xs = hm_part_cnt_xs + hp_offset_part_cnt[:, :, :, 0]
            hm_part_cnt_ys = hm_part_cnt_ys + hp_offset_part_cnt[:, :, :, 1]
        else:
            hm_part_cnt_xs = hm_part_cnt_xs + 0.5
            hm_part_cnt_ys = hm_part_cnt_ys + 0.5

        if wh_part is not None:
            wh_part = _tranpose_and_gather_feat(wh_part, hm_part_cnt_inds.view(batch, -1))
            wh_part = wh_part.view(batch, K, 2)
            part_bboxes = torch.cat([hm_part_cnt_xs.view(batch, K, 1) - wh_part[..., 0:1] / 2,
                                     hm_part_cnt_ys.view(batch, K, 1) - wh_part[..., 1:2] / 2,
                                     hm_part_cnt_xs.view(batch, K, 1) + wh_part[..., 0:1] / 2,
                                     hm_part_cnt_ys.view(batch, K, 1) + wh_part[..., 1:2] / 2], dim=2)

        mask = (hm_part_cnt_score > thresh).float()
        hm_part_cnt_score = (1 - mask) * -1 + mask * hm_part_cnt_score
        hm_part_cnt_ys = (1 - mask) * (-10000) + mask * hm_part_cnt_ys
        hm_part_cnt_xs = (1 - mask) * (-10000) + mask * hm_part_cnt_xs
        hm_part_cnt = torch.stack([hm_part_cnt_xs, hm_part_cnt_ys], dim=-1).unsqueeze(
            2).expand(batch, num_part_joints, K, K, 2)
        dist = (((kps_part_cnt - hm_part_cnt) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_part_cnt_score = hm_part_cnt_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1

        if wh_part is not None:
            part_bboxes = part_bboxes.gather(1, min_ind.view(batch, num_part_joints, K, 1, 1).expand(
                batch, num_part_joints, K, 1, 4).view(1, K, 4))

        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_part_joints, K, 1, 1).expand(
            batch, num_part_joints, K, 1, 2)
        hm_part_cnt = hm_part_cnt.gather(3, min_ind)
        hm_part_cnt = hm_part_cnt.view(batch, num_part_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_part_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_part_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_part_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_part_joints, K, 1)
        mask = (hm_part_cnt[..., 0:1] < l) + (hm_part_cnt[..., 0:1] > r) + \
               (hm_part_cnt[..., 1:2] < t) + (hm_part_cnt[..., 1:2] > b) + \
               (hm_part_cnt_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask_part = (mask > 0).float().expand(batch, num_part_joints, K, 4)

        hm_part_scores = ((1 - (mask > 0).float()) * hm_part_cnt_score + (mask > 0).float() * 0).squeeze(dim=1)
        mask = (mask > 0).float().expand(batch, num_part_joints, K, 2)
        part_cnt = (1 - mask) * hm_part_cnt + mask * part_cnt
        if wh_part is not None:
            part_bboxes = ((1 - mask_part) * part_bboxes + mask_part * 0).squeeze(dim=1)
        part_cnt = part_cnt.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_part_joints * 2)

        part_lms = part_lms.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_part_landmarks * 2)

        return part_bboxes, part_cnt, hm_part_scores, part_lms
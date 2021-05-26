from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch

from src.lib.models.decode import landmark_decode
from src.lib.models.utils import flip_tensor, flip_lr_off, flip_lr
from src.lib.utils.post_process import landmark_post_process

from .base_detector import BaseDetector


class LandmarkDetector(BaseDetector):
    def __init__(self, opt):
        super(LandmarkDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx
        self.flip_idx = opt.flip_idx
        self.face_flip_idx = opt.face_flip_idx
        self.hand_flip_idx = opt.hand_flip_idx
        self.foot_flip_idx = opt.foot_flip_idx

    def process(self, images, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()

            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()

            reg = output['reg'] if self.opt.reg_offset else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()

            if self.opt.flip_test:
                output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
                output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
                output['wh_face'] = (output['wh_face'][0:1] + flip_tensor(output['wh_face'][1:2])) / 2
                output['hps'] = (output['hps'][0:1] +
                                 flip_lr_off(output['hps'][1:2], self.flip_idx, num_kp=26)) / 2

                output['face_lms'] = output['face_lms'][0:1]

                output['hand_lms'] = output['hand_lms'][0:1]

                hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                    if hm_hp is not None else None
                reg = reg[0:1] if reg is not None else None
                hp_offset = hp_offset[0:1] if hp_offset is not None else None

            dets = landmark_decode(
                output['hm'], output['wh'], output['hps'], output['wh_face'],
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K,
                face_lms=output['face_lms'], hand_lms=output['hand_lms'])

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = landmark_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 280)
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:11] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp_face_cnt'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='landmark')
        for ind, bbox in enumerate(results[1]):
            if bbox[4] > self.opt.vis_thresh:
                debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='landmark', text=ind)
                debugger.add_coco_face_hp(bbox[5:11], img_id='landmark', text=ind)
                debugger.add_coco_face_lm(bbox[12:149], img_id='landmark', text=ind)
        debugger.show_all_imgs(pause=self.pause)




from diffusion_policy.policy.inpainting.base_inpainting import BaseInpainting, MSE_inequ_opt, inpaint_vanilla

import numpy as np
import torch

'''
4 sequence
0: b0 t0 b1 t1
1: b0 t1 b1 t0
2: b1 t0 b0 t1
3: b1 t1 b0 t0

[b0, b1, t0, t1]
 0    1   2   3
'''


# correct
# SEQ = [[0, 2, 1, 3], [0, 3, 1, 2], [1, 2, 0, 3], [1, 3, 0, 2] ]
mapping = {0: 'b0', 1: 'b1', 2: 't0', 3: 't1', 4: 'ee', 5: 'ee_target', 6: 'block_center', 7: 'all_center'}

# changed
SEQ = [[0, 2, 1, 3], [1, 2, 0, 3], [6, 2, 6, 3], [7, 3, 7, 2]]

class BlockPushInpainting(BaseInpainting):

    def __init__(self, inpainting_method={}, sequence=0):
        # self.sequence = sequence
        self.reset()

        if 'idx' in inpainting_method:
            self.sequence = inpainting_method['idx']
            print('inpainting method:', inpainting_method)
        else:
            self.sequence = sequence

        if 'vanilla' in inpainting_method:
            self.vanilla = inpainting_method['vanilla']
        else:
            self.vanilla = False
        

    def reset(self):
        self.stage = 0
        self.constraint = 0.01

    def inpaint(self, x_ori):
        
        if self.constraint > 0.0001:
            x = x_ori.detach().cpu().numpy()
            
            mask, cond = self.mask, self.cond
            
            # apply inpainting
            if self.vanilla:
                x_inpaint = inpaint_vanilla(x, mask, cond, self.constraint)
            else:
                x_inpaint = MSE_inequ_opt(x, mask, cond, self.constraint)

            # convert numpy to torch
            x_inpaint = torch.tensor(x_inpaint, dtype=torch.float32).to(x_ori.device)
        else:
            x_inpaint = x_ori

        return x_inpaint



    def create_mask_and_data(self, x, obs, update_state=True):

        # convert torch to numpy
        x = x.detach().cpu().numpy()
        obs = obs.detach().cpu().numpy()

        # parse obs
        block0_translation = obs[:, -1, 0:2]
        block1_translation = obs[:, -1, 3:5]
        effector_translation = obs[:, -1, 6:8]
        effector_target_translation = obs[:, -1, 8:10]
        target0_translation = obs[:, -1, 10:12]
        target1_translation = obs[:, -1, 13:15]

        poses = [block0_translation, block1_translation, target0_translation, target1_translation, effector_translation, effector_target_translation,
                 (block0_translation +  block1_translation) / 2, (block0_translation + block1_translation + target0_translation + target1_translation) / 4]
        ee_pose = effector_translation
        next_pose = poses[SEQ[self.sequence][self.stage]]

        # ==== create mask and data   =====
        # based on the stage

        mask_index = [2,3,4]
        # mask_index = [3, 4]
        mask = np.zeros_like(x)
        cond = np.zeros_like(x)

        mask[:, mask_index, :] = 1 # last is choosen
        diff_pose = next_pose - ee_pose

        # normalize diff_pose
        diff_pose_norm1 = diff_pose / np.linalg.norm(diff_pose) * 0.01

        cond[:, mask_index, :] = np.tile(diff_pose_norm1, (x.shape[0], len(mask_index), 1))

        # adjust
        if self.stage in [0]:
            # self.constraint = 0.01
            self.constraint = 1

        elif self.stage in [1]:
            # self.constraint = 0.001
            self.constraint = 1e-6

        elif self.stage in [2, 3]:
            # self.constraint = 0.005
            self.constraint = 1e-6


        if update_state:
            new_stage = False
            if self.stage in [0, 2]:
                distance = np.linalg.norm(diff_pose)
                if distance < 0.055:
                    print("============ update stage ============", self.stage)
                    print('ee pose', ee_pose)
                    print('block pose', mapping[SEQ[self.sequence][self.stage]], next_pose)
                    print('distance', distance)
                    new_stage = True

            elif self.stage in [1, 3]:
                source_pose = poses[SEQ[self.sequence][self.stage-1]]
                distance = np.linalg.norm(next_pose - source_pose )
            
                if distance < 0.055:
                    print("============ update stage ============", self.stage)
                    print('block pose', mapping[SEQ[self.sequence][self.stage-1]], source_pose)
                    print('target pose', mapping[SEQ[self.sequence][self.stage]], next_pose)
                    print('distance', distance)
                    new_stage = True

            if new_stage:
                if self.stage < 3:
                    self.stage += 1
                    print("stage update to ", self.stage, mapping[SEQ[self.sequence][self.stage]])
                else:
                    print("stage update to ", self.stage, "done")


        return mask, cond
    
    def set_mask_cond(self, mask, cond):
        self.mask = mask
        self.cond = cond
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

import numpy as np
from diffusion_policy.policy.inpainting.block_push_inpaint import SEQ, mapping

# SEQ = [[0, 2, 1, 3], [0, 3, 1, 2], [1, 2, 0, 3], [1, 3, 0, 2] ]
# mapping = {0: 'b0', 1: 'b1', 2: 't0', 3: 't1'}


class BlockPushCondition:
    def __init__(self, cond_method={},  sequence=0):
        self.finish_setup = False
        if 'idx' in cond_method:
            self.sequence = cond_method['idx']
        else:
            self.sequence = sequence

        self.stage = 0
        self._use_condition = True

    def pre_process_condition(self, action_norm, obs, cond):
        # obs is normalized
        # according to the sequence, get target poses, and set it.
        pass

    def get_eval_condition(self, obs, cond):
        poses = self._parse_obs(obs)
        target_pose = poses[SEQ[self.sequence][self.stage]]

        extend_cond = target_pose.repeat(1, cond.shape[1], 1).to(cond.device)
        new_cond = torch.cat((cond, extend_cond), dim=-1)

        self.update_stage(poses)

        return new_cond


    def get_train_condition(self, action, obs, cond):
        poses = self._parse_obs(obs)
        last_pos = poses[4] # ee pose

        extend_cond = last_pos.repeat(1, cond.shape[1], 1).to(cond.device)
        new_cond = torch.cat((cond, extend_cond), dim=-1)

        # print('action', action.shape, 'obs', obs.shape, 'cond', cond.shape, 'new_cond', new_cond.shape)
        #             (B, 5, 2)          (B, 5, 16)              (B, 3, 15)

        return new_cond

    def update_stage(self, poses):
        ee_pose = poses[4]
        next_pose = poses[SEQ[self.sequence][self.stage]]
        diff_pose = next_pose - ee_pose

        new_stage = False
        if self.stage in [0, 2]:
            distance = torch.linalg.norm(diff_pose)
            if distance < 0.055:
                print("============ update stage ============", self.stage)
                print('ee pose', ee_pose)
                print('block pose', mapping[SEQ[self.sequence][self.stage]], next_pose)
                print('distance', distance)
                new_stage = True
                self._use_condition = False

        elif self.stage in [1, 3]:
            source_pose = poses[SEQ[self.sequence][self.stage-1]]
            distance = torch.linalg.norm(next_pose - source_pose )
        
            if distance < 0.055:
                print("============ update stage ============", self.stage)
                print('block pose', mapping[SEQ[self.sequence][self.stage-1]], source_pose)
                print('target pose', mapping[SEQ[self.sequence][self.stage]], next_pose)
                print('distance', distance)
                new_stage = True
                self._use_condition = True


        if new_stage:
            if self.stage < 3:
                self.stage += 1
                print("stage update to ", self.stage, mapping[SEQ[self.sequence][self.stage]])
            else:
                print("stage update to ", self.stage, "done")



    def update_task_finish(self, info):
        pass

    def use_condition(self):
        return self._use_condition

    def _parse_obs(self, obs):
        block0_translation = obs[:, -1:, 0:2]
        block1_translation = obs[:, -1:, 3:5]
        target0_translation = obs[:, -1:, 10:12]
        target1_translation = obs[:, -1:, 13:15]
        effector_translation = obs[:, -1:, 6:8]
        effector_target_translation = obs[:, -1:, 8:10]

        poses = [block0_translation, block1_translation, 
                 target0_translation, target1_translation,
                 effector_translation, effector_target_translation, 
                 (block0_translation + block1_translation) / 2,
                 (block0_translation + block1_translation + target0_translation + target1_translation) / 4]
                 
        return poses
    

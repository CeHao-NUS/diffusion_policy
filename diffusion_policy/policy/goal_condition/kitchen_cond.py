
from diffusion_policy.policy.inpainting.kitchen_inpaint import GAOL_QPOSE, Thresholds, ALL_TASKS_np

'''
ALL_TASKS = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]

'''

import numpy as np 
import torch

class KitchenCondition:

    def __init__(self, cond_method={}, sequence=[0,6,4,2]):
        self.finish_setup = False
        self.sequence = sequence
        self._use_condition = True
        self.stage = 0

        self.reset()

    def reset(self):
        self.idle = 0
        self.STAGE_MACHINE = 0

    def pre_process_condition(self, action_norm, obs, cond):
        target_joint_raw = torch.tensor(GAOL_QPOSE[self.sequence[self.stage]], dtype=torch.float32).to(cond.device)
        self.target_joint = action_norm(target_joint_raw)
        self.finish_setup = True

    def get_eval_condition(self, obs, cond):
        # select one point
        target_joint = self.target_joint
        extend_cond = target_joint.repeat(cond.shape[0], cond.shape[1], 1).to(cond.device)
        new_cond = torch.cat((cond, extend_cond), dim=-1)

        self.update_stage(obs)

        return new_cond

    def get_train_condition(self, action, obs, cond):
        # end of action
        gripper_joint = self._parse_obs(obs)
        extend_cond = gripper_joint.repeat(1, cond.shape[1], 1).to(cond.device)
        new_cond = torch.cat((cond, extend_cond), dim=-1)
        return new_cond

    def update_task_finish(self, info):
        complete_tasks = info['completed_tasks'][-1]

        if self.stage < len(self.sequence): 

            if ALL_TASKS_np[self.sequence][self.stage] in complete_tasks:
                print('1 ============ complete task', ALL_TASKS_np[self.sequence][self.stage])
                self.stage += 1
                self.STAGE_MACHINE = 0 # use condition again
                self._use_condition = True

                if self.stage < len(self.sequence) - 1: 
                    print('stage updated to', self.stage, ALL_TASKS_np[self.sequence][self.stage])

        if self.stage >= len(self.sequence):
            self._use_condition = False

    def update_stage(self, obs):
        now_pose = obs[0,-1, :9]
        next_pose = self.target_joint

        distance = torch.norm(now_pose - next_pose)
        if distance < Thresholds[self.sequence[self.stage]]:
            print('0 ============ finish goal reaching', ALL_TASKS_np[self.sequence][self.stage])
            self.STAGE_MACHINE = 1 # to finish task
            self._use_condition = False
    


    def use_condition(self):
        return self._use_condition

    def _parse_obs(self, obs):
        gripper_joint = obs[..., -1:, :9]
        return gripper_joint

from diffusion_policy.policy.inpainting.kitchen_inpaint import GAOL_QPOSE, Thresholds

import numpy as np 
import torch

class KitchenCondition:

    def __init__(self, cond_method={}, sequence=[0,1]):
        self.finish_setup = False
        self.sequence = sequence
        self.stage = 0

    def reset(self):
        pass

    def pre_process_condition(self, action_norm, obs, cond):
        self.target_joint = torch.tensor(GAOL_QPOSE[self.sequence[self.stage]], dtype=torch.float32).to(cond.device)
        self.finish_setup = True

    def get_eval_condition(self, obs, cond):
        # select one point
        target_joint = self.target_joint
        extend_cond = target_joint.repeat(cond.shape[0], cond.shape[1], 1).to(cond.device)
        new_cond = torch.cat((cond, extend_cond), dim=-1)

        return new_cond

    def get_train_condition(self, action, obs, cond):
        # end of action
        gripper_joint = self._parse_obs(obs)
        extend_cond = gripper_joint.repeat(1, cond.shape[1], 1).to(cond.device)
        new_cond = torch.cat((cond, extend_cond), dim=-1)
        return new_cond

    def _parse_obs(self, obs):
        gripper_joint = obs[..., -1:, :9]
        return gripper_joint
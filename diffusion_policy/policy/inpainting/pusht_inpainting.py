
import torch
import numpy as np

from diffusion_policy.policy.inpainting.base_inpainting import BaseInpainting, MSE_inequ_opt


TRAJ_PT = [
    [120, 256],
    # [380, 350]
    [360, 320]

]


class PushtInpaint(BaseInpainting):
    def __init__(self, traj_pt=TRAJ_PT[1]):
        self.reset()
        self.traj_pt = traj_pt

    def reset(self):
        self.constraint = 0.01

    def inpaint(self, x_ori):
        
        if self.constraint > 0.0001:
            x = x_ori.detach().cpu().numpy()
            
            mask, cond = self.mask, self.cond
            
            # apply inpainting
            x_inpaint = MSE_inequ_opt(x, mask, cond, self.constraint)

            # convert numpy to torch
            x_inpaint = torch.tensor(x_inpaint, dtype=torch.float32).to(x_ori.device)
        else:
            x_inpaint = x_ori

        return x_inpaint

    def create_mask_and_data(self, x, update_state=True):
        

        # convert torch to numpy
        x = x.detach().cpu().numpy()
       


        mask_index = list(np.arange(5, 10)) # 
        mask = np.zeros_like(x)
        cond = np.zeros_like(x)

        mask[:, mask_index, :] = 1
        cond[:, mask_index, :] = np.tile(self.traj_pt, (x.shape[0], len(mask_index), 1))


        return mask, cond

    def update_task_finish(self, info):
        pos_agent = info['pos_agent']

        distance = np.linalg.norm(pos_agent - self.traj_pt)
        if np.any(distance < 10):
            self.constraint = 1e-10
            print('goal reached')
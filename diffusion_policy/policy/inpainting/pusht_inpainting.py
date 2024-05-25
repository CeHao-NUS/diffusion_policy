
import torch
import numpy as np

from diffusion_policy.policy.inpainting.base_inpainting import BaseInpainting, MSE_inequ_opt, inpaint_vanilla


TRAJ_PT = [
    [120, 256], # left
    [360, 320], # right
    [256, 100], # top
    [256, 400] # bottom
]


class PushtInpaint(BaseInpainting):
    def __init__(self, inpainting_method={}, traj_pt=TRAJ_PT[0]):
        self.reset()

        if 'idx' in inpainting_method:
            print('inpainting method:', inpainting_method)
            traj_pt = TRAJ_PT[inpainting_method['idx']]
            self.traj_pt = traj_pt  
        else:
            self.traj_pt = traj_pt

        if 'vanilla' in inpainting_method:
            self.vanilla = inpainting_method['vanilla']
        else:
            self.vanilla = False

    def reset(self):
        self.constraint = 0.01

    def inpaint(self, x_ori):
        
        if self.constraint > 0.0001:
            x = x_ori.detach().cpu().numpy()
            
            mask, cond = self.mask, self.cond
            
            # apply inpainting
            if self.vanilla:
                x_inpaint = inpaint_vanilla(x, mask, cond)
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
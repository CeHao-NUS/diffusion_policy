from diffusion_policy.policy.inpainting.base_inpainting import BaseInpainting, MSE_inequ_opt

import numpy as np
import torch





ALL_TASKS = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]

ALL_TASKS_np = np.array(ALL_TASKS)

# '''
# finish time step result
GAOL_QPOSE_t = [
[-1.36963017, -1.63875601,  1.42989846, -2.31530856,  0.00461365,  1.92396435,  2.30387439,  0.00732828,  0.02427612],
[-1.42594753, -1.52722294, 1.30924367, -2.24714636,  0.10967579,  1.94881617,   2.10800026, -0.00888911,  0.04244188],
[-1.31307317, -1.7677692,   1.17057365, -2.09059942, -0.47231228,  1.91915031,  2.17711429,  0.04863294, -0.00300606],
[-1.932388,   -1.34610765,  1.04401558, -1.97994332,  0.20897186,  2.02404115, 1.09642935,  0.02472089,  0.0108924 ],
[-1.25498712, -1.77191600,  0.845096847, -2.24996938, 1.80051879,  1.43479578, -0.485931738,  0.00149035000,  0.0364637693],
[-0.78073299, -1.77210523,  1.85898166, -1.65057168, -0.80402471,  1.37142738,  2.68452562,  0.04182564,  0.01048662],
[-0.94084406, -1.77315779,  2.00355045, -2.22568889, -0.09239141,  1.19881354,  2.03419926,  0.02368121,  0.01992671],
]
# '''

GAOL_QPOSE = [
[-1.33080228, -1.63429675,  1.34676505, -2.35018303, -0.32023517,  1.99985946,  1.82433575,  0.01062759,  0.02368726],
[-1.45460963, -1.57777519,  1.23190805, -2.25114973, -0.08785895,  2.05993457,  1.64937104,  0.0396748 ,  0.03379816],
[-1.36802454, -1.75327432,  1.22521687, -2.18148695, -0.20625091,  2.06750406,  1.91816893,  0.0448648 ,  0.04164103],
[-1.53479903, -1.44072239,  0.9513158 , -2.20061035,  0.11172501,  1.91618992,  1.10777908,  0.03941613,  0.04522792],
[-1.80844819, -1.5454534 ,  0.86183438, -1.15590004,  0.6092384 ,  1.27898935,  1.14064059,  0.02603787,  0.02930077],
[-1.35607216, -1.75515   ,  1.81620705, -1.30393589, -0.93413364,  1.33984027,  2.69356763,  0.02124246,  0.02253345],
[-0.70989012, -1.76104771,  2.16983408, -2.12385488,  0.41425576,  0.93773832,  1.82686586,  0.03469396,  0.03234254],
]

# GAOL_QPOSE[2] = GAOL_QPOSE_t[2]

# GAOL_QPOSE[2] = [-1.28781428, -1.50308143,  1.42715131, -2.12484371,  0.25217237, 1.87259908,  1.20732494,  0.03120042,  0.02240621]

GAOL_QPOSE = np.array(GAOL_QPOSE)

for goal in GAOL_QPOSE:
    goal[-2:] = 0.041


# dim_mask = list(range(7))
dim_mask = list(range(9))

# THRESHOLD = 0.3
Thresholds = [
0.3, # bottom burner 0
0.3, # top burner 1
0.3, # light switch 2
0.3, # slide cabinet 3
0.3, # hinge cabinet 4
0.8, # microwave 5
0.3, # kettle 6

]

SEQ = [1,0]

class KitchenInpaint(BaseInpainting):
    def __init__(self, inpainting_method={}, sequence=SEQ):
        self.reset()
        if 'idx' in inpainting_method:
            self.sequence = inpainting_method['idx']
        else:
            self.sequence = sequence

    def reset(self):
        self.stage = 0
        # self._reset_constraint()
        self.constraint = 1e1
        self.idle = 0
        self.STAGE_MACHINE = 0
        # 0: approach goal, 1: finish task, 2: idle

    def _reset_constraint(self):
        self.idle = -3
        self.constraint = 1e-5
        # self.constraint = 1e-1

    def _update_constraint(self):
        

        if self.STAGE_MACHINE == 2 :
        
            if self.idle >= 0:
                self.constraint = 1e1
                print('2 ======= return to inpainting again')
                self.STAGE_MACHINE = 0 # to approach goal
            else:
                print(' keep idle', self.idle)
                self.idle += 1


    def inpaint(self, x_ori):
        
        if self.constraint > 0.0001: # 1e-4
            x = x_ori.detach().cpu().numpy()
            
            mask, cond = self.mask, self.cond
            
            # apply inpainting
            x_inpaint = MSE_inequ_opt(x, mask, cond, self.constraint)

            # convert numpy to torch
            x_inpaint = torch.tensor(x_inpaint, dtype=torch.float32).to(x_ori.device)
        else:
            x_inpaint = x_ori

        return x_inpaint
    
    def update_task_finish(self, info):
        complete_tasks = info['completed_tasks'][-1]

        if self.stage < len(self.sequence): 

            if ALL_TASKS_np[self.sequence][self.stage] in complete_tasks:
                print('1 ============ complete task', ALL_TASKS_np[self.sequence][self.stage])
                self._reset_constraint()
                self.stage += 1
                self.STAGE_MACHINE = 2 # to idle

                if self.stage < len(self.sequence) - 1: 
                    print('stage updated to', self.stage, ALL_TASKS_np[self.sequence][self.stage])


    
    def create_mask_and_data(self, x, obs, update_state=True):

        # convert torch to numpy
        x = x.detach().cpu().numpy()
        obs = obs.detach().cpu().numpy()

        # already done
        if self.stage >= len(self.sequence):
            self.constraint = 1e-6
            mask = np.zeros_like(x)
            cond = np.zeros_like(x)
            return mask, cond

        # parse obs
        now_pose = obs[0,-1, :9][dim_mask]
        next_pose = np.array(GAOL_QPOSE[self.sequence[self.stage]])[dim_mask]

        # ==== create mask and data   =====
        # based on the stage


        # mask_index = [11, 12, 13, 14, 15]
        mask_index = list(np.arange(3, 16))
        # mask_index = list(np.arange(5, 16))
        mask = np.zeros_like(x)
        cond = np.zeros_like(x)

        idx1_grid, idx2_grid = np.meshgrid(mask_index, dim_mask, indexing='ij')

        mask[:, idx1_grid, idx2_grid] = 1 # last is choosen
        cond[:, idx1_grid, idx2_grid] = np.tile(next_pose, (x.shape[0], len(mask_index), 1))



        # 1. reach, then enter idle constraint.
        if update_state:
            distance = np.linalg.norm(now_pose - next_pose)
            if distance < Thresholds[self.sequence[self.stage]]:
                print('0 ============ finish goal reaching', ALL_TASKS_np[self.sequence][self.stage])
                self.constraint = 1e-6
                self.STAGE_MACHINE = 1 # to finish task

            else:
                self._update_constraint()

        return mask, cond
    
    def set_mask_cond(self, mask, cond):
        self.mask = mask
        self.cond = cond
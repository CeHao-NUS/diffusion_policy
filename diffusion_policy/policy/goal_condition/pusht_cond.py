
import torch
import numpy as np

dummy_point = [0, 0]

TRAJ_PT = [
    [120, 256],
    # [380, 350]
    [360, 320]

]


class PushtCondition:

    def __init__(self, cond_method={}):
        self.traj_pt = torch.tensor(TRAJ_PT[1])
        # self.traj_pt = torch.tensor(dummy_point)
        self.finish_setup = False
        self._use_condition = True

    def pre_process_condition(self, action_norm, obs, cond):
        if not self.finish_setup:
            self.traj_pt = action_norm(self.traj_pt.to(cond.device))
            self.finish_setup = True

    def get_eval_condition(self, obs, cond):

        # repeat dim0, and dim1 of cond, then put fix_action at the last dim
        # fix_action = self.traj_pt.repeat(cond.shape[0], cond.shape[1], 1).to(cond.device)

        fix_action = self.traj_pt.repeat(cond.shape[0], cond.shape[1], 1).to(cond.device)
        new_cond = torch.cat((cond, fix_action), dim=-1)
        return new_cond

    def get_train_condition(self, action, obs, cond):

        # action and obs are normalized
        last_action = action[:,-1:,:].repeat(1, cond.shape[1], 1)
        new_cond = torch.cat((cond, last_action), dim=-1)
        return new_cond


    def update_task_finish(self, info):
        pos_agent = info['pos_agent']

        distance = np.linalg.norm(pos_agent - self.traj_pt)
        if np.any(distance < 10):
            self._use_condition = False
            print('goal reached')

    def use_condition(self):
        return self._use_condition
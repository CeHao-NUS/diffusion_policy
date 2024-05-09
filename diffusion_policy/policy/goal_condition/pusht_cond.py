
import torch

TRAJ_PT = [
    [120, 256],
    # [380, 350]
    [360, 320]

]


class PushtCondition:

    def __init__(self, cond_method={}):
        self.traj_pt = torch.tensor(TRAJ_PT[0])

    def get_fix_condition(self, obs, cond):

        # repeat dim0, and dim1 of cond, then put fix_action at the last dim
        fix_action = self.traj_pt.repeat(cond.shape[0], cond.shape[1], 1).to(cond.device)
        new_cond = torch.cat((cond, fix_action), dim=-1)


        return new_cond

    def get_train_condition(self, action, obs, cond):
        # action and obs are normalized
        last_action = action[:,-1:,:].repeat(1, cond.shape[1], 1)
        new_cond = torch.cat((cond, last_action), dim=-1)

        return new_cond


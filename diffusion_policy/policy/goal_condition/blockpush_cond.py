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



SEQ = [[0, 2, 1, 3], [0, 3, 1, 2], [1, 2, 0, 3], [1, 3, 0, 2] ]
mapping = {0: 'b0', 1: 'b1', 2: 't0', 3: 't1'}


class BlockPushCondition:
    def __init__(self, cond_method={}):
        self.finish_setup = False
        self.sequence = 0

    def pre_process_condition(self, action_norm, obs, cond):
        # obs is normalized
        # according to the sequence, get target poses, and set it.
        pass

    def get_eval_condition(self, obs, cond):
        poses = self._parse_obs(obs)
        # find target 0 and target 1 pose

        target_pose = torch.cat((poses[0], poses[1]), dim=-1) # B, 1, 4
        extend_cond = target_pose.repeat(1, cond.shape[1], 1).to(cond.device)
        new_cond = torch.cat((cond, extend_cond), dim=-1)

        return new_cond


    def get_train_condition(self, action, obs, cond):
        poses = self._parse_obs(obs)

        # final block0 and block1 pose
        last_pos = torch.concat([poses[1], poses[2]], dim=-1) # B, 1, 4
        extend_cond = last_pos.repeat(1, cond.shape[1], 1).to(cond.device)
        new_cond = torch.cat((cond, extend_cond), dim=-1)

        # print('action', action.shape, 'obs', obs.shape, 'cond', cond.shape, 'new_cond', new_cond.shape)
        #             (B, 5, 2)          (B, 5, 16)              (B, 3, 15)

        return new_cond


    def _parse_obs(self, obs):
        block0_translation = obs[:, -1:, 0:2]
        block1_translation = obs[:, -1:, 3:5]
        effector_translation = obs[:, -1:, 6:8]
        effector_target_translation = obs[:, -1:, 8:10]
        target0_translation = obs[:, -1:, 10:12]
        target1_translation = obs[:, -1:, 13:15]

        poses = [block0_translation, block1_translation, 
                 effector_translation, effector_target_translation, 
                 target0_translation, target1_translation]

        return poses
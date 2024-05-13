

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import yaml
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-cd', '--checkpoint_cond', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')

def main(checkpoint, checkpoint_cond, output_dir, device, manual_cfg):
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    payload_cond = torch.load(open(checkpoint_cond, 'rb'), pickle_module=dill)  

    # ========== vanilla policy
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)

    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # =========== conditional policy
    cfg_cond = payload_cond['cfg']
    cls_cond = hydra.utils.get_class(cfg_cond._target_)

    workspace_cond = cls_cond(cfg_cond, output_dir=output_dir)
    workspace_cond: BaseWorkspace
    workspace_cond.load_payload(payload_cond, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy_cond = workspace_cond.model
    if cfg_cond.training.use_ema:
        policy_cond = workspace_cond.ema_model

    device = torch.device(device)
    policy_cond.to(device)
    policy_cond.eval()

    # =========== run eval in conditional policy
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg_cond.task.env_runner,
        output_dir=output_dir)

    runner_log = env_runner.run(policy, policy_cond) # two policies

    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()

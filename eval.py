"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

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
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-g', '--manual_cfg', default=None)
@click.option('-idx', '--index', default=None)
def main(checkpoint, output_dir, device, manual_cfg, index):
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    

    # process the cfg(change runner, and policy)
    if manual_cfg is not None:
        # manual_cfg is the file path of a yaml, read it and update cfg
        # cfg = yaml.safe_load(manual_cfg.open('r'))
        raw_cfg = yaml.safe_load(open(manual_cfg))
        cfg = OmegaConf.create(raw_cfg)
        print(f"Using manual cfg: {manual_cfg}")
    else:
        cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)

    # check inpainting or cond
    if index is not None:
        if 'inpainting' in cfg.policy:
            cfg.policy.inpainting.inpainting_method.idx = index


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
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


    # save score to txt
    score = runner_log['train/mean_score']
    import wandb.sdk.data_types.video as wv
    file_name = 'score' + wv.util.generate_id() + '.txt'

    with open(os.path.join(output_dir, file_name), 'w') as f:
        f.write(str(score))


    

if __name__ == '__main__':
    main()

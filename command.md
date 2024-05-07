

python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


python train.py --config-name=train_diffusion_unet_ddim_lowdim_workspace task=lift_lowdim training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


Download a checkpoint from the published training log folders, such as [https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt](https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt).

Run the evaluation script:
```console
(robodiff)[diffusion_policy]$ 
python eval.py --checkpoint ./data/outputs/0550-test_mean_score=0.969.ckpt --output_dir data/pusht_eval_output --device cuda:0



======================================== block_pushing

# block

mkdir data && cd data

wget https://diffusion-policy.cs.columbia.edu/data/training/block_pushing.zip

unzip block_pushing.zip && rm -f block_pushing.zip && cd ..

wget -O block_pushing_diffusion_policy_cnn.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/block_pushing/diffusion_policy_cnn/config.yaml

python train.py --config-dir=. --config-name=image_block_pushing_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


## transformer

wget -O block_pushing_diffusion_policy_transformer.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/block_pushing/diffusion_policy_transformer/config.yaml


python train.py --config-dir=. --config-name=image_block_pushing_diffusion_policy_transformer.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'



python eval.py --checkpoint ./data/outputs/6700-test_mean_score=1.000.ckpt --output_dir data/block_push_eval



# ===================== kitchen

wget https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip

unzip kitchen.zip && rm -f kitchen.zip && cd ..


wget -O kitchen_diffusion_policy_transformer.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/kitchen/diffusion_policy_transformer/config.yaml

python train.py --config-dir=. --config-name=low_dim_kitchen.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python eval.py --checkpoint ./data/outputs/3000-test_mean_score=0.574.ckpt --output_dir data/kitchen_eval/test


1750-test_mean_score=0.574.ckpt

2700-test_mean_score=0.574.ckpt

python eval.py --checkpoint ./data/outputs/2700-test_mean_score=0.574.ckpt --output_dir data/kitchen_eval2700/test


# ========================== push T=========
wget -O pusht_diffusion_policy_transformer.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_transformer/config.yaml

python eval.py --checkpoint ./data/outputs/0850-test_mean_score=0.967.ckpt --output_dir data/pusht_eval/test


--------- train -----------

python train.py --config-dir=. --config-name=low_dim_pushT_debug.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# image version =================

wget -O pusht_diffusion_policy_cnn.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn//config.yaml

python eval.py --checkpoint ./data/outputs/0500-test_mean_score=0.884.ckpt --output_dir data/pusht_cnn_eval/test





# ============== test 4 results push t ==============

python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt --output_dir ./files/eval_results/pusht/lowdim_tranformer

python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_cnn_epoch=0550-test_mean_score=0.969.ckpt --output_dir ./files/eval_results/pusht/lowdim_cnn

python eval.py --checkpoint  ./files/pretrain_models/pusht/image_diffusion_policy_transformer__epoch=0100-test_mean_score=0.748.ckpt --output_dir ./files/eval_results/pusht/image_transformer

python eval.py --checkpoint  ./files/pretrain_models/pusht/image_diffusion_policy_cnn_epoch=0500-test_mean_score=0.884.ckpt --output_dir ./files/eval_results/pusht/image_cnn


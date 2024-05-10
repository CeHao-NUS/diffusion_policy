

# ********************* eval ***********************

--device 'cuda:2'

# ====================== pusht ===========================
python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt --output_dir ./files/eval_results/pusht/lowdim_tranformer 

(0.5h)

# ==================== block push ====================
python eval.py --checkpoint ./files/pretrain_models/block_pushing/transformer_epoch=7550-test_mean_score=1.000.ckpt --output_dir ./files/eval_results/block_push/lowdim_tranformer 

(2h)

# ===================== kitchen ======================
python eval.py --checkpoint  ./files/pretrain_models/kitchen/transformer_3000-test_mean_score=0.574.ckpt --output_dir ./files/eval_results/kitchen/lowdim_tranformer

(10h)

# *********************** train ****************************
# pusht

python train.py --config-dir='./files/train_yaml/ori/pusht' --config-name='pusht_diffusion_policy_transformer.yaml' training.seed=42 training.device=cuda:0 hydra.run.dir='files/train_results/pusht'

# block

python train.py --config-dir='./files/train_yaml/ori/block_pushing' --config-name='block_pushing_diffusion_policy_transformer.yaml' training.seed=42 training.device=cuda:0 hydra.run.dir='files/train_results/block'

# kitchen

python train.py --config-dir='./files/train_yaml/ori/kitchen' --config-name='kitchen_diffusion_policy_transformer.yaml' training.seed=42 training.device=cuda:0 hydra.run.dir='files/train_results/kitchen'


# *********************** eval inpainting ****************************

# pusht
python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt --output_dir ./files/eval_results/pusht/inpainting --manual_cfg './files/train_yaml/inpainting/pusht/pusht_diffusion_policy_transformer.yaml'

# block
python eval.py --checkpoint  ./files/pretrain_models/block_pushing/transformer_epoch=7550-test_mean_score=1.000.ckpt --output_dir ./files/eval_results/block_push/inpainting --manual_cfg './files/train_yaml/inpainting/block_push/block_push_diffusion_policy_transformer.yaml'

# kitchen
python eval.py --checkpoint  ./files/pretrain_models/kitchen/transformer_3000-test_mean_score=0.574.ckpt --output_dir ./files/eval_results/kitchen/inpainting --manual_cfg './files/train_yaml/inpainting/kitchen/kitchen_diffusion_policy_transformer.yaml'


# ========================== condition training =========================
python train.py --config-dir='./files/train_yaml/goal_cond/pusht' --config-name='pusht_diffusion_policy_transformer.yaml' training.seed=42 training.device=cuda:0 hydra.run.dir='files/train_results/pusht_condition'


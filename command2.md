

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

# pusht
python train.py --config-dir='./files/train_yaml/goal_cond/pusht' --config-name='pusht_diffusion_policy_transformer.yaml' training.seed=42 training.device=cuda:0 hydra.run.dir='files/train_results/pusht_condition_last0'

python eval.py --checkpoint  ./files/train_results/pusht_condition_last0/checkpoints/latest.ckpt --output_dir ./files/eval_results/pusht/cond_test_last0

=== spefical eval with two checkpoints
python eval_classifier.py --checkpoint ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
--checkpoint_cond ./files/train_results/pusht_condition_last0/checkpoints/latest.ckpt --output_dir ./files/eval_results/pusht/cond_test_last0 \
--manual_cfg './files/train_yaml/goal_cond/pusht/pusht_diffusion_policy_transformer.yaml' \
--output_dir ./files/eval_results/pusht/cond_classifier

# block
python train.py --config-dir='./files/train_yaml/goal_cond/block_pushing' --config-name='block_pushing_diffusion_policy_transformer.yaml' training.seed=42 training.device=cuda:0 hydra.run.dir='files/train_results/blockpush_cond_ee'

python eval.py --checkpoint  ./files/train_results/blochpush_cond/checkpoints/latest.ckpt --output_dir ./files/eval_results/blochpush/cond_test_last0 

=== special eval
python eval_classifier.py  --checkpoint ./files/pretrain_models/block_pushing/transformer_epoch=7550-test_mean_score=1.000.ckpt \
--checkpoint_cond  ./files/train_results/blockpush_cond_ee/checkpoints/latest.ckpt \
--manual_cfg ./files/train_yaml/goal_cond/block_pushing/block_pushing_diffusion_policy_transformer.yaml \
--output_dir ./files/eval_results/blcok_push/cond_classifier


# kitchen
python train.py --config-dir='./files/train_yaml/goal_cond/kitchen' --config-name='kitchen_diffusion_policy_transformer.yaml' training.seed=42 training.device=cuda:0 hydra.run.dir='files/train_results/kitchen_cond'

== special eval
python eval_classifier.py --checkpoint ./files/pretrain_models/kitchen/transformer_3000-test_mean_score=0.574.ckpt \
--checkpoint_cond ./files/train_results/kitchen_cond/checkpoints/latest.ckpt \
--manual_cfg ./files/train_yaml/goal_cond/kitchen/kitchen_diffusion_policy_transformer.yaml \
--output_dir ./files/eval_results/kitchen/cond_classifier


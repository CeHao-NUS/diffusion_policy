

# push-T

locations

1. no cond

python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt --output_dir ./files/eval_results/pusht/uncon


2. inpainting (vanilla)

python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt --output_dir ./files/eval_results/pusht/inpaint_vanilla/right --manual_cfg './files/train_yaml/pusht/vanilla/pusht_diffusion_policy_transformer.yaml'


python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt --output_dir ./files/eval_results/pusht/inpaint_opt/bottom --manual_cfg './files/train_yaml/pusht/inpaint/pusht_diffusion_policy_transformer.yaml'


3. goal_cond


python eval_classifier.py --checkpoint ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
--checkpoint_cond ./files/train_results/pusht_condition_last0/checkpoints/latest.ckpt  \
--manual_cfg './files/train_yaml/pusht/goal/pusht_diffusion_policy_transformer.yaml' \
--output_dir ./files/eval_results/pusht/goal_cond/top




python eval_classifier.py --checkpoint ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
            --checkpoint_cond ./files/train_results/pusht_condition_last0/checkpoints/latest.ckpt  \
            --manual_cfg './files/train_yaml/pusht/goal/pusht_diffusion_policy_transformer.yaml' \
            --output_dir ./files/eval_results/pusht/goal_cond/5 \
            --index 3



#
'''


'''

python eval.py --checkpoint  ./files/pretrain_models/kitchen/transformer_3000-test_mean_score=0.574.ckpt \
     --manual_cfg './files/train_yaml/kitchen/inpaint/kitchen_diffusion_policy_transformer.yaml' \
     --output_dir ./files/eval_results/kitchen/inpaint/8 \
     --index 8 
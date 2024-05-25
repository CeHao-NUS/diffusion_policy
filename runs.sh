
sequence=(0 1 2 3)


out_dir="./files/eval_results/"
run_times=1

max_jobs=10


# ================== push t==================

'''
for idx in "${sequence[@]}"; do
    for ((i=1; i<=run_times; i++)); do
        # uncon
        python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
        --output_dir "./files/eval_results/pusht/uncon/${idx}" \
        --index $idx &

        # vanilla
        python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
        --manual_cfg './files/train_yaml/pusht/vanilla/pusht_diffusion_policy_transformer.yaml' \
        --output_dir ./files/eval_results/pusht/vanilla/${idx} \
        --index $idx &


        # inpaint
        python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
         --manual_cfg './files/train_yaml/pusht/inpaint/pusht_diffusion_policy_transformer.yaml' \
         --output_dir ./files/eval_results/pusht/inpaint/${idx}  \
         --index $idx &

        # goal
        python eval_classifier.py --checkpoint ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
            --checkpoint_cond ./files/train_results/pusht_condition_last0/checkpoints/latest.ckpt  \
            --manual_cfg './files/train_yaml/pusht/goal/pusht_diffusion_policy_transformer.yaml' \
            --output_dir ./files/eval_results/pusht/goal_cond/${idx} \
            --index $idx &

            (( count++ ))
            if [[ $count -ge $max_jobs ]]; then
                wait # Wait for all processes to finish before continuing
                count=0 # Reset count
            fi

    done
done
'''

# =================== block ==================
'''
for idx in "${sequence[@]}"; do
    for ((i=1; i<=run_times; i++)); do

    # un cond
    python eval.py --checkpoint  ./files/pretrain_models/block_pushing/transformer_epoch=7550-test_mean_score=1.000.ckpt \
     --output_dir ./files/eval_results/pushblock/uncon/${idx} \
     --index $idx &

    # vanilla
    python eval.py --checkpoint  ./files/pretrain_models/block_pushing/transformer_epoch=7550-test_mean_score=1.000.ckpt \
     --manual_cfg './files/train_yaml/pushblock/vanilla/block_push_diffusion_policy_transformer.yaml'\
     --output_dir ./files/eval_results/pushblock/vanilla/${idx} \
     --index $idx &

    # inpaint
    python eval.py --checkpoint  ./files/pretrain_models/block_pushing/transformer_epoch=7550-test_mean_score=1.000.ckpt \
     --manual_cfg './files/train_yaml/pushblock/inpaint/block_push_diffusion_policy_transformer.yaml'\
     --output_dir ./files/eval_results/pushblock/inpaint/${idx} \
     --index $idx &

    # goal
    python eval_classifier.py  --checkpoint ./files/pretrain_models/block_pushing/transformer_epoch=7550-test_mean_score=1.000.ckpt \
     --checkpoint_cond  ./files/train_results/blockpush_cond_ee/checkpoints/latest.ckpt \
     --manual_cfg ./files/train_yaml/pushblock/goal/block_pushing_diffusion_policy_transformer.yaml \
     --output_dir ./files/eval_results/pushblock/goal_cond/${idx} \
     --index $idx &

     (( count++ ))
        if [[ $count -ge $max_jobs ]]; then
            wait # Wait for all processes to finish before continuing
            count=0 # Reset count
        fi

    done
done
'''


# =================== kitchen ==================

'''
for idx in "${sequence[@]}"; do
    for ((i=1; i<=run_times; i++)); do

    # un cond
    python eval.py --checkpoint  ./files/pretrain_models/kitchen/transformer_3000-test_mean_score=0.574.ckpt \
     --output_dir ./files/eval_results/kitchen/uncon/${idx} \
     --index $idx &

    # vanilla
    python eval.py --checkpoint  ./files/pretrain_models/kitchen/transformer_3000-test_mean_score=0.574.ckpt \
     --manual_cfg './files/train_yaml/kitchen/vanilla/kitchen_diffusion_policy_transformer.yaml' \
     --output_dir ./files/eval_results/kitchen/vanilla/${idx} \
     --index $idx &

    # inpaint
    python eval.py --checkpoint  ./files/pretrain_models/kitchen/transformer_3000-test_mean_score=0.574.ckpt \
     --manual_cfg './files/train_yaml/kitchen/inpaint/kitchen_diffusion_policy_transformer.yaml' \
     --output_dir ./files/eval_results/kitchen/inpaint/${idx} \
     --index $idx &

    # goal cond
    python eval_classifier.py --checkpoint ./files/pretrain_models/kitchen/transformer_3000-test_mean_score=0.574.ckpt \
     --checkpoint_cond ./files/train_results/kitchen_cond/checkpoints/latest.ckpt \
     --manual_cfg ./files/train_yaml/kitchen/goal/kitchen_diffusion_policy_transformer.yaml \
     --output_dir ./files/eval_results/kitchen/goal_cond/${idx} \
     --index $idx &


     (( count++ ))
        if [[ $count -ge $max_jobs ]]; then
            wait # Wait for all processes to finish before continuing
            count=0 # Reset count
        fi

    done
done
'''
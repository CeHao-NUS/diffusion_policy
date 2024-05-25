
sequence=(0 1 2 3)


out_dir="./files/eval_results/"
run_times=10



# ================== push t==================


for idx in "${sequence[@]}"; do
    for ((i=1; i<=run_times; i++)); do
        # uncon
        python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
        --output_dir "./files/eval_results/pusht/uncon/${idx}" \
        --index $idx 

        # vanilla
        python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
        --manual_cfg './files/train_yaml/pusht/vanilla/pusht_diffusion_policy_transformer.yaml' \
        --output_dir ./files/eval_results/pusht/vanilla/${idx}


        # inpaint
        python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
         --manual_cfg './files/train_yaml/pusht/inpaint/pusht_diffusion_policy_transformer.yaml' \
         --output_dir ./files/eval_results/pusht/inpaint_opt/${idx} 

        # goal
        python eval_classifier.py --checkpoint ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt \
            --checkpoint_cond ./files/train_results/pusht_condition_last0/checkpoints/latest.ckpt  \
            --manual_cfg './files/train_yaml/pusht/goal/pusht_diffusion_policy_transformer.yaml' \
            --output_dir ./files/eval_results/pusht/goal_cond/${idx} \
            --index $idx

    done
    wait
done



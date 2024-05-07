

# ====================== pusht ===========================
python eval.py --checkpoint  ./files/pretrain_models/pusht/lowdim_diffusion_policy_transformer_epoch=0850-test_mean_score=0.967.ckpt --output_dir ./files/eval_results/pusht/lowdim_tranformer


# ==================== block push ====================
python eval.py --checkpoint  ./files/pretrain_models/block_pushing/transformer_epoch=7550-test_mean_score=1.000.ckpt --output_dir ./files/eval_results/block_push/lowdim_tranformer



# ===================== kitchen ======================
python eval.py --checkpoint  ./files/pretrain_models/kitchen/transformer_3000-test_mean_score=0.574.ckpt --output_dir ./files/eval_results/kitchen/lowdim_tranformer


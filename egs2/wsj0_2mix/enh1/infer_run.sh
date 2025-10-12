for en in 2; do
    for re in 3 6 8; do 
        CUDA_VISIBLE_DEVICES=1 ./run.sh --stage 7 --stop_stage 8 \
        --enh_config conf/efficient_train/tisdiss/train_enh_tisdiss_tflocoformer_en-residual_en1x2_re1x6_l1+1x6.yaml \
        --ngpu 1 --gpu_inference true --inference_model valid.loss.ave_5best.pth \
        --inference_enh_config conf/efficient_infer/en_re/en_${en}_re_${re}.yaml \
        --inference_nj 8
    done
done
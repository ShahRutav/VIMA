HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main/train.py \
        algo=llava \
        data_module.task_selection=visual_manipulation \
        prefix=temp \
        suffix=100ep \
        use_wandb=False run_name=temp \
        epochs=100 num_trajs=5000 \
        trainer.accumulate_grad_batches=8 bs=2 gpus=2 \
        trainer.num_sanity_val_steps=2 \
        trainer.strategy=ddp_find_unused_parameters_true \
        data_module.dataloader_num_workers=2 \

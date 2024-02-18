HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python main/train.py \
        algo=llava \
        data_module.task_selection=visual_manipulation \
        prefix=temp \
        suffix=100ep \
        use_wandb=False run_name=temp \
        warmup_epochs=20 decay_epochs=50 epochs=100 gpus=1 \
        trainer.accumulate_grad_batches=1 bs=2 \
        trainer.num_sanity_val_steps=2 \
        trainer.strategy=ddp_find_unused_parameters_true \

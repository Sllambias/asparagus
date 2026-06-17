HYDRA_FULL_ERROR=1 asp_pretrain --config-name 
HYDRA_FULL_ERROR=1 asp_train_seg --config-name test_seg
HYDRA_FULL_ERROR=1 asp_finetune_seg --config-name 
HYDRA_FULL_ERROR=1 asp_train_cls --config-name test_cls
HYDRA_FULL_ERROR=1 asp_finetune_cls --config-name 
HYDRA_FULL_ERROR=1 asp_train_reg --config-name test_reg task=REGR000_LauritSynRegr
HYDRA_FULL_ERROR=1 asp_test_seg test_task= checkpoint_run_id= test_split=


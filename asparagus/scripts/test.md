HYDRA_FULL_ERROR=1 asp_pretrain --config-name pt_lauritsyn_tiny
HYDRA_FULL_ERROR=1 asp_train_seg --config-name seg_lauritsynseg_tiny
HYDRA_FULL_ERROR=1 asp_finetune_seg --config-name ft_seg_lauritsynseg_tiny
HYDRA_FULL_ERROR=1 asp_train_cls --config-name cls_lauritsyncls_tiny
HYDRA_FULL_ERROR=1 asp_finetune_cls --config-name ft_cls_lauritsyncls_tiny
HYDRA_FULL_ERROR=1 asp_train_reg --config-name reg_lauritsynreg_tiny
HYDRA_FULL_ERROR=1 asp_test_seg test_task=Task997_LauritSynSeg checkpoint_run_id=834447 test_split=TEST_75_15_10


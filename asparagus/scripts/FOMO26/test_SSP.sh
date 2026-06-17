python repos/asparagus/asparagus/scripts/FOMO26/Task1_predict.py \
 --model_dir repos/asparagus_data/models/CLS000_LauritSynCLS/unet_clsreg_tiny__3D/script=train_cls/root=base__stem=default_train_cls/leaf=test_cls__clargs=/split_40_10_50__fold=0/run_id=214718 \
 --checkpoint_name best \
 --output Predict1_test.txt \
 --flair datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --adc datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --dwi datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --swi datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --accelerator cpu

python repos/asparagus/asparagus/scripts/FOMO26/Task2_predict.py \
 --output Predict2_test.nii.gz \
 --flair datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --dwi datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --t2s datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --accelerator cpu

python repos/asparagus/asparagus/scripts/FOMO26/Task3_predict.py \
 --model_dir repos/asparagus_data/models/REGR000_LauritSynRegr/unet_clsreg_tiny__3D/script=train_reg/root=base__stem=default_train_reg/leaf=test_reg__clargs=/split_40_10_50__fold=0/run_id=456670 \
 --checkpoint_name best \
 --output Predict3_test.txt \
 --t1 datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --accelerator cpu

python repos/asparagus/asparagus/scripts/FOMO26/Task4_predict.py \
 --model_dir repos/asparagus_data/models/SEG000_LauritSynSeg/unet_tiny__3D/script=train_seg/root=base__stem=default_train_seg/leaf=test_seg__clargs=/split_40_10_50__fold=0/run_id=377904 \
 --checkpoint_name best \
 --output Predict4_test.nii.gz \
 --t2 datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --accelerator cpu

python repos/asparagus/asparagus/scripts/FOMO26/Task5_predict.py \
 --model_dir repos/asparagus_data/models/CLS000_LauritSynCLS/unet_clsreg_tiny__3D/script=train_cls/root=base__stem=default_train_cls/leaf=test_cls__clargs=/split_40_10_50__fold=0/run_id=214718 \
 --checkpoint_name best \
 --output Predict5_test.txt \
 --t1 datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --accelerator cpu

python repos/asparagus/asparagus/scripts/FOMO26/Task6_and_7_predict.py \
 --model_dir repos/asparagus_data/models/PT001_ClevelandCCF/unet_tiny__3D/script=pretrain/root=base__stem=debug_plugins/leaf=DEBUG_Plugins__clargs=/split_40_10_50__fold=0/run_id=908385 \
 --checkpoint_name best \
 --output Predict6_and_7_test.nii.gz \
 --input datasets/CUNMET/sub-1001/ses-01/anat/sub-1001_ses-01_T1w_new_defaced.nii.gz \
 --accelerator cpu
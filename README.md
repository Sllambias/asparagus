# Asparagus

### Environment Variables

### Task Conversion
The idea of task conversion for large pretraining datasets is to convert the data to asparagus format and carry out all preprocessing. One critical assumption here is that preprocessing for a given pretraining dataset is always identical. If this assumption is not true, Yucca may be better suited with its separation of Task Conversion and Preprocessing, which allows any given task converted dataset to be easily preprocessed in multiple ways. 

### Generate data splits
python asparagus/pipeline/run/split.py -t TASK --fn split_80_20


### Run Pretraining with hydra config
python asparagus/pipeline/run/train.py --config-name pretrain  



### Rerun previously executed jobs
python my_app.py --experimental-rerun $OUTPUT_DIR/config.pickle
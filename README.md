# code

## preprocessing
download code to Lian_project_code/
 
## train DCLGAN
python train_pelvic.py --gpu {GPU_ID} --dataroot {DATA_DIR} --log_dir {LOG_DIR} --checkpoints_dir {CHECKPOINT_DIR} 

## test DCLGAN
python test_pelvic.py --gpu {GPU_ID} --dataroot {DATA_DIR} --checkpoints_dir {CHECKPOINT_DIR} --results_dir {OUTPUT_DIR} --epoch final



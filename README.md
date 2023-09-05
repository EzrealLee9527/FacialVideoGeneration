# FacialVideoGeneration

## train
stage1: train landmark adapter model
```
make train_adapter_ddp_sota
```

stage2: train motion model
```
make train_motion_based_adapter_ddp_sota
```

## eval
```
make eval_train_motion_based_adapter
```
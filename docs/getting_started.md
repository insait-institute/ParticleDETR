# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train ParticleDETR with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/particle_detr/particle_detr_base.py 8
```

Eval ParticleDETR with 8 GPUs
```
./tools/dist_test.sh ./projects/configs/particle_detr/particle_detr_base.py ./path/to/ckpts.pth 8
```
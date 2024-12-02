# mcr

### Enviroment

#### Prerequisites

Python = 3.7.13

Pytorch = 1.10.0

CUDA = 11.3

### Prepare Dataset

We use two vision and language datasets: [MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb), [UPMC Food-101](https://visiir.isir.upmc.fr/explore),

### Train

1. Download the pre-trained ViLT model weights from [here](https://github.com/dandelin/ViLT.git).

2. Start to train.

python run.py with data_root=<ARROW_ROOT> \
        num_gpus=<NUM_GPUS> \
        num_nodes=<NUM_NODES> \
        per_gpu_batchsize=<BS_FITS_YOUR_GPU> \
        <task_finetune_mmimdb or task_finetune_food101> \
        load_path=<PRETRAINED_MODEL_PATH> \
        exp_name=<EXP_NAME>
        
2. Start to test.

python run.py with data_root=<ARROW_ROOT> \
        num_gpus=<NUM_GPUS> \
        num_nodes=<NUM_NODES> \
        per_gpu_batchsize=<BS_FITS_YOUR_GPU> \
        <task_finetune_mmimdb or task_finetune_food101> \
        load_path=<PRETRAINED_MODEL_PATH> \
        exp_name=<EXP_NAME>\
        test_only=True

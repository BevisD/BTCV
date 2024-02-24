#!/bin/bash
#SBATCH --qos epsrc
#SBATCH -J BTCVFineTune
#SBATCH --account phwq4930-gbm
#SBATCH --time 1:0:0

module purge
module load baskerville
module load bask-apps/live
module load Python/3.8.6-GCCcore-10.2.0

source .venv/bin/activate

date
which python

python main.py \
--json_list=jsons/neov-seg.json \
--data_dir=/bask/projects/p/phwq4930-gbm/Ines/Ovarian/Data/NeOv \
--feature_size=48 \
--pretrained_model_name='swin_unetr_1_channel.pt' \
--resume_ckpt \
--use_checkpoint \
--batch_size=16 --max_epochs=<total-num-epochs> \
--save_checkpoint

date
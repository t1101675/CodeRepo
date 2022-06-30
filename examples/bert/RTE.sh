#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --rdzv_id=1 \
                  --rdzv_backend=c10d \
                  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

BASE_PATH="/home/guyuxian/ModelCenter"
VERSION="bert-large-cased"
DATASET="RTE"
SAVE_PATH="${BASE_PATH}/results/finetune-bert-large-cased"

OPTS=""
OPTS+=" --model-config ${BASE_PATH}/configs/bert/bert-large-cased"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --dataset_name ${DATASET}"
OPTS+=" --batch-size 64"
OPTS+=" --warmup-iters 40"
OPTS+=" --lr 0.00005"
OPTS+=" --max-encoder-length 512"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --train-iters 400"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 10.0"
OPTS+=" --loss-scale 128"

export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/bert/finetune_bert.py ${OPTS}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/${DATASET}.log

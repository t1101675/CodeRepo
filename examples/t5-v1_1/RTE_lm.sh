#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="/home/guyuxian/ModelCenter/"
VERSION="large"
DATASET="RTE"
SAVE_PATH="${BASE_PATH}/results/finetune-t5-v1_1"

OPTS=""
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${BASE_PATH}/configs/t5/t5-v1_1-${VERSION}"
OPTS+=" --batch-size 16"
OPTS+=" --train-iters 800"
OPTS+=" --save-iters 1000"
OPTS+=" --max-encoder-length 512"
OPTS+=" --max-decoder-length 2"
OPTS+=" --save ${BASE_PATH}/results"
OPTS+=" --save-name finetune-t5-v1_1-ckpt"
OPTS+=" --lr 0.00001"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 4096"
# OPTS+=" --load ${BASE_PATH}/results/t5-v1_1-${VERSION}.pt"

export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/examples/t5-v1_1/finetune_t5-v1_1_lm.py ${OPTS}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/finetune-t5-v1_1-${VERSION}-${DATASET}.log

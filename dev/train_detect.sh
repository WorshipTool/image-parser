
EPOCHS_COUNT=1000

# load path from argument
if [ $# -eq 0 ]
  then
    echo "Dataset path not supplied"
    exit 1
fi

DATASET_PATH=$1

CONTINUE_TRAINING=0
FROM_MODEL_PATH="yolov8n.pt"
if [ $# -eq 2 ]
  then
    CONTINUE_TRAINING=1
    FROM_MODEL_PATH=$2
fi

# device=mps
# batch=4 1.15s/it
# no mps
# batch=16 25s/it
# batch=4 3.3s/it
yolo task=detect mode=train model=$FROM_MODEL_PATH data=$DATASET_PATH epochs=$EPOCHS_COUNT imgsz=640 batch=8 patience=200
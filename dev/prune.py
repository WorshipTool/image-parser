import sys;
import torch;
import torch.nn.utils.prune as tp;
from ultralytics import YOLO
from nni.compression.torch import LevelPruner

if sys.argv[1] == None:
    print("Please provide a path to the model you wish to prune.")
    exit(1)
modelPath = sys.argv[1]

if sys.argv[2] == None:
    print("Please provide a path to the output file.")
    exit(1)
outputPath = sys.argv[2] 


model = YOLO(modelPath)


config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model, config_list)
pruner.compress()
pruner.export_model(outputPath)




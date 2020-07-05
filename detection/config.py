from easydict import EasyDict as Edict 
import torch
import os
def get_config():
    conf=Edict()
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf.confidence_threshold=0.5
    conf.top_k=5000
    conf.nms_threshold=0.4
    conf.keep_top_k=750
    conf.trained_model="detection/weights/mobilenet0.25_new.pth"
    conf.network="mobile0.25" # or resnet50
    conf.cpu=False
    return conf
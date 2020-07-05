from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
from config import get_config

class Retina_Detector:
    def __init__(self):
        torch.set_grad_enabled(False)
        cudnn.benchmark = True
        self.opt=get_config()
        if self.opt.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.opt.network == "resnet50":
            self.cfg = cfg_re50
        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase = 'test')
        self.net = self.load_model(self.net, self.opt.trained_model, self.opt.cpu)
        self.net.eval()
       
        self.net = self.net.to(self.opt.device)


    def check_keys(self,model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self,state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model(self,model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.emove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model
        
    
    def img_process(self, img):
        target_size = self.cfg["image_size"]
        max_size = 1080
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
       
        
        return im, im_scale
    
    def detect(self,img):
        img,imscale=self.img_process(img)
     
        resize=1
        img_raw = img
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.opt.device)
        scale = scale.to(self.opt.device)
       
        tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))
        t1=time.time()
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
       
        priors = priorbox.forward()
        
        priors = priors.to(self.opt.device)
        prior_data = priors.data
        
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.opt.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        
        # ignore low scores
        inds = np.where(scores > self.opt.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.opt.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.opt.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        print("len ",len(dets))
        landms = landms[keep]
        dets/=imscale
        landms /=imscale

        # keep top-K faster NMS
        dets = dets[:self.opt.keep_top_k, :]
        boxes=[list(map(int, x)) for x in dets]

        landms = landms[:self.opt.keep_top_k, :]
        lands=[list(map(int, x)) for x in landms]
        # dets = np.concatenate((dets, landms), axis=1)
        

        return boxes,lands
     

if __name__ == '__main__':
    X=Retina_Detector()
    image=cv2.imread("a.jpg")
    # for i in range(100):
    #     t1=time.time()
    #     dets=X.detect(image)
    #     print(len(dets))
    #     print(time.time()-t1)
    boxes,landms=X.detect(image)
    for b in landms:
                
                text = "{:.4f}".format(b[4])
                # b = list(map(int, b))
                print(b)
                # cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        
                cv2.circle(image, (b[0], b[1]), 1, (0, 0, 255), 4)
                cv2.circle(image, (b[2], b[3]), 1, (0, 255, 255), 4)
                cv2.circle(image, (b[6], b[7]), 1, (255, 0, 255), 4)
                cv2.circle(image, (b[8], b[9]), 1, (0, 255, 0), 4)
            

           

    name = "test"
    cv2.imshow(name, image)
    cv2.waitKey(0)

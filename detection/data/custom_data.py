import os
import os.path
import glob
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json

class Customdata(data.Dataset):
    def __init__(self, dataroot, preproc=None):
        self.preproc = preproc
        self.list_images=glob.glob(os.path.join(dataroot,"*.jpg"))
        self.list_anotations=[x[:-3]+"json" for x in self.list_images]
        self.err=open("err.txt","w+")

    def __len__(self):
        return len(self.list_images)

    def get_labels(self,path_annotation):
        try:
            labels=[]
            with open(path_annotation) as json_file:
                data = json.load(json_file)
                width=data['imageWidth']
                height=data['imageHeight']
                for shape in data['shapes']:
                    points=shape["points"]
                    (x1,y1),(x2,y2),(x3,y3),(x4,y4)=points
                
                    xmin=min(x1,x2,x3,x4)
                    ymin=min(y1,y2,y3,y4)
                    xmax=max(x1,x2,x3,x4)
                    ymax=max(y1,y2,y3,y4)
                    labels.append([xmin,ymin,xmax,ymax,x1,y1,x2,y2,x3,y3,x4,y4])
            return labels # xmin ymin xmax ymax : bounding box , (x1,y1) (x2,y2) (x3,y3) (x4,y4)  : top left,top right,bottom right , bottom left
        except:
            self.err.write(path_annotation+"\n")
            return []

    def __getitem__(self, index):
        img = cv2.imread(self.list_images[index])
        height, width, _ = img.shape

        labels = self.get_labels(self.list_anotations[index])
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # xmin
            annotation[0, 1] = label[1]  # ymin
            annotation[0, 2] = label[2]   # xmax
            annotation[0, 3] = label[3]   # ymax

            # landmarks
            annotation[0, 4] = label[4]    #x1
            annotation[0, 5] = label[5]    #y1
            annotation[0, 6] = label[6]    # x2
            annotation[0, 7] = label[7]    # y2

            annotation[0, 8] = (label[4]+label[8])/2  # x center
            annotation[0, 9] =  (label[5]+label[9])/2 # y_center

            
            annotation[0, 10] = label[8]  # x3
            annotation[0, 11] = label[9]   # y3
            annotation[0, 12] = label[10]  # x4
            annotation[0, 13] = label[11]  # y4

            
            if (annotation[0, 4]<0):
                annotation[0, 14] = 1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

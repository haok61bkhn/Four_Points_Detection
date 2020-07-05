from detection.detect import  Retina_Detector
from alignmet_four_point import Alignment
import cv2
import glob
if __name__ == '__main__':
    X=Retina_Detector()
    A=Alignment()
    for path in glob.glob("test/*"):
        image=cv2.imread(path)
        # for i in range(100):
        #     t1=time.time()
        #     dets=X.detect(image)
        #     print(len(dets))
        #     print(time.time()-t1)
        boxes,landms=X.detect(image)
        print("len boxes",len(boxes))
        for box,land in zip(boxes,landms):
            img=A.align(image,[(land[0],land[1]),(land[2],land[3]),(land[6],land[7]),(land[8],land[9])])    
            name = "test"
            # img1=image[box[0]:box[1],box[2]:box[3]]
            cv2.imshow("image",image)
            cv2.imshow(name, img)
            # cv2.imshow("box",img1)
            cv2.waitKey(0)
           

    


# Four_Points_Detection
------------------------------------------
Demo :

    python3 test.py ( with license plate)
    you can edit config at detection/config.py and detection/data/config.py ( mostly image_target )

------------------------------------------

Train:
 
  Dataset:
      
      make data by labelme tool
      example in data/dataset
      you can edit custom_dataset that has format similar detection/data/custom_data.py
  
  Run:
  
      1) Edit config in train.py ( have 2 backbone mobilenet0.25 and resnet50)
      
      2) python3 train.py    
      
-------------------------------------------

 
 
 References:
  
     https://github.com/wkentaro/labelme
     https://github.com/yhenon/pytorch-retinanet
     
 Goodluck for you!
 
 if you have any problems please contact me (https://www.facebook.com/hoangquoc.hao)

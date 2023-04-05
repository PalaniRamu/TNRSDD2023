# TNRSDD2023
# step 1:  Install yolov7 libraries
# Download YOLOv7 repository and install requirements
!git clone https://github.com/WongKinYiu/yolov7
%cd yolov7
!pip install -r requirements.txt
#!pip install wandb
!pip install torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
import torch
from IPython.display import Image, clear_output  # to display images
#from utils.google_utils import gdrive_download  # to download models/datasets
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt 
clear_output()
print('Setup completed')


#step 2.  Training data access from shared files
# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')
!unzip -u "data in google drive in zi format " -d "specify destination in colab"
drive.flush_and_unmount()
clear_output()

#step 3: Create folder structure , necessary changes in parameter setting
# the dataset slicing 80% for train and 20% for test.
# Commented out IPython magic to ensure Python compatibility.
import os,glob
%cd /content/yolov7/dataset
#!mkdir dataset
!mkdir train
!mkdir test
!mkdir train/images
!mkdir train/labels
!mkdir test/images
!mkdir test/labels


base_path = '/content/yolov7/dataset/Data'

No_of_files = len(glob.glob(base_path+'/*.txt'))

# print('total Number of files :',No_of_files)
# print('80% percent :',round((No_of_files/100)*97))
# print('20% percent :',round((No_of_files/100)*3))

part1=round((No_of_files/100)*80)
part2=round((No_of_files/100)*20)
#part3=round((No_of_files/100)*5)


fcntr=1
file_list = [filename for filename in glob.glob(base_path+'/*.txt')]
for file in file_list:
  if len(glob.glob(file.replace('.txt','*'))) <2 :
    print ('Jpeg file not found',file)
    break
  if fcntr<= part1 :   #  80% files copied in Train files
    try:
      #shutil.copyfile( file, '/content/RSDD2022/dataset/train/labels')
      os.system('cp '+file+ ' /content/yolov7/dataset/train/labels/')
      file=file.replace('.txt','.jpg')
      #shutil.copyfile( file, '/content/RSDD2022/dataset/train/images' )
      os.system('cp '+file+ ' /content/yolov7/dataset/train/images/')
    except IOError:
      print('file not found')
  else:
    try:
      os.system('cp '+file+ ' /conten/yolov7/dataset/test/labels/')
      file=file.replace('.txt','.jpg')
      os.system('cp '+file+ ' /content/yolov7/dataset/test/images/')
    except IOError:
      print('file not found  ' + file)
  fcntr=fcntr+1

print('Total No. of Train Image: ',len(glob.glob('/content/yolov7/dataset/train/images/*.jpg')))
print('Total No. of Test Image: ',len(glob.glob('/content/yolov7/dataset/test/images/*.jpg')))

# Step 4: Optional draw graph - class count
## Model Training -- Training
# Commented out IPython magic to ensure Python compatibility.

import collections
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns


base_path = '/content/yolov7/dataset/Data'
damageTypes=['D00','D10','D20','D40']

# govs corresponds to municipality name.
#govs = ['Sunny', 'Raining', 'Winter', 'Darkness']

# the number of each class labels.
#count_dict = collections.Counter(cls_names)
cls_count = []
total_images=0
cl01=cl02=cl03=cl04=0
file_list = [filename for filename in glob.glob(base_path+'/*.txt')]
for file in file_list:
  total_images = total_images + 1
  f = open(file,'r')
  data = f.read()
  cl01= cl01+data.count("0 ")
  cl02= cl02+data.count("1 ")
  cl03= cl03+data.count("2 ")
  cl04= cl04+data.count("3 ")
  f.close()
  
  cls_count= [cl01,cl02,cl03,cl04]

# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
  
if __name__ == '__main__':
    
    # creating data on which bar chart will be plot
    x = damageTypes
    y = cls_count
      
    # setting figure size by using figure() function 
    plt.figure(figsize = (10, 5))
    # making the bar chart on the data
    plt.bar(x, y,color=['black', 'red', 'green', 'blue'])
    # calling the function to add value labels
    addlabels(x, y)
      
    # giving title to the plot
    plt.title("Road Surface Damage Classes - Training Data")
      
    # giving X and Y labels
    plt.xlabel("Object Class")
    plt.ylabel("Class counts - Training")
      
    # visualizing the plot
    plt.show()

# Step 5 : Create a training model
%cd /content/yolov7
!python train.py  --weights /content/yolov7/yolov7_training.pt --data "data/custom.yaml" --workers 4 --batch-size 8 --img 600 --cfg cfg/training/yolov7x.yaml --name RSDD2023_Jan_Normal --hyp data/hyp.scratch.p5.yaml --epochs 50

# Step 6: - Real time evaluation
%cd /content/yolov7
!python detect.py --weights /content/yolov7/runs/exp/ImageAugmentRSDD2023_trained_model.pt --conf 0.1 --source /content/RSDD2022/Uniqueframes

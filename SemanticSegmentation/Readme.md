## Semantic Segmentation
This project is basically a recreation of pyimagesearch blog post “Semantic segmentation with OpenCV and deep learning”, in this project I performed semantic segmentation on an image using OpenCV and Deep Learning. ENet Deep Learning architecture was used to perform semantic segmentation to image.

 
ENet was trained on Cityscapes dataset with 20-classes including
•	Unlabeled (i.e., background)
•	Road
•	Sidewalk
•	Building
•	Wall
•	Fence
•	Pole
•	TrafficLight
•	TrafficSign
•	Vegetation
•	Terrain
•	Sky
•	Person
•	Rider
•	Car
•	Truck
•	Bus
•	Train
•	Motorcycle
•	Bicycle

## Installation
OpenCV 3.4.1 or higher is required for this project, need to install imutils whish can be installed using

`$ pip install –upgrade imutils`

## Usage
First a legend visualization was created so we can easily visually associate a class label with a color. The legend consists of class label and a colored rectangle next to it. The result:
 

Segment.py performs segmentation on images, we run this file using command

`$ python seg.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --image images/example_01.png`

The results:

Example_01.png
 
Example_02.png
 
Example_03.png
 
Example_04.png
 

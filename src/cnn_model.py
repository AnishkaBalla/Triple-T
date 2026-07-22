import torch
import torch.nn as nn
#goal: predict bounding boxes of the microplastic
class CustomCNN(nn.Module):
   def __init__(self, max_objects = 10):
       super().__init__()
       self.max_objects = max_objects #the max number of microplastics you can detect in one image
       self.conv_first = nn.Conv2d(in_channels= 3, out_channels = 32, kernel_size = 3)
       self.relu = nn.ReLU() #activation function
       self.pool = nn.MaxPool2d(kernel_size = 2) #lower dimensions
       self.conv_second = nn.Conv2d(in_channels= 32, out_channels = 64, kernel_size = 3)
       self.conv_third = nn.Conv2d(in_channels= 64, out_channels = 128, kernel_size = 3)
       self.batch_norm = nn.BatchNorm2d(128) #normalization
       self.flatten = nn.Flatten() # change to 1D
       self.dense = nn.Linear(115200, 128) #brings extracted features together
       self.dropout = nn.Dropout(0.5) #randomly disable 50 percent of the nuerons to prevent overfitting while training
       self.output = nn.Linear(128, max_objects * 5) #the five outputs are x_center, y_center, width, height, and confidence
   def forward(self, x):
       x = self.conv_first(x)
       x = self.relu(x)
       x = self.pool(x)


       x = self.conv_second(x)
       x = self.relu(x)
       x = self.pool(x)


       x = self.conv_third(x)
       x = self.relu(x)
       x = self.batch_norm(x)
       x = self.pool(x)


       x = self.flatten(x)


       x = self.dense(x)


       x = self.relu(x)


       x = self.dropout(x)


       x = self.output(x)
       x = self.sigmoid(x) #to keep confidence between 0 and 1


       x = x.view(
           -1,
           self.max_objects,
           5
       ) # returns (batch_size, number of bounding boxes, 5 values each)
       return x


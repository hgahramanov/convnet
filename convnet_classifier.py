import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from PIL import Image


class conv_net(nn.Module):
    def __init__(self, in_channels, kernelsize):
        super(conv_net, self).__init__()
        self.kernelsize = kernelsize
        self.in_channels = in_channels
        self.out_channels = 2
        
    def build_net(self):
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernelsize),
            nn.Conv2d(self.out_channels, 10, self.kernelsize),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(2420640, 10)
        )
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.net.parameters(), lr = 0.001)


    def train(self, train_data, comparison_data, epochs=10):
        self.build_net()
        model = self.net
        optimizer = self.optimizer
        for i in range(epochs):
            prediction = model(train_data)
            loss = self.loss_func(prediction, comparison_data)
            optimizer.step()
            optimizer.zero_grad()
        self.model = model
        weights = model.parameters()
        return weights

    def test(self, test_data):
       return self.model(test_data)
    
class matricize:
    def __init__(self, path): 
        self.path = path

    def to_tensor(self):
        img = Image.open(self.path)
        img = np.asarray(img)
        img = img[:500, :500]
        img = np.moveaxis(img, -1, 0) #pytorch reuqires the channel dimension to come first, then height and lastly width.
        #however in img channel comes as the last dimension. Therefore, we switch its position to the beginning.
        img = torch.from_numpy(img)[None, ...].float()
        return img

ob = conv_net(3, 5)
ob2 = matricize('mountain.jpg')
img = ob2.to_tensor()
comp_data =torch.from_numpy(np.asarray([0]))
print('SHAPE OF IMAGE: ', img.shape)
print(ob.train(img, comp_data))
print('TRAINING SUCCESSFUL!')
print(ob.test(img))


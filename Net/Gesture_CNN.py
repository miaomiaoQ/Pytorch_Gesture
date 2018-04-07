# -*-coding:utf-8-*-
import torch
from torch.nn import init


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = torch.nn.Conv2d(
            in_channels=2,  # input height
            out_channels=16,  # n_filters
            kernel_size=(1, 7),  # filter size
            stride=(1, 3),  # filter movement/step
            padding=0
        )
        self.conv1 = torch.nn.Sequential(self.c1
                                         ,
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(
                                             kernel_size=(1, 2),  # filter size
                                             stride=(1, 2),  # filter movement/step
                                         ))
        self
        init.xavier_uniform(self.c1.weight)
        init.constant(self.c1.bias, 0.1)

        self.c2 = torch.nn.Conv2d(in_channels=16,  # input height
                                  out_channels=32,  # n_filters
                                  kernel_size=(1, 5),  # filter size
                                  stride=(1, 2),  # filter movement/step
                                  padding=0)
        self.conv2 = torch.nn.Sequential(self.c2
                                         ,
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(kernel_size=(1, 2),
                                                            stride=(1, 2))
                                         )
        init.xavier_uniform(self.c2.weight)
        init.constant(self.c2.bias, 0.1)

        self.c3 = torch.nn.Conv2d(in_channels=32,  # input height
                                  out_channels=64,  # n_filters
                                  kernel_size=(1, 4),  # filter size
                                  stride=(1, 2),  # filter movement/step
                                  padding=0)
        self.conv3 = torch.nn.Sequential(self.c3
                                         ,
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(kernel_size=(1, 2),
                                                            stride=(1, 2))
                                         )
        init.xavier_uniform(self.c3.weight)
        init.constant(self.c3.bias, 0.1)

        self.li1 = torch.nn.Linear(64 * 8 * 5, 128)
        self.li2 = torch.nn.Linear(128, 13)
        self.dense = torch.nn.Sequential(self.li1
                                         ,
                                         torch.nn.ReLU(),
                                         self.li2
                                         )
        init.xavier_uniform(self.li1.weight)
        init.constant(self.li1.bias, 0.1)
        init.xavier_uniform(self.li2.weight)
        init.constant(self.li2.bias, 0.1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        # print(conv1_out.shape)
        conv2_out = self.conv2(conv1_out)
        # print(conv2_out.shape)
        conv3_out = self.conv3(conv2_out)
        # print(conv3_out.shape)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

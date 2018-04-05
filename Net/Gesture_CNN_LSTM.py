# -*-coding:utf-8-*-
import torch


class CNN_LSTM(torch.nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=2,  # input height
                out_channels=16,  # n_filters
                kernel_size=(1, 7),  # filter size
                stride=(1, 3),  # filter movement/step
                padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(1, 2),  # filter size
                stride=(1, 2),  # filter movement/step
            ))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16,  # input height
                            out_channels=32,  # n_filters
                            kernel_size=(1, 5),  # filter size
                            stride=(1, 2),  # filter movement/step
                            padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1, 2),
                               stride=(1, 2))
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,  # input height
                            out_channels=64,  # n_filters
                            kernel_size=(1, 4),  # filter size
                            stride=(1, 2),  # filter movement/step
                            padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1, 2),
                               stride=(1, 2))
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(64 * 8 * 5, 128),
            torch.nn.ReLU(),
        )

        self.rnn = torch.nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=128,
            hidden_size=13,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        fc1_res = conv3_out.view(conv3_out.size(0), -1)
        fc1_out = self.dense(fc1_res)
        r_out, (h_n, h_c) = self.rnn(fc1_out, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        # out = self.out(r_out[:, -1, :])

        return r_out

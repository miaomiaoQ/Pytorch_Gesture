# -*-coding:utf-8-*-
from datasetutils import MyDataset
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch
from torch.autograd import Variable

test_data = MyDataset(path='/home/dmrf/文档/Gesture/New_Data/持续时间为1s的复杂手势/Test_5', transform=transforms.ToTensor())
train_data = MyDataset(path='/home/dmrf/文档/Gesture/New_Data/持续时间为1s的复杂手势/Train_5', transform=transforms.ToTensor())



train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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



        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 8 * 5, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 13)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        #print(conv1_out.shape)
        conv2_out = self.conv2(conv1_out)
        #print(conv2_out.shape)
        conv3_out = self.conv3(conv2_out)
        #print(conv3_out.shape)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

model = Net()
model.cuda()
print(model)

LR=0.001
EPOCH=10

optimizer = torch.optim.Adam(model.parameters(),lr=LR)
loss_func = torch.nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    count=0
    for batch_x, batch_y in train_loader:
        print('train :'+str(count*64))
        count+=1
        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if count%5==0:
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                train_data)), train_acc / (len(train_data))))



    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    count=0
    for batch_x, batch_y in test_loader:
        print('test :' + str(count * 64))
        count+=1
        batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data[0]
        if count%5==0:
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                test_data)), eval_acc / (len(test_data))))




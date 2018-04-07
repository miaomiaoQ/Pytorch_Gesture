# -*-coding:utf-8-*-
from Net.Gesture_CNN import CNN
from Utils.Adjust_learning_rate_inv import adjust_learning_rate_inv
from Utils.DataSetUtils import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import time
import sys

sys.path.append('/home/dmrf/AboutCaffe/caffe/python')
test_data = MyDataset(path='/home/dmrf/文档/Gesture/New_Data/持续时间为1s的复杂手势/Test_5', transform=transforms.ToTensor())
train_data = MyDataset(path='/home/dmrf/文档/Gesture/New_Data/持续时间为1s的复杂手势/Train_5', transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32)

model = CNN()
model.cuda()
print(model)

cnn_base_lr=0.01
cnn_momentum= 0.9
cnn_weight_decay= 0.0005


EPOCH = 1

optimizer = torch.optim.SGD(params=model.parameters(), lr=cnn_base_lr, momentum=cnn_momentum,
                            weight_decay=cnn_weight_decay)

loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    count = 0
    befor = 0.0
    after = 0.0
    befor = time.time()
    for batch_x, batch_y in train_loader:
        after = time.time()
        print('for expend time:' + str(after - befor) + 's')
        print('train :' + str(count * 64))
        count += 1
        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()

        befor = time.time()
        out = model(batch_x)
        after = time.time()
        print('model expend time:' + str(after - befor) + 's')

        loss = loss_func(out, batch_y)
        train_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data[0]
        optimizer.zero_grad()
        loss.backward()

        new_lr = adjust_learning_rate_inv(count * 64)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        if count%5==0:
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                train_data)), train_acc / (len(train_data))))

        befor = time.time()
        print('afer model  expend time:' + str(befor - after) + 's')

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    count = 0
    for batch_x, batch_y in test_loader:
        print('test :' + str(count * 64))
        count += 1
        batch_x, batch_y = Variable(batch_x, volatile=True).cuda(), Variable(batch_y, volatile=True).cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data[0]
        if count % 5 == 0:
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                test_data)), eval_acc / (len(test_data))))

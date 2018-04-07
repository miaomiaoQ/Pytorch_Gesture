# -*-coding:utf-8-*-

base_lr = 0.001
gamma=0.01
power=0.8


def adjust_learning_rate_inv(iter):
    return base_lr * (1 + gamma * iter)**(- power)

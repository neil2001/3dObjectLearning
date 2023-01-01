from models import SimpleNN, PointNet, train, test, num_correct_preds
from preprocessing import preprocess
import random
import numpy as np

import torch
from torch import nn, optim

def testNN():
    batch_size = 256  # batch size
    test_size = 0.2 # test size
    num_epoch = 1 # number of training epochs
    learning_rate = 0.01  # learning rate

    dataloader_train, dataloader_test = preprocess('sphereConeCubeData.csv', batch_size, test_size)

    print('finished preprocessing')

    # model = SimpleNN()
    model = PointNet(num_classes=3)
    # model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) #optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    # print("PRETRAIN:", model.parameters())
    losses = train(model, dataloader_train, loss_func, optimizer, num_epoch, num_correct_preds)
    # print("POST TRAIN", model.parameters())
    loss_train = test(model, dataloader_train, loss_func, num_correct_preds)
    loss_test = test(model, dataloader_test, loss_func, num_correct_preds)

    print('Average Training Loss:', loss_train)
    print('Average Testing Loss:', loss_test)

    return loss_test

def main():
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)  

    testNN()  

if __name__ == "__main__":
    main()
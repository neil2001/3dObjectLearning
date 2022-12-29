from models import SimpleNN, train, test
from preprocessing import preprocess
import random
import numpy as np

import torch
from torch import nn, optim

def testSimpleNN():
    batch_size = 64  # batch size
    test_size = 0.2 # test size
    num_epoch = 25  # number of training epochs
    learning_rate = 0.01  # learning rate

    dataloader_train, dataloader_test = preprocess('sphereConeData.csv', batch_size, test_size)

    print('finished preprocessing')

    model = SimpleNN()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    losses = train(model, dataloader_train, loss_func, optimizer, num_epoch)
    loss_train = test(model, dataloader_train, loss_func)
    loss_test = test(model, dataloader_test, loss_func)
    print('Average Training Loss:', loss_train)
    print('Average Testing Loss:', loss_test)

    return loss_test

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)  

    testSimpleNN()  

if __name__ == "__main__":
    main()
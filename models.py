import torch
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.first_layer = nn.Conv1d(3,64,1)
        self.linear = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x is [batch size, 500, 3]
        # we need [batch size, 3, 500] for conv1d
        # print("x shape", x.shape)
        x = x.transpose(1,2).float()
        x = self.first_layer(x)
        # shape is [batch size, 64, 500]
        x, _ = torch.max(x, 2)
        # print('x shape before linear', x.shape)
        x = self.linear(x)
        # print('x shape after linear', x.shape)
        x = self.softmax(x)
        # print("post softmax", x)
        return x

class TNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.linear1 = nn.Linear(1024, 512) # fc linear layer outputting 512
        self.linear2 = nn.Linear(512, 256) # fc linear layer outputting 256
        self.linear3 = nn.Linear(256, 9) # output a 3x3 matrix!

        self.relu = nn.ReLU()

        self.bn64 = nn.BatchNorm1d(64)
        self.bn128 = nn.BatchNorm1d(128)
        self.bn1024 = nn.BatchNorm1d(1024)
        self.bn512 = nn.BatchNorm1d(512)
        self.bn256 = nn.BatchNorm1d(256)

        pass

    def forward(self, x):
        x = x.transpose(1,2).float()
        x = self.relu(self.bn64(self.conv1(x)))
        x = self.relu(self.bn128(self.conv2(x)))
        x = self.relu(self.bn1024(self.conv3(x)))

        x, _ = torch.max(x, 2)

        x = self.relu(self.bn512(self.linear1(x)))
        x = self.relu(self.bn256(self.linear2(x)))
        x = self.linear3(x)

        return x

class PointNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.tnet = TNet()

        # first multilayer perceptron 
        self.main = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
        )

        self.linear = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(1)
        pass
    
    def forward(self, x):
        # [batch size, n point, 3]
        transform = self.tnet(x)
        x = torch.bmm(x, transform)
        # [batch size, n point, 3]

        x = x.transpose(1,2)
        x = self.main(x)

        # [batch size, 1024, n_points]
        x, _ = torch.max(x, 2)

        x = self.linear(x)
        x = self.softmax(x)
        return x

def train(model, dataloader, loss_func, optimizer, num_epoch, correct_num_func=None, print_info=True):
    losses = []
    accuracy_vals = []
    model.train()

    data_size = len(dataloader.dataset)

    for epoch in range(num_epoch):
        epoch_loss_sum = 0 # total loss in this epoch
        epoch_correct_num = 0 #number of correct predictions
        for X,Y in dataloader:
            out = model.forward(X) # forward pass, get model output
            optimizer.zero_grad() # set grads to zero
            loss = loss_func(out, Y) # loss of this batch
            loss.backward() # backward pass
            optimizer.step() # update parameters
            epoch_loss_sum += loss.item() * X.shape[0] # add (loss * #samples in curret batch)

            if correct_num_func != None:
                epoch_correct_num += correct_num_func(out, Y)
        
        avg_loss = epoch_loss_sum/data_size
        avg_acc = epoch_correct_num/data_size

        losses.append(avg_loss) # append average loss of current epoch 

        if correct_num_func is not None:
            accuracy_vals.append(avg_acc) #append average accuracy of current epoch
        
        if print_info:
            print('Epoch: {} | Loss: {:.4f} '.format(epoch, avg_loss), end="")
            if correct_num_func:
                print('Accuracy: {:.4f}%'.format(avg_acc * 100), end="")            
    
    if correct_num_func is None:
        return losses
    return losses, accuracy_vals

def test(model, dataloader, loss_func, correct_num_func=None):
    loss_sum = 0
    num_correct_preds = 0
    model.eval()
    data_size = len(dataloader.dataset)
    with torch.no_grad(): #don't calculate gradients while testing
        for X,Y in dataloader:
            out = model.forward(X) # get model output
            loss = loss_func(out, Y) # calculate loss
            loss_sum += loss.item() * X.shape[0] #increase loss_sum

            if correct_num_func is not None:
                num_correct_preds += correct_num_func(out, Y)
        
    if correct_num_func is None:
        return loss_sum/data_size
    return loss_sum/data_size, num_correct_preds/data_size

def num_correct_preds(logit, target):
    pred_class_nums = torch.argmax(logit, dim=1)
    num_correct = torch.sum(torch.eq(pred_class_nums, target))
    return num_correct.item()
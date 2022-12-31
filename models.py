import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

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
    def __init__(self, dims):
        self.dim = dims
        super().__init__()
        self.conv1 = nn.Conv1d(self.dim, 64,1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.linear1 = nn.Linear(1024, 512) # fc linear layer outputting 512
        self.linear2 = nn.Linear(512, 256) # fc linear layer outputting 256
        self.linear3 = nn.Linear(256, self.dim**2) # output a 3x3 matrix!

        self.relu = nn.ReLU()

        self.bn64 = nn.BatchNorm1d(64) # nn.LayerNorm((64, 500)) #
        self.bn128 = nn.BatchNorm1d(128) # nn.LayerNorm((128, 500)) #
        self.bn1024 = nn.BatchNorm1d(1024) # nn.LayerNorm((1024, 500)) #
        self.bn512 = nn.BatchNorm1d(512) #nn.LayerNorm((512, )) #
        self.bn256 = nn.BatchNorm1d(256) #nn.LayerNorm((256, )) #

        pass

    def forward(self, x):
        batch_size = x.size()[0]

        x = x.transpose(1,2).float()

        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        x = F.relu(self.bn64(self.conv1(x)))
       
        x = F.relu(self.bn128(self.conv2(x))) 
        
        x = F.relu(self.bn1024(self.conv3(x)))
        
        x, _ = torch.max(x, 2)

        # x = self.relu(self.linear1(x))
        # x = self.relu(self.linear2(x))
        x = F.relu(self.bn512(self.linear1(x)))
        x = F.relu(self.bn256(self.linear2(x)))
        x = self.linear3(x)
        # print("tnet output size", x.shape)
        nxnID = torch.eye(self.dim).flatten()
        identityMat = nxnID.repeat(batch_size, 1)
        x += identityMat
        x = x.view(-1, self.dim, self.dim)
        return x

class PointNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.tnet3 = TNet(3)
        self.tnet64 = TNet(64)

        self.mlp1 = nn.Sequential(
            nn.Conv1d(3,64,1),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64,64,1),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # self.conv1 = nn.Conv1d(3,64,1)
        # self.conv2 = nn.Conv1d(64,64,1)

        # first multilayer perceptron 
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64,64,1),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64,128,1),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128,1024,1),
            # nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, num_classes)
        )

        self.softmax = nn.Softmax(1)
        pass
    
    def forward(self, x):
        # [batch size, n point, 3]
        transform = self.tnet3(x).float()
        x = x.float()
        x = torch.bmm(x, transform)
        # [batch size, n point, 3]
        x = x.transpose(1,2)
        x = self.mlp1(x)

        x = x.transpose(1,2)
        transform64 = self.tnet64(x).float()
        x = torch.bmm(x,transform64)
        x = x.transpose(1,2)

        x = self.mlp2(x)
        # [batch size, 1024, n_points]
        x, _ = torch.max(x, 2)
        x = self.mlp3(x)
        x = self.softmax(x)
        return x

# class STN3d(nn.Module):
#     def __init__(self):
#         super(STN3d, self).__init__()
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 9)
#         self.relu = nn.ReLU()

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)


#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)

#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)

#         iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, 3, 3)
#         return x


# class STNkd(nn.Module):
#     def __init__(self, k=64):
#         super(STNkd, self).__init__()
#         self.conv1 = torch.nn.Conv1d(k, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k*k)
#         self.relu = nn.ReLU()

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)

#         self.k = k

#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)

#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)

#         iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x

# class PointNetfeat(nn.Module):
#     def __init__(self, global_feat = True, feature_transform = False):
#         super(PointNetfeat, self).__init__()
#         self.stn = STN3d()
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.global_feat = global_feat
#         self.feature_transform = feature_transform
#         if self.feature_transform:
#             self.fstn = STNkd(k=64)

#     def forward(self, x):
#         n_pts = x.size()[2]
#         trans = self.stn(x)
#         x = x.transpose(2, 1)
#         x = torch.bmm(x, trans)
#         x = x.transpose(2, 1)
#         x = F.relu(self.bn1(self.conv1(x)))

#         if self.feature_transform:
#             trans_feat = self.fstn(x)
#             x = x.transpose(2,1)
#             x = torch.bmm(x, trans_feat)
#             x = x.transpose(2,1)
#         else:
#             trans_feat = None

#         pointfeat = x
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#         if self.global_feat:
#             return x, trans, trans_feat
#         else:
#             x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
#             return torch.cat([x, pointfeat], 1), trans, trans_feat

# class PointNet(nn.Module):
#     def __init__(self, k=2, feature_transform=False):
#         super(PointNet, self).__init__()
#         self.feature_transform = feature_transform
#         self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k)
#         self.dropout = nn.Dropout(p=0.3)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.dropout(self.fc2(x))))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1), trans, trans_feat

def train(model, dataloader, loss_func, optimizer, num_epoch, correct_num_func=None, print_info=True):
    losses = []
    accuracy_vals = []

    data_size = len(dataloader.dataset)
    model.train()
    for epoch in range(num_epoch):
        epoch_loss_sum = 0 # total loss in this epoch
        epoch_correct_num = 0 #number of correct predictions

        for X,Y in dataloader:
            # print("INPUT", X)
            # print(X.transpose(2,1).double())
            out = model.forward(X) # forward pass, get model output
            optimizer.zero_grad() # set grads to zero
            # print("OUT and TARG", out, Y.flatten().long())
            loss = loss_func(out, Y.flatten().long()) # loss of this batch
            # print("LOSS", loss)
            loss.backward() # backward pass
            optimizer.step() # update parameters
            epoch_loss_sum += loss.item() * X.shape[0] # add (loss * #samples in curret batch)

            if correct_num_func != None:
                num_correct = correct_num_func(out, Y)
                epoch_correct_num += num_correct
                print(loss.item(), num_correct)

        # print("epoch loss sum", epoch_loss_sum, data_size)
        
        avg_loss = epoch_loss_sum/data_size
        avg_acc = epoch_correct_num/data_size

        # print("pred vals", epoch_correct_num, data_size)

        losses.append(avg_loss) # append average loss of current epoch 

        if correct_num_func is not None:
            accuracy_vals.append(avg_acc) #append average accuracy of current epoch
        
        if print_info:
            print('Epoch: {} | Loss: {:.4f} '.format(epoch, avg_loss), end="")
            if correct_num_func:
                print('Accuracy: {:.4f}%'.format(avg_acc * 100), end="") 
            print()

    ### 

    # testCount = 0
    # for X,Y in dataloader:
    #     if testCount > 5:
    #         break
    #     output = model.forward(X)
    #     target = Y.flatten().long()
    #     loss = loss_func(output, target)
    #     correct_preds = correct_num_func(output, Y)
    #     print("POST TESTING PREDICTION", loss.item(), correct_preds/X.shape[0])
    #     testCount += 1

    # testLoss, testAcc = test(model, dataloader, loss_func, correct_num_func)
    # print(testLoss, testAcc)

    ### 
    
    if correct_num_func is None:
        return losses
    return losses, accuracy_vals

def test(model, dataloader, loss_func, correct_num_func=None):
    loss_sum = 0
    num_correct_preds = 0
    for m in model.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm1d:
                child.track_running_stats = False
                child.running_mean = None
                child.running_var = None
    model.eval()
    data_size = len(dataloader.dataset)
    with torch.no_grad(): #don't calculate gradients while testing
        for X,Y in dataloader:
            out = model.forward(X) # get model output
            target = Y.flatten().long()
            # print("test predictions", out, target)

            loss = loss_func(out, target) # calculate loss
            loss_sum += loss.item() * X.shape[0] #increase loss_sum

            if correct_num_func is not None:
                num_correct_preds += correct_num_func(out, Y, True)
    
    if correct_num_func is None:
        return loss_sum/data_size
    return loss_sum/data_size, num_correct_preds/data_size

def num_correct_preds(logit, target, print_info = False):
    pred_class_nums = torch.argmax(logit, dim=1)
    if print_info:
        print("in counting function", torch.sum(pred_class_nums))
    num_correct = torch.sum(torch.eq(pred_class_nums, target.flatten()))
    # print(num_correct.item())
    return num_correct.item()
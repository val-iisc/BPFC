#torch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data.sampler import SubsetRandomSampler

# torch dependencies for data load 
import torchvision
from torchvision import datasets, transforms

# numpy
import numpy as np
# time
import time
import math

###################################### Parse Inputs ###############################################
import getopt
import sys
#READ ARGUMENTS
opts = sys.argv[1::2]
args = sys.argv[2::2]
for  i in range(len(opts)):
    opt = opts[i]
    arg = args[i]
    #Experiment name
    if opt=='-EXP_NAME':
        EXP_NAME = str(arg)
        LOG_FILE_NAME = 'log/'+str(arg)+'.txt'
        print 'EXP_NAME:',EXP_NAME
    if opt=='-MAX_EPOCHS':
        MAX_EPOCHS = int(arg)
        print 'MAX_EPOCHS:',MAX_EPOCHS
    if opt=='-l_ce':
        l_ce = float(arg)
        print 'l_ce:',l_ce
    if opt=='-p_val':
        p_val = int(arg)
        print 'p_val:',p_val
    if opt=='-l_reg':
        l_reg = float(arg)
        print 'l_reg:',l_reg
    if opt=='-mul_ru':
        mul_ru = float(arg)
        print 'mul_ru:',mul_ru

import os
if not os.path.isdir('./results'):
    os.mkdir('./results')
if not os.path.isdir('./log'):
    os.mkdir('./log')
if not os.path.isdir('./models'):
    os.mkdir('./models')
if not os.path.isdir('./data'):
    os.mkdir('./data')

def l2_loss(x,y):
    diff = x-y
    diff = diff*diff
    diff = diff.sum(1)
    diff = diff.mean(0)
    return diff

###################################### Cudnn ######################################################
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True 
print 'Cudnn status:',torch.backends.cudnn.enabled

##################################### Set tensor to CUDA ##########################################
torch.set_default_tensor_type('torch.cuda.FloatTensor')

###################################### Parameters #################################################
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE   = 1024
TEST_BATCH_SIZE  = 1024
BASE_LR          = 1e-2
MODEL_PREFIX     = 'models/mnist_'+EXP_NAME+'_epoch_'

epochs    = MAX_EPOCHS
iteration = 0
loss      = nn.CrossEntropyLoss()
base_lr   = BASE_LR

##################################### Load Network ################################################
execfile('MNIST_Network.py')
model = MNIST_Network()
model.cuda()
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,momentum=0.9,weight_decay=5e-4)
optimizer.zero_grad()

##################################### Load Data ###################################################
transform = transforms.Compose([
        transforms.ToTensor(),])

train_set  = torchvision.datasets.MNIST(root='./data', train=True , download=True, transform=transform)
val_set    = torchvision.datasets.MNIST(root='./data', train=True , download=True, transform=transform)
test_set   = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split training into train and validation
train_size = 50000
valid_size = 10000
test_size  = 10000

#get indices seed
np.random.seed(0)
indices    = np.arange(train_size+valid_size)
np.random.shuffle(indices)
train_indices = indices[0:train_size]
val_indices   = indices[train_size:]

#get data loader ofr train val and test
train_loader = torch.utils.data.DataLoader(train_set,batch_size=TRAIN_BATCH_SIZE ,sampler=SubsetRandomSampler(train_indices))
val_loader   = torch.utils.data.DataLoader(val_set,sampler = SubsetRandomSampler(val_indices),batch_size=VAL_BATCH_SIZE)
test_loader  = torch.utils.data.DataLoader(test_set,batch_size=TEST_BATCH_SIZE)
print('MNIST dataloader: Done')

##################################### Train #######################################################
qnoise_scale = math.pow(2,p_val-2)
for epoch in range(epochs):
    start = time.time()
    iter_loss =0 
    counter =0 
    for data, target in train_loader:
        B,C,H,W = data.size()
        pow_p = torch.ones(B,C,H,W).cuda()*math.pow(2,p_val)
        model.train()
        data   = Variable(data).cuda()
        target = Variable(target).cuda()

        qdata = torch.round(data*255)
        qnoise = torch.Tensor(B,C,H,W).uniform_(-1,1).cuda()*qnoise_scale
        qdata = qdata + qnoise
        qdata = qdata - (qdata % pow_p) + (pow_p/2)
        qdata = torch.clamp(qdata,0,255)
        qdata = qdata/255
 
        optimizer.zero_grad()
        qout  = model(qdata)
        out  = model(data)
        reg = l2_loss(out,qout)
        closs = loss(out,target)
        cost = l_ce*closs + l_reg*reg
        cost.backward()
        optimizer.step()
        
        if iteration%100==0:
            msg = 'iter,'+str(iteration)+',clean loss,'+str(closs.data.cpu().numpy()) \
                                        +',reg loss,'+str(reg.data.cpu().numpy()) \
                                        +',total loss'+str(cost.data.cpu().numpy()) \
                                        +'\n'
            log_file = open(LOG_FILE_NAME,'a')
            log_file.write(msg)
            log_file.close()
            model.train()
        iteration = iteration + 1
        #console log
        counter = counter + 1
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] : Loss:%f \t\t'
                %(epoch, MAX_EPOCHS, counter,
                    (train_size/TRAIN_BATCH_SIZE),cost.data.cpu().numpy()))
    end = time.time()
    print 'Epoch:',epoch,' Time taken:',(end-start)
    
    model_name = MODEL_PREFIX+str(epoch)+'.pkl'
    torch.save(model.state_dict(),model_name)
    
    if epoch ==3*epochs/4:
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr/125
    elif epoch ==2*epochs/4: 
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr/25
    elif epoch ==epochs/4:
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr/5

##################################### Attack ######################################################
model.eval()
def FGSM_Attack_step(model,loss,image,target,eps,steps,bounds=[0,1]):
    tar = Variable(target.cuda())
    img = image.cuda()
    eps = eps/steps
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model(img)
        cost = loss(out,tar)
        cost.backward()
        per = eps * torch.sign(img.grad.data)
        adv = img.data + per.cuda() 
        img = torch.clamp(adv,bounds[0],bounds[1])
    return img

def MSPGD(model,loss,data,target,eps,eps_iter,bounds,steps):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = torch.clamp(noise,-eps,eps)
    img_arr = []
    for step in range(steps[-1]):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass
        out  = model(img)
        #compute loss using true label
        cost = loss(out,tar)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        #convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0,1] - bounds[0,0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1,1] - bounds[1,0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2,1] - bounds[2,0])) * per[:,2,:,:]
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0,0],bounds[0,1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1,0],bounds[1,1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2,0],bounds[2,1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)
        for j in range(len(steps)):
            if step == steps[j]-1:
                img_tmp = data + noise
                img_arr.append(img_tmp)
                break
    return img_arr

##################################### Validation ##################################################
EVAL_LOG_NAME = 'results/'+EXP_NAME+'.txt'
ACC_EPOCH_LOG_NAME = 'results/acc_'+EXP_NAME+'_epoch.txt'
ACC_IFGSM_EPOCH_LOG_NAME = 'results/ifgsm_acc_'+EXP_NAME+'_epoch.txt'
log_file = open(EVAL_LOG_NAME,'a')
msg = '##################### iter.FGSM: steps=40,eps=0.3 ####################\n'
log_file.write(msg)
log_file.close()
accuracy_log = np.zeros(MAX_EPOCHS)
for epoch in range(max(0,MAX_EPOCHS-30),MAX_EPOCHS):
    model_name = MODEL_PREFIX+str(epoch)+'.pkl'
    model.load_state_dict(torch.load(model_name))
    eps=0.3
    accuracy = 0
    accuracy_ifgsm = 0
    count = 0
    for data, target in val_loader:
        data   = Variable(data).cuda()
        target = Variable(target).cuda()
        out = model(data)
        prediction = out.data.max(1)[1] 
        accuracy = accuracy + prediction.eq(target.data).sum().item()
        count += target.size()[0] 
    for data, target in val_loader:
        data = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=40)
        data   = Variable(data).cuda()
        target = Variable(target).cuda()
        out = model(data)
        prediction = out.data.max(1)[1] 
        accuracy_ifgsm = accuracy_ifgsm + prediction.eq(target.data).sum().item()
    acc = (accuracy*100.0) / float(count)
    acc_ifgsm = (accuracy_ifgsm*100.0) / float(count)
    #log accuracy to file
    msg= str(epoch)+','+str(acc)+'\n'
    log_file = open(ACC_EPOCH_LOG_NAME,'a')
    log_file.write(msg)
    log_file.close()
    
    msg1= str(epoch)+','+str(acc_ifgsm)+'\n'
    log_file = open(ACC_IFGSM_EPOCH_LOG_NAME,'a')
    log_file.write(msg1)
    log_file.close()

    accuracy_log[epoch] = acc_ifgsm
    sys.stdout.write('\r')
    sys.stdout.write('| Epoch [%3d/%3d] : Acc:%f \t\t'
            %(epoch, MAX_EPOCHS,acc))
    sys.stdout.flush() 

log_file = open(EVAL_LOG_NAME,'a')
msg = 'Epoch,'+str(accuracy_log.argmax())+',Acc,'+str(accuracy_log.max())+'\n'
log_file.write(msg)
log_file.close()

model_name = MODEL_PREFIX+str(accuracy_log.argmax())+'.pkl'
model.load_state_dict(torch.load(model_name))
model.eval()

##################################### FGSM #############################################
EVAL_LOG_NAME = 'results/'+EXP_NAME+'.txt'
log_file = open(EVAL_LOG_NAME,'a')
msg = '##################### FGSM ####################\n'
log_file.write(msg)
log_file.close()
for eps in np.arange(0,0.301,0.05):
    accuracy = 0
    for data, target in test_loader:
        adv = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=1)
        data   = Variable(adv).cuda()
        target = Variable(target).cuda()
        out = model(data)
        prediction = out.data.max(1)[1] 
        accuracy = accuracy + prediction.eq(target.data).sum().item()
    acc = (accuracy*100.0) / float(test_size)
    log_file = open(EVAL_LOG_NAME,'a')
    msg = 'eps,'+str(eps)+',Acc,'+str(acc)+'\n'
    log_file.write(msg)
    log_file.close()

##################################### iFGSM #############################################
log_file = open(EVAL_LOG_NAME,'a')
msg = '##################### iFGSM: step=40 ####################\n'
log_file.write(msg)
log_file.close()
for eps in np.arange(0.05,0.301,0.05):
    accuracy = 0
    for data, target in test_loader:
        adv = FGSM_Attack_step(model,loss,data,target,eps=eps,steps=40)
        data   = Variable(adv).cuda()
        target = Variable(target).cuda()
        out = model(data)
        prediction = out.data.max(1)[1] 
        accuracy = accuracy + prediction.eq(target.data).sum().item()
    acc = (accuracy*100.0) / float(test_size)
    log_file = open(EVAL_LOG_NAME,'a')
    msg = 'eps,'+str(eps)+',Acc,'+str(acc)+'\n'
    log_file.write(msg)
    log_file.close()

##################################### PGD, steps=[20,40,100,500,1000] #############################################
log_file = open(EVAL_LOG_NAME,'a')
msg = '############## PGD: steps=[20,40,100,500,1000], eps=0.3, eps_iter=0.01 ###############\n'
log_file.write(msg)
log_file.close()
all_steps = [20,40,100,500,1000] 
num_steps = len(all_steps)
eps = 0.3
acc_arr = np.zeros((num_steps))
for data, target in test_loader:
    adv_arr = MSPGD(model,loss,data,target,eps=eps,eps_iter=0.01,bounds=np.array([[0,1],[0,1],[0,1]]),steps=all_steps)     
    target = Variable(target).cuda()
    for j in range(num_steps):
        data   = Variable(adv_arr[j]).cuda()
        out = model(data)
        prediction = out.data.max(1)[1] 
        acc_arr[j] = acc_arr[j] + prediction.eq(target.data).sum().item()
for j in range(num_steps):
    acc_arr[j] = (acc_arr[j]*100.0) / float(test_size)
    log_file = open(EVAL_LOG_NAME,'a')
    msg = 'eps,'+str(eps)+',steps,'+str(all_steps[j])+',Acc,'+str(acc_arr[j])+'\n'
    log_file.write(msg)
    log_file.close()

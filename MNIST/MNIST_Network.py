# conv1-conv11-pool-conv2-conv21

class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()
        self.conv1    = nn.Conv2d(1,32,kernel_size=5,dilation=1, stride=1, padding=2,bias=True)
        self.conv11   = nn.Conv2d(32,32,kernel_size=5,dilation=1, stride=1, padding=2,bias=True) 
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32,64,kernel_size=5,dilation=1, stride=1, padding=2,bias=True)
        self.conv21 = nn.Conv2d(64,64,kernel_size=5,dilation=1, stride=1, padding=2,bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1   = nn.Linear(64*7*7,512,bias=True )
        self.fc2   = nn.Linear(512, 10)
    def forward(self, input):
        out = F.relu((self.conv1(input)))
        out = F.relu((self.conv11(out)))
        out = self.pool1(out)
        
        out = F.relu((self.conv2(out)))
        out = F.relu((self.conv21(out)))
        out = self.pool2(out)
        
        # fc-1
        B,C,H,W = out.size()
        out = out.view(B,-1) 
        out =(F.relu((self.fc1(out))))
        # Logits
        out = self.fc2(out)
        return out 
    
    
class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        """
        conv(64,5,5)+RELU
        conv(64,5,5)+RELU
        Dropout(0.25)
        FC(128)+Relu
        Dropout(0.5)
        FC+Softmax
        """
        self.conv1 = nn.Conv2d(1,64,kernel_size=5,dilation=1, stride=1, padding=0,bias=True)
        self.conv2 = nn.Conv2d(64,64,kernel_size=5,dilation=1, stride=1, padding=0,bias=True)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1      = nn.Linear(64*20*20,128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2      = nn.Linear(128,10)
    def forward(self, input):
        out = F.relu(self.conv1(input))
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0),-1)
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)
        return out
    
class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv2d(1,64,kernel_size=8)
        self.conv2 = nn.Conv2d(64,128,kernel_size=6)
        self.conv3 = nn.Conv2d(128,128,kernel_size=5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1  = nn.Linear(128*12*12,10)
    def forward(self, input):
        B,C,H,W = input.size()
        input = input.view(B,-1) 
        out = self.dropout1(input)
        out = out.view(B,C,H,W)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0),-1)
        out = self.dropout2(out)
        out = self.fc1(out)
        return out

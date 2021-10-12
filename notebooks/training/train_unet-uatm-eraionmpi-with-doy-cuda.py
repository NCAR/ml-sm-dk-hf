from IPython import display

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable


#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# piqa
#from piqa import ssim
from piqa import SSIM
from piqa import MS_GMSD
#from piqa import 
#class SSIMLoss(ssim.SSIM):
#    def forward(self, x, y):
#        return 1. - super().forward(x, y)
#ssim_loss = SSIMLoss().cuda()

ssim_loss = SSIM().cuda()
gmsd_loss = MS_GMSD().cuda()



import cv2
#from skimage.metrics import structural_similarity as ssim
import copy



#############
# Load Data #
#############
# define internal pic res
ires = 16
res = 128

##############
# Chageables #
##############
#ndays = 365
#ndays = 13880    # 1979-01-01 to 2016-12-31
#ndays = 12784    # 1979-01-01 to 2013-12-31

## 2010-2016
#istart = 11323
#ndays = 2557
# 1979-2013
#istart = 0
#ndays = 12784

## 1980-1989
#istart = 365
#ndays = 3653

# 1979-2010
istart = 0
ndays = 11688     # 1979-2010

## 1979-1989
#istart = 0
#ndays = 4018      # 1979-2089

## 1990-1999
#istart = 4018
#ndays = 3652

nvars = 8
mb = 4      # minibatch size

# load files
d = np.load('/glade/work/dkorytin/srgan_data/prec128_gridmetB_1979-2016.npy')[istart:istart+ndays]
dv2 = np.load('/glade/work/dkorytin/srgan_data/tmax128_gridmetB_1979-2016.npy')[istart:istart+ndays]
dv3 = np.load('/glade/work/dkorytin/srgan_data/tmin128_gridmetB_1979-2016.npy')[istart:istart+ndays]
dv4 = np.load('/glade/work/dkorytin/srgan_data/uas128_gridmetB_1979-2016.npy')[istart:istart+ndays]
dv5 = np.load('/glade/work/dkorytin/srgan_data/vas128_gridmetB_1979-2016.npy')[istart:istart+ndays]
dv6 = np.load('/glade/work/dkorytin/srgan_data/huss128_gridmetB_1979-2016.npy')[istart:istart+ndays]
dv7 = np.load('/glade/work/dkorytin/srgan_data/rsds128_gridmetB_1979-2016.npy')[istart:istart+ndays]
dv8 = np.load('/glade/work/dkorytin/srgan_data/miss128_gridmetB_1979-2016.npy')[istart:istart+ndays]

# Find min/max for each variable
v1max = d.max(); v1min = d.min()
v2max = dv2.max(); v2min = dv2.min()
v3max = dv3.max(); v3min = dv3.min()
v4max = dv4.max(); v4min = dv4.min()
v5max = dv5.max(); v5min = dv5.min()
v6max = dv6.max(); v6min = dv6.min()
v7max = dv7.max(); v7min = dv7.min()
v8max = dv8.max(); v8min = dv8.min()
minmaxo = [[v1min,v1max], [v2min,v2max], [v3min,v3max], [v4min,v4max], [v5min,v5max], [v6min,v6max], [v7min,v7max], [v8min,v8max]] # saved w/ model

print("v1 max/min", v1max, v1min)
print("v2 max/min", v2max, v2min)
print("v3 max/min", v3max, v3min)
print("v4 max/min", v4max, v4min)
print("v5 max/min", v5max, v5min)
print("v6 max/min", v6max, v6min)
print("v7 max/min", v7max, v7min)
print("v8 max/min", v8max, v8min)
print("minmaxo", minmaxo)

# run 4th root normalization on prec
d = d**0.25
v1max = d.max(); v1min = d.min()
minmaxo = [[v1min,v1max], [v2min,v2max], [v3min,v3max], [v4min,v4max], [v5min,v5max], [v6min,v6max], [v7min,v7max], [v8min,v8max]] # saved w/ model


# normalize each var to 0-1
d = (d-v1min)/(v1max-v1min)
dv2 = (dv2-v2min)/(v2max-v2min)
dv3 = (dv3-v3min)/(v3max-v3min)
dv4 = (dv4-v4min)/(v4max-v4min)
dv5 = (dv5-v5min)/(v5max-v5min)
dv6 = (dv6-v6min)/(v6max-v6min)
dv7 = (dv7-v7min)/(v7max-v7min)
dv8 = (dv8-v8min)/(v8max-v8min)

# mask & mask_bias
mask = 1-torch.tensor(dv8[0:1,:,:]).cuda().view(1,1,res,res)
invmask = torch.tensor(dv8[0:1,:,:]).cuda().view(1,1,res,res)
b1 = invmask*(0-v1min)/(v1max-v1min)
b2 = invmask*(0-v2min)/(v2max-v2min)
b3 = invmask*(0-v3min)/(v3max-v3min)
b4 = invmask*(0-v4min)/(v4max-v4min)
b5 = invmask*(0-v5min)/(v5max-v5min)
b6 = invmask*(0-v6min)/(v6max-v6min)
b7 = invmask*(0-v7min)/(v7max-v7min)
b8 = mask*(0-v8min)/(v8max-v8min)
mask_bias = torch.cat((b1,b2,b3,b4,b5,b6,b7,b8),1).detach()



###########################
## LOAD ERAI data: daily ##
###########################

# load erai
#mndays = 11688      # 1979-2010
#mndays = 4018      # 1979-2089

## 1990-1999
#mnstart = 4018
#mndays = 3652       

# 1979-2010
mnstart = 0
mndays = 11688 

#mnvars = 8*4
mnvars = 8*1
md = np.load('/glade/scratch/dkorytin/erai-on-mpigrid/U850.ERAI.MPIGRID.1979-2018.npy')[mnstart:mnstart+mndays*1]
mdv2 = np.load('/glade/scratch/dkorytin/erai-on-mpigrid/V850.ERAI.MPIGRID.1979-2018.npy')[mnstart:mnstart+mndays*1]
mdv3 = np.load('/glade/scratch/dkorytin/erai-on-mpigrid/Q850.ERAI.MPIGRID.1979-2018.npy')[mnstart:mnstart+mndays*1]
mdv4 = np.load('/glade/scratch/dkorytin/erai-on-mpigrid/T700.ERAI.MPIGRID.1979-2018.npy')[mnstart:mnstart+mndays*1]
mdv5 = np.load('/glade/scratch/dkorytin/erai-on-mpigrid/Z700.ERAI.MPIGRID.1979-2018.npy')[mnstart:mnstart+mndays*1]
mdv6 = np.load('/glade/scratch/dkorytin/erai-on-mpigrid/Z500.ERAI.MPIGRID.1979-2018.npy')[mnstart:mnstart+mndays*1]
mdv7 = np.load('/glade/scratch/dkorytin/erai-on-mpigrid/U250.ERAI.MPIGRID.1979-2018.npy')[mnstart:mnstart+mndays*1]
mdv8 = np.load('/glade/scratch/dkorytin/erai-on-mpigrid/V250.ERAI.MPIGRID.1979-2018.npy')[mnstart:mnstart+mndays*1]


# Find min/max for each variable
mv1max = md.max(); mv1min = md.min()
mv2max = mdv2.max(); mv2min = mdv2.min()
mv3max = mdv3.max(); mv3min = mdv3.min()
mv4max = mdv4.max(); mv4min = mdv4.min()
mv5max = mdv5.max(); mv5min = mdv5.min()
mv6max = mdv6.max(); mv6min = mdv6.min()
mv7max = mdv7.max(); mv7min = mdv7.min()
mv8max = mdv8.max(); mv8min = mdv8.min()
print("mv1 max/min", mv1max, mv1min)
print("mv2 max/min", mv2max, mv2min)
print("mv3 max/min", mv3max, mv3min)
print("mv4 max/min", mv4max, mv4min)
print("mv5 max/min", mv5max, mv5min)
print("mv6 max/min", mv6max, mv6min)
print("mv7 max/min", mv7max, mv7min)
print("mv8 max/min", mv8max, mv8min)

minmaxi = [[mv1min,mv1max],[mv2min,mv2max],[mv3min,mv3max],[mv4min,mv4max],[mv5min,mv5max],[mv6min,mv6max],[mv7min,mv7max],[mv8min,mv8max]] # saved w/ model


# normalize each var to 0-1
md = (md-mv1min)/(mv1max-mv1min)
mdv2 = (mdv2-mv2min)/(mv2max-mv2min)
mdv3 = (mdv3-mv3min)/(mv3max-mv3min)
mdv4 = (mdv4-mv4min)/(mv4max-mv4min)
mdv5 = (mdv5-mv5min)/(mv5max-mv5min)
mdv6 = (mdv6-mv6min)/(mv6max-mv6min)
mdv7 = (mdv7-mv7min)/(mv7max-mv7min)
mdv8 = (mdv8-mv8min)/(mv8max-mv8min)

mndays = len(mdv8)    #//4
print("mndays", mndays)

# synthetically create 16x16 input samples via bilinear scaling
dd=[]
ddo=[]
for ii in range(mndays):
    # input samples
    rowdd = []
    for iii in [0]: #range(4):
    #for iii in range(4):
       rowdd.append( cv2.resize(md[ii*1+iii],(16,16)) )
       rowdd.append( cv2.resize(mdv2[ii*1+iii],(16,16)) )
       rowdd.append( cv2.resize(mdv3[ii*1+iii],(16,16)) )
       rowdd.append( cv2.resize(mdv4[ii*1+iii],(16,16)) )
       rowdd.append( cv2.resize(mdv5[ii*1+iii],(16,16)) )
       rowdd.append( cv2.resize(mdv6[ii*1+iii],(16,16)) )
       rowdd.append( cv2.resize(mdv7[ii*1+iii],(16,16)) )
       rowdd.append( cv2.resize(mdv8[ii*1+iii],(16,16)) )
       # add doy
       rowdd.append( np.ones((16,16),dtype=np.float32)*np.cos(ii/365.25*3.14159*2.)/2. +0.5)
       #rowdd.append( np.ones((16,16),dtype=np.float32)*np.cos(ii/365.25*3.14159*2*2)/2. +0.5)

       dd.append(rowdd)

    # output samples
    rowddo = []
    rowddo.append( d[ii] )
    rowddo.append( dv2[ii] )
    rowddo.append( dv3[ii] )
    rowddo.append( dv4[ii] )
    rowddo.append( dv5[ii] )
    rowddo.append( dv6[ii] )
    rowddo.append( dv7[ii] )
    rowddo.append( dv8[ii] )
    ddo.append(rowddo)

# add channel for cos
mnvars = 8+1

d = np.array(dd)
do = np.array(ddo)
print("d, do, md",d.shape, do.shape, np.array(md).shape)

# input samples
samples = []
print("ndays, mndays", ndays, mndays)
for t in range(ndays):
    samples.append(d[t])
samples = torch.tensor(np.reshape(samples, (ndays, mnvars, ires, ires)))

# output samples
osamples = []    # 20x20 samples
for t in range(ndays):
    osamples.append(do[t])
osamples = torch.tensor(np.reshape(osamples, (ndays, nvars, res, res)))
print("osamples max min",osamples.max(), osamples.min())






################
## MODELS     ##
################

gaussian_filter = torch.zeros(nvars, nvars, 3, 3).cuda()
for ii in range(nvars):
    #gaussian_filter[ii,ii,:,:]=torch.ones(3,3).cuda()/9.
    gaussian_filter[ii,ii,:,:]=torch.tensor([[1,2,1],[2,4,2],[1,2,1]]).cuda()/16.

class GenerativeNet2(torch.nn.Module):
    
    def __init__(self):
        super(GenerativeNet2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(nvars),
            nn.Conv2d(
                in_channels=nvars, out_channels=128, kernel_size=4, 
                stride=2, padding=1, bias=True
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=True
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=True
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #self.conv4 = nn.Sequential(
        #    nn.Conv2d(
        #        in_channels=512, out_channels=1024, kernel_size=4,
        #        stride=2, padding=1, bias=True
        #    ),
        #    nn.BatchNorm2d(1024),
        #    nn.LeakyReLU(0.2, inplace=True)
        #)        
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=nvars, kernel_size=3,
                stride=1, padding=1, bias=True
            ),
            #nn.BatchNorm2d(nvars),
            nn.LeakyReLU(0.2, inplace=True)
        )        
#        self.out = nn.Sequential(
#            nn.Linear(2048*4*4, 1),
#            nn.Sigmoid(),
#        )
#
#        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)
        x = self.conv5(x)
        ## Flatten and apply sigmoid
        #x = x.view(-1, 2048*4*4)
        #x = self.out(x)
        return x
        
class GenerativeNet(torch.nn.Module):
    
    def __init__(self):
        super(GenerativeNet, self).__init__()
        
        #self.linear = torch.nn.Linear(20*20*1*1, 20*20*1*1)
        
#         self.conv1 = nn.Sequential(
#             nn.BatchNorm2d(nvars),
#             nn.Conv2d(
#                 in_channels=nvars, out_channels=64, kernel_size=4, 
#                 stride=2, padding=1, bias=True
#             ),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        self.conv11 = nn.Sequential(
            #nn.BatchNorm2d(mnvars),
            nn.Conv2d(
                in_channels=mnvars, out_channels=1024, kernel_size=4, 
                stride=2, padding=1, bias=True
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, out_channels=2048, kernel_size=4,
                stride=2, padding=1, bias=True
            ),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True)
        )
        ## smoothing 3x3 stride 1 conv
        #self.conv333 = nn.Sequential(
        #    nn.Conv2d(
        #        in_channels=2048, out_channels=2048, kernel_size=3,
        #        stride=1, padding=1, bias=True
        #    ),
        #    nn.BatchNorm2d(2048),
        #    nn.LeakyReLU(0.2, inplace=True)
        #)

        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2048, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=True
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024+1024*1, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=True
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512+mnvars*1, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=True
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=True
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=nvars, kernel_size=4,
                stride=2, padding=1, bias=True
            ),
            #nn.BatchNorm2d(nvars),
            #nn.LeakyReLU(0.2, inplace=True)
            nn.Sigmoid() # constrain output to 0-1 b/c SSIM does not like them
        )

        # smoothing 3x3 stride 1 conv
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=nvars, out_channels=nvars, kernel_size=3,
                stride=1, padding=1, bias=True
            ),
            #nn.BatchNorm2d(nvars),
            #nn.LeakyReLU(0.2, inplace=True)
            nn.Sigmoid() # constrain output to 0-1 b/c SSIM does not like them
        )

        
#         self.conv7 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=128, out_channels=64, kernel_size=4,
#                 stride=2, padding=1, bias=False
#             ),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.conv8 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=64, out_channels=1, kernel_size=4,
#                 stride=2, padding=1, bias=False
#             ),
#             nn.BatchNorm2d(1),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#
#        self.lr = nn.Sequential(
#            nn.Linear(64*res*res, nvars*res*res)
#        )
#
#        self.out = torch.nn.Tanh()
#        #self.out = torch.nn.Sigmoid()  # allows - values which breaks bce    

    def forward(self, x):
        # Project and reshape
        #x = self.linear(x)
        x1 = x.view(x.shape[0], mnvars, ires, ires)

        #m = nn.Dropout(p=0.1)
        #x1 = m(x1)
          
        # Convolutional layers
        x2 = self.conv11(x1)
        x = self.conv22(x2)
        #x = self.conv333(x)
                
        x = self.conv1(x)

        x = self.conv2(torch.cat((x, x2), dim=1))
        x = self.conv3(torch.cat((x, x1), dim=1))
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.conv6(x)

        # gaussian blur
        #print("EWQQEWEQWQEW", x.shape)
        #conv_filter = torch.ones(nvars, nvars, 3, 3).cuda()/9.
        #print("EWQQEWEQWQEW2", conv_filter.shape)
        #x = torch.nn.functional.conv2d(x, gaussian_filter.detach(), stride=1, padding=1)
        #print("EWQQEWEQWQEW3", x.shape)

#        # Flatten and apply regressor
#        x = x.view(-1, 64*res*res)
#        x = self.lr(x)
#        # reshape
#        x = x.view(-1, nvars, res, res)
#
#         m = nn.Dropout(p=0.1)
#         x = m(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.conv7(x)
#         x = self.conv8(x)
        # Apply Tanh
        #return self.out(x)
        return x

 
# Noise
def noise3d(size, nvars, res):
    n = Variable(torch.randn(size, nvars, res*res))
    if torch.cuda.is_available(): return n.cuda()
    return n

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

# Create Network instances and init weights
print("Create Network instances and init weights")
generator = GenerativeNet()
#generator.half()
generator.apply(init_weights)

generator2 = GenerativeNet2()
generator2.apply(init_weights)


# Enable cuda if available
print("Enable cuda if available")
if torch.cuda.is_available():
    generator.cuda()
    generator2.cuda()



##################
## Optimization ##
##################
# Optimizers
print("Optimizers")
#d_optimizer = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g2_optimizer = Adam(generator2.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Loss function
loss = nn.BCELoss()
mse = nn.MSELoss()

# Number of epochs
num_epochs = 801



##############
## Training ##
##############
def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 2048*4*4))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 2048*4*4))
    if torch.cuda.is_available(): return data.cuda()
    return data

def train_discriminator(optimizer, real_halfdata, real_fulldata, fake_fulldata):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1. Train on Real Data
    prediction_real = discriminator(real_fulldata)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(prediction_real.size(0)))
    #error_real = loss(prediction_real, real_halfdata)
    error_real.backward()    
    
    # 2. Train on Fake Data
    prediction_fake = discriminator(fake_fulldata)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(prediction_fake.size(0)))
    #error_fake = loss(prediction_fake, real_halfdata)
    error_fake.backward()
    
    
#     # 3. Train on Cycle Feedback
#     cycle_prediction_real = discriminator(real_data)
#     # Calculate error and backpropagate    
#     #fullpic = np.reshape(np.array(real_data),(res, res))
#     fullpic = np.reshape(real_data.detach().numpy(),(res, res))    
#     halfpic = cv2.resize(fullpic,(ires,ires)) 
#     halfpic_tensor = torch.tensor(np.reshape(halfpic, (1, 1, ires, ires)))           
#     cycle_error_real = mse(cycle_prediction_real, halfpic_tensor)
#     cycle_error_real.backward()

    
    # Update weights with gradients
    optimizer.step()
    
    return error_real + error_fake, prediction_real, prediction_fake
    return (0, 0, 0)

def train_generator(optimizer, fake_halfdata, fake_fulldata, real_halfdata, real_fulldata):
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    #fake_fulldata = generator(fake_halfdata)
    
#    # try to maximize discriminator error for fake data
#    prediction = discriminator(fake_fulldata)
#    # Calculate error and backpropagate
#    error = loss(prediction, real_data_target(prediction.size(0)))
#    error.backward()
    
    # but minimize image loss for real data
    generated = generator(real_halfdata.detach())

    ## mask 
    #generated = torch.cat((generated[:,0:7,:,:] * (1-real_fulldata[:,7:8,:,:]), generated[:,7:8,:,:]), 1)
    #generated = torch.cat((generated[:,0:7,:,:] * mask, generated[:,7:8,:,:]), 1) + mask_bias   
    generated = torch.cat((generated[:,0:7,:,:] * mask[:,0:1,:,:].detach(), generated[:,7:8,:,:]*invmask[:,0:1,:,:].detach()), 1) + mask_bias   
 
    ## also maximize ssim (1 is perfect similarity)
    #sserr_count = 0
    #sserr = 0
    #for bi in range(generated.size(0)):
    #    for vi in range(nvars):
    #        sserr += 1-ssim(generated[0][0].detach().numpy(), real_fulldata[0][0].detach().numpy())
    #        sserr_count += 1
    #sserr /= sserr_count

    ## pearson coefficient
    #vx = generated - torch.mean(generated)
    y = real_fulldata.detach()
    #vy = y - torch.mean(y)
    ##cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    #cost = 1 - torch.sum(vx * vy) * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2)) 

    ## MSE or BCE
    ##print("generatedmax", generated.max())
    ##print("generatedmin", generated.min())
    #img_loss = loss(generated, y)  #real_fulldata.detach()) #+ sserr
    #img_loss.backward()

    # SSIM loss
    #img_loss = ms_ssim( generated, real_fulldata.detach(), data_range=1, size_average=False )
    aa = generated.view(generated.shape[0]*nvars, 1, res, res)
    bb = real_fulldata.detach().view(real_fulldata.shape[0]*nvars, 1, res, res)
    #print("part1", aa.shape, bb.shape, aa.min(), bb.min())
    # convert to 3-channel grayscale
    aa = torch.cat((aa, aa, aa), dim=1)
    bb = torch.cat((bb, bb, bb), dim=1)
    #print("part2", aa.shape, bb.shape, aa.min(), bb.min())
    img_loss = 1-ssim_loss(aa, bb) #+ 1-gmsd_loss(aa,bb)  #+ mse(generated, real_fulldata.detach())**0.5
    img_loss.backward()


#    # minimize loss of generator2
#    generated2 = generator2(generator(mpi_real_halfdata))
#    #print("generated2",generated2.shape, "mpi_real_halfdata",mpi_real_halfdata.shape)
#    img_loss2 = mse(generated2, mpi_real_halfdata.detach())
#    img_loss2.backward()
    

#     fullpic = np.reshape(real_fulldata.detach().numpy(),(res, res, nvars))    
#     halfpic = cv2.resize(fullpic,(ires,ires)) 
#     halfpic_tensor = torch.tensor(np.reshape(halfpic, (1, nvars, ires, ires)))           
#     error = 14.-loss(prediction, halfpic_tensor)
    
#     # Train on Cycle Feedback
#     cycle_prediction_real = discriminator(real_data)
#     # Calculate error and backpropagate    
#     #fullpic = np.reshape(np.array(real_data),(res, res))
#     fullpic = np.reshape(real_data.detach().numpy(),(res, res))    
#     halfpic = cv2.resize(fullpic,(ires,ires)) 
#     halfpic_tensor = torch.tensor(np.reshape(halfpic, (1, 1, ires, ires)))           
#     cycle_error_real = mse(cycle_prediction_real, halfpic_tensor)
#     cycle_error_real.backward()

    
    # Update weights with gradients
    optimizer.step()
    # Return error
    return img_loss  #error

def train_generator2(optimizer, fake_halfdata, fake_fulldata, real_halfdata, real_fulldata, mpi_real_halfdata):
    # Reset gradients
    optimizer.zero_grad()

    # minimize loss of generator2
    generated2 = generator2(generator(mpi_real_halfdata))
    #print("generated2",generated2.shape, "mpi_real_halfdata",mpi_real_halfdata.shape)
    img_loss2 = mse(generated2, mpi_real_halfdata.detach())
    img_loss2.backward()

    # Update weights with gradients
    optimizer.step()
    # Return error
    return img_loss2  #error


# get random batch
import random
import copy
withheld_samples = 16

def get_random_sample():
    inputlag = 0
    #ii = random.randrange(len(osamples)-withheld_samples)
    ii = random.randrange(len(osamples)-withheld_samples-inputlag) #lagged input
    #osample = osamples[ii:ii+1]
    osample = osamples[ii+inputlag:ii+1+inputlag]  # lagged input
    isample = samples[ii:ii+1]
    return isample, osample

def get_random_batch(batch_sz):
    isample, osample = get_random_sample()
    for ii in range(batch_sz-1):
        isample2, osample2 = get_random_sample()
        isample = torch.cat([isample, isample2], dim=0)
        osample = torch.cat([osample, osample2], dim=0)
    return isample, osample

isample, osample = get_random_batch(4)
print(isample.shape)
print(osample.shape)



####################
## Start training ##
####################

#changeables
num_batches = ndays
#a = enumerate(data_loader)
#print (next(a))
import time
start_time = time.time()

# load state
state = torch.load("/glade/scratch/dkorytin/saved_models/uatm-erai-with-doy-cuda_state")
loaded_epoch = state['epoch']
generator.load_state_dict(state['generator'])
generator2.load_state_dict(state['generator2'])
g_optimizer.load_state_dict(state['g_optimizer'])
g2_optimizer.load_state_dict(state['g2_optimizer'])
#discriminator.load_state_dict(state['discriminator'])
#d_optimizer.load_state_dict(state['d_optimizer'])
 
#for epoch in range(0, num_epochs):
for epoch in range(loaded_epoch, num_epochs):

    #for n_batch, (real_batch,_) in enumerate(data_loader):
    #for n_batch, (real_batch) in enumerate(osamples):  
    for n_batch in range(0, len(osamples)-mb+1-1, mb):  # mb-sized sample start indices 

        # Fetch real training data
        isamples_real_batch, osamples_real_batch = get_random_batch(mb)

        #osamples_real_batch = osamples[n_batch:n_batch+mb]
        real_data = torch.tensor(osamples_real_batch).cuda()
        #real_data = real_data.view(mb, nvars, res, res)

        #isamples_real_batch = samples[n_batch:n_batch+mb]
        real_halfdata = torch.tensor(isamples_real_batch).cuda()
        #real_halfdata = real_halfdata.view(mb, mnvars, ires, ires)


        ## Fetch random mpi data
        #mpi_real_halfdata = msamples[n_batch:n_batch+mb].cuda()
        
        #if torch.cuda.is_available(): real_data = real_data.cuda()

        # Generate fake data
        fake_fulldata = noise3d(mb, nvars, res).view(mb, nvars, res, res)
        fake_halfdata = noise3d(mb, mnvars, ires).view(mb, mnvars, ires, ires)
        
        # Train D
        #d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_halfdata, real_data, fake_fulldata)
        d_error, d_pred_real, d_pred_fake = np.array(0.),np.array(0.),np.array(0.)

        # Train G (note: train_generator() uses the descriminator)
        #fake_data = generator(noise(real_batch.size(0), ires))
        #fake_halfdata = noise3d(batch_size, nvars, ires).view(batch_size, nvars, ires, ires)
        #real_fulldata = Variable(torch.tensor(osamples_real_batch))
        g_error = train_generator(g_optimizer, fake_halfdata, fake_fulldata, real_halfdata, real_data)
        #g_error2 = train_generator2(g2_optimizer, fake_halfdata, fake_fulldata, real_halfdata, real_data, mpi_real_halfdata)

        # debug
        #test_images = generator(test_noise).data.cpu()
        #print(test_images[0,0].shape)
        #%matplotlib inline  
        #import matplotlib.pyplot as plt
        #plt.imshow(test_images[0,0])
        #plt.show()
        
        # Display Progress
        if (n_batch) % 20000 == 0:
            # save state
            state = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'generator2': generator2.state_dict(),
                #'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'g2_optimizer': g2_optimizer.state_dict(),
                #'d_optimizer': d_optimizer.state_dict(),
            }
            ######torch.save(state, "/glade/scratch/dkorytin/saved_models/uatm-erai-with-doy-cuda_state")

            # save generator
            gstate = {
                'minmaxo': minmaxo,
                'minmaxi': minmaxi,
                'generator': generator.state_dict(),
                'generator2': generator2.state_dict(),
            }

            #torch.save(gstate, "/glade/scratch/dkorytin/saved_models/uatm-erai-with-doy-cuda_generator-e"+str(epoch))
            if epoch % 50 == 0 and epoch>0:
                #torch.save(gstate, "/glade/scratch/dkorytin/saved_models/uatm-erai-with-doy-cuda_generator-10s-e"+str(epoch))
                torch.save(state, "/glade/scratch/dkorytin/saved_models/uatm-erai-with-doy-cuda_state")
                torch.save(gstate, "/glade/scratch/dkorytin/saved_models/uatm-erai-with-doy-cuda_generator-e"+str(epoch))


            start_time = time.time()
            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, n_batch, num_batches))
            print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
            print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))
            print("g_error:", float(g_error))

            # out of sample error
            isamples_real_batch = samples[-withheld_samples:None]
            real_halfdata = torch.tensor(isamples_real_batch).cuda()
            osamples_real_batch = osamples[-withheld_samples:None]
            real_fulldata = torch.tensor(osamples_real_batch).cuda()

            #print("JHJHJH", real_halfdata.shape)
            generated = generator(real_halfdata)
            #generated = torch.cat((generated[:,0:7,:,:] * mask, generated[:,7:8,:,:]), 1) + mask_bias  
            generated = torch.cat((generated[:,0:7,:,:] * mask[:,0:1,:,:].detach(), generated[:,7:8,:,:]*invmask[:,0:1,:,:].detach()), 1) + mask_bias

            #os_err = mse(generator(real_halfdata), real_fulldata.detach())
            os_err = mse(generated, real_fulldata.detach())
            print("os_err:", float(os_err))
            if os_err < 0.0015:
                print("SAVING GENERATOR", os_err, epoch)
                torch.save(gstate, "/glade/scratch/dkorytin/saved_models/uatm-erai-with-doy-cuda_generator-e"+str(epoch))


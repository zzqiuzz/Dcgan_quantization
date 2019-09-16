import torch 
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    #@staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        '''tmp = input[input.ge(-1).eq(input.lt(0))]
        input[input.ge(-1).eq(input.lt(0))]= tmp*(2+tmp)
        tmp = input[input.ge(0).eq(input.lt(1))]
        input[input.ge(0).eq(input.lt(1))]= tmp*(2-tmp)
        input[input.lt(-1)] = -1
        input[input.gt(1)] = 1'''
        '''mask1 = input.lt(0.32)
        mask2 = input.ge(0.32).eq(input.lt(1.5*0.6487))
        mask3 = input.ge(1.5*0.6487).eq(input.lt(2.5*0.6487))
        mask4 = input.ge(2.5*0.6487)
        input[mask1] = 0
        input[mask2] = 0.6487
        input[mask3] = 0.6487 * 2
        input[mask4] = 0.6487 * 3'''
        #coe = 255
        #output = torch.round(input.data.mul(coe)).div(coe)
        output = input.sign()
        #print(input)
        return output
    #@staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        '''mask = input.ge(-1).eq(input.lt(0))
        tmp = input[mask]
        grad_input[mask] *= 2 + tmp * 2 
        mask = input.ge(0).eq(input.lt(1))
        tmp = input[mask]
        grad_input[mask] *= 2 - tmp * 2
        grad_input[input.gt(1)] = 0
        grad_input[input.lt(-1)] = 0'''
        '''mask1 = input.le(0)
        mask2 = input.ge(3*0.6487)
        grad_input[mask1] = 0
        grad_input[mask2] = 0'''
        grad_input[input.gt(1)] = 0
        grad_input[input.lt(-1)] = 0
        return grad_input
		
class BinConvTranspose2d(nn.Module):
    def __init__(self,input_channels,output_channels,
		kernel_size,stride,padding,bias=False):
        super(BinConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(input_channels,output_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(output_channels,momentum=0.99)
        
    def forward(self,x):
        x = BinActive()(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
class BinConv2d(nn.Module):
    def __init__(self,input_channels,output_channels,
        kernel_size,stride,padding,bias=False):
        super(BinConv2d,self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(output_channels,momentum=0.99)
    def forward(self,x):
        x = BinActive()(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
class Generator(nn.Module):
    def __init__(self, ngpu,nz,nc,ngf,type):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        if type == 'fwn':
            self.main = nn.Sequential(OrderedDict([
                # input is Z, going into a convolution
                ('conv1',nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)),
                ('bn1',nn.BatchNorm2d(ngf * 8)),
                ('relu1',nn.ReLU(True)),
                # state size. (ngf*8) x 4 x 4
                ('conv2',nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
                ('bn2',nn.BatchNorm2d(ngf * 4)),
                ('relu2',nn.ReLU(True)),
                # state size. (ngf*4) x 8 x 8
                ('conv3',nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
                ('bn3',nn.BatchNorm2d(ngf * 2)),
                ('relu3',nn.ReLU(True)),
                # state size. (ngf*2) x 16 x 16
                ('conv4',nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False)),
                ('bn4',nn.BatchNorm2d(ngf)),
                ('relu4',nn.ReLU(True)),
                # state size. (ngf) x 32 x 32
                ('conv5',nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)), 
                ('tanh',nn.Tanh())
                # state size. (nc) x 64 x 64
            ]))
        elif type == 'fwn_extra':# insert 2 normal convolution layers
            self.main = nn.Sequential(OrderedDict([
                # input is Z, going into a convolution
                ('conv1',nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)),
                ('bn1',nn.BatchNorm2d(ngf * 8)),
                ('relu1',nn.ReLU(True)), 
                # state size. (ngf*8) x 4 x 4
                ######################################
                #insert 1 normal conv 
                #('conv_extra1',nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False)),
                #('bn_extra1',nn.BatchNorm2d(ngf * 8)),
                #('relu_extra1',nn.ReLU(True)),
                ###########################
                ('conv2',nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
                ('bn2',nn.BatchNorm2d(ngf * 4)),
                ('relu2',nn.ReLU(True)), 
                # state size. (ngf*4) x 8 x 8
                #######################################
                #insert 1 normal conv 
                ('conv_extra2',nn.Conv2d(ngf * 4, ngf * 4,3,1,1,bias=False)),
                ('bn_extra2',nn.BatchNorm2d(ngf * 4)),
                ('relu_extra2',nn.ReLU(True)),
                ###########################
                ('conv3',nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
                ('bn3',nn.BatchNorm2d(ngf * 2)),
                ('relu3',nn.ReLU(True)),
                # state size. (ngf*2) x 16 x 16
                #######################################
                #insert 1 normal conv 
                ('conv_extra3',nn.Conv2d(ngf * 2, ngf * 2,3,1,1,bias=False)),
                ('bn_extra3',nn.BatchNorm2d(ngf * 2)),
                ('relu_extra3',nn.ReLU(True)),
                ########################### 
                ('conv4',nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False)),
                ('bn4',nn.BatchNorm2d(ngf)),
                ('relu4',nn.ReLU(True)),
                # state size. (ngf) x 32 x 32
                ('conv5',nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)), 
                ('tanh',nn.Tanh())
                # state size. (nc) x 64 x 64
            ]))
        elif type == 'bnn':
            self.main = nn.Sequential(
			nn.ConvTranspose2d(nz,ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8,momentum=0.99),
			nn.ReLU(True),
            nn.BatchNorm2d(ngf * 8,momentum=0.99),#trial
			BinConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			BinConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			BinConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.ReLU(True),
			nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False), 
            nn.Tanh() 
        )
        elif type == 'bnn_extra':
            self.main = nn.Sequential(
			nn.ConvTranspose2d(nz,ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8,momentum=0.99),
			nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8,ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4,momentum=0.99),
			nn.ReLU(True), 
			BinConv2d(ngf * 4, ngf * 4,3,1,1,bias=False),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
			BinConv2d(ngf * 2, ngf * 2,3,1,1,bias=False),
			nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False), 
            nn.Tanh())
        

    def forward(self, input):
        return self.main(input)
class BinActive_depth(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)  
        '''tmp = input[input.ge(-1).eq(input.lt(0))]
        input[input.ge(-1).eq(input.lt(0))]= tmp*(2+tmp)
        tmp = input[input.ge(0).eq(input.lt(1))]
        input[input.ge(0).eq(input.lt(1))]= tmp*(2-tmp)
        input[input.lt(-1)] = -1
        input[input.gt(1)] = 1'''
        coe = 15
        output = torch.round(input.data.mul(coe)).div(coe)
        output = input.sign()
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone() 
        '''mask = input.ge(-1).eq(input.lt(0))
        tmp = input[mask]
        grad_input[mask] *= 2 + tmp * 2 
        mask = input.ge(0).eq(input.lt(1))
        tmp = input[mask]
        grad_input[mask] *= 2 - tmp * 2'''
        grad_input[input.gt(1)] = 0
        grad_input[input.lt(-1)] = 0
        #print(grad_input)
        return grad_input
        
class Residual_block(nn.Module):
    def __init__(self,input_channels,output_channels,
		kernel_size,stride,padding,output_padding,groups,bias=False):
        super(Residual_block,self).__init__()
        self.conv_tran_dw = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding,output_padding, groups,bias=False)
        self.bn_dw = nn.BatchNorm2d(output_channels)
        self.relu_dw = nn.ReLU(True)
        self.pointwise = nn.Conv2d(output_channels,output_channels // 2,1,1,0,1,1,bias=False)
        self.bn_pw = nn.BatchNorm2d(output_channels // 2)
        self.relu_pw = nn.ReLU(True)
    def forward(self,x):
        #identity = x
        out = self.conv_tran_dw(x)
        out = self.bn_dw(out)
        out = BinActive_depth()(out) 
        out = self.pointwise(out)
        out = self.bn_pw(out)
        out = self.relu_pw(out)
        #out += identity
        return out
              
class Generator_depth(nn.Module):
    def __init__(self, ngpu,nz,nc,ngf,type):
        super(Generator_depth, self).__init__()
        self.ngpu = ngpu
        if type == 'fwn':
            self.main = nn.Sequential(OrderedDict([
            # input is Z, going into a convolution
            #('conv1',nn.ConvTranspose2d( nz, nz, 4, 1, 0,groups=nz, bias=False)),
            #('pointwise1',nn.Conv2d(nz,ngf*8,1,1,0,1,1,bias=False)),
            ('conv1',nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)),
            ('bn1',nn.BatchNorm2d(ngf * 8)),
            ('relu1',nn.ReLU(True)),
                
            ('conv_dw2',nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1,groups=ngf*8,bias=False)),
            ('bn_dw2',nn.BatchNorm2d(ngf * 8)),
            ('relu_dw2',nn.ReLU(True)),           
            
            ('pointwise2',nn.Conv2d(ngf*8,ngf*4,1,1,0,1,1,bias=False)),  
            ('bn2',nn.BatchNorm2d(ngf * 4)),
            ('relu2',nn.ReLU(True)),
            
            ('conv_dw3',nn.ConvTranspose2d( ngf * 4, ngf * 4, 4, 2, 1,groups=ngf*4, bias=False)),
            ('bn_dw3',nn.BatchNorm2d(ngf * 4)),
            ('relu_dw3',nn.ReLU(True)),     
            
            ('pointwise3',nn.Conv2d(ngf*4,ngf*2,1,1,0,1,1,bias=False)),
            ('bn3',nn.BatchNorm2d(ngf * 2)),
            ('relu3',nn.ReLU(True)),
             
            ('conv_dw4',nn.ConvTranspose2d( ngf * 2, ngf*2, 4, 2, 1,groups=ngf*2, bias=False)),
            ('bn_dw4',nn.BatchNorm2d(ngf * 2)),
            ('relu_dw4',nn.ReLU(True)),            
            
            ('pointwise4',nn.Conv2d(ngf*2,ngf,1,1,0,1,1,bias=False)),
            ('bn4',nn.BatchNorm2d(ngf)),
            ('relu4',nn.ReLU(True)),
            
            # state size. (ngf) x 32 x 32
            ('conv5',nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)),  
            ('tanh',nn.Tanh())
            # state size. (nc) x 64 x 64
        ]))
        elif type == 'bnn':
            self.main = nn.Sequential(OrderedDict([
                # input is Z, going into a convolution
                #('conv1',nn.ConvTranspose2d( nz, nz, 4, 1, 0,groups=nz, bias=False)),
                #('pointwise1',nn.Conv2d(nz,ngf*8,1,1,0,1,1,bias=False)),
                ('conv1',nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)),
                ('bn1',nn.BatchNorm2d(ngf * 8)),
                ('relu1',nn.ReLU(True)), 
                ('res1',Residual_block(ngf * 8, ngf * 8 , 4, 2, 1, 0,groups=ngf*8,bias=False)), 
                ('res2',Residual_block(ngf * 4, ngf * 4, 4, 2, 1, 0,groups=ngf*4,bias=False)), 
                ('res3',Residual_block(ngf * 2, ngf * 2, 4, 2, 1, 0,groups=ngf*2,bias=False)), 
                # state size. (ngf) x 32 x 32
                ('conv5',nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False)),  
                ('tanh',nn.Tanh())
                # state size. (nc) x 64 x 64
            ]))

    def forward(self, input):
        return self.main(input)
	
class Discriminator(nn.Module):
    def __init__(self, ngpu,nc,ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
        '''output = self.main(input)
        return output.view(-1, 1).squeeze(1)'''
		

def save_netG_checkpoint(state,out_folder,epoch):
    torch.save(state,'{0}/netG_epoch_{1}.pth'.format(out_folder,epoch))
    
def save_netD_checkpoint(state,out_folder,epoch):
    torch.save(state,'{0}/netD_epoch_{1}.pth'.format(out_folder,epoch))
	
	
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def validate(model,model_path,input):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    output = model(input) 
    return output
    

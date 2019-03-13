# -*- coding: utf-8 -*-
from __future__ import print_function
#%matplotlib inline
import json
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from model import dcgan
import util

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default="/home/zhengzhe/Data/celeb", help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--image_size', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='outputs_fwn', help='folder to output images and model checkpoints')
parser.add_argument('--gpuid',type=int,default=4,help="gpu id")
parser.add_argument('--validate',type=bool,default=False,help="validate model")
parser.add_argument('--G_bnn',action='store_true',help="only binarize weight.")
parser.add_argument('--D_q',action='store_true',help='binarize weight in the Discriminator.')
parser.add_argument('--bit',type=int,default=8,help='Bits to quantize Discriminator.')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--depth',action='store_true',help='perform depthwise convolution')
parser.add_argument('--pretrained_D',action='store_true',help='train bwn resuming from pretrained_D models.')
parser.add_argument('--pretrained_D_path',default='',type=str,help='path to pretrained_D model')
opt = parser.parse_args()
print(opt)
def main():
    
    if not opt.depth: 
        if opt.G_bnn :
            opt.outf = 'outputs_G_bnn' #only binarize G network
            print('only binarize G')
            if opt.pretrained_D:
                opt.outf = 'outputs_G_bnn_pretrained_D'#binarize G but D pretrained_D and fixed
                print('binarize G with D pretrained and fixed.')
            if opt.D_q:
                opt.outf = 'outputs_G_bnn_D_q'
                print("binarize G with D quantized.")
    elif opt.depth:
        opt.outf = 'outputs_G_fwn_depth'
        if opt.G_bnn:
            opt.outf = 'outputs_G_bnn_D_fwn_depth'
            if opt.pretrained_D:
                opt.outf = 'outputs_G_bnn_pretrained_D_depth'
            if opt.D_q:
                opt.outf = 'outputs_G_bnn_D_q_depth'
        else:
            if opt.D_q:
                opt.outf = 'outputs_G_fwn_D_q_depth'
    if not os.path.exists(opt.outf):
        os.system('mkdir {0}'.format(opt.outf))
    if opt.pretrained_D:
        model_path = opt.pretrained_D_path
        checkpoint = torch.load(model_path)
    # Set random seem for reproducibility
    #manualSeed = 999
    manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    grad_data = []    
    layer_name = 'conv_dw2'
    nc = 3     
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.image_size),
                                   transforms.CenterCrop(opt.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.workers)
    id = 'cuda:' + str(opt.gpuid)
    device = torch.device(id if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")
    
    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"image_Size": opt.image_size, "nz": opt.nz, "nc": nc, "ngf": opt.ngf, "ngpu": opt.ngpu}
    with open(os.path.join(opt.outf, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")
        
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    
    if opt.depth:
        netG = dcgan.Generator_depth(opt.ngpu,opt.nz,nc,opt.ngf,'fwn').to(device)
        if opt.G_bnn:
            netG = dcgan.Generator_depth(opt.ngpu,opt.nz,nc,opt.ngf,'bnn').to(device)
    elif not opt.depth:
        netG = dcgan.Generator(opt.ngpu,opt.nz,nc,opt.ngf,'fwn').to(device)
        if opt.G_bnn:
            netG = dcgan.Generator(opt.ngpu,opt.nz,nc,opt.ngf,'bnn').to(device)
    if (device.type == 'cuda') and (opt.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(opt.ngpu)))
    print(netG)
    ##weight init
    if opt.G_bnn:
        for m in netG.modules():
            if isinstance(m,nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    else:
        netG.apply(dcgan.weights_init) 
    netD = dcgan.Discriminator(opt.ngpu,nc,opt.ndf).to(device)
    if opt.pretrained_D:
        netD.load_state_dict(checkpoint['state_dict'])
    else:
        netD.apply(dcgan.weights_init)
    if (device.type == 'cuda') and (opt.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(opt.ngpu)))  
    
    print(netD)
    if opt.G_bnn:
        bin_op_G = util.Bin_G(netG,'bin_G')  
        if opt.depth:
            bin_op_G = util.Bin_G(netG,'bin_G_depth')
    if opt.D_q :
        bit = int(opt.bit)
        print('Quantize D with %d bits',bit)
        bin_op_D = util.Quan_D(netD,bit)
    if opt.validate:
        modelpath = "checkpoint.tar"
        noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)
        with torch.no_grad():
            output = dcgan.validate(netG,modelpath,noise)
            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(vutils.make_grid(output.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()
        return
    
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(opt.num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            if not opt.pretrained_D:
                netD.zero_grad()
                if opt.D_q:
                    bin_op_D.quantization()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0) 
                label = torch.full((b_size,), real_label, device=device)
                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
                if opt.G_bnn :
                    bin_op_G.binarization()
                # train with fake
                noise = torch.randn(b_size, opt.nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                if opt.D_q :
                    bin_op_D.restore()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()
            elif opt.pretrained_D:
                if opt.G_bnn :
                    bin_op_G.binarization()
                b_size = data[0].to(device).size(0)
                label = torch.full((b_size,), real_label, device=device)
                noise = torch.randn(b_size, opt.nz, 1, 1, device=device)
                fake = netG(noise)
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            
            if opt.G_bnn:
                bin_op_G.restore()
                #bin_op_G.updateBinaryGradWeight()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            if not opt.pretrained_D:
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\t \
                        Loss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, opt.num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            else:
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_G: %.4f'
                          % (epoch, opt.num_epochs, i, len(dataloader),
                             errG.item()))
                 #show mean and variance of weights in netG
                util.showWeightsInfo(netG,layer_name ,grad_data)  
            G_losses.append(errG.item())
            if not opt.pretrained_D:
                D_losses.append(errD.item())
            
            if (iters % 500 == 0) or ((epoch == opt.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))                              
            iters += 1
        
        dcgan.save_netG_checkpoint({
                    'epoch':epoch ,
                    'state_dict':netG.state_dict(),
                    },opt.outf,epoch)
        dcgan.save_netD_checkpoint({
                    'epoch':epoch ,
                    'state_dict':netD.state_dict(),
                    },opt.outf,epoch)


    #save grad_data to bin for analysis
    grad_data = np.array(grad_data)
    filename = opt.outf + '/grad_data_' + layer_name + '_' + str(opt.num_epochs) + '.bin'
    grad_data.tofile(filename)
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.title('Specified layer grad During Training')
    plt.plot(grad_data,label="Grad_data_" + layer_name)
    plt.xlabel('iters')
    plt.ylabel('magnitude of grad_data in' + layer_name) 
    plt.legend()
    if opt.G_bnn :
        if not opt.D_q:
            plt.savefig(opt.outf + '/loss_G_bnn_' + str(opt.num_epochs) + '.jpg')
        if opt.D_q:
            plt.savefig(opt.outf + '/loss_G_bnn_D_q_' + str(bit) + '.jpg' )
    else:
        plt.savefig(opt.outf + '/loss_fwn_' + str(opt.num_epochs) + '.jpg')
    plt.show()

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())


    real_batch = next(iter(dataloader))

    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    
    if opt.G_bnn :
        plt.savefig(opt.outf + '/Result_G_bnn_' + str(opt.num_epochs) + '.jpg')
        if opt.D_q:
            plt.savefig(opt.outf + '/Result_G_D_q.jpg')
    else:
        plt.savefig(opt.outf + '/Result_fwn_' + str(opt.num_epochs) + '.jpg')
    plt.show()
if __name__ == '__main__':
    main()

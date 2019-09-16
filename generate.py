from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
#FWN
#python generate.py --gpuid=7 -n=5000 -o=generated_imgs -c=outputs/generator_config.json -w=outputs/netG_epoch_4.pth --cuda
#BWN
#python generate.py --gpuid=7 -n=5000 -o=bwn_generated_imgs -c=outputs_bwn/generator_config.json -w=outputs_bwn/netG_epoch_4.pth --cuda
class RandDataset(Dataset):
    def __init__(self,length,nz):
        self.data = torch.FloatTensor(length, nz, 1, 1).normal_(0, 1)
        self.len = length
    def __getitem__(self,index):
        return self.data[index]
    def __len__(self):
        return self.len

from model import dcgan
if __name__=="__main__":
#python generate.py -c outputs_G_bnn_D_depth/generator_config.json -w outputs_G_bnn_D_depth/netG_epoch_4.pth -o depth_fwn_imgs -bs 2500 -n 50000 --gpuid 2 --type fwn
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='path to generator config .json file')
    parser.add_argument('-w', '--weights', required=True, type=str, help='path to generator weights .pth file')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help="path to to output directory")
    parser.add_argument('-bs', '--batchsize',type=int,default=2500, help="images generated per time.")
    parser.add_argument('-n','--nimages',type=int,default=50000,help="number of images to generate.")
    parser.add_argument('--gpuid',type=int,default=4,help="gpu id")
    parser.add_argument('--type',required=True,type=str,help="fwn or bnn.")
    opt = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpuid)
    with open(opt.config, 'r') as gencfg:
        generator_config = json.loads(gencfg.read())
    if not os.path.exists(opt.output_dir):
        os.system('mkdir {0}'.format(opt.output_dir))
        os.system('mkdir {0}'.format(opt.output_dir + '/imgs'))
    save_img_folder = opt.output_dir + '/imgs'
    imageSize = generator_config["image_Size"]
    nz = generator_config["nz"]
    nc = generator_config["nc"]
    ngf = generator_config["ngf"]
    ngpu = generator_config["ngpu"]
    #id = 'cuda:' + str(opt.gpuid)
    #############
    #device = torch.device(id if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    device = torch.device('cuda')
    #netG_parallel = nn.DataParallel(netG,device_ids=device_id)
    netG = dcgan.Generator(ngpu,nz,nc,ngf,opt.type).to(device)
    # load weights
    model = torch.load(opt.weights)
    netG.load_state_dict(model['state_dict'])
    print(netG)
    #for key,value in netG.state_dict().items():
    #    netG_parallel.state_dict()["module." + key] = value
    # initialize noise
    #epoch = opt.nimages // opt.batchsize
    #kfor epc in range(epoch):
    dataloader = DataLoader(dataset=RandDataset(opt.nimages,nz),batch_size=opt.batchsize)
    #epochs = opt.nimages // opt.batchsize
    #print("Total epochs is: ",epochs)
    #fixed_noise = torch.cuda.FloatTensor(opt.nimages, nz, 1, 1).normal_(0, 1)
    count = 0
    #for epc in range(epochs):
    for data in dataloader:
        input = data.cuda() 

        fake = netG(input)
        fake.data = fake.data.mul(0.5).add(0.5)
        
        for i in range(opt.batchsize):
            vutils.save_image(fake.data[i, ...].reshape((1, nc, imageSize, imageSize)), os.path.join(save_img_folder, "generated_%02d.png"%(i + count * opt.batchsize)))
        count += 1
        print(count * opt.batchsize)

        
    print("Images generated Success!")

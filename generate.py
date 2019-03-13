from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
#FWN
#python generate.py --gpuid=7 -n=5000 -o=generated_imgs -c=outputs/generator_config.json -w=outputs/netG_epoch_4.pth --cuda
#BWN
#python generate.py --gpuid=7 -n=5000 -o=bwn_generated_imgs -c=outputs_bwn/generator_config.json -w=outputs_bwn/netG_epoch_4.pth --cuda


from model import dcgan

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='path to generator config .json file')
    parser.add_argument('-w', '--weights', required=True, type=str, help='path to generator weights .pth file')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help="path to to output directory")
    parser.add_argument('-n', '--nimages', required=True, type=int, help="number of images to generate", default=1)
    parser.add_argument('--gpuid',type=int,default=4,help="gpu id")
    
    opt = parser.parse_args()
    
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
    id = 'cuda:' + str(opt.gpuid)
    device = torch.device(id if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    netG = dcgan.Generator(ngpu,nz,nc,ngf).to(device)

    # load weights
    model = torch.load(opt.weights)
    netG.load_state_dict(model['state_dict'])

    # initialize noise
    fixed_noise = torch.cuda.FloatTensor(opt.nimages, nz, 1, 1,device=device).normal_(0, 1)
 

    fake = netG(fixed_noise)
    fake.data = fake.data.mul(0.5).add(0.5)

    for i in range(opt.nimages):
        vutils.save_image(fake.data[i, ...].reshape((1, nc, imageSize, imageSize)), os.path.join(save_img_folder, "generated_%02d.png"%i))
    print("Images generated Success!")

import torch
import torch.nn as nn
import argparse
import json
import time
import matplotlib.pyplot as plt
from model import dcgan 

#python save_weghts.py -c outputs_G_bnn/generator_config.json -w outputs_G_bnn/netG_epoch_4.pth
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True, type=str, help='path to generator config .json file')
parser.add_argument('-w', '--weights', required=True, type=str, help='path to generator weights .pth file')
#parser.add_argument('-o', '--output_dir', required=True, type=str, help="path to to output directory")
                      
opt = parser.parse_args()
print(opt)
outputFileName = 'weights.bin'

with open(opt.config, 'r') as gencfg:
    generator_config = json.loads(gencfg.read())
nz = generator_config["nz"]
nc = generator_config["nc"]
ngf = generator_config["ngf"]
ngpu = generator_config["ngpu"]
netG = dcgan.Generator(ngpu,nz,nc,ngf,'bnn')
checkpoint = torch.load(opt.weights)
netG.load_state_dict(checkpoint['state_dict'])
print(netG)
for id, m in enumerate(netG.modules()):
    if isinstance(m,nn.ConvTranspose2d):
        out_data = m.weight.data.view(-1)
        out_data = out_data.numpy()
        out_data.tofile(outputFileName)
        break

plt.hist(out_data,50,density=True,facecolor='g',alpha=0.75) 
plt.xlabel('Weights value')
plt.ylabel('Probability')
plt.title('Histogram of weights')
plt.grid(True)
plt.show()

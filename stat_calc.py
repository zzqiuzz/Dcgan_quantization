from model import dcgan
from thop import profile
ngpu = 1
nz = 100
nc = 3 
ngf = 64
netG = dcgan.Generator_depth(ngpu,nz,nc,ngf)
print(netG)
flops,param = profile(netG,input_size=(1,100,1,1))
print('flops is: ',flops)
print('param is: ',param)

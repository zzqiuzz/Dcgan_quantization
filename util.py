import torch
import torch.nn as nn
import numpy
class Quan_D():
    def __init__(self,model,bit):
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                    count_targets = count_targets + 1
        start_range = 1
        end_range = count_targets-2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = [] 
        self.target_modules = []
        self.bit = bit
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                    index = index + 1
                    if index in self.bin_range:
                        tmp = m.weight.data.clone()
                        self.saved_params.append(tmp)
                        self.target_modules.append(m.weight)#save weights for those quantized ones
        print('num of binary layers of D: ',self.num_of_params)
    def quantization(self):  
        self.save_params()#save float version 
        self.quantizeConvParams()
    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
    def quantizeConvParams(self):
        bits = torch.tensor(self.bit)
        coe = torch.pow(2,bits) - 1
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    torch.round(self.target_modules[index].data.mul_(coe)).div(coe)
            
class Bin_G():
    def __init__(self, model,model_type,quan_type,bit):
        # count the number of Conv2d and Linear
        count_targets = 0
        self.bit = bit
        self.quan_type = quan_type
        for m in model.modules():
            if model_type == "bin_G":
                if isinstance(m, nn.ConvTranspose2d):
                    count_targets = count_targets + 1
            elif model_type == "bin_G_extra":
                if isinstance(m,nn.Conv2d):
                    count_targets = count_targets + 1
            elif model_type == "bin_D":
                if isinstance(m, nn.Conv2d):
                    count_targets = count_targets + 1
            elif model_type == "bin_G_depth":
                if isinstance(m, nn.Conv2d):
                    count_targets = count_targets + 1
        start_range = 1
        end_range = count_targets-2
        if model_type == 'bin_G_depth' or model_type == "bin_G_extra":
            flag = False
            end_range = count_targets # 这里量化三层, end_range=count_targets - 3 则量化一层  效果还可以 
         
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = [] 
        self.target_modules = []
        index = -1
        if model_type == 'bin_G_depth' or model_type == "bin_G_extra":
            index = 0
        for m in model.modules():
            if model_type == "bin_G" :
                if isinstance(m, nn.ConvTranspose2d):
                    index = index + 1
                    if index in self.bin_range:
                        tmp = m.weight.data.clone()
                        self.saved_params.append(tmp)
                        self.target_modules.append(m.weight)
            elif model_type == "bin_D" or model_type == "bin_G_extra":
                if isinstance(m, nn.Conv2d):
                    index = index + 1
                    if index in self.bin_range:
                        tmp = m.weight.data.clone()
                        self.saved_params.append(tmp)
                        self.target_modules.append(m.weight)
            elif model_type == "bin_G_depth":
                if isinstance(m, nn.Conv2d):
                    #if not flag:
                    #    flag = True
                    #    index = 0
                    #    continue
                    index = index + 1
                    if index in self.bin_range:
                        tmp = m.weight.data.clone()
                        self.saved_params.append(tmp)
                        self.target_modules.append(m.weight)
        print('num of binary layers of G: ',self.num_of_params)
    def binarization(self):
        #self.meancenterConvParams()
        #self.clampConvParams()
        self.save_params()#save float version 
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            #n = self.target_modules[index].data[0].nelement()
            n = self.target_modules[index].data[:,0,:,:].nelement()
            s = self.target_modules[index].data.size()
            if self.bit == 1:
                if len(s) == 4:
                    m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                            .sum(2, keepdim=True).sum(0, keepdim=True).div(n)
                elif len(s) == 2:
                    m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
                self.target_modules[index].data = \
                        self.target_modules[index].data.sign().mul(m.expand(s))
            elif self.bit > 1:
                if self.quan_type == 'default':
                    coe = torch.pow(2,torch.tensor(self.bit)) - 1 # n bit quantization
                    self.target_modules[index].data.mul_(coe).round_().div_(coe)
                elif self.quan_type == 'jacob':
                    rimax = torch.max(self.target_modules[index].data)
                    rimin = torch.min(self.target_modules[index].data)
                    qmax = 2**self.bit -1
                    qmin = 0
                    r_scale = (rimax - rimin) / (qmax - qmin)
                    r_zero = torch.round(qmax - rimax / r_scale)
                    qvalue = torch.round(self.target_modules[index].data / r_scale + r_zero)
                    ro_ = r_scale * (qvalue - r_zero)
                    self.target_modules[index].data.copy_(ro_)
                else:
                    print("Unimplemented quantization methods!")
            elif self.bit == -1:#binary_relax
                if len(s) == 4:
                    m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                            .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
                elif len(s) == 2:
                    m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
                self.target_modules[index].data = \
                        self.target_modules[index].data.sign().mul(m.expand(s))
                self.target_modules[index].data = \
                        self.target_modules[index].data.sign().mul(m.expand(s)).mul(1 - 0.001) + \
                            self.saved_params[index].data.mul(0.001)

    def show_weight(self):
        n = self.target_modules[0].data[0].nelement()
        s = self.target_modules[0].data.size()
        m = self.target_modules[0].data.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
        self.target_modules[0].data = \
                    self.target_modules[0].data.sign().mul(m.expand(s))
        print(self.target_modules[0].data[0][:][:][:])
            
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
            self.target_modules[index].grad.data = self.target_modules[index].grad.data.mul(1e+9)
            
            
def showWeightsInfo(model,layer_name,grad_data):
    for idx, module in enumerate(model.named_modules()):
        if module[0] == 'main.' + layer_name:
            #print(module[1])    
            #print('mean grad_data of ' + layer_name + ' is: %.6f'% module[1].weight.grad.mean().cpu().data)
            grad_data.append(module[1].weight.grad.mean().cpu().data) 
        
    

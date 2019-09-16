import torch
import os
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import argparse
from torchvision.models.inception import inception_v3
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import entropy 
def inception_score(imgs, workers, gpuid, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    id = 'cuda:' + str(gpuid)
    device = torch.device(id)
    assert batch_size > 0
    assert N > batch_size
    
    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, num_workers=opt.workers)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch[0].to(device) 
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    # python inception_score.py --dataroot=generated_imgs/ --gpuid=4
    # python inception_score.py --dataroot=$DATA/celeb/ --gpuid=4
    # python inception_score.py --dataroot=bwn_generated_imgs/ --gpuid=4
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default=" ", help='path to dataset') #5w
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--gpuid',type=int,default=4,help="gpu id")

    opt = parser.parse_args()
    print(opt)    
    my_images = dset.ImageFolder(root=opt.dataroot, 
                             transform=transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
    print ("Calculating Inception Score...")
    print (inception_score(my_images, opt.workers, opt.gpuid, cuda=True, batch_size=32, resize=True, splits=10))

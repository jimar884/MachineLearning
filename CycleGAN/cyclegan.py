import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torch.nn.modules.padding import ReflectionPad2d
from torch.optim import optimizer
from torch.serialization import load
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import glob as gF
import itertools
import random
import matplotlib.pyplot as plt
import torchvision


img_size = 256
batch_size = 4
inC = 3
outC = 3
lr = 0.0002
beta = 0.5
num_epochs = 200


def weight_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constent_(m.bias.data, 0)

class Summer2WinterDataset(Dataset):
    def __init__(self, img_path='../data/summer2winter/', transform=False, mode='train'):
        self.transform = transform
        self.domainA = self.load_imgs(img_path + mode + 'A/')
        self.domainB = self.load_imgs(img_path + mode + 'B/')
    
    def load_imgs(self, path):
        img_list = []
        glob_path = gF.glob(path+'*.jpg')
        for file_path in sorted(glob_path):
            img = Image.open(file_path)
            img_list.append(img)
        return img_list

    def __len__(self):
        return min(len(self.domainA), len(self.domainB))
    
    def __getitem__(self, idx):
        domainA = self.domainA[idx]
        domainB = self.domainB[idx]
        if self.transform:
            domainA = self.transform(domainA)
            domainB = self.transform(domainB)
        return {'A': domainA, 'B': domainB}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
        )
    
    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=7),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1)
        )
    
    def forward(self, x):
        x = self.model(x)
        return nn.functional.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class ImagePool():
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
    
    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    trainset = Summer2WinterDataset(img_path='../data/summer2winter/', transform=transform, mode='train')
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = Summer2WinterDataset(img_path='../data/summer2winter/', transform=transform, mode='test')
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    GeneratorA2B = Generator(in_channels=inC, out_channels=outC).to(device)
    GeneratorA2B.apply(weight_init)
    GeneratorB2A = Generator(in_channels=inC, out_channels=outC).to(device)
    GeneratorB2A.apply(weight_init)
    DiscriminatorA = Discriminator(in_channels=inC).to(device)
    DiscriminatorA.apply(weight_init)
    DiscriminatorB = Discriminator(in_channels=inC).to(device)
    DiscriminatorB.apply(weight_init)

    optimizerGen = torch.optim.Adam(
        itertools.chain(GeneratorA2B.parameters(), GeneratorB2A.parameters()), lr=lr, betas=(beta, 0.999)
    )
    optimizerDis = torch.optim.Adam(
        itertools.chain(DiscriminatorA.parameters(), DiscriminatorB.parameters()), lr=lr, betas=(beta, 0.999)
    )
    # optimizerGenA2B = torch.optim.Adam(GeneratorA2B.parameters(), lr=lr, betas=(beta, 0.999))
    # optimizerGenB2A = torch.optim.Adam(GeneratorB2A.parameters(), lr=lr, betas=(beta, 0.999))
    # optimizerDisA = torch.optim.Adam(DiscriminatorA.parameters(), lr=lr, betas=(beta, 0.999))
    # optimizerDisB = torch.optim.Adam(DiscriminatorB.parameters(), lr=lr, betas=(beta, 0.999))

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    img_list = []
    loss_G = []
    loss_D = []

    B2A_pool = ImagePool(pool_size=50)
    A2B_pool = ImagePool(pool_size=50)

    for epoch in range(num_epochs):
        lossG_mean = []
        lossD_mean = []
        if epoch > 100:
            optimizerGen.param_groups[0]['lr'] -= lr / 100
            optimizerDis.param_groups[0]['lr'] -= lr / 100
        for i, data in enumerate(trainloader, 0):
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)
            b_size = real_A.size(0)
            target_real = torch.full((b_size, 1), fill_value=1.0, dtype=torch.float, device=device)
            target_fake = torch.full((b_size, 1), fill_value=0.0, dtype=torch.float, device=device)

            # train Generator

            optimizerGen.zero_grad()
            # optimizerGenA2B.zero_grad()
            # optimizerGenB2A.zero_grad()

            # identity
            B2B = GeneratorA2B(real_B)
            loss_identityB = criterion_identity(B2B, real_B) * 5.0
            A2A = GeneratorB2A(real_A)
            loss_identityA = criterion_identity(A2A, real_A) * 5.0

            # adv
            A2B = GeneratorA2B(real_A)
            pred = DiscriminatorB(A2B)
            loss_GAN_A2B = criterion_GAN(pred, target_real)
            B2A = GeneratorB2A(real_B)
            pred = DiscriminatorA(B2A)
            loss_GAN_B2A = criterion_GAN(pred, target_real)

            # cycle
            A2B2A = GeneratorB2A(A2B)
            loss_cycle_A2B2A = criterion_cycle(A2B2A, real_A) * 10.0
            B2A2B = GeneratorA2B(B2A)
            loss_cycle_B2A2B = criterion_cycle(B2A2B, real_B) * 10.0

            lossG = loss_identityA + loss_identityB + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A2B2A + loss_cycle_B2A2B
            lossG.backward()
            optimizerGen.step()

            # train Discriminator

            optimizerDis.zero_grad()

            # real
            pred = DiscriminatorA(real_A)
            loss_real = criterion_GAN(pred, target_real)

            # fake
            B2A = B2A_pool.query(B2A)
            pred = DiscriminatorA(B2A.detach())
            loss_B2A = criterion_GAN(pred, target_fake)

            loss_DA = (loss_real + loss_B2A) * 0.5
            loss_DA.backward()
            optimizerDis.step() 

            # real
            pred = DiscriminatorB(real_B)
            loss_real = criterion_GAN(pred, target_real)

            # fake
            A2B = A2B_pool.query(A2B)
            pred = DiscriminatorB(A2B.detach())
            loss_A2B = criterion_GAN(pred, target_fake)

            loss_DB = (loss_real + loss_A2B) * 0.5
            loss_DB.backward()
            optimizerDis.step()

            lossG_mean.append(lossG)
            lossD_mean.append(loss_DA + loss_DB)

            if i % 50 == 0:
                print(lossG, loss_DA, loss_DB)

        loss_G.append(sum(lossG_mean) / len(lossG_mean))
        loss_D.append(sum(lossD_mean) / len(lossD_mean))
    
    torch.save(GeneratorA2B.state_dict(), 'GeneratorA2B200.pth')
    torch.save(GeneratorB2A.state_dict(), 'GeneratorB2A200.pth')
    torch.save(DiscriminatorA.state_dict(), 'DiscriminatorA200.pth')
    torch.save(DiscriminatorB.state_dict(), 'DiscriminatorB200.pth')

    # save loss
    fig_loss = plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(loss_G,label="G")
    plt.plot(loss_D,label="D")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    fig_loss.savefig("fig_loss_CycleGAN100.png")

    # save images
    GeneratorA2B.eval()
    GeneratorB2A.eval()
    num_iter = 10
    for i, data in enumerate(testloader, 0):
        real_A = data['A'].to(device)
        real_B = data['B'].to(device)

        with torch.no_grad():
            fake_B = 0.5 * (GeneratorA2B(real_A).data + 1.0)
            fake_A = 0.5 * (GeneratorB2A(real_B).data + 1.0)
            real_A = 0.5 * real_A + 0.5
            real_B = 0.5 * real_B + 0.5

            img_A2B = torch.cat([real_A, fake_B], dim=2)
            img_B2A = torch.cat([real_B, fake_A], dim=2)

            save_image(img_A2B, 'image/A2B_%04d.jpg' % (i+1))
            save_image(img_B2A, 'image/B2A_%04d.jpg' % (i+1))
        
        if i == num_iter:
            break

import torch
from torch import nn
from torch import optim
from net import Unet
from data import MaskDataset
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image

module = r'params/module.pth'
img_save_path = r'F:\code\py\UNet\train_img'
dataset = MaskDataset(r'D:\UNET\VOC2012')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

net = Unet().cuda()
opt = optim.Adam(net.parameters())
loss_func = nn.MSELoss()

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
else:
    print("No Params!")

if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

epoch = 1
while True:
    for i, (jpg, png) in enumerate(dataloader):
        jpg = jpg.cuda()
        png = png.cuda()
        out = net(jpg)
        loss = loss_func(out, png)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 5 == 0:
            print("epoch:", epoch, "count:", i, "loss:", loss)
    jpg = jpg[0]
    out = out[0]
    png = png[0]
    img = torch.stack([jpg, out, png], 0)
    save_image(img.cpu(), os.path.join(img_save_path, '{}.png'.format(epoch)))
    epoch += 1
    torch.save(net.state_dict(), module)
    print("saved!")

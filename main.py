import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import DnCNN
from dataset import Dataset
from utils import AverageMeter

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='DnCNN-S', help='DnCNN-S, DnCNN-B, DnCNN-3')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--gaussian_noise_level', type=str)
    parser.add_argument('--downsampling_factor', type=str)
    parser.add_argument('--jpeg_quality', type=str)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_fast_loader', action='store_true')
    opt = parser.parse_args()

    if opt.gaussian_noise_level is not None:
        opt.gaussian_noise_level = list(map(lambda x: int(x), opt.gaussian_noise_level.split(',')))

    if opt.downsampling_factor is not None:
        opt.downsampling_factor = list(map(lambda x: int(x), opt.downsampling_factor.split(',')))

    if opt.jpeg_quality is not None:
        opt.jpeg_quality = list(map(lambda x: int(x), opt.jpeg_quality.split(',')))

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    if opt.arch == 'DnCNN-S':
        model = DnCNN(num_layers=17)
    elif opt.arch == 'DnCNN-B':
        model = DnCNN(num_layers=20)
    elif opt.arch == 'DnCNN-3':
        model = DnCNN(num_layers=20)

    model = model.to(device)
    criterion = nn.MSELoss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataset = Dataset(opt.images_dir, opt.patch_size,
                      opt.gaussian_noise_level, opt.downsampling_factor, opt.jpeg_quality,
                      opt.use_fast_loader)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels) / (2 * len(inputs))

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, epoch)))

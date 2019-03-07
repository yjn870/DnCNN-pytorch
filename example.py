import argparse
import os
import io
import numpy as np
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from model import DnCNN

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='DnCNN-S', help='DnCNN-S, DnCNN-B, DnCNN-3')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--gaussian_noise_level', type=int)
    parser.add_argument('--jpeg_quality', type=int)
    parser.add_argument('--downsampling_factor', type=int)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    if opt.arch == 'DnCNN-S':
        model = DnCNN(num_layers=17)
    elif opt.arch == 'DnCNN-B':
        model = DnCNN(num_layers=20)
    elif opt.arch == 'DnCNN-3':
        model = DnCNN(num_layers=20)

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()

    filename = os.path.basename(opt.image_path).split('.')[0]
    descriptions = ''

    input = pil_image.open(opt.image_path).convert('RGB')

    if opt.gaussian_noise_level is not None:
        noise = np.random.normal(0.0, opt.gaussian_noise_level, (input.height, input.width, 3)).astype(np.float32)
        input = np.array(input).astype(np.float32) + noise
        descriptions += '_noise_l{}'.format(opt.gaussian_noise_level)
        pil_image.fromarray(input.clip(0.0, 255.0).astype(np.uint8)).save(os.path.join(opt.outputs_dir, '{}{}.png'.format(filename, descriptions)))
        input /= 255.0

    if opt.jpeg_quality is not None:
        buffer = io.BytesIO()
        input.save(buffer, format='jpeg', quality=opt.jpeg_quality)
        input = pil_image.open(buffer)
        descriptions += '_jpeg_q{}'.format(opt.jpeg_quality)
        input.save(os.path.join(opt.outputs_dir, '{}{}.png'.format(filename, descriptions)))
        input = np.array(input).astype(np.float32)
        input /= 255.0

    if opt.downsampling_factor is not None:
        original_width = input.width
        original_height = input.height
        input = input.resize((input.width // opt.downsampling_factor,
                              input.height // opt.downsampling_factor),
                             resample=pil_image.BICUBIC)
        input = input.resize((original_width, original_height), resample=pil_image.BICUBIC)
        descriptions += '_sr_s{}'.format(opt.downsampling_factor)
        input.save(os.path.join(opt.outputs_dir, '{}{}.png'.format(filename, descriptions)))
        input = np.array(input).astype(np.float32)
        input /= 255.0

    input = transforms.ToTensor()(input).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input)

    output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
    output = pil_image.fromarray(output, mode='RGB')
    output.save(os.path.join(opt.outputs_dir, '{}{}_{}.png'.format(filename, descriptions, opt.arch)))

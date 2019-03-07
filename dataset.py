import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import io
import numpy as np
import PIL.Image as pil_image

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class Dataset(object):
    def __init__(self, images_dir, patch_size,
                 gaussian_noise_level, downsampling_factor, jpeg_quality,
                 use_fast_loader=False):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.patch_size = patch_size
        self.gaussian_noise_level = gaussian_noise_level
        self.downsampling_factor = downsampling_factor
        self.jpeg_quality = jpeg_quality
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        if self.use_fast_loader:
            clean_image = tf.read_file(self.image_files[idx])
            clean_image = tf.image.decode_jpeg(clean_image, channels=3)
            clean_image = pil_image.fromarray(clean_image.numpy())
        else:
            clean_image = pil_image.open(self.image_files[idx]).convert('RGB')

        # randomly crop patch from training set
        crop_x = random.randint(0, clean_image.width - self.patch_size)
        crop_y = random.randint(0, clean_image.height - self.patch_size)
        clean_image = clean_image.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))

        noisy_image = clean_image.copy()
        gaussian_noise = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)

        # additive gaussian noise
        if self.gaussian_noise_level is not None:
            if len(self.gaussian_noise_level) == 1:
                sigma = self.gaussian_noise_level[0]
            else:
                sigma = random.randint(self.gaussian_noise_level[0], self.gaussian_noise_level[1])
            gaussian_noise += np.random.normal(0.0, sigma, (clean_image.height, clean_image.width, 3)).astype(np.float32)

        # downsampling
        if self.downsampling_factor is not None:
            if len(self.downsampling_factor) == 1:
                downsampling_factor = self.downsampling_factor[0]
            else:
                downsampling_factor = random.randint(self.downsampling_factor[0], self.downsampling_factor[1])

            noisy_image = noisy_image.resize((self.patch_size // downsampling_factor,
                                              self.patch_size // downsampling_factor),
                                             resample=pil_image.BICUBIC)
            noisy_image = noisy_image.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)

        # additive jpeg noise
        if self.jpeg_quality is not None:
            if len(self.jpeg_quality) == 1:
                quality = self.jpeg_quality[0]
            else:
                quality = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
            buffer = io.BytesIO()
            noisy_image.save(buffer, format='jpeg', quality=quality)
            noisy_image = pil_image.open(buffer)

        clean_image = np.array(clean_image).astype(np.float32)
        noisy_image = np.array(noisy_image).astype(np.float32)
        noisy_image += gaussian_noise

        input = np.transpose(noisy_image, axes=[2, 0, 1])
        label = np.transpose(clean_image, axes=[2, 0, 1])

        # normalization
        input /= 255.0
        label /= 255.0

        return input, label

    def __len__(self):
        return len(self.image_files)

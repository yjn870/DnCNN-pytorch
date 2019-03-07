# DnCNN

This repository is implementation of the "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising".

## Requirements
- PyTorch
- Tensorflow
- tqdm
- Numpy
- Pillow

**Tensorflow** is required for quickly fetching image in training phase.

## Results

The DnCNN-3 model is trained for three general image denoising tasks, i.e., blind Gaussian denoising, SISR with multiple upscaling factors, and JPEG deblocking with different quality factors.

<table>
    <tr>
        <td><center>JPEG Artifacts (Quality 40)</center></td>
        <td><center><b>DnCNN-3</b></center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/monarch_jpeg_q40.png" height="300"></center>
    	</td>
    	<td>
    		<center><img src="./data/monarch_jpeg_q40_DnCNN-3.png" height="300"></center>
    	</td>
    </tr>
    <tr>
        <td><center>Gaussian Noise (Level 25)</center></td>
        <td><center><b>DnCNN-3</b></center></td>
    </tr>
    <tr>
        <td>
        	<center><img src="./data/monarch_noise_l25.png" height="300"></center>
        </td>
        <td>
        	<center><img src="./data/monarch_noise_l25_DnCNN-3.png" height="300"></center>
        </td>
    </tr>
    <tr>
        <td><center>Super-Resolution (Scale x3)</center></td>
        <td><center><b>DnCNN-3</b></center></td>
    </tr>
    <tr>
        <td>
        	<center><img src="./data/monarch_sr_s3.png" height="300"></center>
        </td>
        <td>
        	<center><img src="./data/monarch_sr_s3_DnCNN-3.png" height="300"></center>
        </td>
    </tr>
</table>

## Usages

### Train

When training begins, the model weights will be saved every epoch. <br />
If you want to train quickly, you should use **--use_fast_loader** option.

#### DnCNN-S

```bash
python main.py --arch "DnCNN-S" \               
               --images_dir "" \
               --outputs_dir "" \
               --gaussian_noise_level 25 \
               --patch_size 50 \
               --batch_size 16 \
               --num_epochs 20 \
               --lr 1e-3 \
               --threads 8 \
               --seed 123 \
               --use_fast_loader              
```

#### DnCNN-B

```bash
python main.py --arch "DnCNN-B" \               
               --images_dir "" \
               --outputs_dir "" \
               --gaussian_noise_level 0,55 \
               --patch_size 50 \
               --batch_size 16 \
               --num_epochs 20 \
               --lr 1e-3 \
               --threads 8 \
               --seed 123 \
               --use_fast_loader              
```

#### DnCNN-3

```bash
python main.py --arch "DnCNN-3" \               
               --images_dir "" \
               --outputs_dir "" \
               --gaussian_noise_level 0,55 \
               --downsampling_factor 1,4 \
               --jpeg_quality 5,99 \
               --patch_size 50 \
               --batch_size 16 \
               --num_epochs 20 \
               --lr 1e-3 \
               --threads 8 \
               --seed 123 \
               --use_fast_loader              
```

### Test

Output results consist of noisy image and denoised image.

```bash
python example --arch "DnCNN-S" \
               --weights_path "" \
               --image_path "" \
               --outputs_dir "" \
               --jpeg_quality 25               
```

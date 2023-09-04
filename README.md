## Ray Conditioning: Trading Photo-realism for Photo-consistency in Multi-view Image Generation
Official PyTorch implementation of the ICCV 2023 paper</sub>

![Teaser image](https://ray-cond.github.io/assets/teaser.jpg)

**Ray Conditioning: Trading Photo-realism for Photo-consistency in Multi-view Image Generation**<br>
Eric Ming Chen, Sidhanth Holalkere, Ruyu Yan, Kai Zhang, Abe Davis<br>
https://raycond.github.io<br>

Abstract: *Multi-view image generation attracts particular attention these days due to its promising 3D-related applications, e.g., image viewpoint editing. Most existing methods follow a paradigm where a 3D representation is first synthesized, and then rendered into 2D images to ensure photo-consistency across viewpoints. However, such explicit bias for photo-consistency sacrifices photo-realism, causing geometry artifacts and loss of fine-scale details when these methods are applied to edit real images. To address this issue, we propose ray conditioning, a geometry-free alternative that relaxes the photo-consistency constraint. Our method generates multi-view images by conditioning a 2D GAN on a light field prior. With explicit viewpoint control, state-of-the-art photo-realism and identity consistency, our method is particularly suited for the viewpoint editing task.*





## Using networks from Python
This repo is built on top of [stylegan3](https://github.com/NVlabs/stylegan3), and uses the camera conventions of [eg3d](https://github.com/NVlabs/eg3d).

You can use pre-trained networks in your own Python code as follows:

```python
with open('ffhq-raycond2-512.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent codes
c2w # [1, 4, 4] sized Tensor
intrinsics #[1, 3, 3] sized Tensor
c = torch.cat([c2w.view(1, -1), intrinsics.view(1, -1)], dim=-1) # camera parameters
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1], no truncation
```

We also provide visualization notebooks.  There is one for each dataset.
- [`FFHQ`](./notebooks/FFHQ.ipynb)
- [`AFHQ`](./notebooks/AFHQ.ipynb)
- [`ShapeNet Cars`](./notebooks/Cars.ipynb)


### Pretrained Networks
You can download pretrained networks from here: [](), and put them in the  [`checkpoints`](./checkpoints/)` folder.



## Preparing datasets

Datasets are prepared with the [`dataset_preprocessing`](https://github.com/NVlabs/eg3d/tree/main/dataset_preprocessing) scripts from EG3D. The dataset requires camera poses and intrinsics for every image. 

## Training

The training script lies in [`train.py`](./train.py). The training parameters are exactly the same as those of StyleGAN3. Examples of training scripts are stored in the [`slurm_scripts`](./slurm_scripts/) folder. Configurations are provided for both StyleGAN2 and StyleGAN3, and are labeled as:
- `raycond2`
- `raycond3-t`
- `raycond3-r`

Here is an example training command for FFHQ:
```
python train.py --outdir=training-runs --data=/path/to/eg3d-ffhq.zip --cfg=raycond2 --gpus=2 --batch=32 --gamma=1 --snap=20 --cond=1 --aug=noaug --resume=checkpoints/stylegan2-ffhq-512x512.pkl
```


## Citation

```
@inproceedings{chen2023:ray-conditioning,
  author = {Eric Ming Chen and Sidhanth Holalkere and Ruyu Yan and Kai Zhang and Abe Davis},
  title = {Ray Conditioning: Trading Photo-realism for Photo-Consistency in Multi-view Image Generation},
  booktitle = {ICCV},
  year = {2023}
}
```
import torch
import os
from PIL import Image


def load_lfn_data(img_dir, device="cuda"):
    poses = []
    imgs = []
    for i in range(251):
        img_path = os.path.join(img_dir, "rgb", f"{i:06}.png")
        pose_path = os.path.join(img_dir, "pose", f"{i:06}.txt")
        with open(pose_path, "r") as f:
            pose = torch.tensor(
                [float(n) for n in f.read().split(" ")], device=device
            ).reshape(4, 4)

        poses.append(pose)
        img = Image.open(img_path).convert("RGB")
        imgs.append(img)
    return imgs, poses


def load_intrinsics(img_dir, device="cuda"):
    intrinsics_path = os.path.join(img_dir, "intrinsics.txt")
    with open(intrinsics_path, "r") as f:
        first_line = f.read().split("\n")[0].split(" ")
        focal = float(first_line[0])
        cx = float(first_line[1])
        cy = float(first_line[2])

        orig_img_size = (
            512  # cars_train has intrinsics corresponding to image size of 512 * 512
        )
        intrinsics = torch.tensor(
            [
                [focal / orig_img_size, 0.00000000e00, cx / orig_img_size],
                [0.00000000e00, focal / orig_img_size, cy / orig_img_size],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ],
            device=device,
        )
    return intrinsics


def tensor2im(var):
    """
    Converts a tensor image to PIL Image. Taken from the stylegan2-ada-pytorch repo
    Arguments:
        var (Tensor): Tensor representing the input image

    Returns:
        image (PIL.Image): Image displayed

    """
    var = (var.permute(1, 2, 0) * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(var.cpu().numpy(), "RGB")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
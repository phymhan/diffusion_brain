# import random
from pathlib import Path
import cv2
import mediapy
import numpy as np
import torch
from skimage import img_as_ubyte
from models.ddim import DDIMSampler
import fire


def sample_fn(
    cond,
    vae,
    model,
    device="cuda",
):
    latent_shape = [1, 3, 20, 28, 20]
    # cond_crossatten = cond.unsqueeze(1)
    # cond_concat = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(latent_shape[2:]))
    # conditioning = {
    #     "c_concat": [cond_concat.float().to(device)],
    #     "c_crossattn": [cond_crossatten.float().to(device)],
    # }
    conditioning = model.get_learned_conditioning(cond.to(device))

    ddim = DDIMSampler(model)
    num_timesteps = 50
    latent_vectors, _ = ddim.sample(
        num_timesteps,
        conditioning=conditioning,
        batch_size=1,
        shape=list(latent_shape[1:]),
        eta=0,
    )

    with torch.no_grad():
        x_hat = vae.reconstruct_ldm_outputs(latent_vectors).cpu()

    return x_hat.numpy()


def main(
    gender = 1.0,
    age = 50.0,
    ventricular = 0.5,
    brain = 0.5,
    output_dir = "./outputs/",
    device = "cuda",
    verbose = True,
):
    # Load model
    device = torch.device(device)
    vae = torch.load("./trained_models/vae/data/model.pth")
    vae.to(device)
    vae.eval()

    # model = torch.load("./trained_models/ddpm/data/model.pth")
    from omegaconf import OmegaConf
    from utils import instantiate_from_config
    config = OmegaConf.load("./config.yaml")
    model = instantiate_from_config(config.model)
    sd = torch.load("./new_model.pth")
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.to(device)
    model.eval()

    age = (age - 44) / (82 - 44)
    cond = torch.Tensor([[gender, age, ventricular, brain]])

    image_data = sample_fn(cond, vae, model, device=device)
    image_data = image_data[0, 0, 5:-5, 5:-5, :-15]
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    image_data = (image_data * 255).astype(np.uint8)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Write frames to video
    with mediapy.VideoWriter(
        f"{str(output_dir)}/brain_axial.mp4", shape=(150, 214), fps=12, crf=18
    ) as w:
        for idx in range(image_data.shape[2]):
            img = image_data[:, :, idx]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    with mediapy.VideoWriter(
        f"{str(output_dir)}/brain_sagittal.mp4", shape=(145, 214), fps=12, crf=18
    ) as w:
        for idx in range(image_data.shape[0]):
            img = np.rot90(image_data[idx, :, :])
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    with mediapy.VideoWriter(
        f"{str(output_dir)}/brain_coronal.mp4", shape=(145, 150), fps=12, crf=18
    ) as w:
        for idx in range(image_data.shape[1]):
            img = np.rot90(np.flip(image_data, axis=1)[:, idx, :])
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)


if __name__ == "__main__":
    fire.Fire(main)

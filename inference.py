# import random
from pathlib import Path
import torch
from models.ddim import DDIMSampler
import fire
from utils import write_numpy_to_video


def sample_fn(
    cond,
    vae,
    model,
    device="cuda",
):
    latent_shape = [1, 3, 20, 28, 20]
    if hasattr(model, 'get_learned_conditioning'):
        conditioning = model.get_learned_conditioning(cond.to(device))
    else:
        cond_crossatten = cond.unsqueeze(1)
        cond_concat = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(latent_shape[2:]))
        conditioning = {
            "c_concat": [cond_concat.float().to(device)],
            "c_crossattn": [cond_crossatten.float().to(device)],
        }

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
    config = None,
    ckpt_ddpm = "./trained_models/ddpm/data/model.pth",
    ckpt_vae = "./trained_models/vae/data/model.pth",
    backend = 'mediapy',
):
    # Load model
    device = torch.device(device)
    vae = torch.load(ckpt_vae)
    vae.to(device)
    vae.eval()

    if config is None:
        model = torch.load(ckpt_ddpm)
    else:  # load from config
        from omegaconf import OmegaConf
        from utils import instantiate_from_config
        config = OmegaConf.load(config)
        model = instantiate_from_config(config.model)
        sd = torch.load(ckpt_ddpm)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)
    model.to(device)
    model.eval()

    age = (age - 44) / (82 - 44)
    cond = torch.Tensor([[gender, age, ventricular, brain]])

    image_data = sample_fn(cond, vae, model, device=device)
    image_data = image_data[0, 0]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    ext = "gif" if backend == "imageio" else "mp4"
    write_numpy_to_video(image_data, output_dir / f"brain_axial.{ext}", direction="axial", backend=backend)
    write_numpy_to_video(image_data, output_dir / f"brain_sagittal.{ext}", direction="sagittal", backend=backend)
    write_numpy_to_video(image_data, output_dir / f"brain_coronal.{ext}", direction="coronal", backend=backend)


if __name__ == "__main__":
    fire.Fire(main)

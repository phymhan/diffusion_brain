# convert trained model to diffusers state_dict format
import os
import torch
import fire

def main(
    unet=True,
    vae=True,
):
    if vae:
        print("Converting VAE model")
        # load model
        model = torch.load("./trained_models/vae/data/model.pth")
        state_dict = model.state_dict()

        # save state_dict
        torch.save(state_dict, "trained_models/pipe/vae/diffusion_pytorch_model.bin")

    if unet:
        print("Converting UNet model")
        # load model
        model = torch.load("./trained_models/ddpm/data/model.pth")

        # remove "model.model.diffusion_model." from keys
        state_dict = model.model.diffusion_model.state_dict()

        # save state_dict
        torch.save(state_dict, "trained_models/pipe/unet/diffusion_pytorch_model.bin")


if __name__ == "__main__":
    fire.Fire(main)

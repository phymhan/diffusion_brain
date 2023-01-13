# Brain Diffusion
Command line version of the [Brain Diffusion](https://huggingface.co/spaces/Warvito/diffusion_brain) app.

Pretrained weights can be downloaded by either cloning the original [huggingface space](https://huggingface.co/spaces/Warvito/diffusion_brain) or manually downloading them:
```
wget https://huggingface.co/spaces/Warvito/diffusion_brain/resolve/main/trained_models/ddpm/data/model.pth
wget https://huggingface.co/spaces/Warvito/diffusion_brain/resolve/main/trained_models/vae/data/model.pth
```

## Inference
```
python inference.py --gender=0 --output_dir="./outputs"
```

Results will be saved in the `outputs` folder.

## Training
If we want to finetune the pretrained model, convert the saved model to a state dict and load it by passing the path to the `--actual_resume` argument.

```
CUDA_VISIBLE_DEVICES=0 python main.py --base config.yaml -t --gpus 0, --actual_resume model_state_dict.pth
```

## Acknowledgements
Code largely borrowed from [Brain Diffusion](https://huggingface.co/spaces/Warvito/diffusion_brain) and [LDM](https://github.com/CompVis/latent-diffusion).
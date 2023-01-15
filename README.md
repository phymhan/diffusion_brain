# Brain Diffusion
Command line version of the [Brain Diffusion](https://huggingface.co/spaces/Warvito/diffusion_brain) app.

Pretrained weights can be downloaded by either cloning the original [huggingface space](https://huggingface.co/spaces/Warvito/diffusion_brain) or manually downloading them:
```
wget https://huggingface.co/spaces/Warvito/diffusion_brain/resolve/main/trained_models/ddpm/data/model.pth
wget https://huggingface.co/spaces/Warvito/diffusion_brain/resolve/main/trained_models/vae/data/model.pth
```

# Diffusers
[Diffusers](https://github.com/huggingface/diffusers) is now supported! First we need to convert the pretrained model to a state dict:
```shell
python convert_state_dict.py
```
## Inference
To run inference:
```python
from pipeline import BrainDiffusionPipeline
from utils import write_numpy_to_video
pipe = BrainDiffusionPipeline.from_pretrained('trained_models/pipe').to('cuda')
image_data = pipe([0, 0.1, 0.5, 0.5])[0]  # prompt = [gender, age, ventricular, brain]; age is normalized as: age = (age - 44) / (82 - 44)
write_numpy_to_video(image_data[0, 0], "brain_axial.mp4", direction="axial")
```
Please refer to `inference_pipe.py` for more details.

## Training
```shell
CUDA_VISIBLE_DEVICES=0 python train_pipe.py \
  --seed=29 \
  --pretrained_model_name_or_path="trained_models/pipe"  \
  --output_dir="logs/train" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000
```

# Latent Diffusion
## Inference
```shell
python inference.py --gender=0 --output_dir="./outputs"
```

Results will be saved in the `outputs` folder.

## Training
If we want to finetune the pretrained model, convert the saved model to a state dict and load it by passing the path to the `--actual_resume` argument.

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --base config.yaml -t --gpus 0, --actual_resume model_state_dict.pth
```

## Acknowledgements
Code largely borrowed from [Brain Diffusion](https://huggingface.co/spaces/Warvito/diffusion_brain) and [LDM](https://github.com/CompVis/latent-diffusion).
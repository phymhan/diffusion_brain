<!-- ---
title: Brain Diffusion
emoji: 🏢🧠
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 3.3.1
app_file: app.py
pinned: true
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference -->

# Brain Diffusion
Command line version of the [Brain Diffusion](https://huggingface.co/spaces/Warvito/diffusion_brain) app.

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
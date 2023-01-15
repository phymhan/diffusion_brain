# import random
from pathlib import Path
import torch
import fire
from utils import write_numpy_to_video


def main(
    gender = 1.0,
    age = 50.0,
    ventricular = 0.5,
    brain = 0.5,
    pretrained_model_name_or_path = "./trained_models/pipe",
    output_dir = "./outputs/",
    device = "cuda",
    backend = 'mediapy',
    seed = 29,
):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(0)

    age = (age - 44) / (82 - 44)
    cond = [gender, age, ventricular, brain]

    from pipeline import BrainDiffusionPipeline
    pipe = BrainDiffusionPipeline.from_pretrained(pretrained_model_name_or_path).to(device)
    image_data = pipe(cond, generator=generator)[0][0, 0]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    ext = "gif" if backend == "imageio" else "mp4"
    write_numpy_to_video(image_data, output_dir / f"brain_axial.{ext}", direction="axial", backend=backend)
    write_numpy_to_video(image_data, output_dir / f"brain_sagittal.{ext}", direction="sagittal", backend=backend)
    write_numpy_to_video(image_data, output_dir / f"brain_coronal.{ext}", direction="coronal", backend=backend)


if __name__ == "__main__":
    fire.Fire(main)

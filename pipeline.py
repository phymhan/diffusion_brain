""" Modified from diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
"""
import importlib
import inspect
import os
import re
from dataclasses import dataclass
import numpy as np
from typing import Callable, List, Optional, Union
from pathlib import Path

import torch
# from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
import diffusers
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers import StableDiffusionPipeline
from diffusers.utils import (
    CONFIG_NAME,
    DIFFUSERS_CACHE,
    ONNX_WEIGHTS_NAME,
    WEIGHTS_NAME,
    BaseOutput,
    deprecate,
    is_accelerate_available,
    is_safetensors_available,
    is_torch_version,
    is_transformers_available,
    logging,
)
from diffusers.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
    "onnxruntime.training": {
        "ORTModule": ["save_pretrained", "from_pretrained"],
    },
    "custom": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
}
ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


@dataclass
class DiffusionPipelineOutput(BaseOutput):
    images: np.ndarray
    nsfw_content_detected: Optional[List[bool]]


class BrainDiffusionPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler, vae):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.ch_mult) - 1)
        self.register_to_config(requires_safety_checker=False)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        # resume_download = kwargs.pop("resume_download", False)
        # force_download = kwargs.pop("force_download", False)
        # proxies = kwargs.pop("proxies", None)
        # local_files_only = kwargs.pop("local_files_only", False)
        # use_auth_token = kwargs.pop("use_auth_token", None)
        # revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        # custom_pipeline = kwargs.pop("custom_pipeline", None)
        # custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        device_map = kwargs.pop("device_map", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        return_cached_folder = kwargs.pop("return_cached_folder", False)

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        cached_folder = pretrained_model_name_or_path

        config_dict = cls.load_config(cached_folder)

        # 2. Load the pipeline class

        # if custom_pipeline is not None:
        #     if custom_pipeline.endswith(".py"):
        #         path = Path(custom_pipeline)
        #         # decompose into folder & file
        #         file_name = path.name
        #         custom_pipeline = path.parent.absolute()
        #     else:
        #         file_name = CUSTOM_PIPELINE_FILE_NAME

        #     pipeline_class = get_class_from_dynamic_module(
        #         custom_pipeline, module_file=file_name, cache_dir=cache_dir, revision=custom_revision
        #     )
        # elif cls != DiffusionPipeline:
        #     pipeline_class = cls
        # else:
        #     diffusers_module = importlib.import_module(cls.__module__.split(".")[0])
        #     pipeline_class = getattr(diffusers_module, config_dict["_class_name"])
        pipeline_class = cls

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        # define init kwargs
        init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        if len(unused_kwargs) > 0:
            logger.warning(
                f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
            )

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        # import it here to avoid circular import
        from diffusers import pipelines

        # 3. Load each module in the pipeline
        for name, (library_name, class_name) in init_dict.items():
            # 3.1 - now that JAX/Flax is an official framework of the library, we might load from Flax names
            if class_name.startswith("Flax"):
                class_name = class_name[4:]

            is_pipeline_module = hasattr(pipelines, library_name)
            loaded_sub_model = None

            # if the model is in a pipeline module, then we load it from the pipeline
            if name in passed_class_obj:
                # 1. check that passed_class_obj has correct parent class
                # pass

                # set passed class object
                loaded_sub_model = passed_class_obj[name]
            elif is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                class_obj = getattr(pipeline_module, class_name)
            else:
                # else we just import it from the library.
                # NOTE: here I reuse library_name as the module name
                library = importlib.import_module(library_name)
                class_obj = getattr(library, class_name)
            
            if loaded_sub_model is None:
                load_method_name = 'from_pretrained'

                load_method = getattr(class_obj, load_method_name)
                loading_kwargs = {}

                if issubclass(class_obj, torch.nn.Module):
                    loading_kwargs["torch_dtype"] = torch_dtype
                if issubclass(class_obj, diffusers.OnnxRuntimeModel):
                    loading_kwargs["provider"] = provider
                    loading_kwargs["sess_options"] = sess_options

                is_diffusers_model = issubclass(class_obj, diffusers.ModelMixin)

                # When loading a transformers model, if the device_map is None, the weights will be initialized as opposed to diffusers.
                # To make default loading faster we set the `low_cpu_mem_usage=low_cpu_mem_usage` flag which is `True` by default.
                # This makes sure that the weights won't be initialized which significantly speeds up loading.
                if is_diffusers_model:
                    loading_kwargs["device_map"] = device_map
                    loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

                # check if the module is in a subdirectory
                if os.path.isdir(os.path.join(cached_folder, name)):
                    loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
                else:
                    # else load from the root directory
                    loaded_sub_model = load_method(cached_folder, **loading_kwargs)

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        # 4. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components
        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 5. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        if return_cached_folder:
            return model, cached_folder
        return model
    
    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device
    
    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance):
        # batch_size = len(prompt)
        cond = torch.FloatTensor(prompt).to(device)
        return cond, None
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, depth, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor, depth // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def decode_latents(self, latents):
        image = self.vae.decode(latents).sample
        image = image.cpu().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[List[List[float]], List[float]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        depth: Optional[int] = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "numpy",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        guidance_scale = 1.0  # NOTE: hardcoded, no classifier free guidance

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size[0] * self.vae_scale_factor
        width = width or self.unet.config.sample_size[1] * self.vae_scale_factor
        depth = depth or self.unet.config.sample_size[2] * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(prompt, height, width, callback_steps)
        assert isinstance(prompt, (list, tuple))
        if not isinstance(prompt[0], (list, tuple)):
            prompt = [prompt]

        # 2. Define call parameters
        batch_size = len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings, uncond_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance
        )
        dtype = text_embeddings.dtype

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        # num_channels_latents = self.unet.in_channels  # NOTE: move concat to unet
        num_channels_latents = self.unet.out_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            depth,
            dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=uncond_embeddings).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        has_nsfw_concept = [False] * len(image)

        # 10. Convert to PIL
        if output_type == "mp4":
            image = self.numpy_to_mp4(image)  # TODO: add support for mp4

        if not return_dict:
            return (image, has_nsfw_concept)

        return DiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

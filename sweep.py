# The default TQDM usage in the following packages are offensive, mute all of them.
# Unless it knows about casting magic.
import tqdm
TQDM_MAGIC = 0xAE
class _TQDM(tqdm.tqdm):
    def __init__(self, *argv, **kwargs):
        kwargs['disable'] = True
        if kwargs.get('magic', None) == TQDM_MAGIC:
            kwargs['disable'] = False
        kwargs.pop('magic', None)
        super().__init__(*argv, **kwargs)
def real_tqdm(*arg, **kwargs):
    kwargs["magic"] = TQDM_MAGIC
    kwargs["ascii"] = True
    if "desc" in kwargs:
        kwargs["desc"] = kwargs["desc"].rjust(12)
    return tqdm.tqdm(*arg, **kwargs)
tqdm.tqdm = _TQDM

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import os
import PIL
import numpy as np
import copy
import torch
from glob import glob
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from sane_utils import seed_everything
import torch.nn.functional as F
from scripts.util_image import ImageSpliterTh
import threading
import math

from ldm.util import instantiate_from_config
from scripts.wavelet_color_fix import (
    wavelet_reconstruction,
    adaptive_instance_normalization,
)

from cog import BasePredictor, Input, Path

# CONFIG
UNET_CONFIG = "configs/stableSRNew/v2-finetune_text_T_768v.yaml"
UNET_CHECKPOINT = "checkpoints/stablesr_768v_000139.ckpt"
VQGAN_CHECKPOINT = "checkpoints/vqgan_cfw_00011.ckpt"
TILE_SIZE = 768    # Model's native size is preferred?
IMAGE_SIZE_INCREMENT = 32    # Pad (i.e. align) image size to multiples of this value
SCALE = 4

# WARN: VRAM intensive
VQGAN_SIZE = 1536     # Size to VRAM: 1024 -> 15GB ; 1536 -> 19GB; 1736 -> 21G; 2048 -> 24G;
VQGAN_STRIDE = 1472   # VQGAN - 64

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        config = OmegaConf.load(UNET_CONFIG)
        self.model = load_model_from_config(config, UNET_CHECKPOINT)
        device = torch.device("cuda")

        self.model.configs = config
        self.model = self.model.to(device)

        vqgan_config = OmegaConf.load(
            "configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml"
        )
        self.vq_model = load_model_from_config(vqgan_config, VQGAN_CHECKPOINT)
        self.vq_model = self.vq_model.to(device)

    # `image` and `output_image` are fp32 nchw within range [-1, 1]
    def predict(
        self,
        image: torch.Tensor,
        ddpm_steps: int = Input(
            description="Number of DDPM steps for sampling", default=200
        ),
        fidelity_weight: float = Input(
            description="Balance the quality (lower number) and fidelity (higher number)",
            default=0.5,
        ),
        upscale: float = Input(
            description="The upscale for super-resolution, 4x SR by default",
            default=4.0,
        ),
        tile_overlap: int = Input(
            description="The overlap between tiles, betwwen 0 to 64",
            ge=0,
            le=64,
            default=32,
        ),
        colorfix_type: str = Input(
            choices=["adain", "wavelet", "none"], default="adain"
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        pbar_tooltip: str = '',
    ) -> torch.Tensor:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        self.vq_model.decoder.fusion_w = fidelity_weight

        seed_everything(seed)

        n_samples = 1
        device = torch.device("cuda")

        cur_image = image.to(device)
        cur_image = F.interpolate(
            cur_image,
            size=(int(cur_image.size(-2) * upscale), int(cur_image.size(-1) * upscale)),
            mode="bicubic",
        )

        self.model.register_schedule(
            given_betas=None,
            beta_schedule="linear",
            timesteps=1000,
            linear_start=0.00085,
            linear_end=0.0120,
            cosine_s=8e-3,
        )
        self.model.num_timesteps = 1000

        sqrt_alphas_cumprod = copy.deepcopy(self.model.sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = copy.deepcopy(
            self.model.sqrt_one_minus_alphas_cumprod
        )

        use_timesteps = set(space_timesteps(1000, [ddpm_steps]))
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(self.model.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        new_betas = [beta.data.cpu().numpy() for beta in new_betas]
        self.model.register_schedule(
            given_betas=np.array(new_betas), timesteps=len(new_betas)
        )
        self.model.num_timesteps = 1000
        self.model.ori_timesteps = list(use_timesteps)
        self.model.ori_timesteps.sort()
        self.model = self.model.to(device)

        precision_scope = autocast

        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    init_image = cur_image
                    init_image = init_image.clamp(-1.0, 1.0)
                    ori_size = None

                    if (
                        init_image.size(-1) < TILE_SIZE
                        or init_image.size(-2) < TILE_SIZE
                    ):
                        ori_size = init_image.size()
                        new_h = max(ori_size[-2], TILE_SIZE)
                        new_w = max(ori_size[-1], TILE_SIZE)
                        init_template = torch.zeros(
                            1, init_image.size(1), new_h, new_w
                        ).to(init_image.device)
                        init_template[:, :, : ori_size[-2], : ori_size[-1]] = init_image
                    else:
                        init_template = init_image

                    im_spliter = ImageSpliterTh(cur_image, VQGAN_SIZE, VQGAN_STRIDE, sf=1)
                    pbar = real_tqdm(total=len(im_spliter))
                    pbar.set_description(pbar_tooltip)
                    for im_lq_pch, index_infos in im_spliter:
                        seed_everything(seed)
                        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(im_lq_pch))  # move to latent space
                        text_init = [''] * cur_image.size(0)
                        semantic_c = self.model.cond_stage_model(text_init)
                        noise = torch.randn_like(init_latent)
                        # If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
                        t = repeat(torch.tensor([999]), '1 -> b', b=cur_image.size(0))
                        t = t.to(device).long()
                        x_T = self.model.q_sample_respace(
                            x_start=init_latent,
                            t=t,
                            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                            noise=noise,
                        )
                        samples = self.model.sample_canvas(
                            cond=semantic_c,
                            struct_cond=init_latent,
                            batch_size=im_lq_pch.size(0),
                            timesteps=steps,
                            time_replace=steps,
                            x_T=x_T, return_intermediates=False,
                            tile_size=int(TILE_SIZE/8),
                            tile_overlap=tile_overlap,
                            batch_size_sample=1,
                        )
                        _, enc_fea_lq = torch.compile(self.vq_model.encode)(im_lq_pch)
                        x_samples = torch.compile(self.vq_model.decode)(samples * 1. / self.model.scale_factor, enc_fea_lq)
                        if colorfix_type == 'adain':
                            x_samples = torch.compile(adaptive_instance_normalization)(x_samples, im_lq_pch)
                        elif colorfix_type == 'wavelet':
                            x_samples = torch.compile(wavelet_reconstruction)(x_samples, im_lq_pch)
                        im_spliter.update(x_samples, index_infos)
                        pbar.update()

                    im_sr = im_spliter.gather()

                    return torch.clamp(im_sr, min=-1.0, max=1.0)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def read_image(im_path):
    im = np.array(Image.open(im_path).convert("RGB"))
    im = im.astype(np.float32) / 255.0
    im = im[None].transpose(0, 3, 1, 2)
    im = (torch.from_numpy(im) - 0.5) / 0.5

    return im.cuda()


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]  # [250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


if __name__ == '__main__':
    p = Predictor()
    p.setup()

    images = sorted(glob("/argo/sts/sr-in/references/*"))
    images = filter(lambda p: os.path.splitext(p)[1].lower() in [".jpg", ".png"], images)
    images = list(images)[4:]

    for ref_img_path in images:
        basename, _ = os.path.splitext(os.path.basename(ref_img_path))
        ref_img = Image.open(ref_img_path).convert("RGB")
        w, h = ref_img.size

        for prescale in [1, 2]:
            image = ref_img.resize((w//prescale, h//prescale), resample=PIL.Image.LANCZOS)

            # Pad image so each dimension is a multiple of 32 for the model.
            pw, ph = map(lambda x: math.ceil((x // prescale) / IMAGE_SIZE_INCREMENT) * IMAGE_SIZE_INCREMENT, (w, h))
            padded_image = Image.new(image.mode, (pw, ph), (0, 0, 0))
            padded_image.paste(image, (0,0))

            # Prepare Tensor
            padded_image = np.array(padded_image).astype(np.float32)
            padded_image = padded_image / 255.0 * 2.0 - 1.0
            padded_image = padded_image.transpose(2, 0, 1)
            padded_image = torch.from_numpy(padded_image[None])

            for seed in [42, 174, 0xAEAE]:
                for fidelity in [0, 0.25, 0.5, 0.75, 1]:
                    for steps in [10, 20, 40, 80, 160, 240]:
                        key = f"{basename}__scale{prescale}_seed{seed}_steps{steps}_fidelity{fidelity}"

                        output_image = p.predict(
                            padded_image,
                            ddpm_steps = steps,
                            fidelity_weight = fidelity,
                            upscale = SCALE,
                            seed = seed,
                            tile_overlap = 32,
                            colorfix_type = None,
                            pbar_tooltip = key
                        )

                        output_image = torch.clamp((output_image + 1.0) / 2.0 * 255.0, min=0.0, max=255.0)
                        output_image = output_image[0].cpu().numpy().transpose(1, 2, 0)
                        output_image = output_image[:(h//prescale*SCALE),:(w//prescale*SCALE),:]

                        # Write image in the background.
                        save_path = f"/argo/sts/sr-out/references/{key}.png"
                        def write_image(path, numpyImgUint8):
                            Image.fromarray(numpyImgUint8).save(path)
                        threading.Thread(
                            target=write_image,
                            args=(save_path, output_image.astype(np.uint8)),
                        ).start()


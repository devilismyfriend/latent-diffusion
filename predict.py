# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from datetime import datetime
import tempfile, typing
import subprocess

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

sys.path.append("latent-diffusion")

ckpt = "/root/.cache/ldm/text2img-large/model.ckpt"

sizes = [128, 256, 384, 448, 512]


class Predictor(BasePredictor):
    def setup(self):
        subprocess.call(["pip", "install", "-e", "."])
        global config, model, device
        device = torch.device("cuda")
        config = OmegaConf.load("/src/configs/latent-diffusion/txt2img-1p4B-eval.yaml")
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cuda")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)

        model.cuda()
        model.eval()

    def predict(
        self,
        prompt: str = Input(description="Text prompt"),
        scale: float = Input(description="Unconditional guidance scale", default=15.0),
        steps: int = Input(description="Number of sampling steps", default=75),
        plms: bool = Input(description="Use PLMS", default=True),
        eta: float = Input(description="eta for ddim sampling", default=0.0),
        n_samples: int = Input(description="How many samples to produce per iteration", default=3),
        n_iter: int = Input(description="Number of iterations", default=1),
        height: int = Input(description="Height", default=256, choices=sizes),
        width: int = Input(description="Width", default=256, choices=sizes),
    ) -> typing.List[Path]:
        global config, model
        outpath = tempfile.mkdtemp()
        if plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))

        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(n_samples * [""])
                for n in trange(n_iter, desc="Sampling"):
                    c = model.get_learned_conditioning(n_samples * [prompt])
                    shape = [4, height // 8, width // 8]
                    samples_ddim, _ = sampler.sample(
                        S=steps,
                        conditioning=c,
                        batch_size=n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=eta,
                    )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                        sample_file = os.path.join(sample_path, f"{base_count:04}.png")
                        Image.fromarray(x_sample.astype(np.uint8)).save(sample_file)
                        base_count += 1
                        yield Path(sample_file)

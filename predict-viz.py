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
from torchvision.transforms import functional as TF
from datetime import datetime
import tempfile, typing
import subprocess

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

sys.path.append("latent-diffusion")

ckpt = "/root/.cache/ldm/text2img-large/model.ckpt"

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
        image: Path = Input(description="Image")
    ) -> typing.List[Path]:
        global config, model
        output_path = tempfile.mkdtemp()
        ix = 0
        with torch.no_grad():
            with model.ema_scope():
                img = Image.open(str(image))
                img = img.resize((256,256),1)    
                # Remove alpha channel if present                
                if img.mode =="RGBA":
                    bg = Image.new("RGBA", (256,256), (255, 255, 255))
                    img = Image.alpha_composite(bg, img).convert("RGB")                
                resized_file = f'{output_path}/{ix}-resized.png'
                img.save(resized_file)                
                img = TF.to_tensor(img).to(device)[None] * 2 - 1
                latent = model.first_stage_model.encode(img)
                sample = latent.mode()
                sample = torch.nn.functional.normalize(sample)[0]                  
                out_main = 255 * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
                filename = f'{output_path}/{ix}-latent.png'        
                Image.fromarray(out_main.astype(np.uint8)).resize((256,256), 0).save(filename)    
                yield Path(filename)
                for i in [0,1,2,3]:
                    out = 255 * sample[i].cpu().numpy()    
                    filename = f'{output_path}/{ix}-latent-{i}.png'        
                    Image.fromarray(out.astype(np.uint8)).resize((256,256), 0).save(filename)    
                    yield Path(filename)
                ix = ix + 1
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from dkm import DKMv2
from dkm.utils.utils import tensor_to_pil

raise DeprecationWarning("Old demo, not usable at the moment")
dkm_model = DKMv2(pretrained=True, version="outdoor")

im1 = Image.open(f"assets/sacre_coeur_multimodal_query.jpg").resize((512, 384))
im2 = Image.open(f"assets/sacre_coeur_multimodal_support.jpg").resize((512, 384))
im1.save(f"demo/sacre_coeur_query.jpg")
im2.save(f"demo/sacre_coeur_support.jpg")
flow, confidence = dkm_model.match(im1, im2)
confidence = confidence ** (1 / 2)
c_b = confidence / confidence.max()
x2 = (torch.tensor(np.array(im2)) / 255).cuda().permute(2, 0, 1)
im2_transfer_rgb = F.grid_sample(
    x2[None], flow[..., 2:][None], mode="bicubic", align_corners=False
)[0]
white_im = torch.ones_like(x2)
tensor_to_pil(c_b * im2_transfer_rgb + (1 - c_b) * white_im, unnormalize=False).save(
    f"demo/sacre_coeur_warped.jpg"
)

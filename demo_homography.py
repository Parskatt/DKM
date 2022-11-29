from PIL import Image
import torch
import torch.nn.functional as F
from dkm import DKMv2
from dkm.utils.utils import tensor_to_pil
import cv2
from torchvision import transforms

raise DeprecationWarning("Old demo, not usable at the moment")
dkm_model = DKMv2(pretrained=True, version="outdoor")

im1 = Image.open(f"assets/ams_hom_left.jpg").resize((512, 384))
im2 = Image.open(f"assets/ams_hom_right.jpg").resize((512, 384))
im1.save(f"demo/ams_hom_left.jpg")
im2.save(f"demo/ams_hom_right.jpg")
flow, confidence = dkm_model.match(im1, im2)
confidence = confidence ** (1 / 2)
good_matches, _ = dkm_model.sample(flow, confidence, 10000)

H, inliers = cv2.findHomography(
    good_matches[..., :2],
    good_matches[..., 2:],
    method=cv2.RANSAC,
    confidence=0.99999,
    ransacReprojThreshold=1.5 / 512,
)
H = torch.tensor(H).cuda()
to_tensor = transforms.ToTensor()
im1, im2 = to_tensor(im1), to_tensor(im2).cuda()
target_im = torch.zeros((3, 384, 512 * 2)).cuda()
target_im[:, :, :512] = im1.cuda()
h, w = 384, 512
x1_coords = torch.meshgrid(
    (
        torch.linspace(-1 + 1 / h, 1 - 1 / h, 384, device="cuda"),
        torch.linspace(1 + 1 / w, 3 - 1 / w, w, device="cuda"),
    )
)
x1_coords = torch.stack((x1_coords[1], x1_coords[0]), dim=-1)
x2_coords = torch.einsum(
    "dc, hwc -> hwd",
    H,
    torch.cat((x1_coords, torch.ones_like(x1_coords[..., -1:])), dim=-1).double(),
)
x2_coords = x2_coords[..., :-1] / x2_coords[..., -1:]
target_im[:, :, 512:] = F.grid_sample(
    im2[None], x2_coords[None].float(), mode="bicubic", align_corners=False
)[0]
tensor_to_pil(target_im).save("demo/stiched_ams.jpg")

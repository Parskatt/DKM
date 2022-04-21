import torch
import torch.nn as nn
from .dkm import *
from torchvision.models import resnet as tv_resnet
from torchvision.models import ResNet

dkm_pretrained_urls = {"base": {"v11":"https://github.com/Parskatt/storage/releases/download/dkm/dkm_base_v11.pth"}, "v1": ""}


def dkm_base(pretrained=True, version="v11"):
    gp_dim = 256
    dfn_dim = 384
    feat_dim = 256
    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "32": nn.Conv2d(512, feat_dim, 1, 1),
                "16": nn.Conv2d(512, feat_dim, 1, 1),
            }
        ),
        pred_input_modules=nn.ModuleDict(
            {
                "32": nn.Identity(),
                "16": nn.Identity(),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "32": RRB(gp_dim + feat_dim, dfn_dim),
                "16": RRB(gp_dim + feat_dim, dfn_dim),
            }
        ),
        cab_dict=nn.ModuleDict(
            {
                "32": CAB(2 * dfn_dim, dfn_dim),
                "16": CAB(2 * dfn_dim, dfn_dim),
            }
        ),
        rrb_u_dict=nn.ModuleDict(
            {
                "32": RRB(dfn_dim, dfn_dim),
                "16": RRB(dfn_dim, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "32": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
                "16": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
            }
        ),
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512,
                1024,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "8": ConvRefiner(
                2 * 512,
                1024,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "4": ConvRefiner(
                2 * 256,
                512,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "2": ConvRefiner(
                2 * 64,
                128,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
            "1": ConvRefiner(
                2 * 3,
                24,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp32 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"32": gp32, "16": gp16})
    proj = nn.ModuleDict(
        {"16": nn.Conv2d(1024, 512, 1, 1), "32": nn.Conv2d(2048, 512, 1, 1)}
    )
    decoder = Decoder(coordinate_decoder, gps, proj, conv_refiner, detach=True)
    h, w = 384, 512
    encoder = Encoder(
        tv_resnet.resnet50(pretrained=not pretrained)
    )  # only load pretrained weights if not loading a pretrained matcher ;)
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w).cuda()
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(dkm_pretrained_urls["base"][version])
        matcher.load_state_dict(weights)
    return matcher

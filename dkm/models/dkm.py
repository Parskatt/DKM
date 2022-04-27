import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from einops import rearrange


def get_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize))
    if normalize:
        ops.append(TupleToTensorScaled())
        ops.append(
            TupleNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )  # Imagenet mean/std
    else:
        if unscale:
            ops.append(TupleToTensorUnscaled())
        else:
            ops.append(TupleToTensorScaled())
    return TupleCompose(ops)


class ToTensorScaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]"""

    def __call__(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        return "ToTensorScaled(./255)"


class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorScaled(./255)"


class ToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __call__(self, im):
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return "ToTensorUnscaled()"


class TupleToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __init__(self):
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorUnscaled()"


class TupleResize(object):
    def __init__(self, size, mode=InterpolationMode.BICUBIC):
        self.size = size
        self.resize = transforms.Resize(size, mode)

    def __call__(self, im_tuple):
        return [self.resize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleResize(size={})".format(self.size)


class TupleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        return [self.normalize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleNormalize(mean={}, std={})".format(self.mean, self.std)


class TupleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple):
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        bicubic_gradients=False,
        dw=False,
        residual=False,
        groups=1,
        kernel_size=5,
        hidden_blocks=3,
        weight_norm=False,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, groups=groups, kernel_size=kernel_size
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    groups=groups,
                    kernel_size=kernel_size,
                    weight_norm=weight_norm,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        self.bicubic_gradients = bicubic_gradients
        self.residual = residual

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        groups=1,
        residual=False,
        weight_norm=False,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
        )
        norm = nn.BatchNorm2d(out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, x, y, flow):
        """Computes the relative refining displacement in pixels for a given image x,y and a coarse flow-field between them

        Args:
            x ([type]): [description]
            y ([type]): [description]
            flow ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.bicubic_gradients:
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False)
        else:
            with torch.no_grad():
                x_hat = F.grid_sample(
                    y, flow.permute(0, 2, 3, 1), align_corners=False
                )  # Propagating gradients through grid_sample seems to be unstable :/
        d = torch.cat((x, x_hat), dim=1)
        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d)
        certainty, displacement = d[:, :-2], d[:, -2:]
        return certainty, displacement


class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low (old, new)
        x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class RRB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)


class DFN(nn.Module):
    def __init__(
        self,
        internal_dim,
        feat_input_modules,
        pred_input_modules,
        rrb_d_dict,
        cab_dict,
        rrb_u_dict,
        use_global_context=False,
        global_dim=None,
        terminal_module=None,
        upsample_mode="bilinear",
        align_corners=False,
    ):
        super().__init__()
        if use_global_context:
            assert (
                global_dim is not None
            ), "Global dim must be provided when using global context"
        self.align_corners = align_corners
        self.internal_dim = internal_dim
        self.feat_input_modules = feat_input_modules
        self.pred_input_modules = pred_input_modules
        self.rrb_d = rrb_d_dict
        self.cab = cab_dict
        self.rrb_u = rrb_u_dict
        self.use_global_context = use_global_context
        if use_global_context:
            self.global_to_internal = nn.Conv2d(global_dim, self.internal_dim, 1, 1, 0)
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.terminal_module = (
            terminal_module if terminal_module is not None else nn.Identity()
        )
        self.upsample_mode = upsample_mode
        self._scales = [int(key) for key in self.terminal_module.keys()]

    def scales(self):
        return self._scales.copy()

    def forward(self, new_stuff, feats, old_stuff, key):
        feats = self.feat_input_modules[str(key)](feats)
        new_stuff = torch.cat([feats, new_stuff], dim=1)
        new_stuff = self.rrb_d[str(key)](new_stuff)
        new_old_stuff = self.cab[str(key)](
            [old_stuff, new_stuff]
        )  # TODO: maybe use cab output upwards in network instead of rrb_u?
        new_old_stuff = self.rrb_u[str(key)](new_old_stuff)
        preds = self.terminal_module[str(key)](new_old_stuff)
        pred_coord = preds[:, -2:]
        pred_certainty = preds[:, :-2]
        return pred_coord, pred_certainty, new_old_stuff


class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1)
            ),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2), torch.arange(K // 2, w + K // 2)
            ),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        else:
            raise ValueError(
                "No other bases other than fourier currently supported in public release"
            )

    def get_pos_enc(self, y):
        b, c, h, w = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            )
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, x, y, **kwargs):
        b, c, h, w = y.shape
        f = self.get_pos_enc(y)
        b, d, h, w = f.shape
        assert x.shape == y.shape
        x, y, f = self.reshape(x), self.reshape(y), self.reshape(f)
        K_xx = self.K(x, x)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h * w, device=x.device)[None, :, :]
        K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)

        mu_x = K_xy.matmul(K_yy_inv.matmul(f))
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h, w=w)
        if not self.no_cov:
            cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
            cov_x = rearrange(cov_x, "b (h w) (r c) -> b h w r c", h=h, w=w, r=h, c=w)
            local_cov_x = self.get_local_cov(cov_x)
            local_cov_x = rearrange(local_cov_x, "b h w K -> b K h w")
            gp_feats = torch.cat((mu_x, local_cov_x), dim=1)
        else:
            gp_feats = mu_x
        return gp_feats


class Encoder(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet

    def forward(self, x):
        x0 = x

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x1 = self.resnet.relu(x)

        x = self.resnet.maxpool(x1)
        x2 = self.resnet.layer1(x)

        x3 = self.resnet.layer2(x2)

        x4 = self.resnet.layer3(x3)

        x5 = self.resnet.layer4(x4)

        return {32: x5, 16: x4, 8: x3, 4: x2, 2: x1, 1: x0}


class Decoder(nn.Module):
    def __init__(self, embedding_decoder, gps, proj, conv_refiner, detach=False, scales="all"):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales

    def upsample_preds(self, flow, certainty, query, support):
        b, hs, ws, d = flow.shape
        b, c, h, w = query.shape

        flow = flow.permute(0, 3, 1, 2)
        certainty = F.interpolate(
            certainty, size=query.shape[-2:], align_corners=False, mode="bilinear"
        )
        flow = F.interpolate(
            flow, size=query.shape[-2:], align_corners=False, mode="bilinear"
        )
        for k in range(3):
            delta_certainty, delta_flow = self.conv_refiner["1"](query, support, flow)
            boi = delta_certainty.sigmoid()
            delta_flow = boi * delta_flow
            flow = torch.stack(
                (
                    flow[:, 0] + delta_flow[:, 0] / (4 * w),
                    flow[:, 1] + delta_flow[:, 1] / (4 * h),
                ),
                dim=1,
            )
            certainty = (
                certainty + delta_certainty
            )  # predict both certainty and displacement
        flow = flow.permute(0, 2, 3, 1)
        return flow, certainty

    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords

    def forward(self, f1, f2):
        """[summary]

        Args:
            f1:
            f2:
        Returns:
            dict:
                scale (s32,s16,s8,s4,s2,s1):
                    dense_flow: (b,n,h_scale,w_scale)
                    dense_certainty: (b,h_scale,w_scale,2)

        """
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[32].shape[0]
        device = f1[32].device
        old_stuff = torch.zeros(
            b, self.embedding_decoder.internal_dim, *sizes[32], device=f1[32].device
        )
        dense_corresps = {}
        dense_flow = self.get_placeholder_flow(b, *sizes[32], device)
        dense_certainty = 0.0

        for new_scale in all_scales:
            ins = int(new_scale)
            f1_s, f2_s = f1[ins], f2[ins]

            if new_scale in self.proj:
                f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff, size=sizes[ins], mode="bilinear", align_corners=False
                )
                new_stuff = self.gps[new_scale](f1_s, f2_s, dense_flow=dense_flow)
                dense_flow, dense_certainty, old_stuff = self.embedding_decoder(
                    new_stuff, f1_s, old_stuff, new_scale
                )

            if new_scale in self.conv_refiner:
                delta_certainty, displacement = self.conv_refiner[new_scale](
                    f1_s, f2_s, dense_flow
                )  # TODO: concat dense certainty?
                dense_flow = torch.stack(
                    (
                        dense_flow[:, 0] + ins * displacement[:, 0] / (4 * w),
                        dense_flow[:, 1] + ins * displacement[:, 1] / (4 * h),
                    ),
                    dim=1,
                )  # multiply with scale
                dense_certainty = (
                    dense_certainty + delta_certainty
                )  # predict both certainty and displacement

            dense_corresps[ins] = {
                "dense_flow": dense_flow,
                "dense_certainty": dense_certainty,
            }

            if new_scale != "1":
                dense_flow = F.interpolate(
                    dense_flow,
                    size=sizes[ins // 2],
                    align_corners=False,
                    mode="bilinear",
                )

                dense_certainty = F.interpolate(
                    dense_certainty,
                    size=sizes[ins // 2],
                    align_corners=False,
                    mode="bilinear",
                )
                if self.detach:
                    dense_flow = dense_flow.detach()
                    dense_certainty = dense_certainty.detach()
        return dense_corresps


class RegressionMatcher(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=384,
        w=512,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.w_resized = w
        self.h_resized = h
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)

    def extract_backbone_features(self, batch):
        x_q = batch["query"]
        x_s = batch["support"]
        X = torch.cat((x_q, x_s))
        feature_pyramid = self.encoder(X)
        return feature_pyramid
    
    def sample(self, dense_matches, dense_certainty, num = 20000, relative_confidence_threshold = 0.0):
        matches, certainty = (
            dense_matches.reshape(-1, 4).cpu().numpy(),
            dense_certainty.reshape(-1).cpu().numpy(),
        )
        relative_confidence = certainty/certainty.max()
        matches, certainty = (
            matches[relative_confidence > relative_confidence_threshold],
            certainty[relative_confidence > relative_confidence_threshold],
        )
        good_samples = np.random.choice(
            np.arange(len(matches)),
            size=min(num, len(certainty)),
            replace=False,
            p=certainty/np.sum(certainty),
        )
        return matches[good_samples], certainty[good_samples]

    def forward(self, batch):
        feature_pyramid = self.extract_backbone_features(batch)
        f_q_pyramid = {
            scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
        }
        f_s_pyramid = {
            scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
        }
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid)
        return dense_corresps

    def forward_symmetric(self, batch):
        feature_pyramid = self.extract_backbone_features(batch)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]))
            for scale, f_scale in feature_pyramid.items()
        }
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid)
        return dense_corresps

    def cycle_constraint(self, query_coords, query_to_support, support_to_query):
        dist = query_coords - F.grid_sample(
            support_to_query, query_to_support, mode="bilinear"
        )
        return dist

    def stable_neighbours(self, query_coords, query_to_support, support_to_query):
        qts = query_to_support
        for t in range(4):
            _qts = qts
            q = F.grid_sample(support_to_query, qts, mode="bilinear")
            qts = F.grid_sample(
                query_to_support.permute(0, 3, 1, 2),
                q.permute(0, 2, 3, 1),
                mode="bilinear",
            ).permute(0, 2, 3, 1)
        d = (qts - _qts).norm(dim=-1)
        qd = (q - query_coords).norm(dim=1)
        stabneigh = torch.logical_and(d < 1e-3, qd < 5e-3)
        return q, qts, stabneigh

    def match(
        self,
        im1,
        im2,
        batched=False,
        check_cycle_consistency=False,
        do_pred_in_og_res=False,
    ):
        self.train(False)
        with torch.no_grad():
            if not batched:
                b = 1
                w, h = im1.size
                w2, h2 = im2.size
                # Get images in good format
                ws = self.w_resized
                hs = self.h_resized
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                query, support = test_transform((im1, im2))
                batch = {"query": query[None].cuda(), "support": support[None].cuda()}
            else:
                b, c, h, w = im1.shape
                b, c, h2, w2 = im2.shape
                assert w == w2 and h == h2, "wat"
                batch = {"query": im1.cuda(), "support": im2.cuda()}
                hs, ws = self.h_resized, self.w_resized
            finest_scale = 1  # i will assume that we go to the finest scale (otherwise min(list(dense_corresps.keys())) also works)
            # Run matcher
            if check_cycle_consistency:
                dense_corresps = self.forward_symmetric(batch)
                query_to_support, support_to_query = dense_corresps[finest_scale][
                    "dense_flow"
                ].chunk(2)
                query_to_support = query_to_support.permute(0, 2, 3, 1)
                dense_certainty, dc_s = dense_corresps[finest_scale][
                    "dense_certainty"
                ].chunk(
                    2
                )  # TODO: Here we could also use the reverse certainty
            else:
                dense_corresps = self.forward(batch)
                query_to_support = dense_corresps[finest_scale]["dense_flow"].permute(
                    0, 2, 3, 1
                )
                # Get certainty interpolation
                dense_certainty = dense_corresps[finest_scale]["dense_certainty"]

            if do_pred_in_og_res:  # Will assume that there is no batching going on.
                og_query, og_support = self.og_transforms((im1, im2))
                query_to_support, dense_certainty = self.decoder.upsample_preds(
                    query_to_support,
                    dense_certainty,
                    og_query.cuda()[None],
                    og_support.cuda()[None],
                )
                hs, ws = h, w
            # Create im1 meshgrid
            query_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device="cuda:0"),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device="cuda:0"),
                )
            )
            query_coords = torch.stack((query_coords[1], query_coords[0]))
            query_coords = query_coords[None].expand(b, 2, hs, ws)
            dense_certainty = dense_certainty.sigmoid()  # logits -> probs
            if check_cycle_consistency:
                query_coords, query_to_support, stabneigh = self.stable_neighbours(
                    query_coords, query_to_support, support_to_query
                )
                dense_certainty *= stabneigh.float() + 1e-3
            # Return only matches better than threshold
            query_coords = query_coords.permute(0, 2, 3, 1)

            query_to_support = torch.clamp(query_to_support, -1, 1)
            if batched:
                return torch.cat((query_coords, query_to_support), dim=-1), dense_certainty[:, 0]
            else:
                return torch.cat((query_coords, query_to_support), dim=-1)[0], dense_certainty[0, 0]

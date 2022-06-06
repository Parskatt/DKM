from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class HomographyRegressionLoss(nn.Module):
    def __init__(
        self, ce_weight=1.0, local_loss=True, local_dist=4.0, local_largest_scale=8
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale

    def dense_homog_loss(self, dense_certainty, prob, flow_dist, scale, eps=1e-8):
        ce_loss = F.binary_cross_entropy_with_logits(
            dense_certainty[:, 0], prob
        )  # CE loss multiplied by 1 xd
        flow_loss = flow_dist[prob > 0]
        if not torch.any(prob > 0).item():
            flow_loss = (
                flow_dist * 0.0
            ).mean()  # Prevent issues where prob is 0 everywhere
        return {
            f"ce_loss_{scale}": ce_loss.mean(),
            f"flow_loss_{scale}": flow_loss.mean(),
        }

    def forward(self, dense_corresps, batch):
        scales = list(dense_corresps.keys())
        tot_loss = 0.0
        prev_flow_dist = 0.0
        gt_flow = batch["flow"]
        mask = batch["mask"]
        for scale in scales:
            dense_scale_corresps = dense_corresps[scale]
            certainty, flow = (
                dense_scale_corresps["dense_certainty"],
                dense_scale_corresps["dense_flow"],
            )
            flow = rearrange(flow, "b d h w -> b h w d")
            b, h, w, d = flow.shape
            s_prob = F.interpolate(
                mask[:, None], size=(h, w), mode="bilinear", align_corners=False
            )[:, 0]
            s_gt_flow = F.interpolate(
                gt_flow.permute(0, 3, 1, 2),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            flow_dist = (flow - s_gt_flow).norm(dim=-1)
            if (
                scale <= self.local_largest_scale and self.local_loss
            ):  # Thought here is that fine matching loss should not be punished by coarse mistakes, but should identify wrong matching
                s_prob = s_prob * (
                    F.interpolate(prev_flow_dist[:, None], size=(h, w), mode="nearest")[
                        :, 0
                    ]
                    < (2 / 512) * (self.local_dist * scale)
                )
            homog_losses = self.dense_homog_loss(certainty, s_prob, flow_dist, scale)
            scale_loss = (
                self.ce_weight * homog_losses[f"ce_loss_{scale}"]
                + homog_losses[f"flow_loss_{scale}"]
            )  # scale ce loss for coarser scales
            tot_loss = tot_loss + scale_loss
            prev_flow_dist = flow_dist.detach()
        return tot_loss

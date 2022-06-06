import numpy as np
import torch
from dkm.utils import *
from PIL import Image
from tqdm import tqdm


class Megadepth1500Benchmark:
    def __init__(self, data_root="data/megadepth") -> None:
        self.scene_names = [
            "0015_0.1_0.3.npz",
            "0015_0.3_0.5.npz",
            "0022_0.1_0.3.npz",
            "0022_0.3_0.5.npz",
            "0022_0.5_0.7.npz",
        ]
        self.scenes = [
            np.load(f"{data_root}/{scene}", allow_pickle=True)
            for scene in self.scene_names
        ]
        self.data_root = data_root

    def benchmark(self, model, r=2):
        with torch.no_grad():
            data_root = self.data_root
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            for scene_ind in range(len(self.scenes)):
                scene = self.scenes[scene_ind]
                pairs = scene["pair_infos"]
                intrinsics = scene["intrinsics"]
                poses = scene["poses"]
                im_paths = scene["image_paths"]
                pair_inds = range(len(pairs))
                for pairind in tqdm(pair_inds):
                    idx1, idx2 = pairs[pairind][0]
                    K1 = intrinsics[idx1]
                    T1 = poses[idx1]
                    R1, t1 = T1[:3, :3], T1[:3, 3]
                    K2 = intrinsics[idx2]
                    T2 = poses[idx2]
                    R2, t2 = T2[:3, :3], T2[:3, 3]
                    R, t = compute_relative_pose(R1, t1, R2, t2)
                    im1 = im_paths[idx1]
                    im2 = im_paths[idx2]
                    im1 = Image.open(f"{data_root}/{im1}")
                    w1, h1 = im1.size
                    im2 = Image.open(f"{data_root}/{im2}")
                    w2, h2 = im2.size
                    dense_matches, dense_certainty = model.match(
                        im1, im2, do_pred_in_og_res=True
                    )
                    dense_certainty = dense_certainty ** (1 / r)
                    sparse_matches, sparse_confidence = model.sample(
                        dense_matches, dense_certainty, 10000
                    )
                    scale1 = 1200 / max(w1, h1)
                    scale2 = 1200 / max(w2, h2)
                    w1, h1 = scale1 * w1, scale1 * h1
                    w2, h2 = scale2 * w2, scale2 * h2
                    K1 = K1 * scale1
                    K2 = K2 * scale2

                    kpts1 = sparse_matches[:, :2]
                    kpts1 = (
                        np.stack(
                            (
                                w1 * (kpts1[:, 0] + 1) / 2,
                                h1 * (kpts1[:, 1] + 1) / 2,
                            ),
                            axis=-1,
                        )
                        - K1[None, :2, 2]
                    )
                    kpts2 = sparse_matches[:, 2:]
                    kpts2 = (
                        np.stack(
                            (
                                w2 * (kpts2[:, 0] + 1) / 2,
                                h2 * (kpts2[:, 1] + 1) / 2,
                            ),
                            axis=-1,
                        )
                        - K2[None, :2, 2]
                    )
                    try:
                        threshold = 0.5
                        norm_threshold = threshold / (
                            np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2]))
                        )
                        R_est, t_est, mask = estimate_pose(
                            kpts1,
                            kpts2,
                            K1[:2, :2],
                            K2[:2, :2],
                            norm_threshold,
                            conf=0.9999999,
                        )
                        T1_to_2 = np.concatenate((R_est, t_est), axis=-1)  #
                        e_t, e_R = compute_pose_error(T1_to_2, R, t)
                        e_pose = max(e_t, e_R)
                    except:
                        e_t, e_R = 90, 90
                        e_pose = max(e_t, e_R)
                    tot_e_t.append(e_t)
                    tot_e_R.append(e_R)
                    tot_e_pose.append(e_pose)
            tot_e_pose = np.array(tot_e_pose)
            thresholds = [5, 10, 20]
            auc = pose_auc(tot_e_pose, thresholds)
            acc_5 = (tot_e_pose < 5).mean()
            acc_10 = (tot_e_pose < 10).mean()
            acc_15 = (tot_e_pose < 15).mean()
            acc_20 = (tot_e_pose < 20).mean()
            map_5 = acc_5
            map_10 = np.mean([acc_5, acc_10])
            map_20 = np.mean([acc_5, acc_10, acc_15, acc_20])
            return {
                "auc_5": auc[0],
                "auc_10": auc[1],
                "auc_20": auc[2],
                "map_5": map_5,
                "map_10": map_10,
                "map_20": map_20,
            }

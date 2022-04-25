import os.path as osp
import numpy as np
import torch
from dkm.utils import *
from PIL import Image
from tqdm import tqdm


def get_pose(calib):
    w, h = np.array(calib["imsize"])[0]
    return np.array(calib["K"]), np.array(calib["R"]), np.array(calib["T"]).T, h, w


# NOTE: yfcc100m is based on bundler: https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html where the pixel grid is defined on [-n/2, n/2] with the center of the image being at coordinate 0
#
def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


def rotate_intrinsic(K, n):
    base_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rot = np.linalg.matrix_power(base_rot, n)
    return rot @ K


class ScanNetBenchmark:
    def __init__(self, data_root="data/scannet") -> None:
        self.data_root = data_root

    def benchmark_scannet(self, model):
        model.train(False)
        with torch.no_grad():
            data_root = self.data_root
            tmp = np.load(osp.join(data_root, "test.npz"))
            pairs, rel_pose = tmp["name"], tmp["rel_pose"]
            tot_e_t, tot_e_R, tot_e_pose = [], [], []
            pair_inds = np.random.choice(
                range(len(pairs)), size=len(pairs), replace=False
            )
            for pairind in tqdm(pair_inds):
                scene = pairs[pairind]
                scene_name = f"scene0{scene[0]}_00"
                im1 = Image.open(
                    osp.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "color",
                        f"{scene[2]}.jpg",
                    )
                )
                im2 = Image.open(
                    osp.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "color",
                        f"{scene[3]}.jpg",
                    )
                )
                T_gt = rel_pose[pairind].reshape(3, 4)
                R, t = T_gt[:3, :3], T_gt[:3, 3]
                K = np.stack(
                    [
                        np.array([float(i) for i in r.split()])
                        for r in open(
                            osp.join(
                                self.data_root,
                                "scans_test",
                                scene_name,
                                "intrinsic",
                                "intrinsic_color.txt",
                            ),
                            "r",
                        )
                        .read()
                        .split("\n")
                        if r
                    ]
                )
                w1, h1 = im1.size
                w2, h2 = im2.size
                K1 = K.copy()
                K2 = K.copy()
                dense_matches, dense_certainty = model.match(im1, im2)  # , do_pred_in_og_res=True)
                sparse_matches, sparse_certainty = model.sample(dense_matches, dense_certainty, 10000)
                scale1 = 480 / min(w1, h1)
                scale2 = 480 / min(w2, h2)
                w1, h1 = scale1 * w1, scale1 * h1
                w2, h2 = scale2 * w2, scale2 * h2
                K1 *= scale1
                K2 *= scale2

                # best_2000 = np.argsort(confidence)[-2000:]
                offset = 0.5
                kpts1 = sparse_matches[:, :2]
                kpts1 = (
                    np.stack(
                        (
                            w1 * (kpts1[:, 0] + 1) / 2 - offset,
                            h1 * (kpts1[:, 1] + 1) / 2 - offset,
                        ),
                        axis=-1,
                    )
                    - K1[None, :2, 2]
                )  # go from [0,h] -> [h/2, -h/2], [0,w] -> [-w/2, w/2]
                kpts2 = sparse_matches[:, 2:]
                kpts2 = (
                    np.stack(
                        (
                            w2 * (kpts2[:, 0] + 1) / 2 - offset,
                            h2 * (kpts2[:, 1] + 1) / 2 - offset,
                        ),
                        axis=-1,
                    )
                    - K2[None, :2, 2]
                )  # go from [0,h] -> [h/2, -h/2], [0,w] -> [-w/2, w/2]
                try:
                    threshold = 1.0
                    norm_threshold = threshold / (
                        np.mean(K1[:2, :2]) + np.mean(K2[:2, :2])
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

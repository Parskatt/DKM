import json
import os
from glob import glob
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
import kornia.augmentation as K
from dkm.utils.transforms import GeometricSequential
from loguru import logger


class HomographyDataset:
    def __init__(
        self, ims, H_generator=None, photometric_distortion=None, h=384, w=512
    ) -> None:
        self.ims = np.array(ims)
        self.H_generator = H_generator
        self.h = h
        self.w = w
        self.photometric_distortion = photometric_distortion

    def __len__(self):
        return len(self.ims)

    def normalize_H(self, H, h, w):
        T = torch.tensor([[[2 / w, 0, -1], [0, 2 / h, -1], [0, 0, 1]]], device=H.device)
        H_n = T @ H @ torch.linalg.inv(T)
        return H_n

    def flow_from_H(self, query, H):
        b, c, h, w = query.shape
        H = self.normalize_H(H, h, w)
        q_c = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=query.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=query.device),
            )
        )
        q_c = torch.stack((q_c[1], q_c[0], torch.ones_like(q_c[0])), dim=-1)[
            None
        ].expand(b, h, w, 3)
        q_to_s = torch.einsum("bhwc,bdc -> bhwd", q_c, H)
        q_to_s = q_to_s[..., :2] / (q_to_s[..., 2:] + 1e-8)
        return q_to_s

    def covisible_mask(self, query, Hq, Hs):
        ones = torch.ones_like(query)
        mask = self.H_generator.apply_transform(ones, Hq)
        H = Hq @ torch.linalg.inv(Hs)
        mask *= self.H_generator.apply_transform(ones, H)
        return mask

    def __getitem__(self, idx):
        with torch.no_grad():
            source = Image.open(self.ims[idx])
            source = np.array(source)
            if len(source.shape) == 2:
                source = np.repeat(
                    source[:, :, None], 3, -1
                )  # some bw photos need conversion
            assert len(source.shape) == 3, f"{source.shape},{str(self.ims[idx])}"
            query = self.photometric_distortion(image=source)["image"]
            support = self.photometric_distortion(image=source)["image"]
            query, Hq = self.H_generator(torch.tensor(query).permute(2, 0, 1)[None])
            support, Hs = self.H_generator(torch.tensor(support).permute(2, 0, 1)[None])
            H_q_to_s = Hs @ torch.linalg.inv(Hq)
            flow = self.flow_from_H(query, H_q_to_s)
            mask = self.covisible_mask(query, Hq, Hs)
            data_dict = {
                "query": query[0],
                "support": support[0],
                "mask": mask[0, 0],
                "flow": flow[0],
            }
            return data_dict


class MovingObjectsDataset(HomographyDataset):
    def __init__(
        self,
        ims,
        h=384,
        w=512,
        H_generator=None,
        photometric_distortion=None,
        object_sampler=None,
        num_objects=3,
    ) -> None:
        assert H_generator is not None, "Must provide a H_generator"
        assert photometric_distortion is not None, "must provide photometric distortion"
        assert object_sampler is not None, "must provide object sampler"

        super().__init__(
            ims,
            h=h,
            w=w,
            H_generator=H_generator,
            photometric_distortion=photometric_distortion,
        )
        self.object_sampler = object_sampler
        self.num_objects = num_objects
        self.mover = GeometricSequential(
            K.RandomAffine(degrees=45, p=1.0, scale=(0.85, 1.25), translate=(0.3, 0.3))
        )

    def put_object(self, im, obj_im, obj_mask):
        res_im = im * (1 - obj_mask) + obj_im * (obj_mask)
        return res_im

    def move_object(self, obj_im, obj_mask):
        obj_im, M = self.mover(obj_im[None])
        obj_mask = self.mover.apply_transform(obj_mask[None, None].float(), M)
        return obj_im, obj_mask, M

    def __getitem__(self, idx):
        with torch.no_grad():
            data = super().__getitem__(idx)
            objects = [self.object_sampler.sample() for _ in range(self.num_objects)]
            for o in objects:
                obj_im, obj_seg, _ = o
                obj_im = np.array(obj_im)
                if len(obj_im.shape) == 2:
                    obj_im = np.repeat(
                        obj_im[:, :, None], 3, -1
                    )  # some bw photos need conversion
                obj_im = torch.tensor(
                    self.photometric_distortion(image=obj_im)["image"]
                ).permute(2, 0, 1)
                obj_seg = torch.from_numpy(cv2.resize(obj_seg, (self.w, self.h)))
                obj_im_query, obj_seg_query, M_q = self.move_object(obj_im, obj_seg)
                obj_im_support, obj_seg_support, M_s = self.move_object(obj_im, obj_seg)
                M = M_s @ torch.linalg.inv(M_q)
                obj_flow = self.flow_from_H(obj_im[None], M)[0]
                obj_flow_mask = self.covisible_mask(data["mask"][None, None], M_q, M_s)[
                    0, 0
                ]
                data["query"] = self.put_object(
                    data["query"], obj_im_query[0], obj_seg_query[0]
                )
                data["support"] = self.put_object(
                    data["support"], obj_im_support[0], obj_seg_support[0, 0]
                )
                # Where ever something old pointed to the new thing, we set mask to zero (Note: must be done before new object is added)
                obj_match = F.grid_sample(
                    obj_seg_support, data["flow"][None], align_corners=False
                )[0, 0].bool()
                data["mask"][obj_match] = 0
                # In all places where the object is, we set the flow and flow mask, if object goes outside there we don't care about backgrond flow, otherwise use object flow
                obj_seg_query_bool = obj_seg_query[0, 0].bool()
                data["flow"][obj_seg_query_bool] = obj_flow[obj_seg_query_bool]
                data["mask"][obj_seg_query_bool] = obj_flow_mask[obj_seg_query_bool]
            return data


def build_ade20k(data_root="data/homog_data"):
    filenames = json.load(open(os.path.join(data_root, "ade20k_filenames.json"), "r"))
    folders = json.load(open(os.path.join(data_root, "ade20k_folders.json"), "r"))
    ims = [
        os.path.join(data_root, folder, filename)
        for folder, filename in zip(folders, filenames)
    ]
    return ims


def build_cityscapes(data_root="data/homog_data"):
    cityscapes_root = os.path.join(data_root, "leftImg8bit")
    splits = glob(cityscapes_root + "/*")
    ims = []
    for split in splits:
        scenes = glob(split + "/*")
        for scene in scenes:
            ims += glob(scene + "/*")
    return ims


def build_coco(data_root="data/homog_data"):
    coco_root = os.path.join(data_root, "COCO2014")
    splits = glob(coco_root + "/*")
    ims = []
    for split in splits:
        if "annotations" in split:
            continue
        ims += glob(split + "/*")
    return ims


def build_coco_foreground(data_root="data/homog_data"):
    coco_root = os.path.join(data_root, "COCO2014")
    splits = glob(coco_root + "/annotations/*")

    instances = []
    for split in splits:
        if (not os.path.isdir(split)) and ("instances" in split):
            inst = json.load(open(split, "r"))
            split = "train" if "train" in split else "val"
            [i.update({"split": split}) for i in inst["annotations"]]
            instances += inst["annotations"]
    logger.info(f"Found {len(instances)} coco instances")
    return instances


class SampleObject:
    def __init__(self, root_dir, instances, min_area=5000, max_area=100000) -> None:
        self.coco_root = os.path.join(root_dir, "COCO2014")
        self.instances = [
            i
            for i in instances
            if (max_area > i["area"] > min_area) and i["iscrowd"] == 0
        ]
        logger.info(
            f"{len(self.instances)} coco instances remaining after thresholding on {min_area=},{max_area=}, and not using crowds"
        )

    def im_from_split_id(self, split, id):
        file = f"COCO_{split}2014_{id:012d}"
        im = os.path.join(self.coco_root, split + "2014", file + ".jpg")
        return im

    def xywh_to_xyxy(self, bbox):
        return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

    def poly_to_mask(self, instance, w, h):
        from PIL import Image, ImageDraw

        segs = instance["segmentation"]
        img = Image.new("L", (w, h), 0)
        for seg in segs:
            try:
                poly = [int(i) for i in seg]
            except:
                raise ValueError(f"{seg=} is not in polygon format")
            poly += poly[:2]
            ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
        mask = np.array(img)
        return mask

    def sample(self):
        instance = np.random.choice(self.instances)
        im = self.im_from_split_id(instance["split"], instance["image_id"])
        x, y, w_b, h_b = instance["bbox"]
        bbox = self.xywh_to_xyxy((x, y, w_b, h_b))
        im = Image.open(im)
        w, h = im.size
        mask = self.poly_to_mask(instance, w, h)
        return im, mask, bbox

import copy
import torch
import numpy as np
import cv2
import random
import json
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances
from transformers import CLIPTokenizer
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from utils.prompt_engineering import prompt_engineering
from modeling.utils import configurable
import torch.nn.functional as F
__all__ = ["VGDatasetMapper"]


def construct_mask_from_polygon(polygons, img_shape):
    height, width = img_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        polygon = np.array(polygon, dtype=np.int32).reshape((-1, 2))
        cv2.drawContours(mask, [polygon], -1, (1), -1)
    return mask

def resize_mask(mask, target_size=(1024, 1024)):
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)  # 转换为 Tensor

    H, W = mask.shape  # 获取原始尺寸
    target_H, target_W = target_size  # 目标尺寸

    scale = min(target_H / H, target_W / W)
    new_H, new_W = int(H * scale), int(W * scale)

    resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                            size=(new_H, new_W),
                            mode='nearest').squeeze(0).long()

    padded_mask = torch.zeros((1, target_H, target_W), dtype=torch.long)
    padded_mask[:, :new_H, :new_W] = resized

    return padded_mask



def resize_and_pad(image, target_size=1024):

    H, W, C = image.shape
    scale = target_size / max(H, W)  # 计算缩放比例
    new_H, new_W = int(H * scale), int(W * scale)  # 计算缩放后的尺寸

    resized_image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    padded_image = np.zeros((target_size, target_size, C), dtype=image.dtype)

    padded_image[:new_H, :new_W, :] = resized_image

    return padded_image

class VGDatasetMapper:
    @configurable
    def __init__(self, is_train=True, *,image_format,max_grounding_num):
        self.is_train = is_train
        self.image_format = image_format
        self.max_grounding_num = max_grounding_num

    @classmethod
    def from_config(cls, cfg, is_train=True):
        return {
            "is_train": is_train,
            "image_format": cfg['INPUT']['FORMAT'],
            "max_grounding_num": cfg['MODEL']['DECODER']['GROUNDING']['MAX_LEN']
        }

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        image_shape = image.shape[:2]
        height, width = image_shape
        dataset_dict["size"] = image_shape

        image=resize_and_pad(image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["mask"] = resize_mask(construct_mask_from_polygon(dataset_dict["polygons"][0], (height, width)))
        dataset_dict['captions']=[dataset_dict['phrase']+'.']
        dataset_dict['grounding_info']=[]
        dataset_dict['sem_seg_file_name']=''#
        dataset_dict['segments_info']=[]#
        dataset_dict['captions_noun']=[dataset_dict['phrase'].split()]#

        instances = Instances([1024,1024])
        classes = [0]
        is_things = [1]
        masks = dataset_dict["mask"].bool()
        boxes = Boxes(torch.zeros((1, 4)))
        classes = np.array(classes)
        is_things = np.array(is_things)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        instances.is_things = torch.tensor(is_things, dtype=torch.int64)

        instances.gt_masks = masks
        instances.gt_boxes = boxes

        dataset_dict["instances"] = instances


        grounding_len = random.randint(1, self.max_grounding_num-1)
        masks_grd = instances.gt_masks
        mode = 'class'
        if len(masks_grd) == 0:
            masks_grd = torch.tensor([])
            texts_grd = ['none']
            hash_grd = np.array([hash(txt) for txt in texts_grd])
        else:
            texts_grd = [dataset_dict['phrase']]
            hash_grd = [hash(txt) for txt in texts_grd]

        groundings = {'masks': masks_grd, 'texts': texts_grd, 'mode': mode, 'hash': hash_grd}
        dataset_dict["groundings"] = groundings

        return dataset_dict
"""Extract, Transform and Load the data"""

import logging
import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as tx
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence


class MSCOCODataset(Dataset):
    """MS-COCO Captions dataset."""

    def __init__(self, img_paths: List[str], captions: List[str], transform: Optional[tx.Compose] = None):
        """
        Args:
            img_paths: List of Path to the images.
            captions: List of captions.
            transform: Optional transform to be applied an image.
        """
        self.img_paths = img_paths
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with read_image(self.img_paths[idx], ImageReadMode.RGB) as image:
            image /= 255.0
            cap = self.captions[idx]
            if self.transform:
                image = self.transform(image)

            return image, cap

ROOT_DIR = "/dbfs/Users/karan.rampal@hm.com/data/"
ANNOTATION_FILE = os.path.dirname(ROOT_DIR) + "/annotations/captions_train2014.json"
IMG_DIR = os.path.dirname(ROOT_DIR) + "/train2014/"
TRAIN_DATA_PERCENT = 0.9
MIN_FREQ = 10
random.seed(0)


with open(ANNOTATION_FILE, "r") as f:
    annotations = json.load(f)

image_path_to_caption = defaultdict(list)
for val in annotations["annotations"]:
    caption = f"<start> {val['caption']} <end>"
    image_path = IMG_DIR + "COCO_train2014_" + f"{val['image_id']:012d}.jpg"
    image_path_to_caption[image_path].append(caption)

image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)

total = len(image_paths)
print(f"Total images: {total}")
idx = int(total * TRAIN_DATA_PERCENT)
train_image_paths = image_paths[:idx]
print(f"Training images: {len(train_image_paths)}")
test_image_paths = image_paths[idx:]
print(f"Test images: {len(test_image_paths)}")


def make_data_lists(img_pth_to_cap: Dict[str, str], image_paths: List[str]):
    """Make lists of data paths and respective captions
    Args:
        img_pth_to_cap: Dictionary of image paths to captions
        image_paths: List of image paths
    Returns:
        img_list: List of image paths
        cap: List of captions
    """
    cap, img_list = [], []
    for im_pth in image_paths:
        caption_list = img_pth_to_cap[im_pth]
        cap.extend(caption_list)
        img_list.extend([im_pth] * len(caption_list))
    return img_list, cap

train_img_name_list, train_captions = make_data_lists(image_path_to_caption, train_image_paths)
test_img_name_list, test_captions = make_data_lists(image_path_to_caption, test_image_paths)


def yield_tokens(data_iter: List[str]):
    """Yield data tokens
    Args:
        data_iter: torch dataset
    Yield:
        List of tokenized text
    """
    for text in data_iter:
        yield tokenizer(text)

tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(yield_tokens(train_captions),
                                  min_freq=MIN_FREQ,
                                  specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])
print(f"Vocab size: {len(vocab)}")


def collate_batch(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate batch data
    Args:
        batch: batched images and captions
    Returns:
        tuple of images and tokenized captions
    """
    img_list, cap_list = [], []
    for img, cap in batch:
        img_list.append(img)
        processed_cap = vocab(tokenizer(cap))
        cap_list.append(torch.tensor(processed_cap, dtype=torch.long))
    text_tensor = pad_sequence(cap_list, batch_first=True, padding_value=vocab(["<pad>"])[0])
    return torch.stack(img_list, 0), text_tensor



from typing import Protocol, TypeVar, Generic, TypedDict, Literal
from collections.abc import Collection, Sequence
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
import json

TImageId = TypeVar("TImageId")
TCategoryId = TypeVar("TCategoryId")
TAnnotationId = TypeVar("TAnnotationId")
TSegmentation = TypeVar("TSegmentation")

class ImageProtocol(Protocol):
    id: TImageId

class Annotation(Protocol):
    id: TAnnotationId
    image_id: TImageId
    category_id: TCategoryId

class Category(Protocol):
    id: TCategoryId

class GenericDataset(TypedDict, total=False):
    images: list[ImageProtocol]
    annotations: list[Annotation]
    captions: list[dict[str, str]]

class DatasetHandler(Generic[TImageId, TAnnotationId, TCategoryId]):
    def __init__(self, annotation_file: str | Path | None = None, dataset_type=None) -> None:
        self.annotation_file = annotation_file
        self.dataset: dict = {}
        self.dataset_type = dataset_type
        if self.annotation_file is not None:
            self.load()

    def load(self) -> None:
        if self.annotation_file is None:
            raise ValueError("No annotation file provided.")

        path = Path(self.annotation_file)
        with path.open("r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def get_image_ids(self) -> list[TImageId]:
        # if (self.dataset_type == 'rsicd'):
            return [img["imgid"] for img in self.dataset.get("images", [])]
        # if (self.dataset_type == 'coco'):
        #     return [img["image_id"] for img in self.dataset.get("images", [])]
    
    def get_annotation_ids(self, img_id: TImageId) -> list[TAnnotationId]:
        #return [ann["imgid"] for ann in self.dataset.get("annotations", []) if ann["imgid"] == img_id]
        return [img_id]
    
    def load_images(self, ids: list[TImageId]) -> list[dict]:
        id_set = set(ids)
        # if (self.dataset_type == 'coco'):
        #     return [img for img in self.dataset.get("images", []) if img["image_id"] in id_set]
        # if (self.dataset_type == 'rsicd'):
        return [img for img in self.dataset.get("images", []) if img["imgid"] in id_set]
    
    def load_annotations(self, ids: list[TAnnotationId]) -> list[dict]:
        # id_set = set(ids)
        # return [ann for ann in self.dataset.get("annotations", []) if ann["imgid"] in id_set]

        # THIS ONLY WORKS FOR RSCID DATASET
        
        id_set = set(ids)
        annotations = []
        for image in self.dataset.get("images", []):
            if image["imgid"] in id_set and "sentences" in image:
                annotations.extend(image["sentences"])
        return annotations
    
from transformers import CLIPProcessor

class RSICDDataset(torch.utils.data.Dataset):
    def __init__(self, handler, image_dir):
        self.handler = handler
        self.image_dir = Path(image_dir)

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

        self.image_ids = self.handler.get_image_ids()
        self.id_to_filename = {}
        for img in self.handler.dataset.get("images", []):
            self.id_to_filename[img["imgid"]] = img["filename"]

    def __len__(self):
        return len(self.image_ids)

    # def __getitem__(self, idx):
    #     image_id = self.image_ids[idx]  # idx is the index, not the image_id
    #     filename = self.id_to_filename.get(image_id)
    #     if filename is None:
    #         raise ValueError(f"No filename found for image id {image_id}")
    #     image_path = self.image_dir / f"{filename}"  # or .png
    #     image = Image.open(image_path).convert("RGB")

    #     captions = self.handler.load_annotations([image_id])
    #     caption = captions[0]['raw'] if captions else ""

    #     # Transform image and tokenize text
    #     inputs = self.processor(text=[caption], images=image, return_tensors="pt", padding=True)
    #     return {
    #         "image": inputs["pixel_values"].squeeze(0),
    #         "text": inputs["input_ids"].squeeze(0),
    #         "attention_mask": inputs["attention_mask"].squeeze(0),
    #     }
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        filename = self.id_to_filename.get(image_id)
        if filename is None:
            raise ValueError(f"No filename found for image id {image_id}")
        image_path = self.image_dir / f"{filename}"
        image = Image.open(image_path).convert("RGB")

        captions = self.handler.load_annotations([image_id])
        caption = captions[0]['raw'] if captions else ""

        image_tensor = self.image_transform(image)  # use transforms.Compose(...)
        
        return {
            "image": image_tensor,
            "text": caption  # return raw text!
        }
from typing import Protocol, TypeVar, Generic, TypedDict, Literal
from collections.abc import Collection, Sequence
from pathlib import Path
import json

TImageId = TypeVar("TImageId")
TCategoryId = TypeVar("TCategoryId")
TAnnotationId = TypeVar("TAnnotationId")
TSegmentation = TypeVar("TSegmentation")

class Image(Protocol):
    id: TImageId

class Annotation(Protocol):
    id: TAnnotationId
    image_id: TImageId
    category_id: TCategoryId

class Category(Protocol):
    id: TCategoryId

class GenericDataset(TypedDict, total=False):
    images: list[Image]
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
        if (self.dataset_type == 'rscid'):
            return [img["imgid"] for img in self.dataset.get("images", [])]
        if (self.dataset_type == 'coco'):
            return [img["image_id"] for img in self.dataset.get("images", [])]
    
    def get_annotation_ids(self, img_id: TImageId) -> list[TAnnotationId]:
        #return [ann["imgid"] for ann in self.dataset.get("annotations", []) if ann["imgid"] == img_id]
        return [img_id]
    
    def load_images(self, ids: list[TImageId]) -> list[dict]:
        id_set = set(ids)
        if (self.dataset_type == 'coco'):
            return [img for img in self.dataset.get("images", []) if img["image_id"] in id_set]
        if (self.dataset_type == 'rscid'):
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
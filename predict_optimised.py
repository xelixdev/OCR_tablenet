"""Predicting Module."""

from collections import OrderedDict
from typing import List

import click
import numpy as np
import pandas as pd
from albumentations import Compose
from PIL import Image
from pytesseract import image_to_string
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, convex_hull_image
from skimage.transform import resize
from skimage.util import invert

from tablenet import TableNetModule


class Predict:
    """Predict images using pre-trained model."""

    def __init__(self, checkpoint_path: str, transforms: Compose, threshold: float = 0.5, per: float = 0.005):
        """Predict images using pre-trained TableNet model.

        Args:
            checkpoint_path (str): model weights path.
            transforms (Optional[Compose]): Compose object from albumentations used for pre-processing.
            threshold (float): threshold to consider the value as correctly classified.
            per (float): Minimum area for tables and columns to be considered.
        """
        self.transforms = transforms
        self.threshold = threshold
        self.per = per

        self.model = TableNetModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.requires_grad_(False)

    def predict(self, image: Image, extract_data=False) -> List[pd.DataFrame]:
        """Predict a image table values.

        Args:
            image (Image): PIL.Image to

        Returns (List[pd.DataFrame]): Tables in pandas DataFrame format.
        """
        processed_image = self.transforms(image=np.array(image))["image"]

        table_mask, _ = self.model.forward(processed_image.unsqueeze(0))

        table_mask = self._apply_threshold(table_mask)
        
        segmented_tables = self._process_tables(self._segment_image(table_mask))


#             return tables
        return segmented_tables

    def _apply_threshold(self, mask):
        mask = mask.squeeze(0).squeeze(0).numpy() > self.threshold
        return mask.astype(int)

    def _process_tables(self, segmented_tables):
        width, height = segmented_tables.shape
        tables = []
        for i in np.unique(segmented_tables)[1:]:
            table = np.where(segmented_tables == i, 1, 0)
            if table.sum() > height * width * self.per:
                tables.append(convex_hull_image(table))
        return tables


    @staticmethod
    def _segment_image(image):
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(2))
        cleared = clear_border(bw)
        label_image = label(cleared)
        return label_image


# @click.command()
# @click.option('--image_path', default="data/Marmot_data/10.1.1.193.1812_24.bmp")
# @click.option('--model_weights', default="data/best_model.ckpt")
def predict(image_path: str, model_weights: str, transforms) -> List[pd.DataFrame]:
    """Predict table content.

    Args:
        image_path (str): image path.
        model_weights (str): model weights path.

    Returns (List[pd.DataFrame]): Tables in pandas DataFrame format.
    """

    pred = Predict(model_weights, transforms)

    image = Image.open(image_path)
    predictions = pred.predict(image)
    return predictions


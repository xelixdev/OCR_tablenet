import fitz
import json 
from io import BytesIO
from hashlib import md5
from pathlib import Path
from os import makedirs
import os 
import boto3
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
import settings
from predict_optimised import predict
import numpy as np
import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2


def raw_ocr(img: bytes, *, cache_location = None, textract_client):
    """Run Textract ocr with cache support"""
    # Calculate checksum of the image
    img_checksum = md5(img).hexdigest()
    cache_file = Path(cache_location, img_checksum) if cache_location is not None else None

    # Load cache if available
    if cache_location is not None and cache_file is not None and cache_file.is_file():
        with open(cache_file, "r") as f:
            return json.load(f)

    response = textract_client.analyze_document(Document={"Bytes": img}, FeatureTypes=["TABLES"])

    # Write parsed response to cache file
    if cache_location is not None and cache_file is not None:
        makedirs(cache_file.parent, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(response, f)

    return response

def extract_png_page_bytes(file: str, page_number: int, dpi: int) -> bytes:
    with open(file, "rb") as f:
        with fitz.Document(stream=BytesIO(f.read()), filetype="pdf") as document:
            page = document.load_page(page_number)
            return (
                page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72)).getPNGData(), 
                page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72)).width,
                page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72)).height
            )
        
def get_results_from_ocr_dictionary(ocr_dictionary):
    output_dict = [x for x in ocr_dictionary["Blocks"] if x['BlockType'] == 'TABLE']
    keys = ["BlockType","Confidence", "Geometry"]                                                                                                                                                                              
    res = []                                                                                                                                                                                            
    for output in output_dict: 
        result = dict((k, output[k]) for k in keys) 
        res.append(result) 
        
    return res

# Calculate iou
def get_iou(users, preds):
    """
     Calculate the Intersection over Union (IoU) of two bounding boxes.
    :param bb1: dict with Keys: {'x1', 'x2', 'y1', 'y2'}
    :param bb2: dict with Keys: {'x1', 'x2', 'y1', 'y2'}
    :return: float in [0, 1]

    """
    assert users['x1'] < users['x2']
    assert users['y1'] < users['y2']
    assert preds['x1'] < preds['x2']
    assert preds['y1'] < preds['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(users['x1'], preds['x1'])
    y_top = max(users['y1'], preds['y1'])
    x_right = min(users['x2'], preds['x2'])
    y_bottom = min(users['y2'], preds['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both bounding boxes
    user_area = (users['x2'] - users['x1']) * (users['y2'] - users['y1'])
    preds_area = (preds['x2'] - preds['x1']) * (preds['y2'] - preds['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(user_area + preds_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_data_from_s3():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('xelix-area-selection-examples')
    images = []
    image_widths = []
    image_heights = []
    file_info = []
    keys = []
    pdf_keys = []
    for obj in bucket.objects.all():
        key = obj.key
        if key.startswith("pdf"):
            if key.endswith('.pdf'):
                bucket.download_file(key, "pdf_file.pdf")
                extracted_image, image_width, image_height = extract_png_page_bytes("pdf_file.pdf", 0, 300)
                pdf_keys.append(key)
                images.append(extracted_image)
                image_widths.append(image_width)
                image_heights.append(image_height)
            if key.endswith('.json'):
                # print(key)
                keys.append(key)
                body = obj.get()['Body'].read()
                json_content = json.loads(body)['areas']
                file_info.append(json_content)
                
    return images, image_widths, image_heights, file_info, keys, pdf_keys



def get_coordinates(image_data):
    textract_client = boto3.client("textract", region_name=settings.AWS_REGION)
    iou_scores = []
    X_0s = []
    Y_0s = []
    X_1s = []
    Y_1s = []

    X_0_OCRs = []
    Y_0_OCRs = []
    X_1_OCRs = []
    Y_1_OCRs = []

    for image, info, image_width, image_height, identifier in zip(
        image_data["images"], image_data["file_info"], image_data["image_widths"], image_data["image_heights"], image_data["identifier"]
    ):
        ocr_dictionary = raw_ocr(image, cache_location = "cache/", textract_client = textract_client)
        res = get_results_from_ocr_dictionary(ocr_dictionary)

        # Create co-ordinates from labelled data 
        X_0 = info[0][1] 
        Y_0 = info[0][2] 
        X_1 = info[0][3] 
        Y_1 = info[0][4]
        X_0s.append(X_0)
        Y_0s.append(Y_0)
        X_1s.append(X_1)
        Y_1s.append(Y_1)

        dpi_ratio = 72/300
        if len(res) > 0:
            X_0_OCR = res[0]['Geometry']['Polygon'][0]["X"] * image_width * dpi_ratio
            Y_0_OCR = res[0]['Geometry']['Polygon'][0]["Y"] * image_height * dpi_ratio
            X_1_OCR = res[0]['Geometry']['Polygon'][1]["X"] * image_width * dpi_ratio
            Y_1_OCR = res[0]['Geometry']['Polygon'][2]["Y"] * image_height * dpi_ratio
            user = {'x1': X_0, 'x2': X_1, 'y1': Y_0, 'y2': Y_1}
            ocr = {'x1': X_0_OCR, 'x2': X_1_OCR, 'y1': Y_0_OCR, 'y2': Y_1_OCR}
            iou_score = get_iou(user, ocr)

        else:
            iou_score = 0
            X_0_OCR = 0
            Y_0_OCR = 0
            X_1_OCR = 0
            Y_1_OCR = 0

        iou_scores.append(iou_score)
        X_0_OCRs.append(X_0_OCR)
        Y_0_OCRs.append(Y_0_OCR)
        X_1_OCRs.append(X_1_OCR)
        Y_1_OCRs.append(Y_1_OCR)
        
    return iou_scores, X_0_OCRs, Y_0_OCRs, X_1_OCRs, Y_1_OCRs, X_0s, Y_0s, X_1s, Y_1s

def transform_images():


    transforms = album.Compose([
        album.Resize(896, 896, always_apply=True),
        album.Normalize(),
        ToTensorV2()
    ])
    return transforms

def get_model_coordinates(model, transforms, file):
    
    table_mask = predict(
        image_path=file, 
        model_weights=model,
        transforms=transforms 
    )
    if len(table_mask) > 0:
        table_1 = table_mask[1]
        coords = np.column_stack(np.where(table_1 > 0))
        # X_coords = coords[:, 0]
        # Y_coords = coords[:, 1]
        X_coords = [x[0] for x in coords]
        Y_coords = [x[1] for x in coords]
        
        # Y_0_model = min(X_coords) * 2200/896
        # Y_1_model = max(X_coords) * 2200/896
        # X_0_model = min(Y_coords) * 1700/896
        # X_1_model = max(Y_coords) * 1700/896
        # model = {'x1': X_0_model, 'x2': X_1_model, 'y1': Y_0_model, 'y2': Y_1_model}
    else: 
        Y_0_model = 0
        Y_1_model = 0
        X_0_model = 0
        X_1_model = 0
    return X_coords, Y_coords, table_mask
#extractor_images.py

import boto3
import pymupdf as fitz

s3 = boto3.client('s3')
bucket_name = 'hackathon-jr'

def extract_images(doc, page_num, figure_num):
    print("Extracting images...")  # Print "Extracting images...
    figures = figure_num
    for i in range(len(doc)):
        if i + 1 == page_num:
            page = doc[i]  # Get the pages 
            text_data = page.get_text("dict")  # Get the text data

            image_bbox = []
            for block in text_data["blocks"]:
                if block["type"] == 1:  # Skip non-text blocks
                    image_bbox.append(block['bbox'])

            #now use the image_bbox to extract the images
            if (len(image_bbox) > 0):
                height = 60  
                width = 70
                for j, bbox in enumerate(image_bbox):
                    x0, y0, x1, y1 = bbox
                    x0 = max(x0 - width, 0)
                    y0 = max(y0 - height, 0)
                    x1 = min(x1 + width, page.rect.width)
                    y1 = min(y1 + height, page.rect.height)
                    rect = fitz.Rect(x0, y0, x1, y1)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
                    output_path = f"{page_num:03d}_{figures:03d}_000.png"
                    pix.save(output_path)
                    s3.upload_file(output_path, bucket_name, output_path)
                    figures += 1
import io
import zipfile
import numpy as np
from PIL import Image

from server.services.cvat_api import create_cvat_annotation_zip
from flask import Flask


def make_rgb_mask(color, size=(16, 16)):
    img = Image.new('RGB', size, color=color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def make_palette_mask(palette_idx_colors, size=(16, 16)):
    # Create a small palette image using mode 'P' and a small palette
    img = Image.new('P', size)

    # build palette (first three entries) then fill rest zeros
    flat = []
    for c in palette_idx_colors:
        flat.extend(c)
    # pad to 768 values (256*3)
    flat += [0] * (768 - len(flat))
    img.putpalette(flat)

    # Fill image with first index color
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = 0

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def test_create_zip_basic(tmp_path):
    # create a Flask app context so current_app.logger is available
    app = Flask(__name__)
    with app.app_context():
        # Prepare two integer instance masks (numpy arrays)
        mask1 = np.zeros((16, 16), dtype=np.int32)
        mask1[2:10, 2:10] = 1  # single instance

        mask2 = np.zeros((16, 16), dtype=np.int32)
        mask2[1:6, 1:6] = 1
        mask2[8:14, 8:14] = 2  # two instances

        mask_files = {
            'image1.png': mask1,
            'dir/subdir/image2.png': mask2,
        }

        zip_bytes = create_cvat_annotation_zip(mask_files)
        assert zip_bytes and len(zip_bytes) > 0

        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

        namelist = zf.namelist()

        # Ensure SegmentationClass and SegmentationObject entries exist
        assert 'SegmentationClass/image_patches/image1.png' in namelist
        assert 'SegmentationObject/image_patches/image1.png' in namelist
        assert 'SegmentationClass/image_patches/image2.png' in namelist
        assert 'SegmentationObject/image_patches/image2.png' in namelist

        # Ensure ImageSets/Segmentation/default.txt exists and contains basenames without extension
        assert 'ImageSets/Segmentation/default.txt' in namelist
        default_txt = zf.read('ImageSets/Segmentation/default.txt').decode('utf-8').strip().splitlines()
        assert 'image_patches/image1' in default_txt
        assert 'image_patches/image2' in default_txt

        # Ensure labelmap exists and contains nucleus mapping
        assert 'labelmap.txt' in namelist
        labelmap = zf.read('labelmap.txt').decode('utf-8')
        assert 'nucleus' in labelmap

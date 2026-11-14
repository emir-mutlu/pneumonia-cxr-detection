import json

import numpy as np
import pytest
from PIL import Image

from pneumonia_cxr.data.imaging import preprocess_for_model, resize_square, to_uint8_gray
from pneumonia_cxr.data.kermany import parse_kermany_filename
from pneumonia_cxr.data.rsna import load_labels


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("person1086_bacteria_3026.jpeg", ("person1086", "bacteria")),
        ("person1_virus_6.jpeg", ("person1", "virus")),
        ("IM-0115-0001.jpeg", ("IM-0115", "normal")),
        ("NORMAL2-IM-0577-0001.jpeg", ("NORMAL2-IM-0577", "normal")),
        ("NORMAL2-IM-1427-0001-0002.jpeg", ("NORMAL2-IM-1427", "normal")),
    ],
)
def test_parse_kermany_filename(name, expected):
    assert parse_kermany_filename(name) == expected


def test_to_uint8_gray_handles_16bit_and_monochrome1():
    arr = np.linspace(0, 4095, 64 * 64, dtype=np.uint16).reshape(64, 64)
    img = to_uint8_gray(arr)
    out = np.asarray(img)
    assert img.mode == "L"
    assert out.min() == 0 and out.max() == 255
    inverted = np.asarray(to_uint8_gray(arr, invert=True))
    assert np.array_equal(inverted, 255 - out)


def test_rgb_input_becomes_single_channel():
    rgb = Image.new("RGB", (300, 200), (10, 120, 250))
    assert resize_square(to_uint8_gray(rgb), 64).shape == (64, 64)


def test_preprocess_for_model_shape_and_range():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (500, 400), dtype=np.uint8))
    x = preprocess_for_model(img, img_size=224)
    assert x.shape == (1, 1, 224, 224)
    assert x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_rsna_labels_use_final_adjudicated_group(tmp_path):
    def ann(uid, label_id, data=None):
        return {"SOPInstanceUID": uid, "labelId": label_id, "data": data}

    project = {
        "labelGroups": [
            {"name": "Team 1", "labels": [{"id": "T1", "name": "Lung Opacity (High Prob)"}]},
            {
                "name": "Calculated",
                "labels": [
                    {"id": "N", "name": "Normal"},
                    {"id": "X", "name": "No Lung Opacity / Not Normal"},
                    {"id": "L", "name": "Lung Opacity"},
                ],
            },
        ],
        "datasets": [
            {
                "annotations": [
                    ann("a", "T1", {"x": 1, "y": 1, "width": 5, "height": 5}),  # reader label, ignored
                    ann("a", "N"),
                    ann("b", "X"),
                    ann("c", "L", {"x": 10, "y": 20, "width": 30, "height": 40}),
                    ann("c", "L", {"x": 100, "y": 200, "width": 30, "height": 40}),
                ]
            }
        ],
    }
    f = tmp_path / "ann.json"
    f.write_text(json.dumps(project))
    df = load_labels(f).set_index("sop_uid")
    assert df.loc["a", "label"] == 0 and df.loc["a", "rsna_class"] == "Normal"
    assert df.loc["b", "label"] == 0
    assert df.loc["c", "label"] == 1
    assert len(json.loads(df.loc["c", "boxes"])) == 2
    assert json.loads(df.loc["a", "boxes"]) == []

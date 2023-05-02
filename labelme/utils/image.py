import base64

import cv2 as cv
import numpy as np


def img_data_to_arr(img_data):
    try:
        img = cv.imdecode(np.frombuffer(img_data, np.uint16),
                          cv.IMREAD_UNCHANGED)
    except ValueError:
        img = cv.imdecode(np.frombuffer(img_data, np.uint8),
                          cv.IMREAD_UNCHANGED)
    return img


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


def img_arr_to_data(img_arr):
    return cv.imencode(".png", img_arr)[1].tobytes()


def img_arr_to_b64(img_arr):
    img_bin = img_arr_to_data(img_arr)
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


def img_data_to_png_data(img_data):
    img_arr = img_data_to_arr(img_data)
    img_encode = cv.imencode('.png', img_arr)[1]
    arr_encode = np.array(img_encode)
    byte_encode = arr_encode.tobytes()
    return byte_encode

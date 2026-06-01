# modules/preprocessing.py

import cv2
import numpy as np


def crop_oriented_box(frame, obb_box):
    """Crop and straighten one YOLO OBB detection."""
    cx, cy, w, h, r = obb_box
    rect = ((cx, cy), (w, h), np.degrees(r))
    vertices = cv2.boxPoints(rect).astype(np.int32)

    h_img, w_img = frame.shape[:2]
    vertices[:, 0] = np.clip(vertices[:, 0], 0, w_img - 1)
    vertices[:, 1] = np.clip(vertices[:, 1], 0, h_img - 1)

    src = vertices.astype("float32")
    dst = np.array([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_matrix = cv2.getPerspectiveTransform(dst, src)
    crop = cv2.warpPerspective(frame, matrix, (max(1, int(w)), max(1, int(h))))
    return crop, vertices, inverse_matrix

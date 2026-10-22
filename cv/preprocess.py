"""OpenCV preprocessing pipeline for receipt / document images.

Stages
------
1. grayscale conversion
2. denoising (fast non-local means, falls back to median blur on tiny images)
3. adaptive threshold (for contour detection)
4. deskew (Hough-line angle with a projection-profile fallback)
5. contour-based receipt crop (largest plausible quadrilateral, perspective
   corrected; falls back to the full frame when no document is found)

Every function accepts a ``numpy.ndarray`` in BGR or grayscale and returns a
new array.  Nothing here needs a GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """Intermediate images produced by :func:`preprocess`."""

    gray: np.ndarray
    denoised: np.ndarray
    binary: np.ndarray
    deskewed: np.ndarray
    cropped: np.ndarray
    angle: float
    crop_found: bool


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 image."""
    if image is None or image.size == 0:
        raise ValueError("empty image")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"unsupported image shape: {image.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def denoise(gray: np.ndarray) -> np.ndarray:
    """Remove sensor / JPEG noise while keeping text edges."""
    h, w = gray.shape[:2]
    if min(h, w) < 32:
        return cv2.medianBlur(gray, 3)
    return cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)


def adaptive_threshold(gray: np.ndarray, block_size: int = 31, c: int = 10) -> np.ndarray:
    """Adaptive Gaussian threshold; block size is forced odd and >= 3."""
    block = max(3, int(block_size) | 1)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        c,
    )


def _projection_angle(binary: np.ndarray) -> float:
    """Estimate skew from the variance of horizontal ink projections."""
    ink = (binary < 128).astype(np.float32)
    if ink.sum() < 50:
        return 0.0
    best_angle, best_score = 0.0, -1.0
    h, w = ink.shape
    center = (w / 2.0, h / 2.0)
    for angle in np.arange(-8.0, 8.01, 0.5):
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            ink, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        )
        projection = rotated.sum(axis=1)
        score = float(np.var(projection))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def estimate_skew(binary: np.ndarray) -> float:
    """Return the skew angle in degrees (positive = counter-clockwise).

    Hough lines on a Canny edge map give a fast, robust estimate for receipts
    with long horizontal rules.  If too few lines are found we fall back to a
    projection-profile search over ``[-8, 8]`` degrees.
    """
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=80, minLineLength=max(40, binary.shape[1] // 5),
        maxLineGap=12,
    )
    angles: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x2 == x1:
                continue
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) <= 15.0:
                angles.append(angle)
    if len(angles) >= 3:
        return float(np.median(angles))
    return _projection_angle(binary)


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around the centre, expanding the canvas to avoid clipping."""
    if abs(angle) < 0.05:
        return image.copy()
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    border = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(
        image,
        matrix,
        (max(1, new_w), max(1, new_h)),
        flags=cv2.INTER_CUBIC,
        borderValue=border,
    )


def deskew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    """Deskew a binary image (or a grayscale image thresholded internally)."""
    source = binary
    if binary.ndim == 3:
        source = to_gray(binary)
    probe = source
    if probe.dtype != np.uint8:
        probe = probe.astype(np.uint8)
    angle = estimate_skew(probe)
    if abs(angle) < 0.2:
        return binary.copy(), 0.0
    return rotate(binary, angle), angle


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def find_receipt_contour(image: np.ndarray, min_area_ratio: float = 0.08):
    """Find the largest quadrilateral that plausibly bounds a receipt.

    Returns ``(quad, crop_found)``.  When no plausible contour exists the full
    image rectangle is returned so downstream code always gets a crop.
    """
    gray = to_gray(image) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    binary = adaptive_threshold(denoise(gray))
    ink = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    best = None
    best_area = 0.0
    image_area = float(w * h)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = float(cv2.contourArea(contour))
        if area < image_area * min_area_ratio:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            candidate = approx.reshape(4, 2).astype(np.float32)
        else:
            x, y, bw, bh = cv2.boundingRect(contour)
            candidate = np.array(
                [[x, y], [x + bw - 1, y], [x + bw - 1, y + bh - 1], [x, y + bh - 1]],
                dtype=np.float32,
            )
        if area > best_area:
            best, best_area = candidate, area
    if best is None:
        return frame, False
    return _order_quad(best), True


def four_point_crop(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-transform the quadrilateral to a top-down rectangle."""
    rect = _order_quad(quad)
    tl, tr, br, bl = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width, height = max(width, 1), max(height, 1)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (width, height), borderValue=(255, 255, 255))


def crop_receipt(image: np.ndarray, min_area_ratio: float = 0.08) -> tuple[np.ndarray, bool]:
    quad, found = find_receipt_contour(image, min_area_ratio=min_area_ratio)
    if not found:
        return image.copy(), False
    return four_point_crop(image, quad), True


def preprocess(image: np.ndarray, min_area_ratio: float = 0.08) -> PreprocessResult:
    """Run the full pipeline and return every intermediate stage."""
    gray = to_gray(image)
    denoised = denoise(gray)
    binary = adaptive_threshold(denoised)
    deskewed, angle = deskew(denoised)
    cropped, crop_found = crop_receipt(deskewed, min_area_ratio=min_area_ratio)
    return PreprocessResult(
        gray=gray,
        denoised=denoised,
        binary=binary,
        deskewed=deskewed,
        cropped=cropped,
        angle=angle,
        crop_found=crop_found,
    )


def preprocess_file(path: str | Path, min_area_ratio: float = 0.08) -> PreprocessResult:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return preprocess(image, min_area_ratio=min_area_ratio)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess a receipt image with OpenCV")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="write the cropped image here")
    args = parser.parse_args()

    result = preprocess_file(args.image)
    print(
        f"size={result.cropped.shape[1]}x{result.cropped.shape[0]} "
        f"skew={result.angle:.2f} crop_found={result.crop_found}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), result.cropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

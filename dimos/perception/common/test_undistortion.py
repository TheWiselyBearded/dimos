#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import pytest

from dimos.msgs.sensor_msgs.CameraInfo import FISHEYE_DISTORTION_MODELS, CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.perception.common.utils import (
    UndistortionMapCache,
    create_undistortion_maps,
    remap_image,
    rectify_image,
)


def _make_fisheye_camera_info(
    width: int = 640, height: int = 480
) -> CameraInfo:
    """Create a CameraInfo with equidistant (fisheye) distortion."""
    fx, fy = 300.0, 300.0
    cx, cy = width / 2.0, height / 2.0
    return CameraInfo(
        height=height,
        width=width,
        distortion_model="equidistant",
        D=[0.01, -0.02, 0.005, -0.001],
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        frame_id="fisheye_optical",
    )


def _make_plumb_bob_camera_info(
    width: int = 640, height: int = 480
) -> CameraInfo:
    """Create a CameraInfo with plumb_bob distortion."""
    fx, fy = 500.0, 500.0
    cx, cy = width / 2.0, height / 2.0
    return CameraInfo(
        height=height,
        width=width,
        distortion_model="plumb_bob",
        D=[-0.1, 0.05, 0.001, -0.002, 0.0],
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        frame_id="camera_optical",
    )


def _make_test_image(
    width: int = 640,
    height: int = 480,
    fmt: ImageFormat = ImageFormat.BGR,
) -> Image:
    """Create a synthetic test image (checkerboard pattern)."""
    if fmt in (ImageFormat.DEPTH, ImageFormat.DEPTH16):
        data = np.random.default_rng(42).uniform(0.5, 5.0, (height, width)).astype(np.float32)
    else:
        data = np.zeros((height, width, 3), dtype=np.uint8)
        # Simple grid pattern
        data[::20, :, :] = 255
        data[:, ::20, :] = 255
    return Image(data=data, format=fmt, frame_id="test", ts=1234.5)


# ---------------------------------------------------------------------------
# CameraInfo.is_fisheye
# ---------------------------------------------------------------------------


def test_camera_info_is_fisheye() -> None:
    """is_fisheye returns True for all fisheye model strings."""
    for model in FISHEYE_DISTORTION_MODELS:
        ci = CameraInfo(distortion_model=model)
        assert ci.is_fisheye, f"Expected is_fisheye=True for '{model}'"

    for model in ("plumb_bob", "", "none"):
        ci = CameraInfo(distortion_model=model)
        assert not ci.is_fisheye, f"Expected is_fisheye=False for '{model}'"


# ---------------------------------------------------------------------------
# CameraInfo.rectified
# ---------------------------------------------------------------------------


def test_camera_info_rectified_fisheye() -> None:
    """rectified() returns a CameraInfo with zero distortion for fisheye."""
    ci = _make_fisheye_camera_info()
    rect = ci.rectified(balance=0.0)

    assert rect.distortion_model == ""
    assert rect.D == []
    assert rect.width == ci.width
    assert rect.height == ci.height

    Knew = rect.get_K_matrix()
    assert Knew.shape == (3, 3)
    assert Knew[2, 2] == pytest.approx(1.0)
    # Focal length should be positive
    assert Knew[0, 0] > 0
    assert Knew[1, 1] > 0


def test_camera_info_rectified_plumb_bob() -> None:
    """rectified() returns a CameraInfo with zero distortion for plumb_bob."""
    ci = _make_plumb_bob_camera_info()
    rect = ci.rectified(balance=0.0)

    assert rect.distortion_model == ""
    assert rect.D == []

    Knew = rect.get_K_matrix()
    assert Knew.shape == (3, 3)
    assert Knew[0, 0] > 0
    assert Knew[1, 1] > 0


def test_camera_info_rectified_no_distortion() -> None:
    """rectified() on an already-rectified camera returns a copy."""
    ci = CameraInfo(
        height=480, width=640,
        distortion_model="",
        K=[500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
    )
    rect = ci.rectified()
    assert np.allclose(rect.get_K_matrix(), ci.get_K_matrix())


# ---------------------------------------------------------------------------
# create_undistortion_maps
# ---------------------------------------------------------------------------


def test_create_undistortion_maps_fisheye() -> None:
    """Maps are valid numpy arrays with correct shape for fisheye."""
    ci = _make_fisheye_camera_info()
    map1, map2, rect_info = create_undistortion_maps(ci)

    assert map1.shape[:2] == (ci.height, ci.width)
    assert map2.shape[:2] == (ci.height, ci.width)
    assert rect_info.distortion_model == ""


def test_create_undistortion_maps_plumb_bob() -> None:
    """Maps are valid numpy arrays with correct shape for plumb_bob."""
    ci = _make_plumb_bob_camera_info()
    map1, map2, rect_info = create_undistortion_maps(ci)

    assert map1.shape[:2] == (ci.height, ci.width)
    assert map2.shape[:2] == (ci.height, ci.width)
    assert rect_info.distortion_model == ""


# ---------------------------------------------------------------------------
# remap_image
# ---------------------------------------------------------------------------


def test_remap_preserves_metadata() -> None:
    """remap_image preserves format, frame_id, and timestamp."""
    ci = _make_fisheye_camera_info()
    map1, map2, _ = create_undistortion_maps(ci)
    img = _make_test_image(ci.width, ci.height)

    result = remap_image(img, map1, map2)

    assert result.format == img.format
    assert result.frame_id == img.frame_id
    assert result.ts == img.ts
    assert result.data.shape[:2] == (ci.height, ci.width)


# ---------------------------------------------------------------------------
# rectify_image (end-to-end)
# ---------------------------------------------------------------------------


def test_rectify_image_fisheye() -> None:
    """End-to-end fisheye rectification produces valid output."""
    ci = _make_fisheye_camera_info()
    img = _make_test_image(ci.width, ci.height)

    rect_img, rect_info = rectify_image(img, ci)

    assert rect_img.data.shape[:2] == (ci.height, ci.width)
    assert rect_info.distortion_model == ""
    assert rect_info.D == []


def test_rectify_image_plumb_bob() -> None:
    """End-to-end plumb_bob rectification produces valid output."""
    ci = _make_plumb_bob_camera_info()
    img = _make_test_image(ci.width, ci.height)

    rect_img, rect_info = rectify_image(img, ci)

    assert rect_img.data.shape[:2] == (ci.height, ci.width)
    assert rect_info.distortion_model == ""


def test_depth_uses_nearest_interpolation() -> None:
    """Depth images should not have blended values at edges."""
    ci = _make_fisheye_camera_info()
    map1, map2, _ = create_undistortion_maps(ci)

    # Create depth image with two distinct depth values
    depth_data = np.ones((ci.height, ci.width), dtype=np.float32) * 2.0
    depth_data[:, ci.width // 2 :] = 5.0
    depth_img = Image(data=depth_data, format=ImageFormat.DEPTH, frame_id="test", ts=0.0)

    result = remap_image(depth_img, map1, map2)

    # With INTER_NEAREST, all non-zero pixels should be exactly 2.0 or 5.0
    nonzero = result.data[result.data != 0.0]
    if len(nonzero) > 0:
        unique_vals = np.unique(nonzero)
        for v in unique_vals:
            assert v == pytest.approx(2.0) or v == pytest.approx(5.0), (
                f"Depth value {v} is neither 2.0 nor 5.0 — interpolation blended depths"
            )


# ---------------------------------------------------------------------------
# UndistortionMapCache
# ---------------------------------------------------------------------------


def test_undistortion_map_cache_returns_same_maps() -> None:
    """Second call returns the exact same cached map objects."""
    cache = UndistortionMapCache()
    ci = _make_fisheye_camera_info()

    map1_a, map2_a, info_a = cache.get_maps(ci)
    map1_b, map2_b, info_b = cache.get_maps(ci)

    assert map1_a is map1_b
    assert map2_a is map2_b
    assert info_a is info_b


def test_undistortion_map_cache_rectify() -> None:
    """Cache.rectify produces valid output."""
    cache = UndistortionMapCache()
    ci = _make_fisheye_camera_info()
    img = _make_test_image(ci.width, ci.height)

    rect_img, rect_info = cache.rectify(img, ci)

    assert rect_img.data.shape[:2] == (ci.height, ci.width)
    assert rect_info.distortion_model == ""


def test_undistortion_map_cache_different_params() -> None:
    """Different balance values produce different cache entries."""
    cache = UndistortionMapCache()
    ci = _make_fisheye_camera_info()

    _, _, info_a = cache.get_maps(ci, balance=0.0)
    _, _, info_b = cache.get_maps(ci, balance=1.0)

    # Different balance should yield different intrinsics
    assert info_a is not info_b

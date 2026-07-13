# Copyright 2026 Dimensional Inc.
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

"""Monocular depth estimation (Depth Anything 3, Apple Depth Pro).

``estimator`` holds the model wrappers (usable from any script); ``module``
wraps them as a dimos Module with Image/CameraInfo streams in and depth
Image + camera-frame PointCloud2 out.
"""

from dimos.perception.depth.estimator import (
    DA3Estimator,
    DepthEstimator,
    DepthProEstimator,
    filter_depth_edges,
    make_depth_estimator,
)

__all__ = [
    "DA3Estimator",
    "DepthEstimator",
    "DepthProEstimator",
    "filter_depth_edges",
    "make_depth_estimator",
]

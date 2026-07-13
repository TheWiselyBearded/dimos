import json, sys
import numpy as np
sys.path.insert(0, "/Users/reza/Documents/Tools/dimos")
from mac_iphone_spatial_foxglove import load_camera_calibration, make_camera_info_from_K
from pathlib import Path

K_true = np.array([[912.3, 0, 654.2], [0, 910.8, 351.7], [0, 0, 1.0]])
D_true = [0.12, -0.31, 0.001, -0.002, 0.15]

# 1. reachy-solver-style JSON (row-major)
j = {"class_name": "PinholeCameraParameters",
     "intrinsic": {"width": 1280, "height": 720,
                   "intrinsic_matrix": K_true.flatten().tolist()},
     "distortion": {"model": "plumb_bob", "coeffs": D_true}}
p = Path("cal_row.json"); p.write_text(json.dumps(j))
K, D, size = load_camera_calibration(p)
assert np.allclose(K, K_true) and np.allclose(D, D_true) and size == (1280, 720)
print("row-major JSON OK")

# 2. Open3D-style column-major JSON
j["intrinsic"]["intrinsic_matrix"] = K_true.T.flatten().tolist()
p2 = Path("cal_col.json"); p2.write_text(json.dumps(j))
K2, _, _ = load_camera_calibration(p2)
assert np.allclose(K2, K_true), f"column-major transpose failed:\n{K2}"
print("column-major JSON OK")

# 3. ROS CameraInfo YAML
yaml_text = f"""
image_width: 1280
image_height: 720
distortion_model: plumb_bob
camera_matrix:
  rows: 3
  cols: 3
  data: {K_true.flatten().tolist()}
distortion_coefficients:
  rows: 1
  cols: 5
  data: {D_true}
"""
p3 = Path("cal.yaml"); p3.write_text(yaml_text)
K3, D3, size3 = load_camera_calibration(p3)
assert np.allclose(K3, K_true) and np.allclose(D3, D_true) and size3 == (1280, 720)
print("CameraInfo YAML OK")

ci = make_camera_info_from_K(K_true, 1280, 720)
assert abs(ci.K[0] - 912.3) < 1e-9 and abs(ci.K[4] - 910.8) < 1e-9
assert abs(ci.K[2] - 654.2) < 1e-9 and abs(ci.K[5] - 351.7) < 1e-9
print("make_camera_info_from_K OK")

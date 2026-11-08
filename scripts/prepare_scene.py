"""Convert posed RGB-D frames into the ScanNet-style layout Qwen-3D's mappers read.

Inputs (pick one):
  --tum DIR       a TUM RGB-D sequence (rgb.txt, depth.txt, groundtruth.txt)
  --generic DIR   color/*.{jpg,png}, depth/*.png (uint16 mm), pose/*.txt (4x4 cam2world,
                  OpenCV axes), intrinsic.txt (3x3)  -- e.g. written by video_to_rgbd.py

Outputs (under data/custom/):
  frames/<scene>/{color/<i>.jpg, depth/<i>.png, depth_inpainted/<i>.png, pose/<i>.txt, intrinsic.txt}
  mesh/val/<scene>.npy   fused pseudo-mesh, float32 [N,12] = xyz, rgb, ones, segment, sem, inst
  mesh/train_validation_database.yaml   (lists every scene in mesh/val)

The world is re-oriented to match ScanNet's axis-aligned scenes: z up (estimated from the mean
camera "up" direction), floor at z=0, xy starting near 0, metres.
"""
import argparse
import glob
import importlib.util
import os

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(PROJ, "data", "custom")
W, H = 640, 480  # the resolution Qwen-3D was trained on (ScanNet)

# Reuse the exact depth inpainting from the repo's ScanNet preprocessing
_spec = importlib.util.spec_from_file_location(
    "inpaint_depth", os.path.join(PROJ, "qwen3d", "qwen3d", "utils", "inpaint_depth.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
inpaint_depth = _mod.inpaint_depth

TUM_INTRINSICS = {  # https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


def _read_tum_list(path):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    return np.array([float(r[0]) for r in rows]), [r[1:] for r in rows]


def load_tum(seq_dir):
    """Return (color_paths, depth_mm_loader, poses[N,4,4], K[3,3]) for a TUM sequence."""
    t_rgb, rgb = _read_tum_list(os.path.join(seq_dir, "rgb.txt"))
    t_d, dep = _read_tum_list(os.path.join(seq_dir, "depth.txt"))
    t_gt, gt = _read_tum_list(os.path.join(seq_dir, "groundtruth.txt"))
    gt = np.array(gt, dtype=np.float64)

    colors, depths, poses = [], [], []
    for t, (rgb_file,) in zip(t_rgb, rgb):
        i_d, i_gt = np.abs(t_d - t).argmin(), np.abs(t_gt - t).argmin()
        if abs(t_d[i_d] - t) > 0.02 or abs(t_gt[i_gt] - t) > 0.02:
            continue
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(gt[i_gt, 3:7]).as_matrix()  # qx qy qz qw
        pose[:3, 3] = gt[i_gt, :3]
        colors.append(os.path.join(seq_dir, rgb_file))
        depths.append(os.path.join(seq_dir, dep[i_d][0]))
        poses.append(pose)

    cam = next((k for k in TUM_INTRINSICS if k in os.path.basename(os.path.normpath(seq_dir))), None)
    assert cam is not None, f"cannot tell the TUM camera (freiburg1/2/3) from {seq_dir}"
    fx, fy, cx, cy = TUM_INTRINSICS[cam]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # TUM depth PNGs store 5000 units per metre; convert to millimetres like ScanNet
    load_depth = lambda p: (cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 5.0)
    return colors, depths, load_depth, np.stack(poses), K


def load_generic(scene_dir):
    colors = sorted(glob.glob(os.path.join(scene_dir, "color", "*")), key=_frame_key)
    depths = sorted(glob.glob(os.path.join(scene_dir, "depth", "*.png")), key=_frame_key)
    poses = sorted(glob.glob(os.path.join(scene_dir, "pose", "*.txt")), key=_frame_key)
    assert len(colors) == len(depths) == len(poses) > 0, (len(colors), len(depths), len(poses))
    K = np.loadtxt(os.path.join(scene_dir, "intrinsic.txt"))[:3, :3]
    load_depth = lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return colors, depths, load_depth, np.stack([np.loadtxt(p) for p in poses]), K


def _frame_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return float(stem)
    except ValueError:
        return stem


def to_640x480(color, depth, K):
    """Center-crop to 4:3 and resize to 640x480, adjusting the intrinsics."""
    h, w = depth.shape
    K = K.copy()
    if w * 3 > h * 4:  # too wide
        new_w = h * 4 // 3
        x0 = (w - new_w) // 2
        color, depth, K[0, 2] = color[:, x0:x0 + new_w], depth[:, x0:x0 + new_w], K[0, 2] - x0
    elif w * 3 < h * 4:  # too tall
        new_h = w * 3 // 4
        y0 = (h - new_h) // 2
        color, depth, K[1, 2] = color[y0:y0 + new_h], depth[y0:y0 + new_h], K[1, 2] - y0
    sx, sy = W / depth.shape[1], H / depth.shape[0]
    K[0] *= sx
    K[1] *= sy
    color = cv2.resize(color, (W, H), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    return color, depth, K


def backproject(depth_m, K, pose, stride=2, max_depth=4.0):
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[v, u]
    keep = (z > 0.1) & (z < max_depth)
    u, v, z = u[keep], v[keep], z[keep]
    pts_cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    return pts_cam @ pose[:3, :3].T + pose[:3, 3], (v, u)


def fit_floor_normal(points, up, iters=500, thresh=0.03, max_angle_deg=30, min_frac=0.05):
    """RANSAC a plane among the lowest points whose normal is near `up`; None if no floor."""
    rng = np.random.default_rng(0)
    h = points @ up
    low = points[h < np.percentile(h, 30)]
    low = low[rng.choice(len(low), min(len(low), 50000), replace=False)]
    best, best_n = 0, None
    for _ in range(iters):
        a, b, c = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        n = n if n @ up > 0 else -n
        if np.degrees(np.arccos(np.clip(n @ up, -1, 1))) > max_angle_deg:
            continue
        inliers = np.abs((low - a) @ n) < thresh
        if inliers.sum() > best:
            best = inliers.sum()
            # least-squares refit on the inliers
            q = low[inliers] - low[inliers].mean(0)
            best_n = np.linalg.svd(q, full_matrices=False)[2][-1]
            best_n = best_n if best_n @ up > 0 else -best_n
    return best_n if best > min_frac * len(low) else None


def gravity_align(poses, points):
    """Rigid transform making z point up, floor z=0, xy min ~0.

    Up is the floor-plane normal when a floor is visible, else the mean camera -y axis.
    """
    up = -(poses[:, :3, :3] @ np.array([0.0, 1.0, 0.0])).mean(0)
    up /= np.linalg.norm(up)
    floor_n = fit_floor_normal(points, up)
    if floor_n is not None:
        print(f"floor plane found, {np.degrees(np.arccos(floor_n @ up)):.1f} deg from camera-up estimate")
        up = floor_n
    else:
        print("no floor plane found; using the camera-up estimate")
    R = Rotation.align_vectors([[0, 0, 1]], [up])[0].as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    p = points @ R.T
    T[:3, 3] = -np.array([np.percentile(p[:, 0], 1), np.percentile(p[:, 1], 1), np.percentile(p[:, 2], 1)])
    return T


def voxelize(xyz, rgb, voxel):
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    n = counts.shape[0]
    out_xyz = np.stack([np.bincount(inv, xyz[:, i], n) for i in range(3)], -1) / counts[:, None]
    out_rgb = np.stack([np.bincount(inv, rgb[:, i], n) for i in range(3)], -1) / counts[:, None]
    return out_xyz, out_rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tum")
    src.add_argument("--generic")
    ap.add_argument("--scene", required=True, help="scene id, e.g. tum_office (must not contain 'depth')")
    ap.add_argument("--num-frames", type=int, default=90, help="frames kept, evenly spaced")
    ap.add_argument("--voxel", type=float, default=0.02, help="pseudo-mesh voxel size (m)")
    ap.add_argument("--max-depth", type=float, default=4.0, help="max depth fused into the pseudo-mesh (m)")
    args = ap.parse_args()
    assert "depth" not in args.scene, "the repo rewrites every 'depth' in paths; pick another scene id"

    colors, depths, load_depth, poses, K0 = load_tum(args.tum) if args.tum else load_generic(args.generic)
    sel = np.linspace(0, len(colors) - 1, min(args.num_frames, len(colors))).round().astype(int)
    print(f"{len(colors)} posed frames available, keeping {len(sel)}")

    frames = []
    for i in sel:
        color = cv2.imread(colors[i], cv2.IMREAD_COLOR)
        depth_mm = load_depth(depths[i])
        if color.shape[:2] != depth_mm.shape:
            color = cv2.resize(color, depth_mm.shape[::-1], interpolation=cv2.INTER_AREA)
        color, depth_mm, K = to_640x480(color, depth_mm, K0)
        frames.append((color, depth_mm, poses[i]))

    # Fuse frames in the original world frame, then gravity-align everything
    xyz, rgb = [], []
    for color, depth_mm, pose in frames:
        p, (v, u) = backproject(depth_mm / 1000.0, K, pose, max_depth=args.max_depth)
        xyz.append(p)
        rgb.append(color[v, u, ::-1].astype(np.float64))
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    T = gravity_align(np.stack([f[2] for f in frames]), xyz)
    xyz = xyz @ T[:3, :3].T + T[:3, 3]
    xyz, rgb = voxelize(xyz, rgb, args.voxel)

    frame_dir = os.path.join(OUT_ROOT, "frames", args.scene)
    for sub in ("color", "depth", "depth_inpainted", "pose"):
        os.makedirs(os.path.join(frame_dir, sub), exist_ok=True)
    for j, (color, depth_mm, pose) in enumerate(frames):
        depth_u16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(frame_dir, "color", f"{j}.jpg"), color, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(frame_dir, "depth", f"{j}.png"), depth_u16)
        cv2.imwrite(os.path.join(frame_dir, "depth_inpainted", f"{j}.png"),
                    inpaint_depth(depth_u16.astype(np.float32)).astype(np.uint16))
        np.savetxt(os.path.join(frame_dir, "pose", f"{j}.txt"), T @ pose)
    np.savetxt(os.path.join(frame_dir, "intrinsic.txt"), K)

    mesh_dir = os.path.join(OUT_ROOT, "mesh", "val")
    os.makedirs(mesh_dir, exist_ok=True)
    n = len(xyz)
    mesh = np.concatenate(
        [xyz, rgb, np.ones((n, 3)), np.arange(n)[:, None], np.zeros((n, 1)), -np.ones((n, 1))], 1
    ).astype(np.float32)
    np.save(os.path.join(mesh_dir, f"{args.scene}.npy"), mesh)

    db = [
        {"filepath": p, "raw_filepath": f"/custom/{os.path.basename(p)[:-4]}/mesh.ply"}
        for p in sorted(glob.glob(os.path.join(mesh_dir, "*.npy")))
    ]
    with open(os.path.join(OUT_ROOT, "mesh", "train_validation_database.yaml"), "w") as f:
        yaml.safe_dump(db, f)

    ext = xyz.max(0) - xyz.min(0)
    print(f"wrote {len(frames)} frames to {frame_dir}")
    print(f"pseudo-mesh: {n} points, extent {ext.round(2)} m -> {mesh_dir}/{args.scene}.npy")


if __name__ == "__main__":
    main()

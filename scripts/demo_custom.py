"""Run Qwen-3D on a scene prepared by prepare_scene.py.

Grounding (3D mask for a referring expression):
  python scripts/demo_custom.py --scene tum_office --query "the blue ball on the desk"
QA (free-form answer):
  python scripts/demo_custom.py --scene tum_office --question "What color is the chair?"

Outputs go to outputs/demo/<scene>/: a colored .ply with the predicted object in red, and a
PNG showing the mask projected into the input frame that sees it best plus a top view.
Library logs go to outputs/demo/<scene>/last_run.log unless --verbose is given.
"""
import argparse
import contextlib
import glob
import os
import re
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(PROJ, "qwen3d")
os.chdir(REPO)  # the repo opens some files by relative path
sys.path.insert(0, REPO)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoids a fork warning when writing videos
os.environ["SCANNET_DATA_DIR"] =os.path.join(PROJ, "data", "custom", "mesh", "train_validation_database.yaml")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.engine import default_argument_parser  # noqa: E402


@contextlib.contextmanager
def logs_to(log_path, enabled=True):
    """Send everything written to stdout/stderr (including C-level output) to log_path."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(log_path, "a") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_cfg(max_frames, extra_opts):
    from train import setup

    args = default_argument_parser().parse_args(
        [
            "--config-file", "qwen3d/configs/qwen_3d.yaml",
            "--eval-only",
            "OUTPUT_DIR", os.path.join(PROJ, "outputs", "demo"),
            "MODEL.WEIGHTS", os.environ["CKPT_PATH"],
            "QWEN_MODEL", os.environ["QWEN_MODEL"],
            "USE_WANDB", "False",
            "GENERATION", "True",
            "USE_AUTO_NOUN_DETECTION", "False",
            "LOAD_SCANENTS", "False",
            "USE_SCAN_ALIGN_MATRIX", "False",
            "USE_GHOST_POINTS", "True",
            "USE_GT_MASKS", "False",
            "CACHE_QWEN_FEATURES", "False",
            "FORCE_SUBSAMPLE", "False",
            "SCANNET_DATA_DIR", os.environ["SCANNET_DATA_DIR"],
            "MAX_FRAME_NUM", str(max_frames),
            *extra_opts,
        ]
    )
    return setup(args)


def scene_record(scene, frame_root):
    color = sorted(glob.glob(os.path.join(frame_root, scene, "color", "*.jpg")),
                   key=lambda p: int(os.path.basename(p)[:-4]))
    n = len(color)
    assert n > 0, f"no frames for scene {scene} under {frame_root}"
    return {
        "image_id": scene, "height": 480, "width": 640, "length": n,
        "file_names": color,
        "depth_file_names": [p.replace("color", "depth").replace("jpg", "png") for p in color],
        "pose_file_names": [p.replace("color", "pose").replace("jpg", "txt") for p in color],
        "image_ids": list(range(n)), "annotations": [[] for _ in range(n)],
        "segment_file_names": [None] * n, "valid_file_names": [None] * n,
    }


def target_span(utterance, target):
    """Character span of the target noun phrase in the lowercased utterance."""
    from qwen3d.data_video.datasets.ref_coco_utils import consolidate_spans, get_root_and_nouns

    if target:
        beg = utterance.find(target.lower())
        assert beg >= 0, f"--target '{target}' not found in the query"
        return [[[beg, beg + len(target)]]], target.lower()
    _, _, root_spans, _ = get_root_and_nouns(utterance)
    spans = consolidate_spans(root_spans, utterance)
    return [spans], " ".join(utterance[b:e] for b, e in spans)


def save_ply(path, xyz, rgb):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(xyz, rgb.astype(np.uint8)):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def pose_files(frame_root, scene):
    return sorted(glob.glob(os.path.join(frame_root, scene, "pose", "*.txt")),
                  key=lambda p: int(os.path.basename(p)[:-4]))


def visible_uv(points, pose_file, K):
    """Pixel coords of the points visible in this frame (depth test against its depth map)."""
    w2c = np.linalg.inv(np.loadtxt(pose_file))
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    pc = pc[pc[:, 2] > 0.1]
    uv = pc[:, :2] / pc[:, 2:] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
    uv, z = uv[inside], pc[inside, 2]
    depth = cv2.imread(pose_file.replace("pose", "depth_inpainted").replace("txt", "png"),
                       cv2.IMREAD_UNCHANGED) / 1000.0
    return uv[np.abs(depth[uv[:, 1].astype(int), uv[:, 0].astype(int)] - z) < 0.05]


def render(path, xyz, rgb, mask, frame_root, scene, title):
    """Mask projected into the frame that sees the most masked points, plus a top view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = np.loadtxt(os.path.join(frame_root, scene, "intrinsic.txt"))
    best, best_uv, best_i = -1, None, 0
    for i, pf in enumerate(pose_files(frame_root, scene)):
        uv = visible_uv(xyz[mask], pf, K)
        if len(uv) > best:
            best, best_uv, best_i = len(uv), uv, i

    img = cv2.imread(os.path.join(frame_root, scene, "color", f"{best_i}.jpg"))[:, :, ::-1]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].imshow(img)
    ax[0].scatter(best_uv[:, 0], best_uv[:, 1], s=1, c="red", alpha=0.4)
    ax[0].set_title(f"frame {best_i}: predicted mask")
    ax[0].axis("off")
    order = np.argsort(xyz[:, 2])
    ax[1].scatter(xyz[order, 0], xyz[order, 1], c=rgb[order] / 255, s=0.3)
    ax[1].scatter(xyz[mask, 0], xyz[mask, 1], c="red", s=0.6)
    ax[1].set_aspect("equal")
    ax[1].set_title("top view")
    fig.suptitle(title)
    plt.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def caption(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def splat(xyz, rgb, eye, target, K, size=(480, 640)):
    """Z-buffered point rendering from a look-at camera (z up), OpenCV convention."""
    f = target - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(f, right), f])
    pc = (xyz - eye) @ R.T
    keep = pc[:, 2] > 0.1
    pc, col = pc[keep], rgb[keep]
    u = (pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = (pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    order = np.argsort(-pc[:, 2])  # far to near, so near points are written last
    img = np.full((*size, 3), 255, np.uint8)
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = u[order] + du, v[order] + dv
            ok = (uu >= 0) & (uu < size[1]) & (vv >= 0) & (vv < size[0])
            img[vv[ok], uu[ok]] = col[order][ok]
    return img


def make_video(path, xyz, rgb, mask, frame_root, scene, title, fps=15, orbit_frames=150):
    """Walk through the input frames with the mask overlaid, then orbit the object in 3D."""
    import imageio.v2 as iio

    K = np.loadtxt(os.path.join(frame_root, scene, "intrinsic.txt"))
    writer = iio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16)

    for i, pf in enumerate(pose_files(frame_root, scene)):
        img = cv2.imread(os.path.join(frame_root, scene, "color", f"{i}.jpg"))[:, :, ::-1].copy()
        uv = visible_uv(xyz[mask], pf, K).astype(int)
        overlay = img.copy()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                overlay[np.clip(uv[:, 1] + dv, 0, 479), np.clip(uv[:, 0] + du, 0, 639)] = (255, 0, 0)
        img = cv2.addWeighted(img, 0.45, overlay, 0.55, 0)
        writer.append_data(caption(img, f"{title}   [input frame {i}]"))

    vis = rgb.astype(np.uint8).copy()
    vis[mask] = (255, 0, 0)
    obj = xyz[mask]
    target = (obj.min(0) + obj.max(0)) / 2
    radius = max(2.0, 2.5 * np.linalg.norm(obj.max(0) - obj.min(0)))
    for a in np.linspace(0, 2 * np.pi, orbit_frames, endpoint=False):
        eye = target + [radius * np.cos(a), radius * np.sin(a), 0.6 * radius]
        writer.append_data(caption(splat(xyz, vis, eye, target, K), f"{title}   [3D view]"))
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    task = ap.add_mutually_exclusive_group(required=True)
    task.add_argument("--query", help="referring expression to ground in 3D")
    task.add_argument("--question", help="question to answer about the scene")
    ap.add_argument("--target", help="target noun phrase inside --query (default: spaCy root noun)")
    ap.add_argument("--max-frames", type=int, default=-1, help="MAX_FRAME_NUM (-1 = all prepared frames)")
    ap.add_argument("--video", action="store_true", help="also write an MP4 (walkthrough + 3D orbit)")
    ap.add_argument("--verbose", action="store_true", help="show library logs instead of writing them to a file")
    ap.add_argument("opts", nargs=argparse.REMAINDER, help="extra cfg overrides")
    args = ap.parse_args()

    frame_root = os.path.join(PROJ, "data", "custom", "frames")
    out_dir = os.path.join(PROJ, "outputs", "demo", args.scene)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "last_run.log")
    if not args.verbose:
        open(log_path, "w").close()
    quiet = not args.verbose
    t0 = time.time()

    step(f"loading Qwen-3D ({os.path.basename(os.environ['CKPT_PATH'])}, backbone {os.environ['QWEN_MODEL']})")
    with logs_to(log_path, quiet):
        from qwen3d.data_video.dataset_mapper_language import Sr3dDatasetMapper
        from train import Trainer

        cfg = build_cfg(args.max_frames, args.opts)
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval()

    step(f"reading scene '{args.scene}'")
    with logs_to(log_path, quiet):
        # Dataset names steer the repo's code paths: "test" skips all ground-truth handling,
        # "scanqa" switches to answer generation.
        name = "scanqa_ref_scannet_test" if args.question else "custom_ref_scannet_test"
        mapper = Sr3dDatasetMapper(cfg, False, name, scannet_dict=[scene_record(args.scene, frame_root)],
                                   scene_to_id_map={args.scene: 0}, decoder_3d=True)
        K = np.eye(4)
        K[:3, :3] = np.loadtxt(os.path.join(frame_root, args.scene, "intrinsic.txt"))
        mapper.scannet_mapper.get_intrinsics = lambda size, intrinsic_file=None: K.copy()

        if args.question:
            query = {"scan_id": args.scene, "question": args.question, "object_ids": [0], "answers": [""]}
        else:
            utt = re.sub(r"\s+", " ", args.query.strip()).lower()
            tokens_positive, target = target_span(utt, args.target)
            query = {"scan_id": args.scene, "dataset": "custom", "utterance": utt, "target_id": 0,
                     "instance_type": target, "anchor_ids": [], "anchors_types": [],
                     "tokens_positive": tokens_positive}
        inp = mapper([query])
    if not args.question:
        step(f"grounding target span: '{target}'")

    step(f"running inference on {len(inp['images'])} frames")
    with logs_to(log_path, quiet), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model([inp])
    step(f"done in {time.time() - t0:.0f}s total, "
         f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if quiet:
        print(f"           (full logs: {log_path})")

    if args.question:
        print(f"\nQ: {args.question}\nA: {out[0][0]}")
        return

    res = out[0]["instances_3d"]
    scores = res["pred_scores"][:, 0].float().cpu()
    k = int(scores.argmax())
    mask = (res["pred_masks"][k].float().sigmoid() > 0.5).cpu().numpy()
    xyz = np.asarray(inp["scannet_coords"].cpu() if torch.is_tensor(inp["scannet_coords"]) else inp["scannet_coords"])
    rgb = np.asarray(inp["scannet_color"].cpu() if torch.is_tensor(inp["scannet_color"]) else inp["scannet_color"])
    rgb = rgb * 255 if rgb.max() <= 1.0 else rgb
    print(f"\nquery: {args.query}\nbest query {k}/{len(scores)} score={scores[k]:.3f}, "
          f"{mask.sum()} of {len(mask)} points")
    if mask.sum():
        lo, hi = xyz[mask].min(0), xyz[mask].max(0)
        print(f"bbox center {((lo + hi) / 2).round(2)} m, size {(hi - lo).round(2)} m")

    slug = re.sub(r"[^a-z0-9]+", "_", args.query.lower()).strip("_")[:60]
    vis = rgb.copy()
    vis[mask] = [255, 0, 0]
    save_ply(os.path.join(out_dir, f"{slug}.ply"), xyz, vis)
    if mask.sum():
        render(os.path.join(out_dir, f"{slug}.png"), xyz, rgb, mask, frame_root, args.scene,
               f"{args.query}  (score {scores[k]:.2f})")
    print(f"saved {out_dir}/{slug}.ply" + (" and .png" if mask.sum() else ""))
    if args.video and mask.sum():
        make_video(os.path.join(out_dir, f"{slug}.mp4"), xyz, rgb, mask, frame_root, args.scene,
                   f'"{args.query}"  score {scores[k]:.2f}')
        print(f"saved {out_dir}/{slug}.mp4")


if __name__ == "__main__":
    main()

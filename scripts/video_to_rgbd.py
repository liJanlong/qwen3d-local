"""Turn a plain video (or a folder of images) into posed metric RGB-D frames with MapAnything.

Runs in the `mapanything` conda env, and writes the --generic layout read by prepare_scene.py:
  OUT/color/<i>.jpg, OUT/depth/<i>.png (uint16 mm), OUT/pose/<i>.txt (4x4 cam2world, OpenCV),
  OUT/intrinsic.txt (3x3, the median over views)

  PYTHONNOUSERSITE=1 conda run --no-capture-output -n mapanything python scripts/video_to_rgbd.py my_scenes/room.mp4 my_scenes/room_rgbd
"""
import argparse
import glob
import os
import tempfile

import cv2
import numpy as np
import torch


def extract_frames(video, num_frames, out_dir):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert total > 0, f"cannot read frames from {video}"
    paths = []
    for j, idx in enumerate(np.linspace(0, total - 1, min(num_frames, total)).round().astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            paths.append(os.path.join(out_dir, f"{j:05d}.jpg"))
            cv2.imwrite(paths[-1], frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="video file, or a folder of images (sorted by name)")
    ap.add_argument("out", help="output folder")
    ap.add_argument("--num-frames", type=int, default=60, help="frames sampled evenly from the input")
    ap.add_argument("--model", default="facebook/map-anything-apache")
    args = ap.parse_args()

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    with tempfile.TemporaryDirectory() as tmp:
        if os.path.isdir(args.input):
            paths = sorted(p for p in glob.glob(os.path.join(args.input, "*"))
                           if p.lower().endswith((".jpg", ".jpeg", ".png")))
            paths = [paths[i] for i in np.linspace(0, len(paths) - 1, min(args.num_frames, len(paths))).round().astype(int)]
        else:
            paths = extract_frames(args.input, args.num_frames, tmp)
        print(f"reconstructing {len(paths)} frames with {args.model}")

        model = MapAnything.from_pretrained(args.model).to("cuda")
        views = load_images(paths)
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                            apply_mask=True, mask_edges=True)

    for sub in ("color", "depth", "pose"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    Ks = []
    for i, p in enumerate(preds):
        img = (p["img_no_norm"][0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        depth = p["depth_z"][0, ..., 0].float().cpu().numpy()
        depth[~p["mask"][0, ..., 0].bool().cpu().numpy()] = 0  # low confidence / edges -> hole
        cv2.imwrite(os.path.join(args.out, "color", f"{i}.jpg"), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(args.out, "depth", f"{i}.png"), (depth * 1000).clip(0, 65535).astype(np.uint16))
        np.savetxt(os.path.join(args.out, "pose", f"{i}.txt"), p["camera_poses"][0].float().cpu().numpy())
        Ks.append(p["intrinsics"][0].float().cpu().numpy())
    np.savetxt(os.path.join(args.out, "intrinsic.txt"), np.median(np.stack(Ks), 0))
    print(f"wrote {len(preds)} posed RGB-D frames ({img.shape[1]}x{img.shape[0]}) to {args.out}")


if __name__ == "__main__":
    main()

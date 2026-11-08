"""Render a `script -t` terminal capture to MP4 at its real timing, then append result videos.

  script -q -t 2>outputs/recordings/timing.log -c "bash scripts/record_session.sh" outputs/recordings/session.log
  python scripts/render_recording.py outputs/recordings/session.log outputs/recordings/timing.log \
      outputs/recordings/qwen3d_demo.mp4 --append outputs/demo/tum_office/the_teddy_bear.mp4
"""
import argparse

import imageio.v2 as iio
import numpy as np
import pyte
from PIL import Image, ImageDraw, ImageFont

COLS, ROWS = 110, 34
W, H = 1280, 720
FPS = 15
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
BG = (30, 30, 30)
COLORS = {"default": (220, 220, 220), "green": (110, 210, 110), "blue": (100, 150, 240),
          "red": (230, 90, 90), "yellow": (230, 200, 90), "cyan": (90, 200, 210),
          "magenta": (200, 120, 220), "white": (240, 240, 240), "black": (60, 60, 60)}


def read_capture(log_path, timing_path):
    data = open(log_path, "rb").read()
    data = data[data.index(b"\n") + 1:]  # drop the "Script started on ..." header
    chunks, t, pos = [], 0.0, 0
    for line in open(timing_path):
        delay, n = line.split()
        t += float(delay)
        chunks.append((t, data[pos:pos + int(n)]))
        pos += int(n)
    return chunks


def draw(screen, fonts, cell, footer):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x0, y0 = 16, 12
    for y in range(ROWS):
        row = screen.buffer[y]
        for x in range(COLS):
            ch = row[x]
            if ch.data.strip():
                d.text((x0 + x * cell[0], y0 + y * cell[1]), ch.data,
                       fill=COLORS.get(ch.fg, COLORS["default"]), font=fonts[ch.bold])
    d.text((W - 16, H - 10), footer, fill=(150, 150, 150), font=fonts[False], anchor="rb")
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("timing")
    ap.add_argument("out")
    ap.add_argument("--append", nargs="*", default=[], help="videos to play after the terminal part")
    args = ap.parse_args()

    fonts = {False: ImageFont.truetype(FONT, 17), True: ImageFont.truetype(FONT_BOLD, 17)}
    l, t, r, b = fonts[False].getbbox("M")
    cell = (fonts[False].getlength("M"), 20)
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)

    chunks = read_capture(args.log, args.timing)
    end = chunks[-1][0] + 1.0
    writer = iio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    i, last = 0, None
    for k in range(int(end * FPS) + 1):
        now = k / FPS
        changed = False
        while i < len(chunks) and chunks[i][0] <= now:
            stream.feed(chunks[i][1].replace(b"\n", b"\r\n").replace(b"\r\r\n", b"\r\n"))
            i += 1
            changed = True
        footer = f"real-time terminal capture  {int(now // 60):02d}:{now % 60:04.1f}"
        last = draw(screen, fonts, cell, footer) if changed or last is None or k % FPS == 0 else last
        writer.append_data(last)

    for path in args.append:
        reader = iio.get_reader(path)
        for frame in reader:
            canvas = np.full((H, W, 3), BG, np.uint8)
            fh, fw = frame.shape[:2]
            s = min((H - 60) / fh, (W - 40) / fw)
            frame = np.asarray(Image.fromarray(frame).resize((int(fw * s), int(fh * s))))
            y, x = 50, (W - frame.shape[1]) // 2
            canvas[y:y + frame.shape[0], x:x + frame.shape[1]] = frame
            img = Image.fromarray(canvas)
            ImageDraw.Draw(img).text((W // 2, 25), f"output of the run above: {path.split('/')[-1]}",
                                     fill=(220, 220, 220), font=fonts[True], anchor="mm")
            writer.append_data(np.asarray(img))
        reader.close()
    writer.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os

RES = 0.05
ORIGIN_X, ORIGIN_Y = -6.0, -3.0
WIDTH_M, HEIGHT_M = 12.0, 6.0
WIDTH_PX = int(WIDTH_M / RES)
HEIGHT_PX = int(HEIGHT_M / RES)

FREE = 254
OCCUPIED = 0

COLUMNS = [
    (-4.8, 2.0), (-3.4, 2.1), (-2.0, 1.9), (-0.6, 2.0), (0.8, 2.1), (2.2, 1.9), (3.6, 2.0),
    (-4.2, 0.8), (-2.8, 0.6), (-1.2, 0.8), (0.4, 0.6), (1.8, 0.8), (3.2, 0.6),
    (-4.8, -1.0), (-3.2, -1.2), (-1.6, -1.0), (0.0, -1.2), (1.6, -1.0), (3.2, -1.2), (4.4, -1.0),
    (-3.8, -2.2), (-2.2, -2.0), (-0.6, -2.2), (1.0, -2.0), (2.6, -2.2), (4.2, -2.0),
]


def world_to_pixel(x: float, y: float):
    return int((x - ORIGIN_X) / RES), int((y - ORIGIN_Y) / RES)


def draw_rect(grid, x1, y1, x2, y2, value=OCCUPIED):
    px1, py1 = world_to_pixel(x1, y1)
    px2, py2 = world_to_pixel(x2, y2)
    px_min, px_max = max(0, min(px1, px2)), min(WIDTH_PX, max(px1, px2))
    py_min, py_max = max(0, min(py1, py2)), min(HEIGHT_PX, max(py1, py2))
    for py in range(py_min, py_max):
        row = grid[py]
        for px in range(px_min, px_max):
            row[px] = value


def draw_circle(grid, cx, cy, radius, value=OCCUPIED):
    px_c, py_c = world_to_pixel(cx, cy)
    r_px = max(1, int(radius / RES))
    r2 = r_px * r_px
    for py in range(max(0, py_c - r_px), min(HEIGHT_PX, py_c + r_px + 1)):
        row = grid[py]
        dy = py - py_c
        for px in range(max(0, px_c - r_px), min(WIDTH_PX, px_c + r_px + 1)):
            dx = px - px_c
            if dx * dx + dy * dy <= r2:
                row[px] = value


def main():
    grid = [[FREE] * WIDTH_PX for _ in range(HEIGHT_PX)]

    # outer room only
    draw_rect(grid, -6.1, -3.0, -5.9, 3.0)
    draw_rect(grid, 5.9, -3.0, 6.1, 3.0)
    draw_rect(grid, -6.0, 2.9, 6.0, 3.1)
    draw_rect(grid, -6.0, -3.1, 6.0, -2.9)

    # dense columns
    for cx, cy in COLUMNS:
        draw_circle(grid, cx, cy, 0.12)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'maps')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'nonlinear_corridor_map.pgm')
    with open(out_path, 'wb') as f:
        f.write(f"P5\\n{WIDTH_PX} {HEIGHT_PX}\\n255\\n".encode())
        for row in reversed(grid):
            f.write(bytes(row))
    print(f"Generated {out_path} ({WIDTH_PX}x{HEIGHT_PX})")


if __name__ == '__main__':
    main()

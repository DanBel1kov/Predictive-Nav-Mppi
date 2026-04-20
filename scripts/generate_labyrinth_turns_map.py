#!/usr/bin/env python3
import os

RES = 0.05
ORIGIN_X, ORIGIN_Y = -7.0, -5.0
WIDTH_M, HEIGHT_M = 14.0, 10.0
WIDTH_PX = int(WIDTH_M / RES)
HEIGHT_PX = int(HEIGHT_M / RES)
FREE = 254
OCCUPIED = 0
OUTER = [(-7.1, -5.0, -6.9, 5.0), (6.9, -5.0, 7.1, 5.0), (-7.0, 4.9, 7.0, 5.1), (-7.0, -5.1, 7.0, -4.9)]
WALLS = [(-4.6, 2.7, -4.4, 4.5), (-4.6, -2.7, -4.4, -0.9), (-2.8000000000000003, 0.9, -2.6, 2.7), (-2.8000000000000003, -4.5, -2.6, -2.7), (-1.0, 2.7, -0.8, 4.5), (-1.0, -2.7, -0.8, -0.9), (0.8, 0.9, 1.0, 2.7), (0.8, -4.5, 1.0, -2.7), (2.6, 2.7, 2.8000000000000003, 4.5), (2.6, -0.9, 2.8000000000000003, 0.9), (2.6, -2.7, 2.8000000000000003, -0.9), (4.4, 0.9, 4.6, 2.7), (-6.300000000000001, 0.8, -4.5, 1.0), (-4.5, 0.8, -2.7, 1.0), (-4.5, -1.0, -2.7, -0.8), (-2.7, 0.8, -0.9, 1.0), (-2.7, -1.0, -0.9, -0.8), (-0.9, 0.8, 0.9, 1.0), (-0.9, -1.0, 0.9, -0.8), (0.9, 0.8, 2.7, 1.0), (0.9, -1.0, 2.7, -0.8), (2.7, 0.8, 4.5, 1.0), (2.7, -2.8000000000000003, 4.5, -2.6), (4.5, -1.0, 6.300000000000001, -0.8)]


def world_to_pixel(x, y):
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


def main():
    grid = [[FREE] * WIDTH_PX for _ in range(HEIGHT_PX)]
    for wall in OUTER + WALLS:
        draw_rect(grid, *wall)
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'maps')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'labyrinth_turns_map.pgm')
    with open(out_path, 'wb') as f:
        f.write(f"P5\n{WIDTH_PX} {HEIGHT_PX}\n255\n".encode())
        for row in reversed(grid):
            f.write(bytes(row))
    print(f"Generated {out_path} ({WIDTH_PX}x{HEIGHT_PX})")


if __name__ == '__main__':
    main()

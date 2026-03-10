#!/usr/bin/env python3
"""
Generate long_corridor_map.pgm from world geometry.
World: 20m x 6.6m, origin [-10, -3.3], resolution 0.05
Central corridor: |y| < 1.8
Gaps: x=-10..-9 and x=9..10 (ends), x=-4..-3 and x=1..2 (mid)
Outer walls at x=±10, y=±3.3
Corridor walls at y=±1.8 (with gaps)
Side corridors empty (no boxes)
"""
import os

RES = 0.05
ORIGIN_X, ORIGIN_Y = -10.0, -3.3
WIDTH_M, HEIGHT_M = 20.0, 6.6
WIDTH_PX = int(WIDTH_M / RES)
HEIGHT_PX = int(HEIGHT_M / RES)

# ROS2 map_server with negate:0: p = (255-pixel)/255
# pixel=0 (black) → p=1.0 → occupied; pixel=254 (white) → p=0 → free
FREE = 254
OCCUPIED = 0
UNKNOWN = 127  # trinary mode: unknown cells

WALL_THICKNESS = 0.15  # half-thickness for walls in meters


def world_to_pixel(x: float, y: float) -> tuple[int, int]:
    px = int((x - ORIGIN_X) / RES)
    py = int((y - ORIGIN_Y) / RES)
    return px, py


def draw_rect(grid, x1, y1, x2, y2, value=OCCUPIED):
    px1, py1 = world_to_pixel(x1, y1)
    px2, py2 = world_to_pixel(x2, y2)
    px1, px2 = max(0, min(px1, px2)), min(WIDTH_PX, max(px1, px2))
    py1, py2 = max(0, min(py1, py2)), min(HEIGHT_PX, max(py1, py2))
    for py in range(py1, py2):
        for px in range(px1, px2):
            grid[py][px] = value


def main():
    grid = [[FREE] * WIDTH_PX for _ in range(HEIGHT_PX)]

    # Outer walls (thick)
    draw_rect(grid, -10 - 0.2, -3.3, -10 + 0.2, 3.3)  # west
    draw_rect(grid, 10 - 0.2, -3.3, 10 + 0.2, 3.3)   # east
    draw_rect(grid, -10, 3.3 - 0.2, 10, 3.3 + 0.2)   # north
    draw_rect(grid, -10, -3.3 - 0.2, 10, -3.3 + 0.2) # south

    # Corridor walls at y=±1.8, with gaps at ends (x=-10..-9, x=9..10) and mid (x=-4..-3, x=1..2)
    # Top wall (y=1.8): segments from -9 to -4, -3 to 1, 2 to 9
    draw_rect(grid, -9, 1.8 - WALL_THICKNESS, -4, 1.8 + WALL_THICKNESS)
    draw_rect(grid, -3, 1.8 - WALL_THICKNESS, 1, 1.8 + WALL_THICKNESS)
    draw_rect(grid, 2, 1.8 - WALL_THICKNESS, 9, 1.8 + WALL_THICKNESS)
    # Bottom wall (y=-1.8)
    draw_rect(grid, -9, -1.8 - WALL_THICKNESS, -4, -1.8 + WALL_THICKNESS)
    draw_rect(grid, -3, -1.8 - WALL_THICKNESS, 1, -1.8 + WALL_THICKNESS)
    draw_rect(grid, 2, -1.8 - WALL_THICKNESS, 9, -1.8 + WALL_THICKNESS)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'maps')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'long_corridor_map.pgm')

    with open(out_path, 'wb') as f:
        f.write(f"P5\n{WIDTH_PX} {HEIGHT_PX}\n255\n".encode())
        for row in grid:
            f.write(bytes(row))

    print(f"Generated {out_path} ({WIDTH_PX}x{HEIGHT_PX})")


if __name__ == '__main__':
    main()

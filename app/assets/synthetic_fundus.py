"""Synthetic Fundus Image Generator for Offline Testing & Dashboard Presets.

Generates realistic circular retinal fundus images with:
- Circular retinal field-of-view (FOV) and background pigmentation
- Optic Disc and Optic Cup with configurable vertical Cup-to-Disc Ratio (vCDR)
- Retinal vascular arcade branching from the Optic Nerve Head
- Configurable microaneurysms (<10px dark red dots), blot hemorrhages, and hard exudates
- Optional blur or flash glare to test the Tier-0 IQA Gatekeeper
"""

import cv2
import numpy as np


def generate_synthetic_fundus_image(
    size: int = 512,
    laterality: str = "OD",
    num_microaneurysms: int = 0,
    num_hemorrhages: int = 0,
    num_exudates: int = 0,
    vcdr: float = 0.34,
    simulate_severe_blur: bool = False,
    simulate_flash_glare: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic RGB fundus image (H, W, 3) uint8."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.44)

    # 1. Circular retinal background (warm orange-red fundus glow)
    y_grid, x_grid = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2)
    fov_mask = dist_from_center <= radius

    radial_falloff = np.clip(1.0 - 0.35 * (dist_from_center / max(radius, 1)) ** 2, 0.0, 1.0)
    r_ch = (195 * radial_falloff).astype(np.uint8)
    g_ch = (85 * radial_falloff).astype(np.uint8)
    b_ch = (35 * radial_falloff).astype(np.uint8)
    img[:, :, 0] = np.where(fov_mask, r_ch, 0)
    img[:, :, 1] = np.where(fov_mask, g_ch, 0)
    img[:, :, 2] = np.where(fov_mask, b_ch, 0)

    # Macula (darker central-temporal depression)
    macula_offset_x = int(-radius * 0.12) if laterality.upper() == "OD" else int(radius * 0.12)
    macula_cx = center[0] + macula_offset_x
    macula_cy = center[1]
    macula_dist = np.sqrt((x_grid - macula_cx) ** 2 + (y_grid - macula_cy) ** 2)
    macula_darken = np.exp(-(macula_dist ** 2) / (2.0 * (radius * 0.16) ** 2))
    for c, factor in enumerate([0.78, 0.65, 0.65]):
        ch = img[:, :, c].astype(np.float32)
        img[:, :, c] = np.clip(ch * (1.0 - (1.0 - factor) * macula_darken), 0, 255).astype(np.uint8)

    # 2. Optic Nerve Head (Optic Disc & Optic Cup)
    od_offset_x = int(radius * 0.42) if laterality.upper() == "OD" else int(-radius * 0.42)
    od_cx = center[0] + od_offset_x
    od_cy = center[1] - int(radius * 0.04)
    od_rx = int(radius * 0.14)
    od_ry = int(radius * 0.16)
    cv2.ellipse(img, (od_cx, od_cy), (od_rx, od_ry), 0, 0, 360, (242, 195, 115), thickness=-1)

    cup_rx = max(2, int(od_rx * vcdr))
    cup_ry = max(2, int(od_ry * vcdr))
    cv2.ellipse(img, (od_cx, od_cy), (cup_rx, cup_ry), 0, 0, 360, (254, 240, 190), thickness=-1)

    # 3. Retinal vascular arcades radiating from Optic Disc
    for angle_deg in [-65, -35, 35, 65, -110, 110]:
        pts = [(od_cx, od_cy)]
        curr_x, curr_y = float(od_cx), float(od_cy)
        rad = np.deg2rad(angle_deg if laterality.upper() == "OS" else (180 - angle_deg))
        for step in range(1, 12):
            curr_x += np.cos(rad) * (radius * 0.08) + rng.normal(0, 3.0)
            curr_y += np.sin(rad) * (radius * 0.08) + rng.normal(0, 3.0)
            rad += rng.normal(0, 0.08)
            if (curr_x - center[0]) ** 2 + (curr_y - center[1]) ** 2 < (radius * 0.94) ** 2:
                pts.append((int(curr_x), int(curr_y)))
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_arr], isClosed=False, color=(125, 22, 18), thickness=2, lineType=cv2.LINE_AA)

    # 4. Microaneurysms (<10px high-contrast dark red/green-absorbing focal dots)
    for _ in range(num_microaneurysms):
        r_dist = rng.uniform(0.15 * radius, 0.72 * radius)
        theta = rng.uniform(0, 2 * np.pi)
        mx = int(center[0] + r_dist * np.cos(theta))
        my = int(center[1] + r_dist * np.sin(theta))
        ma_rad = int(rng.integers(2, 5))  # 4-8px diameter (<10px)
        cv2.circle(img, (mx, my), ma_rad, (110, 10, 10), thickness=-1)

    # 5. Blot / Flame Hemorrhages
    for _ in range(num_hemorrhages):
        r_dist = rng.uniform(0.20 * radius, 0.78 * radius)
        theta = rng.uniform(0, 2 * np.pi)
        hx = int(center[0] + r_dist * np.cos(theta))
        hy = int(center[1] + r_dist * np.sin(theta))
        cv2.ellipse(
            img,
            (hx, hy),
            (int(rng.integers(6, 14)), int(rng.integers(5, 11))),
            int(rng.integers(0, 180)),
            0,
            360,
            (95, 8, 12),
            thickness=-1,
        )

    # 6. Hard Exudates (bright yellow-white lipid deposits)
    for _ in range(num_exudates):
        r_dist = rng.uniform(0.12 * radius, 0.68 * radius)
        theta = rng.uniform(0, 2 * np.pi)
        ex = int(center[0] + r_dist * np.cos(theta))
        ey = int(center[1] + r_dist * np.sin(theta))
        cv2.circle(img, (ex, ey), int(rng.integers(3, 7)), (248, 232, 110), thickness=-1)

    # Add subtle physiological sensor texture inside FOV so sharpness IQA passes on normal images
    noise = rng.normal(0, 3.5, size=img.shape)
    img = np.where(fov_mask[:, :, None], np.clip(img.astype(np.float32) + noise, 0, 255), 0).astype(np.uint8)

    if simulate_severe_blur:
        img = cv2.GaussianBlur(img, (35, 35), sigmaX=14.0)
        img = np.where(fov_mask[:, :, None], img, 0)

    if simulate_flash_glare:
        cv2.circle(img, center, int(radius * 0.55), (255, 255, 255), thickness=-1)

    return img

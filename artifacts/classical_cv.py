"""Classical computer-vision module for P5 fine-grained vehicle recognition."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

HOG_SIZE = (160, 96)  # (W, H) — cars are wider than tall


# ---------------- preprocessing / normalisation ----------------
def to_bgr(img):
    """PIL image, RGB array (from PIL) or BGR array (from cv2) / file path -> BGR uint8."""
    if isinstance(img, str):
        im = cv2.imread(img)
        if im is None:
            raise ValueError(f'Cannot read {img}')
        return im
    if hasattr(img, 'mode'):
        img = np.array(img.convert('RGB'))[:, :, ::-1]
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def resize_max(bgr, max_side):
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return bgr


def apply_clahe(bgr, clip=2.0, grid=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def brightness(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2].mean())


def adaptive_enhance(bgr, dark_thr=80, bright_thr=200):
    """Illumination normalisation: gamma-lift + CLAHE for dark images, CLAHE for washed-out ones."""
    v = brightness(bgr)
    if v < dark_thr:
        gamma = max(0.4, v / dark_thr)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        return apply_clahe(cv2.LUT(bgr, lut), clip=2.5), 'dark: gamma+CLAHE'
    if v > bright_thr:
        return apply_clahe(bgr, clip=1.5), 'bright: CLAHE'
    return bgr, 'none'


def normalize_image(bgr):
    """Per-channel z-score normalisation; returns float image and a displayable uint8 version."""
    f = bgr.astype(np.float32)
    mean, std = f.reshape(-1, 3).mean(0), f.reshape(-1, 3).std(0) + 1e-6
    z = (f - mean) / std
    disp = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return z, disp


# ---------------- edges / contours / ROI ----------------
def auto_canny(gray, sigma=0.33):
    v = float(np.median(gray))
    return cv2.Canny(gray, int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v)))


def edge_contour_analysis(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(blur)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    closed = cv2.dilate(closed, k, iterations=1)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return dict(gray=gray, blur=blur, edges=edges, closed=closed, contours=cnts)


def classical_roi(img, max_side=512):
    """Classical vehicle localisation: CLAHE -> Canny -> closing -> best (large, central, car-shaped) contour.
    Returns (box_xyxy in original coords or None, score in [0,1], debug dict)."""
    bgr = to_bgr(img)
    H, W = bgr.shape[:2]
    small = apply_clahe(resize_max(bgr, max_side))
    s = small.shape[1] / W
    ea = edge_contour_analysis(small)
    h, w = small.shape[:2]
    best, best_score = None, 0.0
    for c in ea['contours']:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch / (w * h)
        if area < 0.05:
            continue
        cx, cy = (x + cw / 2) / w, (y + ch / 2) / h
        centrality = 1 - min(1.0, np.hypot(cx - 0.5, cy - 0.5) / 0.71)
        ar = cw / max(ch, 1)
        ar_score = 1.0 if 0.8 <= ar <= 3.5 else 0.5
        score = area * (0.5 + 0.5 * centrality) * ar_score
        if score > best_score:
            best_score, best = score, (x, y, x + cw, y + ch)
    if best is None:
        return None, 0.0, ea
    return tuple(v / s for v in best), float(min(1.0, best_score)), ea


def crop_box(bgr, box, pad=0.0):
    if box is None:
        return bgr
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    x2, y2 = int(min(W, x2 + pad * bw)), int(min(H, y2 + pad * bh))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return bgr
    return bgr[y1:y2, x1:x2]


# ---------------- feature extraction ----------------
def color_features(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch, bins, rng in [(0, 18, 180), (1, 8, 256), (2, 8, 256)]:
        h = cv2.calcHist([hsv], [ch], None, [bins], [0, rng]).ravel()
        feats.append(h / (h.sum() + 1e-6))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    feats.append(np.concatenate([lab.mean(0), lab.std(0)]) / 255.0)
    return np.concatenate(feats)


def texture_features(gray):
    g = cv2.resize(gray, (128, 128))
    feats = []
    for P, R in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(g, P, R, method='uniform')
        h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2))
        feats.append(h / (h.sum() + 1e-6))
    q = (g // 4).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=64, symmetric=True, normed=True)
    for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        feats.append(graycoprops(glcm, p).ravel())
    return np.concatenate(feats)


def hog_features(gray, visualize=False):
    g = cv2.resize(gray, HOG_SIZE)
    return hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               block_norm='L2-Hys', visualize=visualize, feature_vector=True)


def geometric_features(bgr):
    h, w = bgr.shape[:2]
    ea = edge_contour_analysis(resize_max(bgr, 256))
    edges, cnts = ea['edges'], ea['contours']
    hh, ww = edges.shape
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, cw, ch = cv2.boundingRect(c)
        hull_a = cv2.contourArea(cv2.convexHull(c)) + 1e-6
        per = cv2.arcLength(c, True) + 1e-6
        f = [area / (hh * ww), area / (cw * ch + 1e-6), area / hull_a, 4 * np.pi * area / per ** 2, cw / max(ch, 1)]
        hu = cv2.HuMoments(cv2.moments(c)).ravel()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    else:
        f, hu = [0.0] * 5, np.zeros(7)
    gx = cv2.Sobel(ea['blur'], cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(ea['blur'], cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    oh, _ = np.histogram(ang % 180, bins=8, range=(0, 180), weights=mag)
    oh = oh / (oh.sum() + 1e-6)
    col = edges.mean(0)
    sym = float(np.clip(1 - np.abs(col - col[::-1]).mean() / (2 * col.mean() + 1e-6), 0, 1))
    # layout: [aspect, edge_density, n_contours/50, area_ratio, extent, solidity, circularity, contour_ar, hu*7, orient*8, symmetry]
    return np.concatenate([[w / max(h, 1), edges.mean() / 255.0, len(cnts) / 50.0], f, hu, oh, [sym]])


def extract_classical_features(img):
    """Full hand-crafted descriptor (~1.75k dims): HOG + colour + texture + geometry."""
    bgr = resize_max(to_bgr(img), 320)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    f = np.concatenate([hog_features(gray), color_features(bgr), texture_features(gray), geometric_features(bgr)])
    return np.nan_to_num(f.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# multiprocessing helpers (top-level so they pickle)
def features_from_path(args):
    path, box, pad = args
    bgr = cv2.imread(path)
    return extract_classical_features(crop_box(bgr, box, pad))


def roi_from_path(path):
    box, score, _ = classical_roi(cv2.imread(path))
    return box, score


# ---------------- visualisation of intermediate stages ----------------
def _rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _to_u8(a):
    a = a.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return (a * 255).astype(np.uint8)


def processing_stages(img, box=None, max_side=640):
    """Returns ({stage_name: RGB uint8 image}, {feature_name: value}) for the app's 'intermediate results' panel.
    `box` (optional) is the detector box in ORIGINAL image coordinates."""
    src = to_bgr(img)
    W0 = src.shape[1]
    bgr = resize_max(src, max_side)
    s = bgr.shape[1] / W0
    st = {'Original': bgr}
    enh, mode = adaptive_enhance(bgr)
    st[f'Illumination normalisation ({mode})'] = enh
    _, nd = normalize_image(enh)
    st['Z-score normalised'] = nd
    ea = edge_contour_analysis(enh)
    st['Grayscale + Gaussian blur'] = ea['blur']
    st['Canny edges'] = ea['edges']
    st['Morphological closing'] = ea['closed']
    cimg = bgr.copy()
    cv2.drawContours(cimg, ea['contours'], -1, (0, 255, 0), 1)
    roi, rscore, _ = classical_roi(bgr)
    if roi is not None:
        cv2.rectangle(cimg, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (255, 0, 0), 2)
    dbox = [v * s for v in box] if box is not None else None
    if dbox is not None:
        cv2.rectangle(cimg, (int(dbox[0]), int(dbox[1])), (int(dbox[2]), int(dbox[3])), (0, 0, 255), 2)
    st['Contours + ROI (blue=classical, red=detector)'] = cimg
    use = dbox if dbox is not None else roi
    crop = crop_box(bgr, use, 0.05) if use is not None else bgr
    st['Region of interest'] = crop
    g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (320, 192))
    _, hv = hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    st['HOG (shape / gradients)'] = _to_u8(hv)
    st['LBP texture'] = _to_u8(local_binary_pattern(g, 8, 1, 'uniform'))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = np.dstack([hsv[:, :, 0], np.full_like(hsv[:, :, 0], 255), np.full_like(hsv[:, :, 0], 255)])
    st['Hue map (colour)'] = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)
    gf = geometric_features(crop)
    c = crop[crop.shape[0] // 4: 3 * crop.shape[0] // 4, crop.shape[1] // 4: 3 * crop.shape[1] // 4]
    info = dict(brightness=round(brightness(bgr), 1), illumination_fix=mode,
                roi_aspect_ratio=round(float(gf[0]), 3), edge_density=round(float(gf[1]), 4),
                num_contours=len(ea['contours']), contour_solidity=round(float(gf[5]), 3),
                lr_symmetry=round(float(gf[-1]), 3),
                mean_rgb=[int(v) for v in c.reshape(-1, 3).mean(0)[::-1]],
                classical_roi_score=round(rscore, 3))
    return {k: _rgb(v) for k, v in st.items()}, info

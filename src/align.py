"""Map between the reference background and a video's own view.

The camera framing shifts slightly between recordings (about 100 px in 4K, plus small zoom and
rotation), so zones drawn on scene/background.jpg are moved onto each video with a homography
estimated from SIFT matches between the two backgrounds.
"""
import cv2
import numpy as np

WORK_WIDTH = 1920  # match features at half of 4K: plenty of detail, 4x cheaper


def median_background(video_path, n_frames=15):
    """Median of frames spread over the video: removes moving traffic, keeps the road."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, total - 1, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    stack = np.stack(frames)
    bg = np.empty(stack.shape[1:], np.uint8)
    for r in range(0, bg.shape[0], 120):  # row chunks keep memory low on 4K
        bg[r:r + 120] = np.median(stack[:, r:r + 120], axis=0)
    return bg


def _features(img, sift):
    scale = WORK_WIDTH / img.shape[1]
    gray = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    keypoints, desc = sift.detectAndCompute(gray, None)
    return np.float32([k.pt for k in keypoints]) / scale, desc


def estimate_homography(ref_img, img, min_inliers=40):
    """Return (H, n_inliers) with H mapping reference pixels to `img` pixels."""
    sift = cv2.SIFT_create(4000)
    ref_pts, ref_desc = _features(ref_img, sift)
    img_pts, img_desc = _features(img, sift)
    good = [a for a, b in cv2.BFMatcher().knnMatch(ref_desc, img_desc, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < min_inliers:
        raise RuntimeError(f"only {len(good)} feature matches with the reference view")
    H, mask = cv2.findHomography(ref_pts[[m.queryIdx for m in good]], img_pts[[m.trainIdx for m in good]],
                                 cv2.RANSAC, 6.0)
    n_inliers = int(mask.sum()) if mask is not None else 0
    if H is None or n_inliers < min_inliers:
        raise RuntimeError(f"homography has only {n_inliers} inliers")
    return H, n_inliers


def warp_points(points, H):
    """Apply homography H to an (N, 2) array of pixel coordinates."""
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

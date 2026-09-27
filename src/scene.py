"""Scene layout for one video: hand-drawn zones moved from the reference view onto this video's view."""
import json
from pathlib import Path

import cv2
import numpy as np

from align import estimate_homography, median_background, warp_points

ROOT = Path(__file__).resolve().parent.parent
ZONES = ROOT / "scene" / "zones.json"
REFERENCE = ROOT / "scene" / "background.jpg"
PED_MARGIN = 45  # px (4K) around crossings / pavements / islands where pedestrians still count as off-road


class Scene:
    """Zones in video pixels plus the masks the rules need."""

    def __init__(self, shapes, size):
        self.shapes = shapes      # [{"type", "name", "note", "points": float32 (N, 2)}] in video pixels
        self.size = size          # (height, width)
        self.road = self.mask({"road"}) & ~self.mask({"island"})
        self.junction = self.mask({"intersection"})
        self.exempt_stop = self.mask({"bus_stop", "parking"})
        self.walk_ok = self.mask({"crosswalk", "sidewalk", "island", "bus_stop"}, margin=PED_MARGIN)

    @classmethod
    def for_video(cls, video_path, background=None, **kwargs):
        """Align the reference zones to `video_path` (its median background unless `background` is given)."""
        return cls.from_image(background if background is not None else median_background(video_path), **kwargs)

    @classmethod
    def from_image(cls, image, zones_path=ZONES, reference_path=REFERENCE):
        """Align the reference zones to a view of this camera (a frame or a background) by SIFT homography."""
        size = image.shape[:2]
        ref = cv2.imread(str(reference_path))
        aligned = True
        try:
            H, _ = estimate_homography(ref, image)
        except RuntimeError:
            # alignment failed (e.g. a very different view): fall back to scaling the reference to this size
            H = np.diag([size[1] / ref.shape[1], size[0] / ref.shape[0], 1.0])
            aligned = False
        zones = json.loads(Path(zones_path).read_text())
        shapes = [{**s, "points": warp_points(s["points"], H).astype(np.float32)} for s in zones["shapes"]]
        scene = cls(shapes, size)
        scene.aligned = aligned
        return scene

    def of_type(self, *types):
        return [s for s in self.shapes if s["type"] in types]

    def mask(self, types, margin=0, shapes=None):
        """Boolean (h, w) mask of the polygons of `types`, grown by `margin` px."""
        m = np.zeros(self.size, np.uint8)
        for s in shapes if shapes is not None else self.of_type(*types):
            cv2.fillPoly(m, [s["points"].astype(np.int32)], 1)
        if margin:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
        return m.astype(bool)


def lookup(mask, x, y):
    """Mask values at pixel coordinates (arrays), clipped to the frame."""
    h, w = mask.shape
    return mask[np.clip(np.asarray(y).astype(int), 0, h - 1), np.clip(np.asarray(x).astype(int), 0, w - 1)]

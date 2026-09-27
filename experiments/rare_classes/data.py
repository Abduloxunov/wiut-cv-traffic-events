"""All 4 sample videos (tracks + aligned scene + signal timeline) for checking rare-class rules: none of the videos
contains a labelled red_light / wrong_way / accident / illegal_u_turn / road_obstacle / fire_smoke, so every firing of
those rules on them is a false alarm."""
import pickle
import cv2
import pandas as pd
from common import *
from rules import add_motion
from scene import Scene

CACHE = DATA / "runs" / "jaywalking_v2" / "cache4.pkl"
VIDEOS = ["C3897", "C3902", "C3905", "C3896"]
fmt = lambda t: f"{int(t // 60)}:{t % 60:04.1f}"


def load_all():
    try:
        return pickle.load(open(CACHE, "rb"))
    except FileNotFoundError:
        out = {}
        base = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
        for v in VIDEOS:
            if f"{v}.MP4" in base:
                df, scene, _ = base[f"{v}.MP4"]
            else:
                df = add_motion(pd.read_csv(DATA / "runs" / v / "tracks.csv"), FPS)
                scene = Scene.from_image(cv2.imread(str(DATA / "runs" / v / "background.jpg")))
            sig = pd.read_csv(DATA / "runs" / "signals" / f"{v}.csv")
            out[v] = (df, scene, sig)
        pickle.dump(out, open(CACHE, "wb"))
        return out

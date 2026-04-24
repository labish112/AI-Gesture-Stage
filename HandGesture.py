# =============================================================================
#   AI GESTURE STAGE  v3.0  — Fixed for MediaPipe 0.10+
#   Works in: Spyder, VS Code, PyCharm, Terminal
#
#   INSTALL (run in Anaconda Prompt or CMD):
#       pip install opencv-python mediapipe numpy
#
#   HOW TO RUN IN SPYDER:
#       1. Open this file in Spyder
#       2. Press F5
#       3. First run downloads the hand model (~8 MB) automatically
#       4. Show your hand to the webcam
#       5. Press Q to quit
#
#   GESTURES:
#       Open Palm   ->  Radial burst
#       Fist        ->  Explosion
#       Index Up    ->  Beam upward
#       Peace       ->  Dual stream
#       Thumbs Up   ->  Wave
#       Pinch       ->  Collapse inward
#       OK Sign     ->  Spiral
#       Rock On     ->  Lightning
#       Call Me     ->  Network pulse
# =============================================================================

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import threading
import urllib.request
import os
from collections import Counter

# New MediaPipe 0.10+ Tasks API imports
from mediapipe.tasks.python                 import vision          as mp_vision
from mediapipe.tasks.python.vision          import (HandLandmarker,
                                                    HandLandmarkerOptions,
                                                    HandLandmarkerResult,
                                                    RunningMode)
from mediapipe.tasks.python.core.base_options import BaseOptions


# =============================================================================
#  SETTINGS  — Change these if something doesn't work
# =============================================================================
CAMERA_INDEX     = 0       # Try 1 or 2 if camera not found
WINDOW_W         = 1280
WINDOW_H         = 720
STABLE_FRAMES    = 8       # Frames before gesture fires (lower = faster)
GESTURE_HOLD_SEC = 2.5     # Seconds animation stays on screen
DETECT_CONF      = 0.70
TRACK_CONF       = 0.60
MAX_HANDS        = 2

# Hand landmark model — downloaded automatically on first run
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"   # Saved next to this .py file


# =============================================================================
#  MODEL DOWNLOADER
# =============================================================================
def download_model(path, url):
    """Download the MediaPipe hand model if not already on disk."""

    # Check if file already exists and is valid (> 1 MB)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f"  Model ready: {path}")
        return

    print(f"  Downloading hand landmark model (~8 MB)...")
    print(f"  Saving to: {os.path.abspath(path)}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, int(downloaded * 100 / total_size))
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    try:
        # Add a browser-like User-Agent header
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, path, progress)
        print(f"\n  Download complete!")

        # Verify size
        size = os.path.getsize(path)
        if size < 1_000_000:
            os.remove(path)
            raise RuntimeError(f"Downloaded file too small ({size} bytes). "
                               "Check internet connection.")
    except Exception as e:
        # Clean up partial download
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(
            f"Could not download model automatically.\n"
            f"  Error: {e}\n\n"
            f"  MANUAL FIX:\n"
            f"  1. Open this URL in your browser:\n"
            f"     {url}\n"
            f"  2. Save the file as:  hand_landmarker.task\n"
            f"  3. Put it in the SAME folder as gesture_stage.py\n"
            f"  4. Run the script again."
        )


# =============================================================================
#  GESTURE DEFINITIONS
# =============================================================================
GESTURES = {
    "open_palm":  {"name": "Open Palm",   "color": (251, 158,  23), "style": "radial",    "desc": "Expanding energy outward"},
    "fist":       {"name": "Power Fist",  "color": ( 30,  80, 249), "style": "burst",     "desc": "Maximum force unleashed"},
    "point":      {"name": "Pointing",    "color": (250, 139, 167), "style": "beam",      "desc": "Beam of precision targeting"},
    "peace":      {"name": "Peace",       "color": ( 58, 211,  52), "style": "dual",      "desc": "Harmony between two forces"},
    "thumbs_up":  {"name": "Thumbs Up",   "color": (250, 165,  96), "style": "wave",      "desc": "Positive signal confirmed"},
    "pinch":      {"name": "Pinch",       "color": (155,  73, 236), "style": "collapse",  "desc": "Drawing energy inward"},
    "ok":         {"name": "OK Sign",     "color": (212, 182,   6), "style": "spiral",    "desc": "The loop is complete"},
    "rock":       {"name": "Rock On",     "color": (249, 121, 232), "style": "lightning", "desc": "Maximum intensity ignited"},
    "call":       {"name": "Call Me",     "color": ( 80, 222,  74), "style": "network",   "desc": "Establishing connection"},
}


# =============================================================================
#  PARTICLE — One animated dot on screen
# =============================================================================
class Particle:
    def __init__(self, x, y, vx=None, vy=None, color=(255,255,255),
                 size=3.0, decay=0.015, gravity=0.08, shape="circle", trail=False):
        self.x       = float(x)
        self.y       = float(y)
        self.vx      = vx if vx is not None else random.uniform(-4, 4)
        self.vy      = vy if vy is not None else random.uniform(-4, 4)
        self.color   = color
        self.size    = size
        self.life    = 1.0
        self.decay   = decay
        self.gravity = gravity
        self.shape   = shape
        self.trail   = trail
        self.trail_pts = []
        self.angle   = random.uniform(0, math.pi * 2)
        self.spin    = random.uniform(-0.2, 0.2)

    def update(self):
        if self.trail:
            self.trail_pts.append((int(self.x), int(self.y)))
            if len(self.trail_pts) > 8:
                self.trail_pts.pop(0)
        self.x    += self.vx
        self.y    += self.vy
        self.vy   += self.gravity
        self.vx   *= 0.985
        self.angle += self.spin
        self.life  -= self.decay

    def draw(self, frame):
        if self.life <= 0:
            return
        alpha = max(0.0, self.life)
        r, g, b = self.color

        if self.trail and len(self.trail_pts) > 1:
            for i in range(1, len(self.trail_pts)):
                t_a = alpha * i / len(self.trail_pts)
                cv2.line(frame, self.trail_pts[i-1], self.trail_pts[i],
                         (int(b*t_a), int(g*t_a), int(r*t_a)), 1)

        cx_i, cy_i = int(self.x), int(self.y)
        sz = max(1, int(self.size * alpha))

        if self.shape == "star":
            for k in range(4):
                ang = k * math.pi/2 + self.angle
                ex = int(cx_i + math.cos(ang) * sz * 2)
                ey = int(cy_i + math.sin(ang) * sz * 2)
                cv2.line(frame, (cx_i, cy_i), (ex, ey),
                         (int(b*alpha), int(g*alpha), int(r*alpha)), 1)
        else:
            cv2.circle(frame, (cx_i, cy_i), sz,
                       (int(b*alpha), int(g*alpha), int(r*alpha)), -1)


# =============================================================================
#  RING — Expanding circle that fades
# =============================================================================
class Ring:
    def __init__(self, x, y, color):
        self.x     = int(x)
        self.y     = int(y)
        self.r     = 5
        self.max_r = random.randint(120, 220)
        self.color = color
        self.life  = 1.0
        self.speed = random.uniform(2.5, 4.5)

    def update(self):
        self.r    += self.speed
        self.life  = max(0.0, 1.0 - self.r / self.max_r)

    def draw(self, frame):
        if self.life <= 0 or self.r < 1:
            return
        r, g, b = self.color
        alpha = self.life
        cv2.circle(frame, (self.x, self.y), int(self.r),
                   (int(b*alpha), int(g*alpha), int(r*alpha)), 2)


# =============================================================================
#  PARTICLE SPAWNER — Creates effect for each gesture style
# =============================================================================
def spawn_particles(style, cx, cy, color, particles, rings):
    if style == "radial":
        for i in range(12):
            angle = (i/12) * math.pi * 2
            for j in range(6):
                spd = 2.0 + j * 1.2
                col = color if j % 2 == 0 else (255, 255, 255)
                particles.append(Particle(cx, cy,
                    vx=math.cos(angle)*spd, vy=math.sin(angle)*spd,
                    color=col, size=1.5+random.random()*2.5,
                    gravity=0.02, decay=0.007+random.random()*0.008))
        for _ in range(3):
            rings.append(Ring(cx, cy, color))

    elif style == "burst":
        for i in range(100):
            angle = random.uniform(0, math.pi*2)
            spd   = 3 + random.random() * 10
            col   = color if i % 3 != 0 else (255, 255, 255)
            particles.append(Particle(cx, cy,
                vx=math.cos(angle)*spd, vy=math.sin(angle)*spd,
                color=col, size=1.5+random.random()*5,
                gravity=0.12, shape="star" if i%4==0 else "circle",
                trail=i%3==0, decay=0.01+random.random()*0.02))
        for i in range(3):
            rings.append(Ring(cx, cy, color if i%2==0 else (255,255,255)))

    elif style == "beam":
        for i in range(80):
            spread = random.uniform(-30, 30)
            particles.append(Particle(cx + spread, cy,
                vx=random.uniform(-1, 1), vy=-(4 + random.random()*9),
                color=color, size=1.5+random.random()*3,
                gravity=0.04, trail=True,
                decay=0.006+random.random()*0.010))

    elif style == "dual":
        cols = [color, (96, 165, 250)]
        for i in range(60):
            side = -1 if i % 2 == 0 else 1
            particles.append(Particle(cx + side*50, cy,
                vx=side*(1+random.random()*3), vy=-(2+random.random()*5),
                color=cols[i%2], size=2+random.random()*3,
                gravity=0.06, trail=True,
                decay=0.009+random.random()*0.013))
        rings.append(Ring(cx-50, cy, cols[0]))
        rings.append(Ring(cx+50, cy, cols[1]))

    elif style == "wave":
        for i in range(55):
            angle = (i/55) * math.pi * 2
            spd   = 2 + random.random() * 4
            particles.append(Particle(cx, cy,
                vx=math.cos(angle)*spd, vy=math.sin(angle)*spd - 1,
                color=color, size=2+random.random()*3,
                gravity=0.04, trail=random.random()>0.6,
                decay=0.008+random.random()*0.012))
        rings.append(Ring(cx, cy, color))
        rings.append(Ring(cx, cy, color))

    elif style == "collapse":
        for i in range(80):
            angle = random.uniform(0, math.pi*2)
            r     = 100 + random.random() * 80
            px    = cx + math.cos(angle) * r
            py    = cy + math.sin(angle) * r
            spd   = 3 + random.random() * 4
            particles.append(Particle(px, py,
                vx=(cx-px)/r*spd, vy=(cy-py)/r*spd,
                color=color, size=1.5+random.random()*3,
                gravity=0, trail=True,
                decay=0.008+random.random()*0.012))
        rings.append(Ring(cx, cy, color))

    elif style == "spiral":
        for i in range(100):
            t = (i/50) * math.pi * 6
            r = t * 8
            a = t
            spd = 1 + random.random() * 2
            particles.append(Particle(
                cx + math.cos(a)*r*0.2,
                cy + math.sin(a)*r*0.2,
                vx=math.cos(a+math.pi/2)*spd,
                vy=math.sin(a+math.pi/2)*spd - 0.5,
                color=color, size=1.5+random.random()*2.5,
                gravity=0.03, trail=True,
                decay=0.007+random.random()*0.010))
        rings.append(Ring(cx, cy, color))
        rings.append(Ring(cx, cy, color))

    elif style == "lightning":
        for i in range(80):
            offset = random.uniform(-25, 25)
            particles.append(Particle(cx + offset, cy,
                vx=random.uniform(-8, 8),
                vy=-(3 + random.random()*15),
                color=color if i%5!=0 else (255,255,255),
                size=1+random.random()*4,
                gravity=0.15, trail=i%2==0,
                decay=0.015+random.random()*0.025))
        for i in range(3):
            rings.append(Ring(cx + random.uniform(-40,40),
                              cy + random.uniform(-40,40),
                              color if i%2==0 else (255,255,255)))

    elif style == "network":
        for i in range(8):
            angle = (i/8) * math.pi * 2
            nx = cx + math.cos(angle) * 90
            ny = cy + math.sin(angle) * 90
            for j in range(6):
                particles.append(Particle(nx, ny,
                    vx=random.uniform(-3, 3),
                    vy=random.uniform(-3, 3),
                    color=color, size=2+random.random()*2,
                    gravity=0, decay=0.01+random.random()*0.01))
            rings.append(Ring(nx, ny, color))


# =============================================================================
#  GESTURE RECOGNIZER — Landmarks -> gesture name
# =============================================================================
class GestureRecognizer:
    def __init__(self, stable_frames=8):
        self.history       = []
        self.stable_frames = stable_frames

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _finger_up(lm, tip_id, mcp_id):
        # Fingertip must be above (smaller Y) the knuckle
        return (lm[mcp_id].y - lm[tip_id].y) > 0.04

    def recognize_raw(self, lm):
        idx = self._finger_up(lm, 8,  5)
        mid = self._finger_up(lm, 12, 9)
        rng = self._finger_up(lm, 16, 13)
        pky = self._finger_up(lm, 20, 17)

        pinch_d   = self._dist(lm[4], lm[8])
        hand_sz   = self._dist(lm[0], lm[9]) + 1e-6
        thumb_out = self._dist(lm[4], lm[5]) / hand_sz > 0.38
        up_count  = sum([idx, mid, rng, pky])

        if pinch_d < 0.06 and not mid and not rng:   return "pinch"
        if up_count == 4:                              return "open_palm"
        if up_count == 0 and not thumb_out:            return "fist"
        if idx and not mid and not rng and not pky:    return "point"
        if idx and mid and not rng and not pky:        return "peace"
        if up_count == 0 and thumb_out:                return "thumbs_up"
        if idx and not mid and not rng and pky:        return "rock"
        if not idx and not mid and not rng and pky and thumb_out: return "call"
        if pinch_d < 0.09 and idx and mid:             return "ok"
        return None

    def recognize(self, lm):
        raw = self.recognize_raw(lm)
        if raw:
            self.history.append(raw)
        if len(self.history) > self.stable_frames:
            self.history.pop(0)
        if len(self.history) < 3:
            return None
        best, n = Counter(self.history).most_common(1)[0]
        if n >= max(3, int(self.stable_frames * 0.75)):
            return best
        return None


# =============================================================================
#  HUD RENDERER — Draws text / UI on the frame
# =============================================================================
class HUDRenderer:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_gesture_name(self, frame, gesture_key, alpha):
        if not gesture_key or alpha <= 0:
            return
        g    = GESTURES[gesture_key]
        name = g["name"]
        desc = g["desc"]
        r, gg, b = g["color"]
        col  = (int(b*alpha), int(gg*alpha), int(r*alpha))
        gray = (int(140*alpha), int(140*alpha), int(140*alpha))

        # Gesture name — centered
        scale = 2.2
        (tw, _), _ = cv2.getTextSize(name, self.font, scale, 3)
        tx = (self.w - tw) // 2
        ty = self.h // 2 - 30
        cv2.putText(frame, name, (tx, ty), self.font, scale, (0,0,0), 7, cv2.LINE_AA)
        cv2.putText(frame, name, (tx, ty), self.font, scale, col, 3, cv2.LINE_AA)

        # Description below
        (dw, _), _ = cv2.getTextSize(desc, self.font, 0.7, 1)
        dx = (self.w - dw) // 2
        cv2.putText(frame, desc, (dx, ty+52), self.font, 0.7, gray, 1, cv2.LINE_AA)

    def draw_landmarks(self, frame, lm_list, gesture_key):
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        TIPS = {4, 8, 12, 16, 20}

        if gesture_key and gesture_key in GESTURES:
            r, g, b = GESTURES[gesture_key]["color"]
            tip_col = (int(b*0.9), int(g*0.9), int(r*0.9))
        else:
            tip_col = (120, 120, 160)

        # lm_list is list of landmark objects from new Tasks API
        pts = [(int((1 - lm.x) * self.w), int(lm.y * self.h))
               for lm in lm_list]

        for a, b_i in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b_i], (50, 50, 65), 1, cv2.LINE_AA)

        for i, (px, py) in enumerate(pts):
            if i in TIPS:
                cv2.circle(frame, (px, py), 7, tip_col, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 7, (200, 200, 220), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 4, (70, 70, 95), -1, cv2.LINE_AA)

    def draw_status_bar(self, frame, fps, hand_count, gesture_key):
        y = self.h - 44
        cv2.rectangle(frame, (0, y), (self.w, self.h), (12, 12, 16), -1)
        cv2.line(frame, (0, y), (self.w, y), (35, 35, 40), 1)

        cv2.putText(frame, f"FPS: {fps:.0f}", (14, self.h-14),
                    self.font, 0.52, (70, 70, 70), 1, cv2.LINE_AA)

        dot_col = (60, 200, 70) if hand_count > 0 else (50, 50, 50)
        cv2.circle(frame, (self.w//2 - 65, self.h-20), 4, dot_col, -1)
        cv2.putText(frame, f"Hands: {hand_count}", (self.w//2-55, self.h-14),
                    self.font, 0.52, (90, 90, 90), 1, cv2.LINE_AA)

        g_txt = GESTURES[gesture_key]["name"] if gesture_key else "No gesture"
        cv2.putText(frame, g_txt, (self.w-200, self.h-14),
                    self.font, 0.52, (90, 90, 90), 1, cv2.LINE_AA)

        cv2.putText(frame, "Q = Quit", (self.w-100, 26),
                    self.font, 0.48, (50, 50, 50), 1, cv2.LINE_AA)

    def draw_grid(self, frame):
        for x in range(0, self.w, 60):
            cv2.line(frame, (x, 0), (x, self.h), (14, 14, 18), 1)
        for y in range(0, self.h, 60):
            cv2.line(frame, (0, y), (self.w, y), (14, 14, 18), 1)

    def draw_no_hand_hint(self, frame):
        msg = "Show your hand to the camera"
        (tw, _), _ = cv2.getTextSize(msg, self.font, 0.7, 1)
        cv2.putText(frame, msg, ((self.w-tw)//2, self.h//2),
                    self.font, 0.7, (50, 50, 50), 1, cv2.LINE_AA)

    def draw_bg_glow(self, frame, gesture_key, alpha):
        if not gesture_key or alpha < 0.05:
            return
        r, g, b = GESTURES[gesture_key]["color"]
        overlay = frame.copy()
        cx, cy = self.w//2, self.h//2
        for rad, strength in [(300, 18), (180, 14), (90, 10)]:
            col = (int(b * strength * alpha / 100),
                   int(g * strength * alpha / 100),
                   int(r * strength * alpha / 100))
            cv2.circle(overlay, (cx, cy), rad, col, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


# =============================================================================
#  MAIN APPLICATION  (MediaPipe 0.10+ Tasks API with LIVE_STREAM)
# =============================================================================
class GestureStageApp:
    """
    Uses the NEW MediaPipe Tasks API (0.10+):
      - HandLandmarker with RunningMode.LIVE_STREAM
      - detect_async() is called each frame (non-blocking)
      - Results arrive in _on_result() callback from a background thread
      - We store the latest result in self._latest_result (thread-safe lock)
    """

    def __init__(self):
        print("\n" + "="*60)
        print("  AI GESTURE STAGE  v3.0")
        print("  MediaPipe 0.10+ / Tasks API")
        print("="*60)

        # ── Download model if not present ────────────────────────────────
        download_model(MODEL_PATH, MODEL_URL)

        # ── Thread-safe storage for async MediaPipe results ──────────────
        self._latest_result = None   # type: HandLandmarkerResult
        self._lock          = threading.Lock()

        # ── Build HandLandmarker (LIVE_STREAM mode) ──────────────────────
        #
        # LIVE_STREAM = non-blocking: detect_async() returns immediately,
        # result is delivered to _on_result() from a background thread.
        # timestamp_ms must be strictly increasing — we use a counter.
        #
        print("  Loading hand landmark model...")
        options = HandLandmarkerOptions(
            base_options      = BaseOptions(model_asset_path=MODEL_PATH),
            running_mode      = RunningMode.LIVE_STREAM,
            num_hands         = MAX_HANDS,
            min_hand_detection_confidence  = DETECT_CONF,
            min_hand_presence_confidence   = DETECT_CONF,
            min_tracking_confidence        = TRACK_CONF,
            result_callback   = self._on_result,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        print("  Model loaded!")

        # ── Camera ───────────────────────────────────────────────────────
        print(f"  Opening camera {CAMERA_INDEX}...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {CAMERA_INDEX}. "
                "Change CAMERA_INDEX at top of the file."
            )

        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Camera: {self.W} x {self.H}")

        # ── Sub-systems ──────────────────────────────────────────────────
        self.recognizer = GestureRecognizer(STABLE_FRAMES)
        self.hud        = HUDRenderer(self.W, self.H)
        self.particles  = []
        self.rings      = []

        # ── State ────────────────────────────────────────────────────────
        self.current_g  = None
        self.g_alpha    = 0.0
        self.g_timer    = 0.0
        self.fps_timer  = time.time()
        self.fps        = 0.0
        self._ts        = 0    # Monotonically increasing timestamp for MediaPipe

        print("\n  Ready! Gestures:")
        print("    Open Palm | Fist | Point | Peace | Thumbs Up")
        print("    Pinch | OK | Rock On | Call Me")
        print("  Press Q in the window to quit.")
        print("="*60 + "\n")

    # ── MediaPipe callback (called from background thread) ────────────────
    def _on_result(self, result: HandLandmarkerResult,
                   output_image: mp.Image, timestamp_ms: int):
        with self._lock:
            self._latest_result = result

    # ── Trigger a gesture animation ───────────────────────────────────────
    def _trigger(self, key):
        if key == self.current_g:
            return
        self.current_g = key
        self.g_timer   = time.time()
        self.g_alpha   = 1.0
        g  = GESTURES[key]
        cx = self.W // 2
        cy = int(self.H * 0.42)
        spawn_particles(g["style"], cx, cy, g["color"],
                        self.particles, self.rings)
        print(f"  {g['name']}")

    # ── Main loop ─────────────────────────────────────────────────────────
    def run(self):
        while True:
            # 1. Read frame from camera
            ok, frame = self.cap.read()
            if not ok:
                continue

            # 2. Mirror (so left hand = left on screen)
            frame = cv2.flip(frame, 1)

            # 3. Build dark stage background
            bg = np.full((self.H, self.W, 3), (6, 6, 10), dtype=np.uint8)
            self.hud.draw_grid(bg)

            # Add ghosted camera feed (very dim, just for hand reference)
            ghost = (frame.astype(np.float32) * 0.10).astype(np.uint8)
            cv2.add(bg, ghost, bg)

            # 4. Send frame to MediaPipe (async — non-blocking)
            #
            # Steps:
            #   a) Convert BGR -> RGB  (MediaPipe expects RGB)
            #   b) Wrap in mp.Image
            #   c) Call detect_async with strictly increasing timestamp
            #
            self._ts += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=rgb_frame)
            self._landmarker.detect_async(mp_image, self._ts)

            # 5. Read latest result (thread-safe)
            with self._lock:
                result = self._latest_result

            hand_count = 0

            if result and result.hand_landmarks:
                hand_count = len(result.hand_landmarks)

                # Draw skeleton for each detected hand
                for lm_list in result.hand_landmarks:
                    self.hud.draw_landmarks(bg, lm_list, self.current_g)

                # Recognize gesture from first hand
                detected = self.recognizer.recognize(result.hand_landmarks[0])
                if detected:
                    self._trigger(detected)
            else:
                self.recognizer.history.clear()

            # 6. Fade out gesture name over time
            if self.current_g:
                elapsed      = time.time() - self.g_timer
                self.g_alpha = max(0.0, 1.0 - elapsed / GESTURE_HOLD_SEC)
                if self.g_alpha <= 0:
                    self.current_g = None

            # 7. Draw effects
            self.hud.draw_bg_glow(bg, self.current_g, self.g_alpha)

            # Update + draw particles
            for i in range(len(self.rings)-1, -1, -1):
                self.rings[i].update()
                self.rings[i].draw(bg)
                if self.rings[i].life <= 0:
                    self.rings.pop(i)

            for i in range(len(self.particles)-1, -1, -1):
                self.particles[i].update()
                self.particles[i].draw(bg)
                if self.particles[i].life <= 0:
                    self.particles.pop(i)

            # Gesture name text
            if self.current_g:
                self.hud.draw_gesture_name(bg, self.current_g, self.g_alpha)
            elif hand_count == 0:
                self.hud.draw_no_hand_hint(bg)

            # 8. FPS + status bar
            now            = time.time()
            self.fps       = 1.0 / max(now - self.fps_timer, 1e-6)
            self.fps_timer = now
            self.hud.draw_status_bar(bg, self.fps, hand_count, self.current_g)

            # 9. Show frame
            cv2.imshow("AI Gesture Stage", bg)

            # 10. Quit on Q or Escape
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\nQuitting — goodbye!")
                break

        self._cleanup()

    def _cleanup(self):
        self.cap.release()
        self._landmarker.close()
        cv2.destroyAllWindows()


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        app = GestureStageApp()
        app.run()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
    except KeyboardInterrupt:
        print("\n[Stopped]")
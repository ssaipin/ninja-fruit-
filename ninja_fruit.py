import cv2
import numpy as np
import random
import time
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# CONFIG 
WIDTH, HEIGHT  = 900, 600
ASSETS         = "assets"
MODEL_PATH     = "hand_landmarker.task"
GRAVITY        = 900      
SPAWN_INTERVAL = 0.8      
SLICE_RADIUS   = 40       
BOMB_CHANCE    = 0.20     
BOMB_LIVES     = 3        

# change fruit & bomb size
FRUIT_SIZE = 450
BOMB_SIZE  = 500

FRUIT_LIST = [
    {"whole": "Apple.png",      "sliced": "Apple_slice.png"},
    {"whole": "Banana.png",     "sliced": "Banana_slice.png"},
    {"whole": "Orange.png",     "sliced": "Orange_slice.png"},
    {"whole": "Watermelon.png", "sliced": "Watermelon_slice.png"},
]
BOMB_DEF = {"whole": "Bomb.png", "sliced": "Bomb_explode.png"}


# HELPERS

def load_png(filename, size):
    #Load a PNG from the assets folder and resize it
    path = os.path.join(ASSETS, filename)
    img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Warning: could not load {path}")
        return np.zeros((size, size, 4), dtype=np.uint8)

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    return cv2.resize(img, (size, size))


def draw_png(canvas, img, x, y):
    #Alpha-blend a transparent PNG onto the canvas at position (x, y)
    h, w = img.shape[:2]

    x1 = max(0, x);           y1 = max(0, y)
    x2 = min(WIDTH,  x + w);  y2 = min(HEIGHT, y + h)
    if x2 <= x1 or y2 <= y1:
        return

    sx1 = x1 - x;  sy1 = y1 - y
    sx2 = sx1 + (x2 - x1);  sy2 = sy1 + (y2 - y1)

    patch = img[sy1:sy2, sx1:sx2]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    roi   = canvas[y1:y2, x1:x2].astype(np.float32)

    canvas[y1:y2, x1:x2] = (patch[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)


# FRUIT

class Fruit:
    def __init__(self, img_whole, img_sliced, size, is_bomb=False):
        self.img_whole  = img_whole
        self.img_sliced = img_sliced
        self.size       = size
        self.is_bomb    = is_bomb    # bombs behave differently when hit

        self.x = float(random.randint(100, WIDTH - 100))
        self.y = float(HEIGHT + 50)

        self.vx = random.uniform(-120, 120)
        self.vy = -random.uniform(700, 900)  # ← higher = launches further up 

        self.is_sliced = False

    def update(self, dt):
        #Move the fruit and apply gravity
        self.vy += GRAVITY * dt
        self.x  += self.vx * dt
        self.y  += self.vy * dt

    def draw(self, canvas):
        #Draw whole or sliced/exploded sprite.
        img = self.img_sliced if self.is_sliced else self.img_whole
        draw_png(canvas, img,
                 int(self.x - self.size // 2),
                 int(self.y - self.size // 2))

    def is_hit(self, finger_x, finger_y):
        return abs(self.x - finger_x) < SLICE_RADIUS and \
               abs(self.y - finger_y) < SLICE_RADIUS

    def is_offscreen(self):
        return self.y > HEIGHT + 100


# GAME

class Game:
    def __init__(self):
        self._load_assets()
        self._setup_camera()
        self._setup_hand_tracker()

        self.fruits     = []
        self.score      = 0
        self.bomb_hits  = 0   # counts bomb hits; resets game at BOMB_LIVES
        self.last_spawn = time.time()
        self.game_over  = False
        self.final_score = 0

    def _load_assets(self):
        #Load all fruits and the bomb.
        self.fruit_images = []
        for entry in FRUIT_LIST:
            whole  = load_png(entry["whole"],  FRUIT_SIZE)
            sliced = load_png(entry["sliced"], FRUIT_SIZE)
            self.fruit_images.append((whole, sliced))

        # bomb loaded separately so we always know which is which
        self.bomb_whole  = load_png(BOMB_DEF["whole"],  BOMB_SIZE)
        self.bomb_sliced = load_png(BOMB_DEF["sliced"], BOMB_SIZE)

    def _setup_camera(self):
        self.cap = cv2.VideoCapture(0)

    def _setup_hand_tracker(self):
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.hand_detector = mp_vision.HandLandmarker.create_from_options(options)
        self.timestamp     = 0

    # HAND TRACKING 

    def get_finger_position(self, frame):
        #Returns (x, y) of index fingertip, or None if no hand found
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.timestamp += 1
        result = self.hand_detector.detect_for_video(mp_image, self.timestamp)

        if result.hand_landmarks:
            tip = result.hand_landmarks[0][8]   # index fingertip
            return int(tip.x * WIDTH), int(tip.y * HEIGHT)

        return None

    # SPAWNING

    def maybe_spawn(self):
        #Spawn a random fruit or bomb if enough time has passed
        if time.time() - self.last_spawn > SPAWN_INTERVAL:
            if random.random() < BOMB_CHANCE:
                # spawn a bomb
                self.fruits.append(
                    Fruit(self.bomb_whole, self.bomb_sliced, BOMB_SIZE, is_bomb=True)
                )
            else:
                # spawn a random fruit
                whole, sliced = random.choice(self.fruit_images)
                self.fruits.append(
                    Fruit(whole, sliced, FRUIT_SIZE, is_bomb=False)
                )
            self.last_spawn = time.time()

    def _reset_game(self):
        #Reset all game state back to the start
        self.fruits      = []
        self.score       = 0
        self.bomb_hits   = 0
        self.last_spawn  = time.time()
        self.game_over   = False
        self.final_score = 0

    # HUD

    def draw_hud(self, canvas):
        """Draw score and lives in the top corners."""
        cv2.putText(canvas, f"Score: {self.score}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (51, 0, 0), 2)

        # show bomb hits as red circles (fills up to BOMB_LIVES)
        for i in range(BOMB_LIVES):
            cx    = WIDTH - 30 - i * 36
            color = (0, 0, 200) if i < self.bomb_hits else (80, 80, 80)
            cv2.circle(canvas, (cx, 28), 12, color, -1)
            cv2.circle(canvas, (cx, 28), 12, (255, 255, 255), 1)

        cv2.putText(canvas, "Bombs", (WIDTH - 30 - (BOMB_LIVES - 1) * 36 - 68, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def draw_game_over(self, canvas):
        #Dark overlay with GAME OVER, final score, and Play Again button
        # dim the background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        # white border box
        box_x1, box_y1 = WIDTH // 2 - 200, HEIGHT // 2 - 130
        box_x2, box_y2 = WIDTH // 2 + 200, HEIGHT // 2 + 130
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 2)

        # GAME OVER
        cv2.putText(canvas, "GAME  OVER",
                    (WIDTH // 2 - 155, HEIGHT // 2 - 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 3)

        # Final score
        cv2.putText(canvas, f"Final Score:  {self.final_score}",
                    (WIDTH // 2 - 145, HEIGHT // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

        # Play Again button
        btn_x1, btn_y1 = WIDTH // 2 - 110, HEIGHT // 2 + 35
        btn_x2, btn_y2 = WIDTH // 2 + 110, HEIGHT // 2 + 90
        cv2.rectangle(canvas, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
        cv2.putText(canvas, "Play Again",
                    (WIDTH // 2 - 82, HEIGHT // 2 + 72),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

        # Play again 
        cv2.putText(canvas, "press  SPACE  to restart",
                    (WIDTH // 2 - 148, HEIGHT // 2 + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # MAIN LOOP 
    def run(self):
        prev_time = time.time()

        while True:
            # timing
            now = time.time()
            dt  = now - prev_time
            prev_time = now

            # read camera
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Camera not working")
                continue
            frame = cv2.flip(frame, 1)

            # hand tracking
            finger = self.get_finger_position(frame)

            # canvas: camera feed so you see yourself
            canvas = cv2.resize(frame, (WIDTH, HEIGHT))

            # draw finger dot
            if finger is not None:
                fx, fy = finger
                cv2.circle(canvas, (fx, fy), 15, (0, 255, 0), -1)
            else:
                cv2.putText(canvas, "No hand detected", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # spawn & update (skip when game is over)
            if not self.game_over:
                self.maybe_spawn()

            if not self.game_over:
                for fruit in self.fruits:
                    fruit.update(dt)

                    if finger and fruit.is_hit(fx, fy) and not fruit.is_sliced:
                        fruit.is_sliced = True
                        if fruit.is_bomb:
                            self.bomb_hits += 1
                            if self.bomb_hits >= BOMB_LIVES:
                                self.final_score = self.score
                                self.game_over   = True
                        else:
                            self.score += 1

                    fruit.draw(canvas)

                self.fruits = [f for f in self.fruits if not f.is_offscreen()]

            if self.game_over:
                # show game over screen instead of playing
                self.draw_game_over(canvas)
            else:
                # HUD on top of everything
                self.draw_hud(canvas)

            cv2.imshow("Fruit Game", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:                        # ESC = quit
                break
            if key == ord(' ') and self.game_over:  # SPACE = play again
                self._reset_game()

        self.cap.release()
        cv2.destroyAllWindows()


# RUN
if __name__ == "__main__":
    Game().run()
**Ninja Fruit**
* A hand tracking fruit slicing game inspired by Fruit Ninja
* built from scratch with Python, OpenCV, and MediaPipe
* Slice fruits flying across the screen using just index finger

**How to Play**
- Point your index finger at the webcam
- Swipe through fruits to slice them and earn points
- Hit 3 bombs and the game over
- Press SPACE to play again after game over
- Press ESC to quit

**Features**
* Real time hand tracking via MediaPipe (no mouse or keyboard needed)
* 4 fruits: Apple, Banana, Orange, Watermelon — each with a whole and sliced sprite
* Bombs randomly mixed in — slice them and lose a life
* Live webcam feed as the game background so you see yourself playing
* Score tracker and bomb hit counter displayed on screen
* Game Over screen with your final score and Play Again option
* Custom hand drawn sprites made in Procreate

**Getting Started**
1. git clone https://github.com/ssaipin/ninja-fruit-.git
   cd ninja-fruit-
2. Install dependencies
   pip install opencv-python mediapipe numpy
3. Download the hand tracking model
   curl -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
4. Run the game
   python3 ninja_fruit-.py

**Built With**
* Python
* OpenCV
* MediaPipe
* Numpy
* Procreate 

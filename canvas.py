import cv2
import numpy as np
import mediapipe as mp
import math
import time
import threading

# Try to import Windows sound library (optional feature)
try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# ------------------------------
# Configuration settings
# ------------------------------
class Config:
    WIDTH, HEIGHT = 1280, 720
    
    # Sensitivity settings
    PINCH_THRESHOLD = 40       # Distance required to start drawing
    SMOOTHING = 0.6            # Lower = smoother but laggy, Higher = faster but jittery
    
    # Visual settings
    BRUSH_SIZE = 8
    NEON_GLOW = True
    HUD_COLOR = (255, 255, 0)  # Cyan color for HUD elements
    
    # Arc color palette settings
    ARC_CENTER = (640, 0)      # Top center of the screen
    ARC_RADIUS = 150
    ARC_THICKNESS = 60


# ------------------------------
# Sound engine (runs in a thread)
# ------------------------------
class SoundEngine:
    def __init__(self):
        self.active = False
        self.velocity = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()

    # Update drawing state and movement speed
    def set_drawing(self, is_drawing, velocity):
        self.active = is_drawing
        self.velocity = velocity

    # Background loop that generates sound while drawing
    def _loop(self):
        while not self.stop_event.is_set():
            if AUDIO_AVAILABLE and self.active:
                try:
                    # Sound pitch changes based on movement speed
                    freq = int(200 + (self.velocity * 5))
                    freq = max(100, min(freq, 800))  # Clamp frequency range
                    winsound.Beep(freq, 40)
                except:
                    pass
            else:
                time.sleep(0.05)


# ------------------------------
# Hand tracking and HUD drawing
# ------------------------------
class HandSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.prev_pos = (0,0)

    # Process camera frame and extract hand landmark points
    def process(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            h, w, c = img.shape
            
            # Convert normalized landmarks into pixel coordinates
            points = []
            for lm in landmarks:
                points.append((int(lm.x * w), int(lm.y * h)))
            return points
        return None

    # Draw a sci-fi styled HUD overlay on the hand
    def draw_sci_fi_hud(self, img, points, pinch_dist):
        if not points: 
            return img
        
        overlay = img.copy()
        
        # Hand skeleton connections
        connections = [[0,1],[1,2],[2,3],[3,4],       # Thumb
                       [0,5],[5,6],[6,7],[7,8],       # Index
                       [0,9],[9,10],[10,11],[11,12],  # Middle
                       [0,13],[13,14],[14,15],[15,16],# Ring
                       [0,17],[17,18],[18,19],[19,20]]# Pinky
        
        # Draw connecting lines between joints
        for p1, p2 in connections:
            pt1 = points[p1]
            pt2 = points[p2]
            cv2.line(overlay, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)
            
        # Draw joint points
        for pt in points:
            cv2.circle(overlay, pt, 3, (0, 165, 255), -1)
            cv2.circle(overlay, pt, 6, (0, 255, 255), 1)

        # Target circle on index finger tip
        idx_x, idx_y = points[8]
        cv2.circle(overlay, (idx_x, idx_y), 10, (255, 255, 255), 1)
        
        # Pinch indicator bar
        bar_len = 40
        bar_height = 6
        fill = min(1.0, (100 - pinch_dist) / 60)
        fill = max(0.0, fill)
        
        bar_color = (0, 0, 255)
        if pinch_dist < Config.PINCH_THRESHOLD:
            bar_color = (0, 255, 0)
            cv2.putText(overlay, "ON", (idx_x + 20, idx_y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, bar_color, 2)
            
        cv2.rectangle(overlay, (idx_x + 15, idx_y),
                      (idx_x + 15 + bar_len, idx_y + bar_height),
                      (50,50,50), -1)
        cv2.rectangle(overlay, (idx_x + 15, idx_y),
                      (idx_x + 15 + int(bar_len * fill), idx_y + bar_height),
                      bar_color, -1)

        return cv2.addWeighted(overlay, 0.7, img, 0.3, 0)


# ------------------------------
# Arc color palette UI
# ------------------------------
class ArcPalette:
    def __init__(self):
        self.colors = [
            ((0, 0, 255), "RED"),
            ((0, 165, 255), "ORANGE"),
            ((0, 255, 255), "YELLOW"),
            ((0, 255, 0), "GREEN"),
            ((255, 255, 0), "CYAN"),
            ((255, 0, 255), "PURPLE"),
            ((255, 255, 255), "WHITE"),
            ((0, 0, 0), "CLEAR")
        ]
        self.selected_index = 4

    # Draw the arc palette and detect hover selection
    def draw(self, img, hover_pt):
        overlay = img.copy()
        
        num_colors = len(self.colors)
        sector_angle = 180 / num_colors
        
        cx, cy = Config.ARC_CENTER
        radius = Config.ARC_RADIUS
        
        hover_index = -1
        
        # Detect which color segment the cursor is hovering over
        if hover_pt:
            hx, hy = hover_pt
            dist = math.hypot(hx - cx, hy - cy)
            if radius < dist < radius + Config.ARC_THICKNESS:
                dx, dy = hx - cx, hy - cy
                angle = math.degrees(math.atan2(dy, dx))
                if angle < 0: 
                    angle += 360
                
                if 0 <= angle <= 180:
                    hover_index = int(angle / sector_angle)

        # Render arc segments
        for i in range(num_colors):
            start_ang = i * sector_angle
            end_ang = (i + 1) * sector_angle
            color, name = self.colors[i]
            
            thickness = Config.ARC_THICKNESS
            shift = 0
            
            # Highlight selected color
            if i == self.selected_index:
                shift = 15
                cv2.ellipse(img, (cx, cy),
                            (radius + shift, radius + shift),
                            0, start_ang, end_ang,
                            (255,255,255), -1)
            
            # Highlight hovered color and show label
            if i == hover_index:
                thickness += 10
                cv2.putText(img, name,
                            (cx - 40, cy + radius + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw color arc segment
            cv2.ellipse(img, (cx, cy),
                        (radius + shift + (thickness//2),
                         radius + shift + (thickness//2)),
                        0, start_ang, end_ang, color, thickness)

        return hover_index


# ------------------------------
# Main application loop
# ------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, Config.WIDTH)
    cap.set(4, Config.HEIGHT)
    
    hand_sys = HandSystem()
    palette = ArcPalette()
    sound = SoundEngine()
    
    # Drawing canvas
    canvas = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)
    
    # State variables
    smooth_x, smooth_y = 0, 0
    current_color = (255, 255, 0)
    
    print("IRON CANVAS ACTIVATED")
    
    while True:
        success, img = cap.read()
        if not success: 
            break
        
        img = cv2.flip(img, 1)
        points = hand_sys.process(img)
        
        is_drawing = False
        velocity = 0
        
        if points:
            # Get index and thumb tip positions
            idx_tip = points[8]
            thm_tip = points[4]
            
            # Smooth cursor movement
            cx, cy = idx_tip
            if smooth_x == 0: 
                smooth_x, smooth_y = cx, cy
            
            smooth_x = int(smooth_x * (1 - Config.SMOOTHING) + cx * Config.SMOOTHING)
            smooth_y = int(smooth_y * (1 - Config.SMOOTHING) + cy * Config.SMOOTHING)
            
            # Measure pinch distance
            dist = math.hypot(idx_tip[0] - thm_tip[0],
                              idx_tip[1] - thm_tip[1])
            
            # Draw HUD overlay
            img = hand_sys.draw_sci_fi_hud(img, points, dist)
            
            # Handle palette selection
            hover_idx = palette.draw(img, (smooth_x, smooth_y))
            
            if hover_idx != -1 and dist < Config.PINCH_THRESHOLD:
                color, name = palette.colors[hover_idx]
                if name == "CLEAR":
                    canvas[:] = 0
                else:
                    palette.selected_index = hover_idx
                    current_color = color
            
            # Drawing logic (avoid drawing over palette area)
            elif dist < Config.PINCH_THRESHOLD and smooth_y > 200:
                is_drawing = True
                
                velocity = math.hypot(smooth_x - cx, smooth_y - cy)
                
                cv2.line(canvas, (smooth_x, smooth_y), (cx, cy),
                         current_color, Config.BRUSH_SIZE)
                cv2.circle(canvas, (cx, cy),
                           Config.BRUSH_SIZE // 2, current_color, -1)
                
            # Update reference point
            smooth_x, smooth_y = cx, cy
            
        else:
            # Draw palette even when no hand is detected
            palette.draw(img, None)

        # Update sound engine
        sound.set_drawing(is_drawing, velocity)

        # Create glow effect
        canvas_small = cv2.resize(canvas, (0,0), fx=0.2, fy=0.2)
        blur = cv2.GaussianBlur(canvas_small, (15, 15), 0)
        blur_up = cv2.resize(blur, (Config.WIDTH, Config.HEIGHT))
        
        final_canvas = cv2.addWeighted(canvas, 1.0, blur_up, 1.5, 0)
        
        # Merge canvas with camera feed
        gray = cv2.cvtColor(final_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        img = cv2.add(img_bg, final_canvas)
        
        cv2.imshow("Iron Canvas Pro", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sound.stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

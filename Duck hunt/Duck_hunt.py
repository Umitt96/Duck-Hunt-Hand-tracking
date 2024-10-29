"""
@author: Ritalin
problem: Duck hunt with OPENCV
"""
import cv2
import mediapipe as mp
import numpy as np
import random
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Arkaplan ve imleç resimlerini yükleme
background_path = "bg.png"  
background = cv2.imread(background_path)
background = cv2.resize(background, (640, 480)) if background is not None else np.zeros((480, 640, 3), dtype=np.uint8)

# İmleç resimlerini yükleme
cursor_img = cv2.imread("cursor.png", cv2.IMREAD_UNCHANGED)
fire_cursor_img = cv2.imread("cursor-boom.png", cv2.IMREAD_UNCHANGED)

# İmleç resimlerini uygun boyuta getirme
cursor_img = cv2.resize(cursor_img, (50, 50))
fire_cursor_img = cv2.resize(fire_cursor_img, (50, 50))

# Kuş resmi
bird_img = cv2.imread("bird.png", cv2.IMREAD_COLOR) 
bird_img = cv2.resize(bird_img, (50, 50))

# Kuş sayısı
bird_count = 5
birds = [(random.randint(200, 440), random.randint(100, 200)) for _ in range(bird_count)]
bird_speed = 2
bird_direction = [1] * bird_count

# Skor ve Mermi Sayısı
score = 0
shots_left = 20

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    fire_mode = False
    fire_start_time = 0
    fire_duration = 0.5  
    fire_position = (0, 0)
    fire_cooldown = 1
    last_fire_time = 0
    last_missed_shot_time = 0
    missed_shot_cooldown = 1
    game_over = False
    all_birds_shot = False

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        canvas = background.copy()
        cursor_to_use = cursor_img  

        # Kuşları çizme ve hareket ettirme
        if not game_over:
            for i, bird_pos in enumerate(birds):
                bird_x, bird_y = bird_pos
                
                # Kuşlar hareket etsin
                bird_x += bird_speed * bird_direction[i]

                # Kenarlara çarpmasını önleme
                if bird_x > 440 - 25 or bird_x < 200:
                    bird_direction[i] *= -1

                # Kuşlar rastgele yukarı-aşağı hareket etsin
                bird_y += random.randint(-2, 2)

                # Ekranın dışına çıkmasını engelle
                bird_y = max(100, min(bird_y, 200))

                birds[i] = (bird_x, bird_y)

                canvas[bird_y:bird_y + 50, bird_x:bird_x + 50] = bird_img

                # Ateş etmeyi kontrol et
                if fire_mode and (
                    fire_position[0] - 35 <= bird_x <= fire_position[0] + 35 
                    and fire_position[1] - 35 <= bird_y <= fire_position[1] + 35 
                ):
                    birds.pop(i)
                    bird_direction.pop(i)
                    score += 1
                    shots_left += 2
                    fire_mode = False
                    last_fire_time = time.time()
                    print("Skor:", score)
                elif fire_mode:
                    fire_mode = False
                    last_missed_shot_time = time.time()
                    print("Başarısız Atış!", "Kalan mermi:", shots_left)

        if results.multi_hand_landmarks and time.time() - last_fire_time > fire_cooldown and time.time() - last_missed_shot_time > missed_shot_cooldown and not game_over:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip_y = hand_landmarks.landmark[8].y
                thumb_base_y = hand_landmarks.landmark[5].y
                h, w, _ = image.shape
                x, y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

                # Ateş etme modu
                if index_finger_tip_y > thumb_base_y and not fire_mode and shots_left > 0:
                    fire_mode = True
                    fire_start_time = time.time()
                    fire_position = (x, y)  
                    cursor_to_use = fire_cursor_img
                    shots_left -= 1
                elif fire_mode:
                    fire_mode = False

                # İmleci çizme
                if cursor_to_use is not None:
                    cursor_h, cursor_w, _ = cursor_to_use.shape
                    top_left_x = x - cursor_w // 2
                    top_left_y = y - cursor_h // 2

                    if 0 <= top_left_x <= w - cursor_w and 0 <= top_left_y <= h - cursor_h:
                        alpha_s = cursor_to_use[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s

                        for c in range(3):
                            canvas[top_left_y:top_left_y + cursor_h, top_left_x:top_left_x + cursor_w, c] = (
                                alpha_s * cursor_to_use[:, :, c] + alpha_l * canvas[top_left_y:top_left_y + cursor_h, top_left_x:top_left_x + cursor_w, c]
                            )

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Ateş etme animasyonu
        if fire_mode:
            if time.time() - fire_start_time > fire_duration:
                fire_mode = False
            fire_duration = 0.2
            if time.time() - fire_start_time > fire_duration:
                cursor_to_use = cursor_img

        # Kalan mermi sayısını ve skoru ekrana yazdırma
        cv2.putText(canvas, f"Kalan mermi: {shots_left}", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(canvas, f"Skor: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Oyun bitti mi?
        if shots_left == 0:
            game_over = True
            cv2.putText(canvas, "OYUN BITTI :(", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        elif len(birds) == 0:
            all_birds_shot = True
            cv2.putText(canvas, "TEBRIKLER!", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

        cv2.imshow('MediaPipe Hands', image)
        cv2.imshow('Duck hunt game', canvas)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.set_page_config(page_title="AirTicTac - Gesture Game", layout="wide")  # ✅ wider layout

# Game state variables
if "board" not in st.session_state:
    st.session_state.board = [[None for _ in range(3)] for _ in range(3)]
    st.session_state.current_turn = "X"
    st.session_state.last_move_time = 0
    st.session_state.winner = None
    st.session_state.game_over = False

# Title
st.title("✋ AirTicTac - Gesture Tic Tac Toe (High Clarity)")

# Instructions
with st.expander("📘 How to Play"):
    st.markdown("""
    - Make a **fist to move your hand** to your desired column.
    - **Extend your index finger** to place your move.
    - Players alternate between **X (red)** and **O (blue)**.
    - Raise your finger and hold steady for 1 second to place your move.
    - Press **Reset** to play again.
    """)

# Reset Button
if st.button("🔄 Reset Game"):
    st.session_state.board = [[None for _ in range(3)] for _ in range(3)]
    st.session_state.current_turn = "X"
    st.session_state.last_move_time = 0
    st.session_state.winner = None
    st.session_state.game_over = False


def draw_grid(frame):
    h, w = frame.shape[:2]
    for i in range(1, 3):
        cv2.line(frame, (0, i * h // 3), (w, i * h // 3), (255, 255, 255), 2)
        cv2.line(frame, (i * w // 3, 0), (i * w // 3, h), (255, 255, 255), 2)


def draw_marks(frame):
    h, w = frame.shape[:2]
    cw, ch = w // 3, h // 3
    for r in range(3):
        for c in range(3):
            center = (c * cw + cw // 2, r * ch + ch // 2)
            if st.session_state.board[r][c] == "X":
                cv2.line(frame, (c * cw + 20, r * ch + 20), ((c + 1) * cw - 20, (r + 1) * ch - 20), (0, 0, 255), 5)
                cv2.line(frame, (c * cw + 20, (r + 1) * ch - 20), ((c + 1) * cw - 20, r * ch + 20), (0, 0, 255), 5)
            elif st.session_state.board[r][c] == "O":
                cv2.circle(frame, center, min(cw, ch) // 3, (255, 0, 0), 5)


def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0]:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i]:
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0]:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2]:
        return board[0][2]
    return None


def fingers_up(hand_landmarks):
    fingers = []
    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Index, Middle, Ring, Pinky
    for tip_id in [8, 12, 16, 20]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


# Set up camera and MediaPipe
cap = cv2.VideoCapture(0)

# ✅ Set highest resolution for clarity
cap.set(3, 1920)  # width
cap.set(4, 1080)  # height

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
frame_placeholder = st.empty()

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Cannot access camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        draw_grid(frame)
        draw_marks(frame)

        if results.multi_hand_landmarks and not st.session_state.game_over:
            for hand_landmarks in results.multi_hand_landmarks:

                finger_status = fingers_up(hand_landmarks)
                
                # ✅ Ignore input if all fingers are down (fist)
                if sum(finger_status[1:]) == 0:
                    continue

                # ✅ Process only if index finger is up
                if finger_status[1] == 1:
                    x = int(hand_landmarks.landmark[8].x * w)
                    y = int(hand_landmarks.landmark[8].y * h)
                    row, col = int(y // (h / 3)), int(x // (w / 3))

                    if 0 <= row < 3 and 0 <= col < 3:
                        now = time.time()
                        if st.session_state.board[row][col] is None and (now - st.session_state.last_move_time) > 1.0:
                            st.session_state.board[row][col] = st.session_state.current_turn
                            st.session_state.winner = check_winner(st.session_state.board)
                            if st.session_state.winner:
                                st.session_state.game_over = True
                            st.session_state.current_turn = "O" if st.session_state.current_turn == "X" else "X"
                            st.session_state.last_move_time = now

                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Text display
        status_text = f"{st.session_state.winner} wins!" if st.session_state.game_over else f"Turn: {st.session_state.current_turn}"
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        # ✅ Increase displayed image size in Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)


except Exception as e:
    st.error(f"An error occurred: {e}")

finally:
    cap.release()
    hands.close()

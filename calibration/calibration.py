import cv2
import mediapipe as mp
import math
import os

# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "face_landmarker.task"
    )
)

print("Model path:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# EYE RATIO
# ============================================================

def eye_ratio(face, top, bottom, left, right):

    vertical = distance(
        face[top],
        face[bottom]
    )

    horizontal = distance(
        face[left],
        face[right]
    )

    return vertical / horizontal


# ============================================================
# MOUTH RATIO
# ============================================================

def mouth_ratio(face):

    vertical = distance(
        face[13],
        face[14]
    )

    horizontal = distance(
        face[61],
        face[291]
    )

    return vertical / horizontal


# ============================================================
# SMILE RATIO
# ============================================================

def smile_ratio(face):

    mouth_width = distance(
        face[61],
        face[291]
    )

    face_width = distance(
        face[234],
        face[454]
    )

    return mouth_width / face_width


# ============================================================
# IRIS POSITION
# ============================================================

def iris_position(face, iris, left_corner, right_corner):

    eye_left = face[left_corner].x
    eye_right = face[right_corner].x

    iris_x = face[iris].x

    eye_width = eye_right - eye_left

    if abs(eye_width) < 0.0001:
        return 0.5

    position = (
        iris_x - eye_left
    ) / eye_width

    return position


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("❌ Could not access camera.")

    exit()


# ============================================================
# MEDIAPIPE
# ============================================================

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        # ----------------------------------------------------
        # Read camera
        # ----------------------------------------------------

        success, frame = camera.read()

        if not success:

            print("❌ Could not read camera.")

            break


        # ----------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Create MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # Detect face
        # ----------------------------------------------------

        result = landmarker.detect(
            mp_image
        )


        # ====================================================
        # FACE FOUND
        # ====================================================

        if result.face_landmarks:

            face = result.face_landmarks[0]


            # ------------------------------------------------
            # Eyes
            # ------------------------------------------------

            left_eye = eye_ratio(
                face,
                159,
                145,
                33,
                133
            )

            right_eye = eye_ratio(
                face,
                386,
                374,
                362,
                263
            )


            # ------------------------------------------------
            # Mouth
            # ------------------------------------------------

            mouth_open = mouth_ratio(
                face
            )


            # ------------------------------------------------
            # Smile
            # ------------------------------------------------

            smile = smile_ratio(
                face
            )


            # ------------------------------------------------
            # Iris
            # ------------------------------------------------

            left_iris = iris_position(
                face,
                468,
                33,
                133
            )

            right_iris = iris_position(
                face,
                473,
                362,
                263
            )


            # =================================================
            # DISPLAY VALUES
            # =================================================

            cv2.putText(
                frame,
                f"Left Eye: {left_eye:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Right Eye: {right_eye:.3f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Mouth Open: {mouth_open:.3f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Smile: {smile:.3f}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Left Iris: {left_iris:.3f}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Right Iris: {right_iris:.3f}",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            frame,
            "Make expression and record values",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Reaction Calibration",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()


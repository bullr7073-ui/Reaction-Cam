import cv2
import mediapipe as mp
import math
import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_landmarker.task"
)

REACTIONS_DIR = os.path.join(
    BASE_DIR,
    "reactions"
)


# ============================================================
# REACTION IMAGES
# ============================================================

reaction_files = {
    "Larry": "larry.jpeg",
    "Ronaldo": "Ronaldo.jpeg",
    "Cute": "Cute.jpeg",
    "Rizz": "Rizz.jpeg",
    "Monkey": "Monkey.jpeg",
    "Goofy": "Goofy.jpeg",
    "What": "What.jpeg",
    "Dog": "Dog.jpeg"
}


reaction_images = {}

for name, filename in reaction_files.items():

    path = os.path.join(
        REACTIONS_DIR,
        filename
    )

    image = cv2.imread(path)

    if image is None:

        print(f"WARNING: Could not load {filename}")

    else:

        reaction_images[name] = image


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# EYE RATIO
# ============================================================

def eye_ratio(
    face,
    top,
    bottom,
    left,
    right
):

    vertical = distance(
        face[top],
        face[bottom]
    )

    horizontal = distance(
        face[left],
        face[right]
    )

    if horizontal == 0:
        return 0

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

    if horizontal == 0:
        return 0

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

    if face_width == 0:
        return 0

    return mouth_width / face_width


# ============================================================
# IRIS POSITION
# ============================================================

def iris_position(
    face,
    iris,
    left_corner,
    right_corner
):

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
# CALIBRATED REACTION VALUES
# ============================================================

reaction_values = {

    "Ronaldo": [
        0.342,
        0.308,
        0.405,
        0.519,
        0.388,
        0.504
    ],

    "Rizz": [
        0.399,
        0.365,
        0.214,
        0.450,
        0.372,
        0.485
    ],

    "Larry": [
        0.480,
        0.451,
        0.008,
        0.370,
        0.439,
        0.543
    ],

    "Monkey": [
        0.429,
        0.422,
        0.333,
        0.517,
        0.444,
        0.520
    ],

    "Goofy": [
        0.407,
        0.419,
        0.671,
        0.412,
        0.415,
        0.522
    ],

    "What": [
        0.468,
        0.542,
        0.008,
        0.392,
        0.355,
        0.468
    ],

    "Dog": [
        0.630,
        0.475,
        0.012,
        0.394,
        0.536,
        0.617
    ]
}


# ============================================================
# VALUE WEIGHTS
# ============================================================

# Mouth is especially useful for:
#
# Rizz
# Monkey
# Goofy
#
# Iris is especially useful for:
#
# What
# Dog
#
# Smile helps distinguish:
#
# Larry
# Cute

weights = [
    1.0,   # Left Eye
    1.0,   # Right Eye
    2.0,   # Mouth
    1.5,   # Smile
    1.2,   # Left Iris
    1.2    # Right Iris
]


# ============================================================
# FIND CLOSEST REACTION
# ============================================================

def detect_reaction(values):

    best_reaction = "Larry"

    best_score = float("inf")

    for reaction, target in reaction_values.items():

        score = 0

        for i in range(len(values)):

            difference = (
                values[i] - target[i]
            )

            score += (
                difference *
                difference *
                weights[i]
            )

        if score < best_score:

            best_score = score

            best_reaction = reaction


    # --------------------------------------------------------
    # Confidence protection
    # --------------------------------------------------------
    #
    # If the expression is too different from every
    # calibration sample, use Larry instead.
    #

    if best_score > 0.035:

        return "Larry"

    return best_reaction


# ============================================================
# RESIZE REACTION IMAGE
# ============================================================

def resize_reaction(image, width=500):

    height, original_width = image.shape[:2]

    ratio = width / original_width

    new_height = int(
        height * ratio
    )

    return cv2.resize(
        image,
        (width, new_height)
    )


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("❌ Could not access camera.")

    exit()


# ============================================================
# START MEDIAPIPE
# ============================================================

with FaceLandmarker.create_from_options(
    options
) as landmarker:

    current_reaction = "Larry"

    while True:

        # ----------------------------------------------------
        # CAMERA FRAME
        # ----------------------------------------------------

        success, frame = camera.read()

        if not success:

            print("❌ Could not read camera.")

            break


        # ----------------------------------------------------
        # MIRROR CAMERA
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MEDIAPIPE IMAGE
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # DETECT FACE
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
            # Measurements
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


            mouth_open = mouth_ratio(
                face
            )


            smile = smile_ratio(
                face
            )


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


            # ------------------------------------------------
            # Put values together
            # ------------------------------------------------

            values = [

                left_eye,

                right_eye,

                mouth_open,

                smile,

                left_iris,

                right_iris
            ]


            # ------------------------------------------------
            # Detect reaction
            # ------------------------------------------------

            detected = detect_reaction(
                values
            )


            current_reaction = detected


        # ====================================================
        # DISPLAY REACTION NAME
        # ====================================================

        cv2.putText(

            frame,

            f"Reaction: {current_reaction}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            "Press Q to quit",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(

            "Reaction Cam",

            frame
        )


        # ====================================================
        # SHOW REACTION IMAGE
        # ====================================================

        if current_reaction in reaction_images:

            reaction_image = reaction_images[
                current_reaction
            ]

            reaction_image = resize_reaction(
                reaction_image
            )

            cv2.imshow(

                "Reaction",

                reaction_image
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

print("Reaction Cam closed.")
import cv2
import numpy as np
from PIL import Image
import gradio as gr

# Try to import MediaPipe; if unavailable (slow to build on some hosts),
# fall back to a lightweight Haar-cascade heuristic.
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

# Landmark index groups used when MediaPipe is available. When falling back
# to Haar cascades we synthesize these indices into a landmarks list so the
# rest of the pipeline can remain unchanged.
FACE_LANDMARKS = {
    "left_eye": [33, 133, 160, 159, 158, 157, 173, 246],
    "right_eye": [362, 263, 387, 386, 385, 384, 398, 466],
    "nose_tip": [1, 2, 98, 327],
    "mouth": [61, 291, 78, 308, 95, 324]
}


def to_cv2(image: Image.Image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def to_pil(image_cv):
    return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))


def get_landmarks(img):
    h, w = img.shape[:2]
    if HAS_MEDIAPIPE:
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
            results = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None
            coords = []
            for lm in results.multi_face_landmarks[0].landmark:
                coords.append((int(lm.x * w), int(lm.y * h)))
            return coords

    # Fallback: use Haar cascade to detect a face and synthesize approximate
    # landmark positions. This is less accurate than MediaPipe but fast and
    # doesn't require heavy native dependencies.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, fw, fh = faces[0]

    # Create a large landmarks array and fill only indices referenced by
    # FACE_LANDMARKS so existing helpers can operate unchanged.
    coords = [(0, 0)] * 500

    # approximate left/right eye centers
    left_eye_center = (int(x + fw * 0.3), int(y + fh * 0.35))
    right_eye_center = (int(x + fw * 0.7), int(y + fh * 0.35))
    nose_center = (int(x + fw * 0.5), int(y + fh * 0.5))
    mouth_center = (int(x + fw * 0.5), int(y + fh * 0.75))

    for i, pt in enumerate(FACE_LANDMARKS["left_eye"]):
        coords[pt] = (left_eye_center[0] + np.random.randint(-6, 6), left_eye_center[1] + np.random.randint(-4, 4))
    for i, pt in enumerate(FACE_LANDMARKS["right_eye"]):
        coords[pt] = (right_eye_center[0] + np.random.randint(-6, 6), right_eye_center[1] + np.random.randint(-4, 4))
    for i, pt in enumerate(FACE_LANDMARKS["nose_tip"]):
        coords[pt] = (nose_center[0] + np.random.randint(-4, 4), nose_center[1] + np.random.randint(-4, 4))
    for i, pt in enumerate(FACE_LANDMARKS["mouth"]):
        coords[pt] = (mouth_center[0] + np.random.randint(-8, 8), mouth_center[1] + np.random.randint(-6, 6))

    return coords


def bounding_rect_from_idxs(landmarks, idxs, pad=10):
    pts = [landmarks[i] for i in idxs if i < len(landmarks)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = max(min(xs) - pad, 0), min(max(xs) + pad, 10**5)
    y1, y2 = max(min(ys) - pad, 0), min(max(ys) + pad, 10**5)
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def paste_scaled_patch(base, rect, scale, flags=cv2.NORMAL_CLONE):
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return base
    patch = base[y:y+h, x:x+w].copy()
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    patch_resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Create mask for seamlessClone
    mask = 255 * np.ones(patch_resized.shape[:2], patch_resized.dtype)

    center = (x + w // 2, y + h // 2)
    try:
        out = cv2.seamlessClone(patch_resized, base, mask, center, flags)
    except Exception:
        out = base
    return out


def stylize_cartoon(img, strength=0.6):
    # bilateral filtering and edge preservation for a cartoon look
    num_bilateral = 5
    img_color = img.copy()
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=75, sigmaSpace=75)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    edges = cv2.adaptiveThreshold(img_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)

    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img_color, edges_colored)
    return cv2.addWeighted(img, 1 - strength, cartoon, strength, 0)


def generate_caricature(inp_img: Image.Image, eye_scale: float = 1.5, nose_scale: float = 1.15,
                        mouth_scale: float = 0.95, stylize_strength: float = 0.6):
    img_cv = to_cv2(inp_img)
    landmarks = get_landmarks(img_cv)
    if landmarks is None:
        return inp_img

    out = img_cv.copy()

    # Eyes
    lx, ly, lw, lh = bounding_rect_from_idxs(landmarks, FACE_LANDMARKS["left_eye"], pad=8)
    rx, ry, rw, rh = bounding_rect_from_idxs(landmarks, FACE_LANDMARKS["right_eye"], pad=8)
    out = paste_scaled_patch(out, (lx, ly, lw, lh), eye_scale)
    out = paste_scaled_patch(out, (rx, ry, rw, rh), eye_scale)

    # Nose
    nx, ny, nw, nh = bounding_rect_from_idxs(landmarks, FACE_LANDMARKS["nose_tip"], pad=6)
    out = paste_scaled_patch(out, (nx, ny, nw, nh), nose_scale)

    # Mouth
    mx, my, mw, mh = bounding_rect_from_idxs(landmarks, FACE_LANDMARKS["mouth"], pad=10)
    out = paste_scaled_patch(out, (mx, my, mw, mh), mouth_scale)

    # Stylize
    out = stylize_cartoon(out, strength=stylize_strength)

    return to_pil(out)


def main():
    iface = gr.Interface(
        fn=generate_caricature,
        inputs=[
            gr.Image(type='pil', label='Input Image'),
            gr.Slider(1.0, 2.5, value=1.5, step=0.05, label='Eye scale'),
            gr.Slider(0.7, 1.5, value=1.15, step=0.01, label='Nose scale'),
            gr.Slider(0.7, 1.2, value=0.95, step=0.01, label='Mouth scale'),
            gr.Slider(0.0, 1.0, value=0.6, step=0.05, label='Stylize strength')
        ],
        outputs=gr.Image(type='pil', label='Caricature'),
        title='Gerador de Caricatura (heurístico)',
        description='Protótipo local: usa MediaPipe + OpenCV para exagerar traços e aplicar estilo cartoon.'
    )
    iface.launch()


if __name__ == '__main__':
    main()

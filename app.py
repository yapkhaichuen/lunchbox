import streamlit as st
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import io

# -------- MTCNN FACE DETECTION --------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.85, 0.90, 0.95])  # Less sensitive, more accurate

def detect_faces(image):
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return []
    return [tuple(map(int, box)) for box in boxes]

# -------- LIGHT LEAK FUNCTIONS --------
def random_color():
    palette = [
        np.array([1.0, np.random.uniform(0.2, 0.4), 0.0]),    # Orange
        np.array([1.0, 0.0, np.random.uniform(0.2, 0.4)]),    # Reddish
        np.array([np.random.uniform(0.9, 1.0), np.random.uniform(0.3, 0.5), 0.0]),  # Orange-Yellow
        np.array([np.random.uniform(0.9, 1.0), 0.0, np.random.uniform(0.3, 0.5)]),  # Red-Orange
        np.array([np.random.uniform(0.8, 1.0), np.random.uniform(0.2, 0.3), 0.0]),  # Deeper Orange
        np.array([np.random.uniform(0.7, 0.9), 0.0, np.random.uniform(0.2, 0.3)]),  # Deeper Red
        np.array([1.0, np.random.uniform(0.1, 0.2), 0.0]),  # Bright Orange
        np.array([np.random.uniform(0.7, 0.8), 0.0, np.random.uniform(0.2, 0.3)]),  # Muted Red-Orange
        np.array([np.random.uniform(0.6, 0.8), np.random.uniform(0.2, 0.3), 0.0]),  # Dull Orange
        np.array([np.random.uniform(0.6, 0.7), 0.0, np.random.uniform(0.2, 0.3)]),  # Dull Red
        np.array([0.9, 0.4, 0.6]),  # Pink, rare
    ]
    # Weighted choice to reduce pink
    weights = [1] * 10 + [0.1]
    idx = np.random.choice(len(palette), p=np.array(weights) / np.sum(weights))
    return palette[idx], palette[np.random.randint(len(palette) - 1)]

def fast_blur_mask(mask, sigma, upscale_shape):
    down_factor = 12
    small_shape = (upscale_shape[1] // down_factor, upscale_shape[0] // down_factor)
    small_mask = cv2.resize(mask, small_shape, interpolation=cv2.INTER_AREA)
    ksize = int(sigma / down_factor) * 2 + 1
    blurred = cv2.GaussianBlur(small_mask, (ksize, ksize), sigmaX=sigma / down_factor)
    return cv2.resize(blurred, (upscale_shape[1], upscale_shape[0]), interpolation=cv2.INTER_LINEAR)

def generate_light_leak_mask(shape, faces=None, num_leaks=3):
    h, w = shape[:2]
    h_small, w_small = h // 2, w // 2
    final_mask = np.zeros((h_small, w_small), dtype=np.float32)

    y, x = np.ogrid[:h_small, :w_small]
    cx, cy = w_small / 2, h_small / 2
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    prob_map = (distance / distance.max()) ** 2

    face_mask = np.ones((h_small, w_small), dtype=np.float32)
    if faces:
        for (x1, y1, x2, y2) in faces:
            x1, y1, x2, y2 = x1 // 2, y1 // 2, x2 // 2, y2 // 2
            face_mask[max(0, y1):min(h_small, y2), max(0, x1):min(w_small, x2)] = 0.0
        prob_map *= face_mask

    prob_map /= prob_map.sum() + 1e-8
    coords = np.argwhere(prob_map > 0)

    for _ in range(num_leaks):
        for _ in range(8):
            idx = np.random.choice(len(coords))
            cy, cx = coords[idx]
            temp = np.zeros_like(final_mask)
            shape_type = np.random.choice(['ellipse', 'streak', 'radial'])

            if shape_type == 'ellipse':
                axes = (np.random.randint(w_small // 5, w_small // 2),
                        np.random.randint(h_small // 5, h_small // 2))
                angle = np.random.uniform(0, 360)
                cv2.ellipse(temp, (cx, cy), axes, angle, 0, 360, color=1, thickness=-1)

            elif shape_type == 'streak':
                temp_streak = np.zeros_like(temp)
                length = np.random.randint(w_small // 3, int(w_small * 0.9))
                thickness = np.random.randint(10, 40)
                if np.random.rand() < 0.5:
                    x1 = max(0, cx - length // 2)
                    x2 = min(w_small - 1, cx + length // 2)
                    y1 = y2 = np.clip(cy, 0, h_small - 1)
                    cv2.line(temp_streak, (x1, y1), (x2, y2), color=1, thickness=thickness)
                else:
                    y1 = max(0, cy - length // 2)
                    y2 = min(h_small - 1, cy + length // 2)
                    x1 = x2 = np.clip(cx, 0, w_small - 1)
                    cv2.line(temp_streak, (x1, y1), (x2, y2), color=1, thickness=thickness)
                sigma = np.random.uniform(15, 40)
                temp = fast_blur_mask(temp_streak.astype(np.float32), sigma, (h_small, w_small))

            else:  # radial
                radial = np.zeros_like(temp)
                cv2.circle(radial, (cx, cy),
                           np.random.randint(h_small // 6, int(h_small * 0.6)),
                           color=1, thickness=-1)
                temp = fast_blur_mask(radial, sigma=np.random.uniform(50, 120),
                                      upscale_shape=(h_small, w_small))

            if np.all(temp * face_mask == temp):
                final_mask += temp * np.random.uniform(0.7, 1.5)
                break

    final_mask = cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    blurred = fast_blur_mask(final_mask, sigma=np.random.uniform(60, 180), upscale_shape=(h, w))
    return np.clip(blurred / (blurred.max() + 1e-8) * np.random.uniform(0.5, 1.1), 0, 1)

def generate_colored_leak(mask):
    color_a, color_b = random_color()
    t = np.random.rand()
    gradient_color = color_a + t * (color_b - color_a)
    gradient_color = np.clip(gradient_color * np.random.uniform(1.0, 1.6), 0, 1)
    noise = np.random.normal(scale=np.random.uniform(0.01, 0.03), size=mask.shape + (1,))
    return np.clip(mask[..., None] * gradient_color + noise, 0, 1)

def blend_screen(base, leak):
    return 1 - (1 - base) * (1 - leak)

def apply_light_leak_pil(file_bytes):
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    rgb_img = np.array(pil_img)
    img_float = rgb_img / 255.0
    faces = detect_faces(rgb_img)

    combined_leak = np.zeros_like(img_float)
    for _ in range(np.random.randint(2, 4)):
        mask = generate_light_leak_mask(rgb_img.shape, faces=faces, num_leaks=1)
        colored_leak = generate_colored_leak(mask)
        combined_leak = blend_screen(combined_leak, colored_leak)

    result = blend_screen(img_float, combined_leak)
    result_img = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    output_pil = Image.fromarray(result_img)
    output_bytes = io.BytesIO()
    output_pil.save(output_bytes, format="JPEG")
    return output_bytes.getvalue(), len(faces)

# -------- STREAMLIT UI --------
st.set_page_config(page_title="Film Effects Generator")
st.title("Film Effects Generator")
st.markdown("Upload multiple images to apply artistic light leak effects. Faces will be detected with machine learning and avoided (beta, might not be entirely accurate). Enjoy light leaks, film burn and more random surprises. Image manipulation tool of the Lunchbox toolkit.")

uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Apply Light Effect"):
        with st.spinner("Processing images..."):
            for file in uploaded_files:
                filename = file.name
                orig_bytes = file.getvalue()
                output_bytes, face_count = apply_light_leak_pil(orig_bytes)

                st.markdown(f"#### {filename}")
                col1, col2 = st.columns(2)

                with col1:
                    st.image(orig_bytes, caption="Original", use_container_width=True)

                with col2:
                    st.image(output_bytes, caption=f"Effects Applied (Faces Detected: {face_count})", use_container_width=True)
                    st.download_button(
                        "Download",
                        data=output_bytes,
                        file_name=f"effects_{filename}",
                        mime="image/jpeg",
                        key=filename
                    )

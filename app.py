
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from io import BytesIO
from itertools import combinations
import hashlib

st.set_page_config(
    page_title="GAN Image Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("GAN Image Quality & Diversity Analyzer")
st.write("Analyze generated images without training a GAN.")

st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Similarity threshold (%)", 70, 99, 90
) / 100

uploaded_files = st.file_uploader(
    "Upload generated images",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    images = []
    names = []
    hashes = []

    for file in uploaded_files:
        try:
            raw = file.getvalue()
            image = Image.open(BytesIO(raw))
            image = ImageOps.exif_transpose(image).convert("RGB")

            images.append(image)
            names.append(file.name)
            hashes.append(hashlib.sha256(raw).hexdigest())
        except Exception:
            st.warning(f"Could not read {file.name}")

    n = len(images)

    if n == 0:
        st.error("No valid images found.")
        st.stop()

    unique_count = len(set(hashes))
    duplicate_count = n - unique_count

    st.subheader("1. Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Images", n)
    c2.metric("Unique Files", unique_count)
    c3.metric("Exact Duplicates", duplicate_count)

    st.subheader("2. Image Gallery")

    cols = st.columns(4)
    for i, image in enumerate(images):
        with cols[i % 4]:
            st.image(image, use_container_width=True)
            st.caption(names[i])

    st.subheader("3. Similarity Analysis")

    vectors = []
    for image in images:
        small = image.resize((64, 64))
        arr = np.asarray(small, dtype=np.float32) / 255.0
        vectors.append(arr.flatten())

    scores = []
    similar_pairs = []

    for i, j in combinations(range(n), 2):
        difference = np.mean(np.abs(vectors[i] - vectors[j]))
        score = 1.0 - difference
        scores.append(score)

        if score >= threshold:
            similar_pairs.append((names[i], names[j], score))

    if scores:
        avg = float(np.mean(scores))
        similar_pct = len(similar_pairs) / len(scores) * 100

        a, b = st.columns(2)
        a.metric("Average Pixel Similarity", f"{avg * 100:.2f}%")
        b.metric("Highly Similar Pairs", len(similar_pairs))

        fig, ax = plt.subplots()
        ax.hist(np.array(scores) * 100, bins=15, edgecolor="black")
        ax.set_title("Image Similarity Distribution")
        ax.set_xlabel("Pixel Similarity (%)")
        ax.set_ylabel("Number of Pairs")
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.bar(
            ["Unique Files", "Duplicate Files"],
            [unique_count, duplicate_count]
        )
        ax.set_title("Unique Images vs Duplicates")
        ax.set_ylabel("Number of Images")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("4. Automatic Report")

        repetitive = (
            duplicate_count / n >= 0.10
            or similar_pct >= 25
        )

        if repetitive:
            st.warning("The image collection may be repetitive.")
        else:
            st.success(
                "The collection appears reasonably diverse "
                "under the basic screening rules."
            )

        st.write(f"Images analyzed: {n}")
        st.write(f"Unique file contents: {unique_count}")
        st.write(f"Exact duplicates: {duplicate_count}")
        st.write(f"Average pixel similarity: {avg * 100:.2f}%")
        st.write(f"Pairs above threshold: {similar_pct:.2f}%")

        for name1, name2, score in similar_pairs[:30]:
            st.write(
                f"{name1} and {name2}: {score * 100:.2f}% similarity"
            )

        st.info(
            "This is a basic pixel-based analyzer. It does not "
            "yet measure perceptual image quality or use a "
            "pretrained GAN."
        )

    else:
        st.info("Upload at least two images to compare similarity.")

else:
    st.info("Upload multiple images using the uploader above.")

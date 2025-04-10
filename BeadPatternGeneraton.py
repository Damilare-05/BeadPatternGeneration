
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from PIL import Image

# ------------------------ CONFIG ------------------------
BEAD_COST_DEFAULT = 0.05

# ------------------------ CUSTOM STYLING ------------------------
st.markdown("""
    <style>
        .main {
            background-color: #F9F9F9;
        }
        h1, h2, h3 {
            color: #3F51B5;
        }
        .stButton>button {
            background-color: #FF7043;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------ HELPERS ------------------------
def extract_color_palette(image, k=5):
    pixels = image.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    palette = np.uint8(kmeans.cluster_centers_)
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)
    return [tuple(color) for color in palette[sorted_idx]]

# ------------------------ PATTERN GENERATORS ------------------------
def generate_vertical_symmetric_grid(palette, rows=8, cols=8):
    grid = [[None]*cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols // 2):
            color = palette[(i + j) % len(palette)]
            grid[i][j] = grid[i][cols - j - 1] = color
    return grid

def generate_horizontal_symmetric_grid(palette, rows=8, cols=8):
    grid = [[None]*cols for _ in range(rows)]
    for i in range(rows // 2):
        for j in range(cols):
            color = palette[(i + j) % len(palette)]
            grid[i][j] = grid[rows - i - 1][j] = color
    return grid

def generate_radial_symmetric_grid(palette, rows=8, cols=8):
    grid = [[None]*cols for _ in range(rows)]
    for i in range(rows // 2):
        for j in range(cols // 2):
            color = palette[(i + j) % len(palette)]
            grid[i][j] = grid[i][cols - j - 1] = grid[rows - i - 1][j] = grid[rows - i - 1][cols - j - 1] = color
    return grid

def generate_random_grid(palette, rows=8, cols=8, seed=42):
    rng = np.random.default_rng(seed)
    return [[palette[rng.integers(len(palette))] for _ in range(cols)] for _ in range(rows)]

def generate_linear_pattern(palette, pattern_length=12, mode="symmetric"):
    colors = [tuple(color) for color in palette]
    if mode == "symmetric":
        half = pattern_length // 2
        pattern = [colors[i % len(colors)] for i in range(half)]
        return pattern + pattern[::-1]
    elif mode == "alternating":
        pattern, forward, i = [], True, 0
        while len(pattern) < pattern_length:
            pattern.append(colors[i % len(colors)])
            i += 1 if forward else -1
            if i == len(colors) or i < 0:
                forward = not forward
                i = max(min(i, len(colors)-1), 0)
        return pattern
    elif mode == "gradient":
        return [colors[i % len(colors)] for i in range(pattern_length)]
    elif mode == "zigzag":
        pattern, reverse = [], False
        while len(pattern) < pattern_length:
            seq = colors[::-1] if reverse else colors
            for color in seq:
                if len(pattern) >= pattern_length:
                    break
                pattern.append(color)
            reverse = not reverse
        return pattern
    elif mode == "burst":
        half = pattern_length // 2
        half_pattern = [colors[i % len(colors)] for i in range(half)]
        pattern = half_pattern[::-1] + half_pattern
        if pattern_length % 2 != 0:
            pattern.insert(half, colors[half % len(colors)])
        return pattern
    return []

# ------------------------ VISUALIZATION ------------------------
def plot_grid(grid, title="Bead Pattern"):
    rows, cols = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    for i in range(rows):
        for j in range(cols):
            cell = grid[i][j]
            if cell is None:
                continue
            color = np.array(cell) / 255
            circle = patches.Circle((j + 0.5, rows - i - 0.5), 0.45, color=color)
            ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    return fig


def plot_linear_pattern(pattern, title="1D Pattern"):
    fig, ax = plt.subplots(figsize=(len(pattern), 2))
    for i, color in enumerate(pattern):
        circle = patches.Circle((i + 0.5, 0.5), 0.45, color=np.array(color)/255)
        ax.add_patch(circle)
    ax.set_xlim(0, len(pattern))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    return fig

# ------------------------ COST ESTIMATION ------------------------
def estimate_total_cost(pattern, bead_cost=0.05):
    if isinstance(pattern[0], list):
        total_beads = sum(len(row) for row in pattern)
    else:
        total_beads = len(pattern)
    return round(total_beads * bead_cost, 2)


# ------------------------ STREAMLIT APP ------------------------
st.title("🎨 Bead Pattern Generator")

uploaded_file = st.file_uploader("Upload an image to generate a bead pattern:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    palette = extract_color_palette(image, k=5)

    # Display image and palette
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.subheader("Extracted Color Palette")
    fig_palette = plt.figure(figsize=(5, 1))
    ax = fig_palette.add_subplot(111)
    for i, color in enumerate(palette):
        ax.fill_between([i, i+1], 0, 1, color=np.array(color)/255)
    ax.set_xlim(0, len(palette))
    ax.axis('off')
    st.pyplot(fig_palette)

    # Pattern Settings
    st.subheader("Pattern Settings")
    pattern_type = st.radio("Choose a layout style:", ["1D String / Bracelet Layout", "2D Woven Surface / Grid Layout"])
    st.markdown("""
    - **1D String / Bracelet Layout**: Suitable for necklaces, bracelets, and linear beadwork.
    - **2D Woven Surface / Grid Layout**: Ideal for wall hangings, mats, and surface designs.
    """)
    if pattern_type == "2D Woven Surface / Grid Layout":
        symmetry_type = st.selectbox("2D Symmetry Type", ["Vertical", "Horizontal", "Radial", "Random"])
    else:
        symmetry_type = None
    if pattern_type == "1D String / Bracelet Layout":
        pattern_style = st.selectbox("1D Pattern Style", ["symmetric", "alternating", "gradient", "zigzag", "burst"])
    else:
        pattern_style = None
    rows = st.slider("Rows (for 2D)", 4, 20, 8)
    cols = st.slider("Columns (for 2D / Length for 1D)", 4, 20, 8)
    bead_cost = st.number_input("Bead Cost ($)", min_value=0.01, max_value=1.00, value=BEAD_COST_DEFAULT)

    # Generate pattern
    if pattern_type == "2D Woven Surface / Grid Layout":
        if symmetry_type == "Vertical":
            grid = generate_vertical_symmetric_grid(palette, rows, cols)
        elif symmetry_type == "Horizontal":
            grid = generate_horizontal_symmetric_grid(palette, rows, cols)
        elif symmetry_type == "Radial":
            grid = generate_radial_symmetric_grid(palette, rows, cols)
        else:
            grid = generate_random_grid(palette, rows, cols)
        pattern = grid
        fig = plot_grid(grid, title=f"{symmetry_type} Symmetry Pattern")
    else:
        pattern = generate_linear_pattern(palette, pattern_length=cols, mode=pattern_style)
        fig = plot_linear_pattern(pattern, title=f"1D {pattern_style.capitalize()} Pattern")

    # Show pattern and cost
    st.subheader("Generated Bead Pattern")
    st.pyplot(fig)
    total_cost = estimate_total_cost(pattern, bead_cost)
    st.subheader("Estimated Cost")
    st.write(f"💰 This pattern would cost approximately **${total_cost}**")
else:
    st.warning("📸 Please upload an image to get started.")

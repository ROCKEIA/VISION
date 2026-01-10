import os
import cv2
import numpy as np

from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC


# ------------------------------
# Parámetros globales
# ------------------------------
IMAGE_SIZE = (64, 64)
N_COMPONENTS_PCA = 120


# ------------------------------
# 1. Cargar dataset
# ------------------------------
def load_dataset(dataset_dir):
    X = []
    y = []
    classes = []

    for label, class_name in enumerate(sorted(os.listdir(dataset_dir))):
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        classes.append(class_name)

        for file in os.listdir(class_dir):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(class_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, IMAGE_SIZE)

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y), classes



# ------------------------------
# 2. Extraer HOG
# ------------------------------
def extract_hog_features(images):
    features = []

    for img in images:
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        features.append(feat)

    return np.array(features)



# ------------------------------
# 3. PCA
# ------------------------------
def apply_pca(features, n_components=N_COMPONENTS_PCA):
    pca = PCA(n_components=n_components, svd_solver="randomized")
    reduced = pca.fit_transform(features)
    return reduced, pca



# ------------------------------
# 4. Entrenar clasificador
# ------------------------------
def train_classifier(X_train, y_train):
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    return clf


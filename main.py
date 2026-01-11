
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from funcion import (
    load_dataset,
    extract_hog_features,
    extract_hog_features_opencv,
    apply_pca,
    train_classifier
)

# Ruta del dataset
DATASET_PATH = r"D:\IgRob 25i26\Vision\Final\archive (1)\animals\train"

def main():

    print("Cargando dataset...")
    X, y, labels = load_dataset(DATASET_PATH)

    print("Clases detectadas:", labels)
    print("Número total de imágenes:", len(X))

    if len(np.unique(y)) < 2:
        print("ERROR: se necesita al menos 2 clases para entrenar")
        return

    # ---------------------------
    # HOG
    # ---------------------------
    print("\nExtrayendo HOG...")
    hog_features = extract_hog_features_opencv(X)  # Usar versión OpenCV
    # ---------------------------
    # PCA
    # ---------------------------
    print("\nAplicando PCA (reducción de dimensionalidad)...")
    X_pca, pca = apply_pca(hog_features)

    # ---------------------------
    # Train/Test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------
    # Clasificador
    # ---------------------------
    print("\nEntrenando LinearSVC (rápido)...")
    clf = train_classifier(X_train, y_train)

    # ---------------------------
    # Evaluación
    # ---------------------------
    print("\nEvaluando modelo...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\nAccuracy: {acc:.2f}%")

    cm = confusion_matrix(y_test, y_pred)

    print("\nMatriz de confusión:")
    print(cm)

    # ---------------------------
    # Graficar matriz
    # ---------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# Import modul-modul yang dibutuhkan
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib  # Buat simpan dan load model

# Fungsi buat load data
def prepare_data(data_dir, img_size=(128, 128), test_split=0.2):
    """Load gambar, resize, dan bagi jadi data latih sama uji."""
    data, labels = [], []
    class_names = os.listdir(data_dir)

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    data.append(img.flatten())
                    labels.append(idx)

    data = np.array(data) / 255.0  # Normalisasi pixel ke [0, 1]
    labels = np.array(labels)

    return data, labels, class_names

# Klasifikasi dengan algoritma
def Knn(X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

def Svm(X_train, y_train, X_test):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return svm.predict(X_test)

def Random_forest(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    return rf.predict(X_test)

def save_model(model, model_name):
    """Simpan model yang udah dilatih."""
    joblib.dump(model, model_name)
    print(f"Model disimpan sebagai '{model_name}'")

def predict_image(model, img_path, img_size=(128, 128)):
    """Prediksi kelas dari satu gambar."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img.flatten() / 255.0  # Normalisasi
    img = img.reshape(1, -1)  # Ubah bentuk untuk prediksi
    return model.predict(img)

def display_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # Load data
    X, y, class_names = prepare_data('Data')  # Sesuaikan path sesuai kebutuhan

    # Bagi dataset jadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Klasifikasi dengan KNN
    y_pred_knn = Knn(X_train, y_train, X_test)
    print("Laporan Klasifikasi KNN:")
    print(classification_report(y_test, y_pred_knn))
    display_confusion_matrix(y_test, y_pred_knn, "Matriks Kebingungan KNN")
    save_model(KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train), 'model/knn_model.pkl')

    # Klasifikasi dengan SVM
    y_pred_svm = Svm(X_train, y_train, X_test)
    print("Laporan Klasifikasi SVM:")
    print(classification_report(y_test, y_pred_svm))
    display_confusion_matrix(y_test, y_pred_svm, "Matriks Kebingungan SVM")
    save_model(SVC(kernel='linear').fit(X_train, y_train), 'model/svm_model.pkl')

    # Klasifikasi dengan Random Forest
    y_pred_rf = Random_forest(X_train, y_train, X_test)
    print("Laporan Klasifikasi Random Forest:")
    print(classification_report(y_test, y_pred_rf))
    display_confusion_matrix(y_test, y_pred_rf, "Matriks Kebingungan Random Forest")
    save_model(RandomForestClassifier(n_estimators=100).fit(X_train, y_train), 'model/rf_model.pkl')

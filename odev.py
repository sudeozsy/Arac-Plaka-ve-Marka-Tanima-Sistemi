import cv2
import tkinter as tk  # Tkinter - Arayüz oluşturmak için
from tkinter import filedialog, messagebox  # Dosya seçimi ve hata mesajı göstermek için
from PIL import Image, ImageTk  # Görüntüyü Tkinter'da göstermek için
from ultralytics import YOLO  # YOLO modeli kullanımı için
import pytesseract  # OCR işlemi için
import os  # Dosya yolu işlemleri
import numpy as np
import time
import re  # Regex ile plaka tanıma işlemi

# Tesseract OCR yolu
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # OCR motorunun yolu

# Model yolları
MARKA_MODEL_PATH = "best.pt"
PLAKA_MODEL_PATH = "car_dataset/runs/detect/train2/weights/best.pt"

marka_model = YOLO(MARKA_MODEL_PATH)  # Marka modeli yükleniyor
plaka_model = YOLO(PLAKA_MODEL_PATH)  # Plaka modeli yükleniyor

def preprocess_plate(roi):  # Plaka görüntüsünü OCR için ön işleme fonksiyonu
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Griye çevir
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Görseli büyüt
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Eşikleme uygula
    return thresh

def extract_plate_text(thresh_img):  # Tesseract ile plaka yazısını okuma
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # OCR ayarları
    text = pytesseract.image_to_string(thresh_img, config=config)  # Metin çıkarımı
    clean = text.strip().replace(" ", "")  # Boşlukları kaldır
    candidates = re.findall(r'[0-9]{2}[A-Z]{1,3}[0-9]{1,4}', clean)  # Türk plakası formatına uyan metni bul
    if candidates:
        return candidates[0]  # Eğer format uyuyorsa ilk eşleşmeyi döndür
    else:
        corrections = str.maketrans("OIZSB", "01258")  # Hatalı karakter düzeltmeleri
        corrected = clean.translate(corrections)
        return corrected[:10]  # Maksimum 10 karakterlik düzeltme

root = tk.Tk()  # Ana pencere
root.title("Araç Marka ve Plaka Tespit Sistemi")  # Başlık
root.geometry("1000x600")  # Pencere boyutu
root.configure(bg="#f0f0f0")  # Arkaplan rengi

frame_left = tk.Frame(root, bg="#f0f0f0")  # Sol panel
frame_left.pack(side="left", fill="y", padx=20, pady=20)

frame_right = tk.Frame(root, bg="#dcdcdc")  # Sağ panel (görselin gösterileceği)
frame_right.pack(side="right", fill="both", expand=True, padx=20, pady=20)

history_label = tk.Label(frame_left, text="Tespit Geçmişi", font=("Arial", 12, "bold"), bg="#f0f0f0")  # Geçmiş başlığı
history_label.pack(anchor="w", pady=(0, 10))

history_text = tk.Text(frame_left, width=40, height=30, font=("Arial", 10))  # Geçmiş metin alanı
history_text.pack()

canvas = tk.Canvas(frame_right, bg="#dcdcdc")  # Görüntülerin gösterileceği tuval
canvas.pack(fill="both", expand=True)

current_img_tk = None  # Global değişken: gösterilecek görüntü

def process_image(image_path):  # Fotoğrafı işleyen fonksiyon
    global current_img_tk
    image = cv2.imread(image_path)  # Fotoğrafı yükle
    result = process_frame(image)  # İşlenmiş görseli al
    bgr_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # Renk dönüşümü
    im = Image.fromarray(bgr_image)  # PIL Image objesi oluştur

    root.update_idletasks()  # Canvas boyutunu güncelle
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    im_ratio = im.width / im.height  # Görsel oranı
    canvas_ratio = canvas_width / canvas_height  # Canvas oranı

    if im_ratio > canvas_ratio:
        new_width = canvas_width
        new_height = int(canvas_width / im_ratio)
    else:
        new_height = canvas_height
        new_width = int(canvas_height * im_ratio)

    im = im.resize((new_width, new_height), Image.LANCZOS)  # Görseli yeniden boyutlandır

    current_img_tk = ImageTk.PhotoImage(im)  # Tkinter uyumlu görsel objesi
    canvas.delete("all")  # Önceki görseli temizle
    canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2, anchor="nw", image=current_img_tk)  # Görseli ortala ve yerleştir

def process_frame(frame):  # Tespit işlemini gerçekleştiren fonksiyon
    marka_results = marka_model.predict(source=frame, conf=0.4, verbose=False)  # Marka tahmini
    plaka_results = plaka_model.predict(source=frame, conf=0.4, verbose=False)  # Plaka tahmini

    brand_text = ""
    plate_text = ""

    if marka_results and marka_results[0].boxes:
        best_box = max(marka_results[0].boxes, key=lambda b: b.conf[0])  # En güvenilir marka kutusu
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cls = int(best_box.cls[0])
        label = marka_model.names[cls]  # Marka adı
        brand_text = label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Kutu çiz
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)  # Marka yazısı

    if plaka_results:
        for box in plaka_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]  # Plaka bölgesi
            preprocessed = preprocess_plate(roi)  # Ön işleme
            text = extract_plate_text(preprocessed)  # OCR ile metin okuma
            plate_text = text if text else "Plaka?"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Sonuçları geçmişe yazdır
    history_text.insert(tk.END, f"Plaka: {plate_text if plate_text else 'Belirlenemedi'}\n")
    history_text.insert(tk.END, f"Marka: {brand_text if brand_text else 'Belirlenemedi'}\n\n")
    history_text.see(tk.END)

    return frame

def select_file():  # Dosya seçim penceresi
    initial_dir = os.getcwd()
    file_path = filedialog.askopenfilename(initialdir=initial_dir, title="Fotoğraf Seç",
                                           filetypes=[("Görsel Dosyaları", "*.jpg *.jpeg *.png")])
    if file_path:
        process_image(file_path)
    else:
        messagebox.showerror("Hata", "Desteklenmeyen dosya formatı")

select_btn = tk.Button(frame_left, text="Fotoğraf Seç", command=select_file, font=("Arial", 14), bg="#007acc", fg="white", padx=20, pady=10)  # Seçim butonu
select_btn.pack(anchor="w", pady=(0, 10))

root.mainloop()  # Arayüzü başlat

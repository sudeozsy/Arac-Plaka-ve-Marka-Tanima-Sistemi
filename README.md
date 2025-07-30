# Araç Marka ve Plaka Tespit Sistemi

Bu proje, **Python** ve **görüntü işleme teknikleri** kullanılarak geliştirilen bir araç marka ve plaka tespit sistemidir. Sistem, görsellerdeki araçların **plakasını** ve **markasını** otomatik olarak tespit edebilmektedir.

---

## Projenin Amacı

Görsel girdiler üzerinden araçlara ait:

- Plaka bilgilerini
- Marka bilgilerini  
otomatik olarak tespit eden bir sistem geliştirmek.

---

## Kullanılan Yöntemler

-  Görüntü İşleme (**OpenCV**)
-  Derin Öğrenme (YOLOv8)
  - Plaka tespiti için: Özel olarak eğitilmiş YOLO modeli
  - Marka tespiti için: Hazır YOLO modeli
-  OCR (Optik Karakter Tanıma) – **Tesseract**
-  Regex ile plaka format kontrolü ve karakter düzeltme

---

## Kullanılan Teknolojiler

-  Python  
-  Ultralytics YOLOv8  
-  Tesseract OCR  
-  OpenCV  
-  NumPy  
-  Matplotlib

---


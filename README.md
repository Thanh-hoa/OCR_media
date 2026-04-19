# Hệ thống trích xuất thông tin bệnh án tiếng Việt (YOLO + Tesseract OCR)

Dự án dùng **YOLO** (Ultralytics) để phân vùng các khối trên ảnh bệnh án, sau đó **Tesseract OCR** đọc chữ từng vùng và trả về JSON dễ tích hợp với **Spring Boot** hoặc client khác.

## Yêu cầu

- Python 3.10+ (khuyến nghị)
- Model: `models/weights/best.pt`
- Tesseract OCR: cài đặt và chỉnh đường dẫn trong `src/api/main.py` (mặc định `D:\HK2_4\DoAn\OCR\tesseract.exe`)

## Cài đặt

```bash
cd Project_OCR
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Kiểm tra: `GET /health`

## Endpoint chính

| Method | Đường dẫn | Mô tả |
|--------|-----------|--------|
| `POST` | `/extract` | Upload ảnh, trả về `filename` + `result` (gom theo nhãn) |
| `POST` | `/v1/ocr/upload` | Alias tương thích cũ, cùng hành vi với `/extract` |

### Postman / client

- Body: **form-data**
- Key: `file` (kiểu **File**)
- Chọn file ảnh (`.jpg`, `.jpeg`, `.png`, …)

### Ví dụ phản hồi (rút gọn)

```json
{
  "filename": "mau.png",
  "result": {
    "hospital_header": { "count": 1, "confidence_avg": 0.91, "text": "...", "boxes": [[...]] },
    "patient_info": { ... },
    "diagnosis_block": { ... },
    "test_table": { ... },
    "footer_signature": { ... }
  },
  "debug_output": {
    "annotated": "outputs/yolo_regions/20260419_143022_mau_annotated.jpg",
    "crops_dir": "outputs/yolo_regions/crops/20260419_143022_mau"
  }
}
```

Trường `debug_output` chỉ có khi bật lưu ảnh debug (mặc định bật). Xem mục dưới.

## Ảnh debug YOLO (phân vùng + crop)

Sau mỗi request thành công, server có thể lưu:

1. **`outputs/yolo_regions/<timestamp>_<tên_file>_annotated.jpg`** — ảnh gốc có vẽ khung bbox + nhãn + độ tin cậy.
2. **`outputs/yolo_regions/crops/<timestamp>_<tên_file>/`** — từng vùng đã cắt (`00_label_conf.jpg`, …).

Dùng để kiểm tra model có bao đúng vùng hay không trước khi tin vào OCR.

### Tắt ghi file (ví dụ production)

```bash
set YOLO_DEBUG_SAVE=0
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Cấu trúc thư mục chính

```
Project_OCR/
├── models/
│   └── weights/
│       └── best.pt
├── outputs/
│   └── yolo_regions/     # ảnh debug (annotated + crops), tạo khi gọi API
├── src/
│   ├── api/
│   │   └── main.py
│   ├── core/
│   │   ├── detector.py   # MedicalDetector (YOLO)
│   │   └── reader.py     # MedicalReader (Tesseract)
│   └── utils/
│       ├── image_processing.py
│       └── yolo_debug.py
├── requirements.txt
└── README.md
```

## Gợi ý tích hợp Spring Boot

Gửi `multipart/form-data` với part tên `file` tới `http://<host>:8000/extract`, nhận JSON và map vào DTO tương ứng với object `result`.

## Ghi chú

- Chất lượng OCR phụ thuộc ảnh (nét, nghiêng, ánh sáng) và cấu hình Tesseract.
- Có thể tinh chỉnh tiền xử lý trong `src/utils/image_processing.py` và `MedicalReader`.

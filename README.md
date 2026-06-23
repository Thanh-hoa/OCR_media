# Hệ thống trích xuất thông tin bệnh án tiếng Việt

Dự án triển khai AI Service bằng **FastAPI** để nhận ảnh bệnh án, phát hiện vùng thông tin bằng **YOLO11n-OBB**, đọc chữ bằng pipeline **OCR lai PaddleOCR/Tesseract**, hiệu chỉnh kết quả bằng **Gemini** nếu có API key, rồi chuẩn hóa dữ liệu thành JSON để tích hợp với Backend Spring Boot hoặc client khác.

## Pipeline chính

1. Nhận ảnh upload qua API `/v1/ocr/upload`.
2. Giải mã ảnh, deskew ảnh đầu vào bằng OpenCV.
3. Dùng `models/weights/best.pt` để phát hiện 5 vùng:
   - `hospital_header`
   - `patient_info`
   - `diagnosis_block`
   - `test_table`
   - `footer_signature`
4. Crop từng vùng. Nếu YOLO trả về OBB thì crop bằng perspective transform theo 4 góc thật của vùng.
5. Tiền xử lý crop bằng OpenCV.
6. OCR bằng PaddleOCR tiếng Việt; nếu kết quả thấp hoặc rỗng thì fallback sang Tesseract `vie`.
7. Nếu có `GEMINI_API_KEY`, gọi Gemini batch một lần để hiệu chỉnh OCR theo ảnh crop.
8. Parse text thành `parsedData.extractedData` và `parsedData.labData`.

## Yêu cầu

- Python 3.10+.
- Model YOLO: `models/weights/best.pt`.
- Tesseract OCR 5.x và gói ngôn ngữ tiếng Việt `vie`.
- Đường dẫn Tesseract cấu hình bằng biến môi trường `TESSERACT_CMD`.

- Tùy chọn: cấu hình `GEMINI_API_KEY` trong file `.env` để bật Gemini validation.

## Cài đặt

```bash
cd Project_OCR
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Hoặc dùng Makefile:

```bash
make run
```

- Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Kiểm tra trạng thái: `GET /health`

## Endpoint

| Method | Đường dẫn | Mô tả |
|--------|-----------|-------|
| `GET` | `/health` | Kiểm tra model path, Tesseract path và trạng thái Gemini |
| `POST` | `/v1/ocr/upload` | Upload ảnh bệnh án và nhận kết quả OCR |

Định dạng ảnh hỗ trợ: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`.

### Postman / client

- Body: **form-data**
- Key: `file`
- Type: **File**
- URL: `http://127.0.0.1:8000/v1/ocr/upload`

### Ví dụ phản hồi rút gọn

```json
{
  "filename": "mau.png",
  "result": {
    "hospital_header": {
      "count": 1,
      "confidence_avg": 0.91,
      "text": "BỆNH VIỆN ...",
      "boxes": [[10, 20, 600, 120]]
    },
    "patient_info": {
      "count": 1,
      "confidence_avg": 0.88,
      "text": "Họ tên: ...",
      "boxes": [[...]]
    },
    "diagnosis_block": { "count": 1, "confidence_avg": 0.86, "text": "...", "boxes": [[...]] },
    "test_table": { "count": 1, "confidence_avg": 0.9, "text": "WBC | 7.2 | K/uL | 4.0-10.0", "boxes": [[...]] },
    "footer_signature": { "count": 1, "confidence_avg": 0.8, "text": "...", "boxes": [[...]] }
  },
  "parsedData": {
    "extractedData": {
      "hospital_name": "...",
      "patient_name": "...",
      "patient_dob": "1990-05-15",
      "patient_gender": "Nam",
      "diagnosis": "..."
    },
    "labData": [
      {
        "testName": "WBC",
        "testValue": "7.2",
        "unit": "K/uL",
        "referenceRange": "4.0-10.0",
        "isAbnormal": false
      }
    ]
  }
}
```

## Cấu trúc thư mục chính

```text
Project_OCR/
├── models/
│   ├── configs/
│   │   └── args.yaml          # cấu hình train YOLO11n-OBB
│   └── weights/
│       └── best.pt            # trọng số model đang dùng
├── outputs/                   # dữ liệu output/debug nếu có
├── src/
│   ├── api/
│   │   └── main.py            # FastAPI app, endpoint upload
│   ├── core/
│   │   ├── detector.py        # MedicalDetector, YOLO11n-OBB + crop OBB
│   │   ├── reader_hybrid.py   # PaddleOCR + Tesseract fallback + Gemini
│   │   ├── gemini_validator.py
│   │   └── data_parser.py     # parse OCR text thành extractedData/labData
│   └── utils/
│       └── image_processing.py
├── Makefile
├── requirements.txt
└── README.md
```

## Gợi ý tích hợp Spring Boot

Gửi `multipart/form-data` với part tên `file` tới:

```text
http://<ai-service-host>:8000/v1/ocr/upload
```

Backend nên map:

- `result`: text OCR theo từng vùng YOLO, kèm số lượng vùng, độ tin cậy trung bình và bbox.
- `parsedData.extractedData`: thông tin bệnh viện, bệnh nhân, chẩn đoán, chữ ký.
- `parsedData.labData`: danh sách chỉ số xét nghiệm đã parse.

## Ghi chú

- Chất lượng OCR phụ thuộc độ nét, ánh sáng, độ nghiêng và bố cục biểu mẫu.
- Nếu không có `GEMINI_API_KEY`, hệ thống vẫn chạy và trả về kết quả OCR thô đã qua xử lý.
- Nếu đổi vị trí cài Tesseract, cấu hình biến môi trường `TESSERACT_CMD`.

## Deploy Docker

Sao chép file env mẫu và cấu hình secret:

```bash
cp .env.example .env
```

Build và chạy container:

```bash
docker build -t project-ocr-api .
docker run --rm -p 8000:8000 --env-file .env project-ocr-api
```

Trong production, nên đặt các biến môi trường sau:

```env
TESSERACT_CMD=/usr/bin/tesseract
YOLO_MODEL_PATH=/app/models/weights/best.pt
GEMINI_API_KEY=
MAX_UPLOAD_MB=10
```

Lệnh chạy production không dùng `--reload`:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

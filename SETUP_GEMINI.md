# Setup Gemini Vision API cho OCR Validation

## Bước 1: Lấy Gemini API Key

1. Truy cập: https://aistudio.google.com/app/apikeys
2. Click **"Create API key"** (chọn **"Create API key in existing project"** nếu là lần đầu)
3. Copy API key

## Bước 2: Set Environment Variable

### Cách 1: Dùng `.env` file (Recommended)

```bash
# Copy .env.example → .env
cp .env.example .env

# Mở .env, thay thế:
# GEMINI_API_KEY=your_gemini_api_key_here
# →
# GEMINI_API_KEY=abc123xyz...
```

### Cách 2: Set OS environment variable (Windows PowerShell)

```powershell
$env:GEMINI_API_KEY="abc123xyz..."
```

### Cách 3: Set OS environment variable (Command Prompt)

```cmd
set GEMINI_API_KEY=abc123xyz...
```

## Bước 3: Test Setup

```bash
python -c "
import os
key = os.getenv('GEMINI_API_KEY')
if key:
    print(f'OK: Gemini API key loaded ({len(key)} chars)')
else:
    print('ERROR: GEMINI_API_KEY not set')
"
```

## Free Tier Limits

- **15 requests/phút**
- **1,500 requests/ngày**
- Đủ cho testing & development
- Sau production, có thể upgrade (cost rất rẻ, ~$0.075 per 1M input tokens)

## Flow Sau Setup

```
YOLO detect crop
    ↓
PaddleOCR đọc
    ↓
✨ Gemini Vision API validate & fix ✨
    ↓
Final text (chính xác 95%+)
```

### Ví dụ: test_table

**Before (PaddleOCR):**
```
TENXETNGHEM kếTqui T...
```

**After (Gemini validation):**
```
TEN XET NGHEM kết quả T...
```

---

**Note:** Nếu không set GEMINI_API_KEY, validation sẽ bị disable. System vẫn chạy normal (fallback Tesseract).


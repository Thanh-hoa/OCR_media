# Prompt tạo ground truth OCR

Dùng prompt này khi đưa ảnh phiếu xét nghiệm cho AI đọc. Mục tiêu là tạo ground truth nháp cho file `evaluation/ground_truth.json`, sau đó người dùng kiểm tra và sửa lại các chỗ AI đọc sai.

```text
Bạn là trợ lý tạo ground truth cho hệ thống OCR bệnh án tiếng Việt.

Hãy đọc nội dung trong ảnh phiếu xét nghiệm và trả về JSON đúng theo schema bên dưới.

Yêu cầu bắt buộc:
- Chỉ trả về JSON, không giải thích.
- Không bọc JSON trong markdown.
- Giữ nguyên tiếng Việt có dấu nếu đọc được.
- Nếu không chắc một ký tự, một từ, hoặc một giá trị, hãy ghi chuỗi rỗng "" thay vì đoán bừa.
- Không tự thêm thông tin không có trong ảnh.
- Đầu vào là ảnh nguyên phiếu xét nghiệm, không phải ảnh crop.
- Tách nội dung theo 5 vùng:
  1. hospital_header: tên bệnh viện, khoa, địa chỉ, điện thoại, PID, số bệnh phẩm, mã bệnh án.
  2. patient_info: họ tên, ngày sinh, giới tính, địa chỉ, BHYT.
  3. diagnosis_block: chẩn đoán, khoa/phòng, người lấy mẫu, người nhận mẫu, thời gian mẫu, loại bệnh phẩm, tình trạng mẫu, bác sĩ chỉ định.
  4. test_table: bảng kết quả xét nghiệm.
  5. footer_signature: người thực hiện, người duyệt, chữ ký, chức danh cuối phiếu.
- Trường regions là text đầy đủ theo từng vùng để tính CER/WER.
- Trong regions, giữ xuống dòng tự nhiên theo nội dung trên phiếu. Không gom toàn bộ một vùng thành một dòng nếu trên phiếu có nhiều dòng.
- Trường fields là dữ liệu đã chuẩn hóa để tính Field Accuracy.
- patient_dob phải chuẩn hóa dạng yyyy-MM-dd nếu đọc được.
- Nếu không đọc được ngày sinh thì để patient_dob là "".
- diagnosis trong fields chỉ lấy nội dung chẩn đoán, không lấy nhãn "Chẩn đoán:".

Quy tắc riêng cho test_table:
- Đọc kỹ bảng xét nghiệm.
- Mỗi dòng xét nghiệm phải có dạng:
  Tên xét nghiệm | Kết quả | Giá trị tham chiếu | Đơn vị
- Giữ đúng thứ tự cột như trên vì đây là thứ tự cột thường thấy trên ảnh phiếu xét nghiệm.
- Không đổi đơn vị lên trước giá trị tham chiếu.
- Không bỏ dòng nếu vẫn đọc được tên xét nghiệm và kết quả.
- Nếu một ô không đọc rõ, để trống ô đó nhưng vẫn giữ dấu phân cách |.

Schema cần trả về cho một ảnh:

{
  "image": "sample_001.jpg",
  "regions": {
    "hospital_header": "",
    "patient_info": "",
    "diagnosis_block": "",
    "test_table": "",
    "footer_signature": ""
  },
  "fields": {
    "hospital_name": "",
    "patient_name": "",
    "patient_dob": "",
    "patient_gender": "",
    "diagnosis": ""
  }
}
```

Khi làm nhiều ảnh, đổi giá trị `image` thành đúng tên ảnh, ví dụ `sample_002.jpg`, `sample_003.jpg`.

Sau khi AI trả JSON, cần kiểm tra lại bằng mắt. Ground truth sai thì CER/WER và Field Accuracy sẽ sai theo.

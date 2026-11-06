# Đối chiếu nguồn cell 8 và đề xuất sửa nhãn

## Nguồn thực sự đã đọc

Agent đọc output cell training-v2-repair (cell 8) đã lưu trong notebooks/colab-training-preparation.ipynb, execution_count=2. Log có V2_REPAIR_SAVED và V2_REPAIR_PREVIEW_READY, không có traceback; thư mục theo log là dataset-v2-preparation/20261104T173852Z-f54a30/ bên trong run hiện tại.

Đây là ảnh và nhãn từ prepared-candidate.zip của Industrial Watch v2, SHA-256 28ee24280bce8184acd1cdf9a97e850788e908109bfa662efecb07a8a7b179eb, đã dùng train. “Ảnh gốc” ở cell 8 nghĩa là byte ảnh lấy nguyên từ candidate đó, không phải ảnh raw nguyên bản từ camera trước mọi preprocessing. Cả bảy ảnh được xuất có kích thước 640×640.

Agent trích bảy ảnh SOURCE nhúng trong output và bảy chuỗi original_yolo. SHA-256 của từng ảnh và chuỗi nhãn đều bằng giá trị image_sha256/label_sha256 được xuất cùng record. Đây là kiểm tra tính nhất quán của output với record đã xuất, không phải đọc Drive độc lập và không chứng minh nhãn đúng về ngữ nghĩa.

Không ảnh nào trong bảy mẫu có EXIF orientation. Ảnh Mountain Dew/chai móp nằm ngang ngay trong pixel nguồn, không phải bị code tự xoay khi đọc.

## Đọc file đúng không có nghĩa là nhận diện đúng

Cell 8 không tải YOLO hoặc gọi model.predict. CURRENT_LABELS là hình vẽ các hàng YOLO có sẵn trong dataset; các file đó có thể hợp lệ về SHA/cấu trúc nhưng vẫn bỏ sót hoặc khoanh sai vật thể.

Ví dụ train-005: hàng bottle nguồn 0 0.53984375 0.4796875 0.36875 0.896875 trên ảnh 640×640 cho box pixel [227,5; 20; 463,5; 594]. Vị trí đường box xanh trong ảnh output phù hợp với tọa độ đó sau raster hóa/JPEG. Chính box nguồn thừa nền trắng bên phải và cắt sát đáy chai; parser/vẽ đúng công thức không sửa được chất lượng box nguồn.

## Kết quả xem ảnh nguồn

- train-005: thiếu label cho phần nhãn phủ thân; box bottle nguồn chưa ôm sát chai. Đã chuẩn bị box đề xuất mới và giữ nguyên box cap nguồn.
- train-016: xác nhận nắp chai trái nhìn thấy nhưng chưa có annotation. Chai giữa thấy ren/miệng mở, không được thêm cap cho chai đó. Đề xuất chỉ thêm cap trái, giữ các box còn lại.
- train-018: xác nhận nhãn Mountain Dew nhìn thấy nhưng file không có hàng class 2. Đã vẽ box label đề xuất trong hệ tọa độ ảnh nằm ngang.
- train-027: xác nhận có nắp trắng ở đầu trái, file chỉ có bottle. Đã vẽ box cap đề xuất, tránh lấy cả vòng xanh của cổ chai.
- train-044 và train-046: là ảnh quảng cáo/collage, thiếu annotation label; train-046 còn thiếu nhiều cap. Đề xuất loại riêng hai ảnh này khỏi candidate tập trung vào camera dây chuyền, giữ nguyên nguồn. Đây không phải hạn mức loại ảnh.
- train-020: ảnh cận cap/cổ và một phần thân. Box cap hiện tại còn bao cả phần cổ/vòng xanh; quy tắc bottle khi chỉ thấy một phần vẫn chưa thống nhất. Giữ unresolved, chưa sửa hoặc tự loại.

## Sản phẩm đã chuẩn bị và giới hạn

tests/fixtures/notebook-reviews/candidate-v2-corrections-2026-11-05-v1.json chứa bốn replacement_yolo đề xuất, hai đề xuất exclude và một trường hợp unresolved. Mỗi action ràng buộc đường dẫn, SHA ảnh và SHA nhãn hiện tại, không dựa vào tên ID đơn lẻ. Các tọa độ mới được kiểm tra nằm trong ảnh; ảnh so sánh nguồn/nhãn cũ/nhãn đề xuất đã được xem lại.

Đây là sửa thủ công dựa trên ảnh candidate đã xuất, không phải nhãn sinh từ model và chưa phải ground truth đầy đủ của toàn bộ dataset. Patch còn PROPOSED_CORRECTIONS, training_ready=False, applied_to_drive=False. Chưa áp dụng lên ZIP/Drive, chưa tạo candidate v2 và chưa train lại. Cần rà các nhóm tương tự và dữ liệu dây chuyền trước khi công bố sẵn sàng train.

## Ảnh đối chiếu

Ảnh nguồn và overlay đã giữ cùng báo cáo trong [thư mục bằng chứng](original-label-review-evidence/). Các file `*-current.jpg` là ảnh candidate; file cùng ID không có hậu tố `-current` là overlay xem box hiện có. Chúng là bản minh họa được xuất cho lần review này, không phải ảnh raw gốc từ camera.

# Review detector trên video kiểm tra chai PET của MV Center

Ngày review: 04/11/2026. Đây là quan sát đầu ra thật đã lưu trong notebook, không phải phép đo accuracy hoặc kiểm chứng PASS/FAIL.

## Bằng chứng và phạm vi

- Notebook: notebooks/colab-training-preparation.ipynb, cell training-video, execution_count=2; VIDEO_DEMO_SAVED và VIDEO_PREVIEW_READY, không có traceback.
- Model: YOLO11n, SHA-256 4839c128b8b13b5785b55db2d0a15d061e424b3abfbca3653f48a1b9a5533030; 0=bottle, 1=cap, 2=label.
- [Nguồn MV Center](https://mv-center.com/en/automatic-inspection-of-pet-bottles/); SHA-256 nguồn 017fa285bc3325fbd82a7b2d30a335e731154af0601fb2a0175857192321d172.
- CPU Colab, imgsz=640, conf=0.25, stride=4, bắt đầu giây 10. Đã đọc 2258 frame nguồn và nhận diện 415 frame, đầu ra 15 fps dài 27,67 giây.
- Run theo output: MyDrive/BottleQC/training-runs/bottleqc-yolo11n-640-v1/video-tests/factory-line-20261104T034556Z-b5c9c1/.
- Đã trích video nhúng và xác minh SHA-256 khớp báo cáo: 1ab4b30e955f89ee3e6c0af5ffbb5f6edb11089240396ce8b0ddb9022aa22618, 1.996.917 byte. Chưa đọc Drive độc lập.
- Đã xem 27 frame, mỗi giây từ giây 0 đến 26 của đầu ra; xem riêng ảnh lớn ở giây 1, 6, 17, 20, 22. Đây là lấy mẫu định tính, chưa có ground truth và không kiểm tra từng frame.
- Thời gian nguồn = thời gian đầu ra + 10 giây tại các mốc mẫu trên. CSV lưu thời gian nguồn; đầu phát video bắt đầu từ 0.

## Lỗi quan sát được

| Đầu ra | Nguồn | Quan sát |
|---|---|---|
| 1 giây | 11 giây | Nhận ra một số chai vàng, nhưng bỏ nhiều chai/nắp khác trong ảnh. Chưa thấy nhãn rõ ở góc này để kết luận thiếu nhãn hay lỗi nhận nhãn. |
| 3–5 giây | 13–15 giây | Các frame mẫu có chai đỏ/nắp đỏ trên băng chuyền nhưng không có box nhận diện. Có nhòe, ánh sáng đỏ và thân bị thanh chắn che. |
| 6 giây | 16 giây | Một box bottle 0,35 chỉ ôm cổ chai; một vùng gồm thanh vàng/nền bị gọi bottle 0,41. |
| 11 giây | 21 giây | Nhận ra một chai đỏ phía xa và nắp; cho thấy model vẫn có một số dự đoán phù hợp, không phải hoàn toàn mất khả năng nhận chai. |
| 17 giây | 27 giây | Box bottle trùm nền/máy; chi tiết máy phía trên bị gọi cap 0,76. |
| 20 giây | 30 giây | Cụm thanh kim loại ở tiền cảnh bị gọi bottle 0,48 trong khi nhiều chai thật phía sau không có box. |
| 22 giây | 32 giây | Outro/logo không có chai vẫn có bottle 0,91. Đây là cảnh đồ họa ngoài phạm vi kiểm hàng; không đưa vào chỉ số dây chuyền, nhưng ghi nhận phản ứng trên nền không có chai. |

Đoạn đầu và giữa clip có cảnh màn hình phần mềm của hãng; đoạn cuối có outro. Các vật thể hiển thị trên màn hình không được tính là chai đang chạy trên dây chuyền. Clip có đúng chủ đề ứng dụng nhưng chưa phải một bộ kiểm thử có nhãn, camera cố định và ca lỗi được xác nhận.

## Nhận định và phần chưa rõ

Nhận sai và bỏ sót xuất hiện trong các cảnh quay trực tiếp chai, nên không thể giải thích toàn bộ bằng intro/outro hoặc việc chọn clip. Tăng ngưỡng confidence đơn thuần không xử lý được ví dụ nhận nhầm cap 0,76, đồng thời có nguy cơ loại thêm dự đoán đúng confidence thấp.

Đã xem lại bốn trang gallery nguồn chuẩn bị dữ liệu: overview-01, overview-09, overview-17, overview-25. Các trang chứa ảnh chai chụp trên bàn/sàn/vải, ảnh sản phẩm nền trắng, chai móp và rác. Đây là căn cứ nghi ngờ chênh lệch giữa dữ liệu và môi trường ứng dụng. Gallery thuộc bước review trước khi tạo candidate cuối, không phải thống kê phân bố của toàn bộ 3849 ảnh train hiện tại. Chưa kiểm tra đầy đủ overlay nhãn của candidate cuối; chưa thể kết luận nhãn sai hay dữ liệu là nguyên nhân duy nhất.

Nhòe chuyển động, kích thước nắp nhỏ, ánh sáng và che khuất cũng cần kiểm tra. Nguồn chỉ 854×480; tăng kích thước inference không khôi phục được chi tiết vốn đã mất. Chưa có thí nghiệm để tách ảnh hưởng độ phân giải, dữ liệu và kích cỡ model.

## Đề xuất lượt train tiếp theo

1. Audit candidate đang train trên Colab bằng gallery ảnh kèm box: kiểm tra đủ nhãn và nhất quán phạm vi bottle/cap/label, gồm ảnh nhiều chai và ảnh nghi vấn. Giữ nguồn, làm candidate mới; ghi rõ ảnh/nhãn nào cần sửa hoặc chưa giải quyết. Không chỉ xem class count hoặc file YOLO hợp lệ rồi coi nhãn đã đúng.
2. Xây dữ liệu đại diện cho camera kiểm hàng: chai đủ nắp/nhãn, thiếu nắp hoặc thiếu nhãn đã xác nhận, nhiều khoảng cách/góc nhìn, màu chai/nắp, phản xạ, ánh sáng và mức nhòe tương ứng. Khởi đầu một đợt vài trăm ảnh khác biệt từ nhiều video/phiên quay; số lượng này là quy mô thử, không phải cam kết đủ. Dùng nguồn được phép huấn luyện; clip MV Center hiện giữ làm demo phát triển, chưa đưa vào train.
3. Thêm ảnh nền máy dễ gây nhận nhầm. Chỉ dùng nhãn rỗng khi ảnh thật sự không có bottle/cap/label; ảnh có chai phải gán đủ đối tượng mục tiêu. Giữ đúng ba lớp, không thêm lớp missing-cap/missing-label. Bộ phận bị che không tự trở thành ca thiếu bộ phận; thống nhất quy tắc gán nhãn trước.
4. Tách theo video/phiên quay: tạo bộ validation dây chuyền để chọn cấu hình, giữ một tập kiểm tra cuối từ nguồn độc lập chưa dùng điều chỉnh model. Không chia ngẫu nhiên các frame cạnh nhau vào train và val/test. Điểm test cũ chỉ mô tả bộ test cũ; không dùng để chọn đi chọn lại run mới.
5. Lượt A: fine-tune từ best.pt hiện tại bằng dữ liệu đã kiểm tra và dữ liệu bổ sung phù hợp, giữ YOLO11n/imgsz=640 để đánh giá tác động thay đổi dữ liệu. Tạo run mới và binding dataset mới; không resume run 92 epoch đã hoàn tất, không ghi đè model hiện tại. Số epoch/lr/batch được chọn trên validation và GPU Colab thực tế, có checkpoint lên Drive.
6. Nếu lượt A vẫn yếu: thử tăng kích thước inference trên cùng validation; sau đó cân nhắc thử YOLO11s từ pretrained trên cùng dữ liệu. Đổi một yếu tố mỗi thí nghiệm, đo cả độ trễ. Chưa có bằng chứng model lớn hơn hoặc nhiều epoch hơn sẽ giải quyết các lỗi hiện tại.
7. So sánh model cũ/mới trên cùng bộ validation đã gán nhãn: precision/recall và định vị từng lớp, số nhận nhầm trên nền máy, bỏ sót chai/nắp/nhãn; theo dõi khả năng cũ để tránh giảm chất lượng. Chỉ sau đó đánh giá PASS/FAIL với ground truth từng chai ở vị trí quan sát phù hợp. Detector tốt chưa tự chứng minh sản phẩm đạt/lỗi đúng.

Chưa sửa code train hoặc khởi chạy huấn luyện trong lượt review này.

## Tài liệu tham khảo

- [Ultralytics: dữ liệu đại diện, nhãn đầy đủ/chính xác và ảnh nền](https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results/).
- [Ultralytics: tách theo video/camera/phiên chụp và giữ test độc lập](https://academy.ultralytics.com/courses/dataset-readiness-for-yolo/split-the-dataset-correctly).
- [Ultralytics: train và resume](https://docs.ultralytics.com/modes/train/).

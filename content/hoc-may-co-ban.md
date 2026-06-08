# Học máy cơ bản

> Học máy (Machine Learning) là nhánh của AI cho phép máy tính **học quy luật từ dữ liệu** thay vì được lập trình tường minh cho từng tình huống.

## Ý tưởng cốt lõi

Thay vì viết luật `nếu... thì...`, ta đưa cho mô hình rất nhiều ví dụ và để nó tự tìm ra hàm ánh xạ từ đầu vào $x$ sang đầu ra $y$:

$$y = f_\theta(x)$$

trong đó $\theta$ là tập tham số mà mô hình điều chỉnh trong quá trình huấn luyện.

## Ba kiểu học

### 1. Học có giám sát (Supervised Learning)

Dữ liệu có nhãn: mỗi $x$ đi kèm đáp án $y$ đúng.

- **Hồi quy (Regression)** — dự đoán giá trị liên tục (giá nhà, nhiệt độ).
- **Phân loại (Classification)** — dự đoán nhãn rời rạc (thư rác / không phải).

### 2. Học không giám sát (Unsupervised Learning)

Dữ liệu **không có nhãn**; mô hình tự tìm cấu trúc ẩn.

- **Phân cụm (Clustering)** — nhóm các điểm tương tự.
- **Giảm chiều (Dimensionality Reduction)** — nén dữ liệu giữ thông tin chính.

### 3. Học tăng cường (Reinforcement Learning)

Tác nhân (agent) tương tác với môi trường, nhận **phần thưởng** và học chính sách hành động tối đa hóa phần thưởng tích lũy.

## Quy trình huấn luyện một mô hình

1. **Thu thập & làm sạch dữ liệu.**
2. **Chia tập dữ liệu** thành train / validation / test.
3. **Chọn mô hình** và khởi tạo tham số.
4. **Huấn luyện** — tối thiểu hóa hàm mất mát.
5. **Đánh giá** trên tập test.
6. **Tinh chỉnh** siêu tham số và lặp lại.

## Quá khớp và dưới khớp

- **Underfitting (dưới khớp)** — mô hình quá đơn giản, sai cả trên tập train.
- **Overfitting (quá khớp)** — mô hình học thuộc lòng dữ liệu train, kém trên dữ liệu mới.

> Mục tiêu là **tổng quát hóa** (generalization): hoạt động tốt trên dữ liệu chưa từng thấy. Các kỹ thuật như regularization, dropout và tăng cường dữ liệu giúp đạt điều này.

Chuyên đề tiếp theo sẽ làm rõ "mô hình học như thế nào" qua **hàm mất mát và tối ưu hóa**.

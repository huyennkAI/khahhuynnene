# Mạng nơ-ron nhân tạo

> Mạng nơ-ron nhân tạo (Artificial Neural Network) là mô hình tính toán lấy cảm hứng từ não bộ, là nền tảng của **học sâu** (deep learning).

## Nơ-ron nhân tạo

Đơn vị cơ bản nhận nhiều đầu vào, tính tổng có trọng số rồi đưa qua một hàm kích hoạt:

$$z = \sum_{i} w_i x_i + b, \qquad a = \sigma(z)$$

- $w_i$ — trọng số (weights), thể hiện tầm quan trọng của mỗi đầu vào.
- $b$ — độ lệch (bias).
- $\sigma$ — hàm kích hoạt phi tuyến.

## Hàm kích hoạt

Phi tuyến là điều kiện để mạng học được quan hệ phức tạp:

| Hàm | Công thức | Ghi chú |
| --- | --- | --- |
| Sigmoid | $\dfrac{1}{1+e^{-z}}$ | Đầu ra trong $(0,1)$ |
| Tanh | $\dfrac{e^z - e^{-z}}{e^z + e^{-z}}$ | Đầu ra trong $(-1,1)$ |
| ReLU | $\max(0, z)$ | Phổ biến, tính nhanh |

> Nếu bỏ hàm kích hoạt, một mạng nhiều lớp chỉ tương đương một phép biến đổi tuyến tính duy nhất, dù sâu đến đâu.

## Mạng nhiều lớp (MLP)

Một **Multi-Layer Perceptron** gồm:

- **Lớp đầu vào** (input layer) — nhận dữ liệu.
- **Các lớp ẩn** (hidden layers) — trích xuất đặc trưng cấp cao dần.
- **Lớp đầu ra** (output layer) — đưa ra dự đoán.

Mỗi lớp biến đổi đầu vào theo:

$$h^{(l)} = \sigma\!\left( W^{(l)} h^{(l-1)} + b^{(l)} \right)$$

Càng nhiều lớp, mạng càng học được các đặc trưng trừu tượng: cạnh → hình khối → vật thể.

## Vì sao "học sâu" mạnh mẽ?

1. **Tự học đặc trưng** — không cần thiết kế đặc trưng thủ công.
2. **Khả năng biểu diễn lớn** — xấp xỉ được hầu hết hàm liên tục (universal approximation).
3. **Mở rộng theo dữ liệu** — càng nhiều dữ liệu và tính toán, hiệu năng càng tăng.

## Những kiến trúc tiêu biểu

- **CNN** — xử lý ảnh, dùng phép tích chập.
- **RNN / LSTM** — xử lý chuỗi tuần tự.
- **Transformer** — xử lý chuỗi bằng cơ chế attention (chuyên đề tiếp theo).

> Mạng nơ-ron biến ý tưởng "học từ ví dụ" thành một cỗ máy có thể mở rộng tới hàng tỷ tham số.

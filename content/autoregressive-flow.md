# Autoregressive & Normalizing Flows

> Hai họ mô hình này có chung một điểm mạnh hiếm có: **tính được likelihood chính xác** $p_\theta(x)$, thay vì chỉ xấp xỉ như VAE hay né tránh như GAN.

## Mô hình tự hồi quy (Autoregressive)

Ý tưởng: phân tích xác suất của toàn bộ dữ liệu thành tích các xác suất có điều kiện theo quy tắc chuỗi:

$$p_\theta(x) = \prod_{i=1}^{n} p_\theta(x_i \mid x_1, \dots, x_{i-1})$$

Mỗi phần tử được dự đoán dựa trên các phần tử **đứng trước** nó. Đây chính là cách các mô hình ngôn ngữ lớn (GPT) sinh văn bản: dự đoán token kế tiếp.

### Ví dụ tiêu biểu

- **PixelCNN / PixelRNN** — sinh ảnh từng điểm ảnh một.
- **WaveNet** — sinh âm thanh từng mẫu một.
- **GPT** — sinh văn bản từng token một.

### Ưu và nhược

- **Ưu:** likelihood chính xác, chất lượng cao, huấn luyện ổn định.
- **Nhược:** lấy mẫu **tuần tự** nên chậm, vì phải sinh lần lượt từng phần tử.

## Normalizing Flows

Ý tưởng: biến đổi một phân phối đơn giản (như chuẩn) thành phân phối dữ liệu phức tạp qua một chuỗi **hàm khả nghịch** $f = f_K \circ \dots \circ f_1$.

Dùng công thức đổi biến, likelihood tính được chính xác:

$$\log p_\theta(x) = \log p_z\big(f^{-1}(x)\big) + \log \left| \det \frac{\partial f^{-1}}{\partial x} \right|$$

Để khả thi, mỗi phép biến đổi phải:

1. **Khả nghịch** — để cả suy diễn và sinh mẫu đều thực hiện được.
2. **Có Jacobian dễ tính định thức** — thường thiết kế dạng tam giác.

### Ví dụ tiêu biểu

- **RealNVP / NICE** — dùng các lớp coupling khả nghịch.
- **Glow** — thêm phép tích chập khả nghịch $1 \times 1$, sinh khuôn mặt chất lượng cao.

### Ưu và nhược

- **Ưu:** likelihood chính xác, lấy mẫu một bước, không gian ẩn diễn giải được.
- **Nhược:** ràng buộc khả nghịch hạn chế khả năng biểu diễn; mô hình thường rất lớn.

## So sánh nhanh trong họ mô hình sinh

| Tiêu chí | VAE | GAN | Diffusion | Autoregressive | Flow |
| --- | --- | --- | --- | --- | --- |
| Likelihood chính xác | Không | Không | Không | Có | Có |
| Chất lượng mẫu | Trung bình | Cao | Rất cao | Cao | Trung bình |
| Tốc độ lấy mẫu | Nhanh | Nhanh | Chậm | Chậm | Nhanh |
| Độ ổn định huấn luyện | Cao | Thấp | Cao | Cao | Cao |

> Không có "người chiến thắng" tuyệt đối: mỗi họ tối ưu một góc khác nhau của tam giác chất lượng — likelihood — tốc độ.

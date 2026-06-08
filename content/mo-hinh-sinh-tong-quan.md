# Tổng quan mô hình sinh

> Mô hình sinh sâu (Deep Generative Models) học **phân phối của dữ liệu** để có thể tạo ra các mẫu mới giống thật: ảnh, âm thanh, văn bản, phân tử...

## Mô hình sinh vs mô hình phân biệt

- **Mô hình phân biệt (Discriminative)** học $p(y \mid x)$ — ranh giới giữa các lớp. Ví dụ: bộ phân loại ảnh.
- **Mô hình sinh (Generative)** học $p(x)$ hoặc $p(x \mid y)$ — cách dữ liệu được tạo ra. Từ đó **lấy mẫu** $x \sim p(x)$ để sinh dữ liệu mới.

## Bài toán cốt lõi

Cho tập dữ liệu $\{x_1, \dots, x_n\}$ rút từ phân phối thật $p_{\text{data}}(x)$ chưa biết, ta muốn học một mô hình $p_\theta(x)$ sao cho $p_\theta \approx p_{\text{data}}$. Tiêu chí phổ biến là **hợp lý cực đại** (maximum likelihood):

$$\theta^\* = \arg\max_{\theta} \; \mathbb{E}_{x \sim p_{\text{data}}} \big[ \log p_\theta(x) \big]$$

Khó khăn: với dữ liệu nhiều chiều (như ảnh), việc tính $p_\theta(x)$ trực tiếp thường bất khả thi. Mỗi họ mô hình sinh là một cách **né tránh hoặc xấp xỉ** khó khăn này.

## Bốn họ mô hình chính

| Họ mô hình | Ý tưởng | Đặc điểm |
| --- | --- | --- |
| **VAE** | Mã hóa vào không gian ẩn, học qua chặn dưới ELBO | Huấn luyện ổn định, mẫu hơi mờ |
| **GAN** | Trò chơi đối kháng giữa sinh và phân biệt | Mẫu sắc nét, khó huấn luyện |
| **Diffusion** | Khử nhiễu dần từ nhiễu thuần | Chất lượng cao, lấy mẫu chậm |
| **Autoregressive / Flow** | Phân tích $p(x)$ theo tích xác suất có điều kiện | Tính được likelihood chính xác |

## Đánh đổi cơ bản

Mọi mô hình sinh đều xoay quanh ba yếu tố khó cùng đạt:

1. **Chất lượng mẫu** — mẫu sinh ra có sắc nét, đa dạng không.
2. **Likelihood tính được** — có ước lượng được $p_\theta(x)$ không.
3. **Lấy mẫu nhanh** — sinh một mẫu tốn bao nhiêu bước.

Không họ nào hoàn hảo cả ba; lựa chọn phụ thuộc bài toán.

## Cách đánh giá

- **FID (Fréchet Inception Distance)** — so khoảng cách đặc trưng giữa ảnh thật và ảnh sinh; càng thấp càng tốt.
- **Inception Score** — đo độ rõ ràng và đa dạng của mẫu.
- **Log-likelihood / bits-per-dim** — cho các mô hình tính được likelihood.

> Các bài tiếp theo đi sâu vào từng họ: **Autoencoder & VAE**, **GAN**, **Diffusion**, và **Autoregressive & Normalizing Flows**.

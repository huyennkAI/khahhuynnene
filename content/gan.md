# Mạng đối kháng sinh (GAN)

> Generative Adversarial Network (GAN) huấn luyện hai mạng cạnh tranh nhau: một bên cố tạo dữ liệu giả như thật, một bên cố vạch trần. Cuộc đua này đẩy chất lượng mẫu lên rất cao.

## Hai người chơi

- **Bộ sinh (Generator)** $G$ — nhận nhiễu $z \sim p(z)$ và tạo mẫu giả $G(z)$.
- **Bộ phân biệt (Discriminator)** $D$ — nhận một mẫu và ước lượng xác suất nó là **thật**.

Generator muốn đánh lừa Discriminator; Discriminator muốn phân biệt đúng. Đây là một **trò chơi tổng bằng không**.

## Hàm mục tiêu minimax

$$\min_{G} \max_{D} \; \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]$$

- $D$ cực đại hóa: gán điểm cao cho dữ liệu thật, điểm thấp cho dữ liệu giả.
- $G$ cực tiểu hóa: làm $D(G(z))$ tiến gần 1.

Tại điểm cân bằng lý tưởng, $G$ tái tạo đúng $p_{\text{data}}$ và $D$ chỉ còn đoán mò với xác suất $0.5$.

## Quy trình huấn luyện

```text
Lặp:
  1. Cố định G, cập nhật D trên một lô mẫu thật và mẫu giả
  2. Cố định D, cập nhật G để làm D đánh giá mẫu giả là thật
```

Trên thực tế, $G$ thường tối ưu mục tiêu **non-saturating** $\max_G \mathbb{E}\big[\log D(G(z))\big]$ để có gradient mạnh hơn ở giai đoạn đầu.

## Những khó khăn kinh điển

- **Sụp đổ chế độ (Mode collapse)** — $G$ chỉ sinh ra vài kiểu mẫu, mất đa dạng.
- **Mất ổn định huấn luyện** — hai mạng dao động, không hội tụ.
- **Gradient biến mất** — khi $D$ quá mạnh, $G$ không còn tín hiệu học.

## Các cải tiến quan trọng

| Biến thể | Đóng góp |
| --- | --- |
| **DCGAN** | Dùng tích chập, ổn định huấn luyện ảnh |
| **WGAN** | Thay mục tiêu bằng khoảng cách Wasserstein, giảm mode collapse |
| **Conditional GAN** | Sinh có điều kiện theo nhãn |
| **StyleGAN** | Điều khiển phong cách theo từng tầng, ảnh khuôn mặt cực thực |

## Ưu và nhược

- **Ưu:** mẫu **sắc nét**, lấy mẫu nhanh (một lần truyền qua $G$).
- **Nhược:** huấn luyện khó, không tính được likelihood, dễ mất đa dạng.

> GAN từng là tiêu chuẩn vàng cho sinh ảnh suốt 2014–2020, trước khi mô hình Diffusion vươn lên dẫn đầu về chất lượng và độ ổn định.

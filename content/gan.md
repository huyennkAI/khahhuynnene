# Mạng đối kháng sinh (GAN)

> Mạng đối kháng sinh (Generative Adversarial Network - GAN) huấn luyện hai mạng cạnh tranh: một bên (generator) tạo dữ liệu giả như thật, một bên (discriminator) cố vạch trần. Cuộc đua đẩy chất lượng mẫu lên rất cao.

## Trực giác ra đời

VAE và normalizing flow đều cố **mô hình hóa mật độ (density)** $p(x)$ rõ ràng, và phải trả giá: VAE cho mẫu mờ, flow bị ràng buộc khả nghịch. Ý tưởng đột phá của GAN (Goodfellow, 2014):

> Để sinh ảnh đẹp, ta đâu cần biết $p(x)$ là gì. Ta chỉ cần một **giám khảo (critic)** không phân biệt được thật với giả.

Đây là chuyển từ mô hình sinh tường minh (explicit generative model) sang **ngầm định (implicit)**: ta không bao giờ viết ra $p_\theta(x)$, chỉ định nghĩa cách lấy mẫu từ nó.

## Khắc phục điều gì

- **Né hoàn toàn việc tính mật độ và hằng số chuẩn hóa** — không cần likelihood.
- **Mẫu sắc nét (sharp samples):** mục tiêu đối kháng phạt thẳng những chi tiết "trông giả", thay vì trung bình hóa như lỗi tái tạo (reconstruction loss) của VAE.
- **Lấy mẫu một bước (one-step sampling):** chỉ một lần truyền qua generator.

## Hai người chơi

- **Bộ sinh (generator)** $G$ — nhận nhiễu $z \sim p(z)$ và tạo mẫu giả $G(z)$.
- **Bộ phân biệt (discriminator)** $D$ — ước lượng xác suất một mẫu là **thật (real)**.

Đây là một trò chơi đối kháng (adversarial game): $G$ muốn đánh lừa $D$; $D$ muốn phân biệt đúng.

## Lý thuyết: mục tiêu minimax

$$\min_{G} \max_{D} \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]$$

### Bước 1: Discriminator tối ưu

Cố định $G$ (sinh ra phân phối $p_g$), tìm $D$ tối ưu. Viết lại theo $x$:

$$V(D, G) = \int_x \Big[ p_{\text{data}}(x) \log D(x) + p_g(x) \log\big(1 - D(x)\big) \Big] dx$$

Với mỗi $x$, cực đại hàm $a\log D + b\log(1-D)$ theo $D$. Đạo hàm bằng 0:

$$\frac{a}{D} - \frac{b}{1-D} = 0 \;\Rightarrow\; D^{*}(x) = \frac{a}{a+b} = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

### Bước 2: Thay vào, hiện ra JS divergence

Thay $D^{*}$ vào $V$:

$$
\begin{aligned}
V(D^{*}, G) &= \mathbb{E}_{p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}}{p_{\text{data}} + p_g}\right] + \mathbb{E}_{p_g}\!\left[\log \frac{p_g}{p_{\text{data}} + p_g}\right] \\
&= -\log 4 + D_{\mathrm{KL}}\!\Big(p_{\text{data}} \,\Big\|\, \tfrac{p_{\text{data}} + p_g}{2}\Big) + D_{\mathrm{KL}}\!\Big(p_g \,\Big\|\, \tfrac{p_{\text{data}} + p_g}{2}\Big) \\
&= -\log 4 + 2 \cdot D_{\mathrm{JS}}\!\big(p_{\text{data}} \,\|\, p_g\big)
\end{aligned}
$$

### Kết luận

Cực tiểu theo $G$ chính là **cực tiểu phân kỳ Jensen–Shannon (Jensen–Shannon divergence)** giữa $p_g$ và $p_{\text{data}}$. Vì JS divergence $\ge 0$ và bằng 0 khi và chỉ khi hai phân phối trùng nhau, **tối ưu toàn cục đạt tại $p_g = p_{\text{data}}$**, khi đó $D^{*} \equiv \tfrac{1}{2}$ (giám khảo chỉ còn đoán mò). $\blacksquare$

## Huấn luyện thực tế

```text
Lặp:
  1. Cố định G, cập nhật D trên một lô mẫu thật và mẫu giả
  2. Cố định D, cập nhật G để D đánh giá mẫu giả là thật
```

$G$ thường tối ưu mục tiêu **non-saturating** $\max_G \mathbb{E}[\log D(G(z))]$ thay vì $\min_G \log(1 - D(G(z)))$, vì mục tiêu gốc có gradient gần như biến mất (vanishing gradient) ở giai đoạn đầu khi $D$ dễ dàng phân biệt.

## Những khó khăn kinh điển

- **Sụp đổ chế độ (mode collapse):** $G$ chỉ sinh vài kiểu mẫu để lừa $D$, mất đa dạng.
- **Mất ổn định (training instability):** hai mạng dao động, cân bằng Nash khó đạt.
- **Gradient biến mất:** khi JS divergence bão hòa lúc hai phân phối không chồng lấn, $G$ mất tín hiệu học.

## Các cải tiến quan trọng

| Biến thể | Đóng góp |
| --- | --- |
| **DCGAN** | Kiến trúc tích chập, ổn định huấn luyện ảnh |
| **WGAN** | Thay JS bằng khoảng cách Wasserstein, gradient mượt, giảm mode collapse |
| **Conditional GAN (cGAN)** | Sinh có điều kiện theo nhãn |
| **StyleGAN** | Điều khiển phong cách theo từng tầng, khuôn mặt cực thực |

> WGAN ra đời chính để khắc phục gradient biến mất: Wasserstein distance cho gradient có ý nghĩa cả khi hai phân phối **không chồng lấn**, điều mà JS divergence không làm được.

## Ưu điểm

- **Mẫu sắc nét** nhất trong các mô hình tường minh suốt 2014–2020.
- **Lấy mẫu nhanh** — một lần truyền qua $G$.
- **Khung linh hoạt** — chỉ cần định nghĩa được generator khả vi.

## Nhược điểm

- **Huấn luyện khó và nhạy cảm** siêu tham số, dễ mất ổn định.
- **Không tính được likelihood**, khó đánh giá định lượng.
- **Mode collapse** làm giảm đa dạng mẫu.

> GAN từng là tiêu chuẩn vàng cho sinh ảnh, trước khi mô hình khuếch tán (diffusion) vươn lên dẫn đầu nhờ ổn định và đa dạng hơn — chủ đề bài tiếp theo.

# Mô hình khuếch tán (Diffusion)

> Mô hình khuếch tán (diffusion model) học cách **khử nhiễu dần dần (iterative denoising)**: bắt đầu từ nhiễu thuần (pure noise) và từng bước biến nó thành mẫu sắc nét. Đây là nền tảng của Stable Diffusion, DALL·E và phần lớn hệ sinh ảnh hiện đại.

## Trực giác ra đời

GAN cho mẫu đẹp nhưng huấn luyện bất ổn và hay sụp đổ chế độ (mode collapse). VAE ổn định nhưng mẫu mờ. Trực giác của diffusion:

> Sinh một bức ảnh hoàn chỉnh trong một bước là quá khó. Nhưng **khử một chút nhiễu** thì dễ. Vậy hãy chia bài toán khó thành hàng trăm bước nhỏ dễ.

Đây là tư duy "chia để trị" trên miền xác suất: thay vì học một phép nhảy lớn từ nhiễu sang dữ liệu (như GAN), ta học rất nhiều phép nhảy nhỏ, mỗi phép chỉ cần khử nhiễu nhẹ.

## Khắc phục điều gì

- **So với GAN:** huấn luyện **ổn định (stable)** — chỉ là hồi quy bình phương (regression), không có trò chơi đối kháng (adversarial game); đa dạng mẫu (sample diversity) tốt hơn nhiều.
- **So với VAE:** mẫu **sắc nét hơn hẳn** nhờ tinh chỉnh qua nhiều bước.
- Đổi lại: **lấy mẫu chậm** vì cần nhiều bước suy luận.

## Hai quá trình

### Quá trình thuận (forward / diffusion process) — thêm nhiễu

Dần phá hủy dữ liệu bằng cách cộng nhiễu Gauss qua $T$ bước, theo lịch nhiễu (noise schedule) $\beta_t$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left( x_t; \sqrt{1 - \beta_t}\, x_{t-1}, \; \beta_t I \right)$$

**Tính chất nhảy bước (closed-form):** với $\alpha_t = 1 - \beta_t$ và $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$, có thể lấy mẫu $x_t$ thẳng từ $x_0$:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

Khi $t \to T$, $\bar{\alpha}_t \to 0$ nên $x_T$ gần như là nhiễu chuẩn $\mathcal{N}(0, I)$.

### Quá trình nghịch (reverse process) — khử nhiễu

Mô hình học đảo ngược, từng bước đưa nhiễu về dữ liệu:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left( x_{t-1};\, \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t) \right)$$

Một kết quả quan trọng: nếu $\beta_t$ đủ nhỏ, bước nghịch thật sự **cũng là Gauss**, nên giả định trên là hợp lý.

## Lý thuyết: từ ELBO đến mục tiêu đơn giản

Giống VAE, diffusion tối ưu một chặn dưới bằng chứng (ELBO) trên $\log p_\theta(x_0)$. Khai triển cho ra tổng các số hạng KL giữa hậu nghiệm thuận $q(x_{t-1}\mid x_t, x_0)$ và bước nghịch học được $p_\theta(x_{t-1}\mid x_t)$:

$$\mathcal{L} = \mathbb{E}_q\!\Big[ \underbrace{D_{\mathrm{KL}}(q(x_T\mid x_0)\,\|\,p(x_T))}_{\text{không tham số}} + \sum_{t>1} D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big) - \log p_\theta(x_0 \mid x_1) \Big]$$

Vì cả hai phân phối trong KL đều là Gauss, KL rút về khoảng cách bình phương giữa hai trung bình. Tham số hóa $\mu_\theta$ khéo léo để **dự đoán nhiễu (noise prediction)** $\epsilon$ thay vì trung bình, mục tiêu rút gọn (Ho et al., 2020) thành:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\, \epsilon,\, t} \left[ \big\lVert \epsilon - \epsilon_\theta(x_t, t) \big\rVert^2 \right]$$

> Diễn giải đẹp: huấn luyện diffusion chỉ là **dạy một mạng đoán xem đã thêm bao nhiêu nhiễu** vào ảnh tại mức $t$ bất kỳ. Đơn giản, ổn định, không đối kháng.

```python
t = torch.randint(0, T, (batch,))
eps = torch.randn_like(x0)
x_t = sqrt_acp[t] * x0 + sqrt_one_minus_acp[t] * eps
loss = mse(model(x_t, t), eps)
```

Mạng $\epsilon_\theta$ thường là kiến trúc **U-Net** với điều kiện thời gian (time embedding).

## Liên hệ score matching

Có thể chứng minh dự đoán nhiễu tương đương ước lượng **hàm điểm (score function)** $\nabla_x \log p_t(x)$. Cách nhìn này (score-based model) dẫn tới công thức phương trình vi phân ngẫu nhiên (stochastic differential equation - SDE), thống nhất diffusion dưới một khung liên tục.

## Lấy mẫu

Bắt đầu từ $x_T \sim \mathcal{N}(0, I)$, lặp bước khử nhiễu từ $t = T$ về $1$ để thu $x_0$.

- **DDPM** — lấy mẫu ngẫu nhiên, cần hàng trăm đến hàng nghìn bước.
- **DDIM** — lấy mẫu xác định (deterministic), rút còn vài chục bước.

## Sinh có điều kiện (conditional generation)

- **Classifier-free guidance** — huấn luyện đồng thời có và không điều kiện, rồi ngoại suy để điều khiển mức bám prompt.
- **Latent Diffusion (Stable Diffusion)** — chạy khuếch tán trong không gian ẩn nén (latent space) của một autoencoder, giảm mạnh chi phí tính toán.

## Ưu điểm

- **Chất lượng và đa dạng mẫu (sample quality & diversity)** dẫn đầu hiện nay.
- **Huấn luyện ổn định** — chỉ là hồi quy, không mode collapse.
- Khung lý thuyết phong phú (ELBO, score matching, SDE).

## Nhược điểm

- **Lấy mẫu chậm (slow sampling):** cần nhiều bước, chi phí suy luận lớn.
- **Tốn tính toán** khi huấn luyện và sinh ở độ phân giải cao (giảm bớt nhờ latent diffusion).
- Không cho likelihood chính xác (chỉ chặn dưới ELBO).

> Diffusion hiện dẫn đầu sinh ảnh, và đang lan sang âm thanh, video, sinh phân tử. Bí quyết: biến một bài toán sinh khó thành chuỗi bài toán khử nhiễu dễ và ổn định.

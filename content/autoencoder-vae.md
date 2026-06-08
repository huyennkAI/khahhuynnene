# Autoencoder & VAE

> Autoencoder nén dữ liệu vào một không gian ẩn rồi tái tạo lại. Biến thể xác suất của nó — **Variational Autoencoder (VAE)** — biến ý tưởng này thành một mô hình sinh đúng nghĩa.

## Autoencoder

Một autoencoder gồm hai phần:

- **Bộ mã hóa (Encoder)** $z = f_\phi(x)$ — nén $x$ thành vector ẩn $z$ ít chiều.
- **Bộ giải mã (Decoder)** $\hat{x} = g_\theta(z)$ — tái tạo lại $x$ từ $z$.

Huấn luyện bằng cách tối thiểu hóa sai số tái tạo:

$$L = \lVert x - g_\theta(f_\phi(x)) \rVert^2$$

Vấn đề: không gian ẩn không có cấu trúc xác suất, nên **không lấy mẫu sinh mới được** một cách có ý nghĩa.

## Từ Autoencoder đến VAE

VAE áp một **phân phối tiên nghiệm** lên không gian ẩn, thường là chuẩn $p(z) = \mathcal{N}(0, I)$. Encoder không trả về một điểm mà trả về một **phân phối** $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi(x))$.

Để sinh dữ liệu mới: lấy $z \sim \mathcal{N}(0, I)$ rồi cho qua decoder.

## Hàm mục tiêu ELBO

Likelihood $\log p_\theta(x)$ không tính trực tiếp được, nên ta tối ưu **chặn dưới bằng chứng** (Evidence Lower Bound):

$$\log p_\theta(x) \ge \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big]}_{\text{tái tạo}} - \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\text{điều chuẩn}}$$

- **Số hạng tái tạo** — buộc mẫu giải mã giống đầu vào.
- **Số hạng KL** — kéo phân phối ẩn về gần $\mathcal{N}(0, I)$, giúp không gian ẩn liên tục và lấy mẫu được.

## Mẹo tái tham số hóa (Reparameterization Trick)

Không thể lan truyền ngược qua phép lấy mẫu ngẫu nhiên. VAE viết lại:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

Nhờ đó tính ngẫu nhiên nằm ở $\epsilon$, còn $\mu, \sigma$ vẫn khả vi để huấn luyện bằng gradient descent.

```python
mu, log_var = encoder(x)
eps = torch.randn_like(mu)
z = mu + torch.exp(0.5 * log_var) * eps
x_hat = decoder(z)
recon = mse(x_hat, x)
kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
loss = recon + kld
```

## Ưu và nhược

- **Ưu:** huấn luyện ổn định, có không gian ẩn diễn giải được, dễ nội suy giữa các mẫu.
- **Nhược:** mẫu sinh thường **mờ** hơn GAN hay Diffusion, do giả định phân phối đơn giản và mục tiêu trung bình hóa.

> VAE đặt nền cho tư duy "biến ẩn + suy diễn biến phân", xuất hiện lại trong nhiều mô hình sinh hiện đại.

# Autoencoder & VAE

> Bộ tự mã hóa (autoencoder) nén dữ liệu vào không gian ẩn (latent space) rồi tái tạo lại. Biến thể xác suất của nó — bộ tự mã hóa biến phân (Variational Autoencoder - VAE) — biến ý tưởng đó thành một mô hình sinh (generative model) đúng nghĩa.

## Trực giác ra đời

Mô hình tự hồi quy (autoregressive) lấy mẫu chậm; normalizing flow bị ràng buộc khả nghịch (invertibility). Câu hỏi đặt ra: **có thể giả định dữ liệu sinh ra từ một vài biến ẩn (latent variables) đơn giản, rồi để mạng nơ-ron tự do ánh xạ từ biến ẩn sang dữ liệu không?**

Trực giác sâu xa: một khuôn mặt được mô tả gọn bằng vài yếu tố ẩn (góc nghiêng, cảm xúc, ánh sáng). Nếu học được không gian ẩn đó, ta chỉ cần lấy mẫu một điểm trong không gian nhỏ này rồi "giải mã" thành ảnh.

## Khắc phục điều gì

- **Né hằng số chuẩn hóa (normalizing constant) $Z_\theta$:** thay vì tính trực tiếp $p_\theta(x)$, VAE tối ưu một chặn dưới (lower bound) bằng suy diễn biến phân (variational inference).
- **So với autoencoder thường:** áp một cấu trúc xác suất (probabilistic structure) lên không gian ẩn để **lấy mẫu sinh mới** được — điều autoencoder thường không làm được.
- **So với flow:** decoder được **tự do** (không cần khả nghịch), nên biểu diễn mạnh hơn.

## Từ Autoencoder đến VAE

Bộ tự mã hóa gồm:

- **Bộ mã hóa (encoder)** $z = f_\phi(x)$ — nén $x$ thành vector ẩn $z$ ít chiều.
- **Bộ giải mã (decoder)** $\hat{x} = g_\theta(z)$ — tái tạo $x$ từ $z$.

Huấn luyện bằng cách cực tiểu sai số tái tạo (reconstruction error) $\lVert x - g_\theta(f_\phi(x)) \rVert^2$. Vấn đề: không gian ẩn không có cấu trúc xác suất nên **không sinh mẫu mới có ý nghĩa được**.

VAE sửa điều này: áp một phân phối tiên nghiệm (prior) $p(z) = \mathcal{N}(0, I)$ lên không gian ẩn, và để encoder trả về một **phân phối** thay vì một điểm:

$$q_\phi(z \mid x) = \mathcal{N}\big(\mu_\phi(x),\, \sigma_\phi^2(x)\big)$$

## Lý thuyết: suy ra ELBO

Ta muốn cực đại log-likelihood biên (marginal log-likelihood) $\log p_\theta(x)$, với $p_\theta(x) = \int p_\theta(x \mid z)\,p(z)\,dz$ — tích phân này bất khả thi. Đưa vào phân phối hậu nghiệm xấp xỉ (approximate posterior) $q_\phi(z \mid x)$:

$$
\begin{aligned}
\log p_\theta(x) &= \log \int p_\theta(x \mid z)\, p(z)\, dz \\
&= \log \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)} \right] \\
&\ge \mathbb{E}_{q_\phi(z \mid x)}\!\left[ \log \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)} \right] \quad \text{(bất đẳng thức Jensen)}
\end{aligned}
$$

Vế cuối là **chặn dưới bằng chứng (Evidence Lower Bound - ELBO)**, tách thành:

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big]}_{\text{tái tạo (reconstruction)}} - \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\text{điều chuẩn (regularization)}}$$

### Vì sao đây là chặn dưới chặt? (chứng minh khoảng hở)

Có thể chứng minh đẳng thức chính xác:

$$\log p_\theta(x) = \mathcal{L}_{\text{ELBO}} + D_{\mathrm{KL}}\!\big(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x)\big)$$

Vì KL luôn $\ge 0$, ta có $\log p_\theta(x) \ge \mathcal{L}_{\text{ELBO}}$, và **khoảng hở (gap) đúng bằng** KL giữa hậu nghiệm xấp xỉ và hậu nghiệm thật $p_\theta(z\mid x)$. Tối đa ELBO vừa tăng likelihood, vừa kéo $q_\phi$ về gần hậu nghiệm thật. $\blacksquare$

Diễn giải hai số hạng:

- **Số hạng tái tạo (reconstruction term)** — buộc $z$ giải mã lại đúng $x$.
- **Số hạng KL (KL term)** — kéo phân phối ẩn về gần prior $\mathcal{N}(0, I)$, giữ không gian ẩn liên tục và lấy mẫu được.

## Mẹo tái tham số hóa (reparameterization trick)

Không thể lan truyền ngược (backpropagation) qua phép lấy mẫu ngẫu nhiên (stochastic sampling). VAE viết lại phép lấy mẫu thành hàm khả vi:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

Tính ngẫu nhiên dồn vào $\epsilon$ (không phụ thuộc tham số), còn $\mu, \sigma$ vẫn khả vi để huấn luyện bằng gradient descent.

```python
mu, log_var = encoder(x)
eps = torch.randn_like(mu)
z = mu + torch.exp(0.5 * log_var) * eps
x_hat = decoder(z)
recon = mse(x_hat, x)
kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
loss = recon + kld
```

Số hạng KL với prior chuẩn có **dạng đóng (closed form)** như trên.

## Ưu điểm

- **Huấn luyện ổn định (stable training)** — chỉ tối ưu một hàm mục tiêu trơn, không có động lực đối kháng (adversarial dynamics).
- **Không gian ẩn diễn giải được**, dễ nội suy (interpolation) và thao tác thuộc tính.
- Cho **ước lượng likelihood** (qua ELBO), hữu ích để phát hiện bất thường (anomaly detection).

## Nhược điểm

- **Mẫu mờ (blurry samples):** giả định Gaussian và mục tiêu trung bình hóa khiến ảnh kém sắc nét so với GAN/Diffusion.
- **Hậu nghiệm sụp đổ (posterior collapse):** khi decoder quá mạnh, $q_\phi(z\mid x)$ rơi về đúng prior và biến ẩn bị bỏ qua.
- **Khoảng hở ELBO:** chỉ tối ưu chặn dưới, không phải likelihood thật.

> VAE đặt nền cho tư duy "biến ẩn (latent variable) + suy diễn biến phân (variational inference)", xuất hiện lại trong nhiều mô hình sinh hiện đại, kể cả latent diffusion. Bài tiếp theo — **GAN** — đi theo hướng ngược lại: bỏ hẳn likelihood để đổi lấy mẫu sắc nét.

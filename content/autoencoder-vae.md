# Autoencoder & VAE

> Bộ tự mã hóa (autoencoder) nén dữ liệu vào không gian ẩn (latent space) rồi tái tạo lại. Biến thể xác suất của nó — bộ tự mã hóa biến phân (Variational Autoencoder - VAE) — biến ý tưởng đó thành một mô hình sinh (generative model) đúng nghĩa qua suy diễn biến phân (variational inference).

## 1. Trực giác ra đời

Mô hình tự hồi quy (autoregressive) lấy mẫu chậm; normalizing flow bị ràng buộc khả nghịch (invertibility) và bảo toàn số chiều. Câu hỏi:

> Có thể giả định dữ liệu sinh ra từ một vài **biến ẩn (latent variables)** đơn giản, rồi để mạng nơ-ron **tự do** ánh xạ từ biến ẩn sang dữ liệu không?

Trực giác sâu xa: một khuôn mặt được mô tả gọn bằng vài yếu tố ẩn (góc nghiêng, cảm xúc, ánh sáng). Nếu học được không gian ẩn nhỏ đó, ta chỉ cần lấy mẫu một điểm $z$ rồi "giải mã" thành ảnh.

## 2. Khắc phục điều gì (lập luận lý thuyết)

- **Né hằng số chuẩn hóa $Z_\theta$:** thay vì tính trực tiếp $p_\theta(x) = \int p_\theta(x\mid z) p(z)\, dz$ (bất khả thi), VAE tối ưu một **chặn dưới (lower bound)** bằng suy diễn biến phân.
- **So với normalizing flow:** flow buộc decoder song ánh và $\dim z = \dim x$. VAE cho decoder **tự do** và $\dim z \ll \dim x$ (nén thật sự), biểu diễn mạnh hơn — đổi lại chỉ được chặn dưới, không phải likelihood chính xác.
- **So với autoencoder thường:** áp một cấu trúc xác suất (probabilistic structure) lên không gian ẩn để **lấy mẫu sinh mới** được.

## 3. Từ Autoencoder đến VAE

Bộ tự mã hóa gồm: **bộ mã hóa (encoder)** $z = f_\phi(x)$ và **bộ giải mã (decoder)** $\hat{x} = g_\theta(z)$, huấn luyện bằng cực tiểu sai số tái tạo (reconstruction error) $\lVert x - g_\theta(f_\phi(x))\rVert^2$.

**Vấn đề lý thuyết:** không gian ẩn của autoencoder thường **không có cấu trúc xác suất** — lấy một điểm $z$ ngẫu nhiên rồi giải mã thường ra rác, vì các $z$ hợp lệ nằm rải rác không theo phân phối nào. VAE sửa điều này bằng cách:

1. Áp **phân phối tiên nghiệm (prior)** $p(z) = \mathcal{N}(0, I)$ lên không gian ẩn.
2. Để encoder trả về một **phân phối** thay vì một điểm: $q_\phi(z \mid x) = \mathcal{N}\big(\mu_\phi(x),\, \operatorname{diag}(\sigma_\phi^2(x))\big)$.

Khi đó sinh mẫu mới rất đơn giản: $z \sim \mathcal{N}(0, I)$ rồi $x \sim p_\theta(x \mid z)$.

## 4. Lý thuyết: suy ra ELBO (hai cách)

Ta muốn cực đại log-likelihood biên (marginal log-likelihood) $\log p_\theta(x)$.

### 4.1. Cách 1 — bất đẳng thức Jensen

Đưa vào phân phối hậu nghiệm xấp xỉ (approximate posterior) $q_\phi(z \mid x)$ bất kỳ:

$$
\begin{aligned}
\log p_\theta(x) &= \log \int p_\theta(x, z)\, dz = \log \int q_\phi(z\mid x)\, \frac{p_\theta(x, z)}{q_\phi(z\mid x)}\, dz \\
&= \log \mathbb{E}_{q_\phi(z\mid x)}\!\left[\frac{p_\theta(x, z)}{q_\phi(z\mid x)}\right] \\
&\ge \mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{p_\theta(x, z)}{q_\phi(z\mid x)}\right] \quad\text{(Jensen, vì } \log \text{ lõm)} \\
&= \underbrace{\mathbb{E}_{q_\phi}\big[\log p_\theta(x\mid z)\big] - D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\textstyle \mathcal{L}_{\text{ELBO}}(\theta, \phi; x)}
\end{aligned}
$$

(Bước cuối tách $\log\frac{p_\theta(x\mid z) p(z)}{q_\phi(z\mid x)} = \log p_\theta(x\mid z) - \log\frac{q_\phi(z\mid x)}{p(z)}$.)

### 4.2. Cách 2 — phân rã chính xác (cho thấy khoảng hở)

**Định lý.** Với mọi $q_\phi$:

$$\log p_\theta(x) = \mathcal{L}_{\text{ELBO}}(\theta, \phi; x) + D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)$$

**Chứng minh.** Xuất phát từ KL giữa hậu nghiệm xấp xỉ và hậu nghiệm thật $p_\theta(z\mid x) = \frac{p_\theta(x, z)}{p_\theta(x)}$:

$$
\begin{aligned}
D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)
&= \mathbb{E}_{q_\phi}\!\left[\log \frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}\right]
= \mathbb{E}_{q_\phi}\!\left[\log \frac{q_\phi(z\mid x)\, p_\theta(x)}{p_\theta(x, z)}\right] \\
&= \mathbb{E}_{q_\phi}\!\left[\log \frac{q_\phi(z\mid x)}{p_\theta(x, z)}\right] + \log p_\theta(x) \\
&= -\mathcal{L}_{\text{ELBO}} + \log p_\theta(x)
\end{aligned}
$$

Chuyển vế được điều phải chứng minh. $\blacksquare$

> **Hệ quả quan trọng:** vì $D_{\mathrm{KL}} \ge 0$, ta có $\log p_\theta(x) \ge \mathcal{L}_{\text{ELBO}}$, và **khoảng hở (gap) đúng bằng** KL giữa hậu nghiệm xấp xỉ và hậu nghiệm thật. Tối đa ELBO theo $\phi$ vừa làm $q_\phi$ tiến gần hậu nghiệm thật, vừa siết chặt chặn dưới; tối đa theo $\theta$ tăng likelihood. ELBO chặt (gap = 0) khi và chỉ khi $q_\phi = p_\theta(z\mid x)$.

### 4.3. Diễn giải hai số hạng ELBO

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{tái tạo (reconstruction)}} - \underbrace{D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{điều chuẩn (regularization)}}$$

- **Số hạng tái tạo** — buộc $z$ giải mã lại đúng $x$.
- **Số hạng KL** — kéo phân phối ẩn về gần prior $\mathcal{N}(0, I)$, giữ không gian ẩn liên tục và lấy mẫu được. Đây chính là thứ autoencoder thường thiếu.

## 5. KL với prior chuẩn có dạng đóng (closed form)

Với $q_\phi(z\mid x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$ và $p(z) = \mathcal{N}(0, I)$ trong $d$ chiều:

$$D_{\mathrm{KL}}\big(\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))\,\|\,\mathcal{N}(0, I)\big) = \frac{1}{2}\sum_{j=1}^{d}\Big(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\Big)$$

### 5.1. Chứng minh (một chiều, rồi cộng theo chiều)

Với hai Gauss một chiều $q = \mathcal{N}(\mu, \sigma^2)$, $p = \mathcal{N}(0, 1)$:

$$
\begin{aligned}
D_{\mathrm{KL}}(q\,\|\,p) &= \mathbb{E}_q\!\left[\log q(z) - \log p(z)\right] \\
&= \mathbb{E}_q\!\left[-\tfrac{1}{2}\log(2\pi\sigma^2) - \tfrac{(z-\mu)^2}{2\sigma^2} + \tfrac{1}{2}\log(2\pi) + \tfrac{z^2}{2}\right] \\
&= -\tfrac{1}{2}\log\sigma^2 - \tfrac{1}{2\sigma^2}\underbrace{\mathbb{E}_q[(z-\mu)^2]}_{=\,\sigma^2} + \tfrac{1}{2}\underbrace{\mathbb{E}_q[z^2]}_{=\,\mu^2+\sigma^2} \\
&= -\tfrac{1}{2}\log\sigma^2 - \tfrac{1}{2} + \tfrac{1}{2}(\mu^2 + \sigma^2) = \tfrac{1}{2}\big(\mu^2 + \sigma^2 - 1 - \log\sigma^2\big)
\end{aligned}
$$

Vì $q_\phi$ có hiệp phương sai chéo (các chiều độc lập), KL cộng theo chiều. $\blacksquare$

## 6. Mẹo tái tham số hóa (reparameterization trick)

Số hạng tái tạo $\mathbb{E}_{q_\phi(z\mid x)}[\cdots]$ phụ thuộc $\phi$ qua **phân phối lấy kỳ vọng**, nên không lan truyền ngược (backpropagation) trực tiếp qua phép lấy mẫu ngẫu nhiên được. VAE viết lại phép lấy mẫu thành hàm khả vi của một nhiễu độc lập:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

### 6.1. Vì sao mẹo này hợp lệ?

Phép biến đổi affine một Gauss chuẩn cho ra đúng $\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$, nên phân phối của $z$ không đổi. Nhưng giờ kỳ vọng lấy theo $\epsilon$ (không phụ thuộc $\phi$), nên gradient chui được vào trong:

$$\nabla_\phi \mathbb{E}_{q_\phi}\big[h(z)\big] = \nabla_\phi \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\big[h(\mu_\phi + \sigma_\phi \odot \epsilon)\big] = \mathbb{E}_{\epsilon}\big[\nabla_\phi\, h(\mu_\phi + \sigma_\phi \odot \epsilon)\big]$$

Ước lượng Monte Carlo của vế phải có **phương sai thấp hơn nhiều** so với ước lượng score-function (REINFORCE) — đây là lý do lý thuyết khiến VAE huấn luyện được hiệu quả.

```python
mu, log_var = encoder(x)
eps = torch.randn_like(mu)
z = mu + torch.exp(0.5 * log_var) * eps
x_hat = decoder(z)
recon = mse(x_hat, x)
kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
loss = recon + kld
```

## 7. Vì sao mẫu VAE bị mờ? (giải thích lý thuyết)

Đây là dẫn chứng cho nhược điểm chính. Giả định decoder Gauss $p_\theta(x\mid z) = \mathcal{N}(g_\theta(z), \sigma^2 I)$ khiến số hạng tái tạo thành sai số bình phương (L2). Cực tiểu kỳ vọng L2 cho nghiệm là **trung bình có điều kiện**:

$$g_\theta(z)^{*} = \mathbb{E}[x \mid z]$$

Khi một $z$ tương ứng nhiều ảnh sắc nét hợp lý (do hậu nghiệm xấp xỉ không hoàn hảo), nghiệm tối ưu là **trung bình** của chúng — và trung bình của các ảnh sắc nét là một ảnh **mờ**. GAN tránh điều này vì mục tiêu đối kháng phạt thẳng ảnh trung bình hóa.

## 8. Ưu điểm (có dẫn chứng lý thuyết)

- **Huấn luyện ổn định** — chỉ tối ưu một hàm mục tiêu trơn (ELBO), không có động lực đối kháng; gradient phương sai thấp nhờ reparameterization (mục 6.1).
- **Không gian ẩn diễn giải được** — số hạng KL ép $q_\phi$ về prior trơn, nên nội suy (interpolation) trong $z$ cho chuyển tiếp mượt.
- **Cho ước lượng likelihood** (qua ELBO), hữu ích cho phát hiện bất thường (anomaly/OOD detection).

## 9. Nhược điểm (có dẫn chứng lý thuyết)

- **Mẫu mờ (blurry samples)** — chứng minh ở mục 7: decoder Gauss + mục tiêu L2 ⇒ hồi quy về trung bình có điều kiện.
- **Hậu nghiệm sụp đổ (posterior collapse)** — khi decoder quá mạnh (ví dụ autoregressive), nó tự mô hình hóa $x$ mà bỏ qua $z$; tối ưu kéo $q_\phi(z\mid x) \to p(z)$ làm số hạng KL về 0 và $z$ vô dụng.
- **Khoảng hở ELBO** — chỉ tối ưu chặn dưới; nếu họ $q_\phi$ (Gauss chéo) quá nghèo so với hậu nghiệm thật, gap lớn và likelihood thực bị bỏ xa (mục 4.2).

## 10. Vị trí trong bức tranh chung

VAE đặt nền cho tư duy "biến ẩn (latent variable) + suy diễn biến phân (variational inference)", tái xuất trong nhiều mô hình hiện đại — đặc biệt **latent diffusion** chạy diffusion trong không gian ẩn của một VAE.

> Bài tiếp theo — **GAN** — đi ngược lại hoàn toàn: bỏ hẳn likelihood và ELBO, học bằng đối kháng để đổi lấy mẫu sắc nét.

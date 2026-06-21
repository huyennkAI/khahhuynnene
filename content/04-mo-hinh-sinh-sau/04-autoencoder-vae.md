# Bộ tự mã hóa và VAE (Autoencoder & VAE)

> Bộ tự mã hóa (autoencoder) nén dữ liệu vào một **không gian ẩn (latent space)** nhỏ rồi tái tạo lại. Biến thể xác suất của nó — bộ tự mã hóa biến phân (Variational Autoencoder - VAE) — biến ý tưởng nén đó thành một **mô hình sinh (generative model)** đúng nghĩa thông qua suy diễn biến phân (variational inference):
>
> $$z \sim p(z) \;\longrightarrow\; x \sim p_\theta(x \mid z)$$
>
> Đây là nền tảng của tư duy "biến ẩn + suy diễn biến phân", và là thành phần lõi của các mô hình hiện đại như latent diffusion.

---

# 1. Động cơ ra đời

Mô hình tự hồi quy (autoregressive) lấy mẫu chậm với độ trễ $O(D)$. Normalizing flow lấy mẫu nhanh nhưng bị ràng buộc khả nghịch (invertibility) và buộc giữ nguyên số chiều.

Câu hỏi tự nhiên đặt ra:

> Liệu có thể giả định dữ liệu được sinh ra từ một vài **biến ẩn (latent variables)** đơn giản, rồi để mạng nơ-ron **tự do** ánh xạ từ biến ẩn sang dữ liệu hay không?

Trực giác sâu xa nằm ở chỗ dữ liệu nhiều chiều thường có cấu trúc thấp chiều ẩn bên dưới. Một khuôn mặt người, dù được mô tả bằng hàng trăm nghìn pixel, thực ra có thể tóm gọn bằng vài yếu tố: góc nghiêng, cảm xúc, ánh sáng, giới tính, tuổi.

Nếu học được không gian ẩn nhỏ đó, việc sinh dữ liệu mới trở nên đơn giản: chỉ cần lấy mẫu một điểm $z$ trong không gian ẩn rồi "giải mã (decode)" nó thành ảnh.

---

# 2. VAE khắc phục điều gì?

Mục tiêu của mô hình sinh là cực đại log-likelihood biên (marginal log-likelihood):

$$\log p_\theta(x), \qquad p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz$$

Tích phân này lấy trên toàn bộ không gian ẩn nên **bất khả thi** khi $z$ nhiều chiều. VAE giải quyết bằng ba lập luận lý thuyết:

**Né tích phân chuẩn hóa.** Thay vì tính trực tiếp $p_\theta(x)$, VAE tối ưu một **chặn dưới (lower bound)** của nó bằng suy diễn biến phân — số hạng ta sẽ gọi là ELBO.

**So với normalizing flow.** Flow buộc bộ giải mã song ánh và $\dim z = \dim x$. VAE cho bộ giải mã **tự do** và $\dim z \ll \dim x$ (nén thật sự), nên biểu diễn mạnh hơn. Đổi lại, VAE chỉ thu được chặn dưới, không phải likelihood chính xác.

**So với autoencoder thường.** VAE áp một **cấu trúc xác suất (probabilistic structure)** lên không gian ẩn, nhờ đó mới **lấy mẫu sinh mới** được — điều mà autoencoder thường không làm được.

Bảng so sánh nhanh:

| Tiêu chí | Autoregressive | Normalizing Flow | VAE |
| --- | --- | --- | --- |
| Likelihood | Chính xác | Chính xác | Chặn dưới (ELBO) |
| Tốc độ lấy mẫu | Chậm $O(D)$ | Một bước | Một bước |
| Số chiều ẩn | — | $\dim z = \dim x$ | $\dim z \ll \dim x$ |
| Ràng buộc decoder | — | Song ánh, khả nghịch | Tự do |

---

# 3. Từ Autoencoder đến VAE

Bộ tự mã hóa cổ điển gồm hai phần:

* **Bộ mã hóa (encoder)** $z = f_\phi(x)$ — nén dữ liệu thành mã ẩn.
* **Bộ giải mã (decoder)** $\hat{x} = g_\theta(z)$ — khôi phục dữ liệu từ mã ẩn.

Nó được huấn luyện bằng cực tiểu sai số tái tạo (reconstruction error):

$$\mathcal{L}_{\text{AE}} = \lVert x - g_\theta(f_\phi(x)) \rVert^2$$

**Vấn đề lý thuyết.** Không gian ẩn của autoencoder thường **không có cấu trúc xác suất**. Nếu ta lấy một điểm $z$ ngẫu nhiên rồi giải mã, kết quả thường ra rác — vì các $z$ "hợp lệ" nằm rải rác lộn xộn, không tuân theo phân phối nào.

VAE sửa lỗi này bằng hai thay đổi cốt lõi:

1. Áp một **phân phối tiên nghiệm (prior)** chuẩn lên không gian ẩn:
   $$p(z) = \mathcal{N}(0, I)$$
2. Để encoder trả về một **phân phối** thay vì một điểm duy nhất:
   $$q_\phi(z \mid x) = \mathcal{N}\big(\mu_\phi(x),\, \operatorname{diag}(\sigma_\phi^2(x))\big)$$

Khi đó việc sinh mẫu mới trở nên cực kỳ đơn giản: lấy $z \sim \mathcal{N}(0, I)$ rồi $x \sim p_\theta(x \mid z)$.

Bảng đối chiếu hai mô hình:

| | Autoencoder thường | VAE |
| --- | --- | --- |
| Encoder trả về | Một điểm $z$ | Một phân phối $q_\phi(z \mid x)$ |
| Cấu trúc không gian ẩn | Không có | Prior $\mathcal{N}(0, I)$ |
| Sinh mẫu mới | Không (ra rác) | Có (lấy mẫu từ prior) |

---

# 4. Suy ra ELBO (chặn dưới biến phân)

Ta muốn cực đại log-likelihood biên $\log p_\theta(x)$ nhưng tích phân chuẩn hóa bất khả thi. Cách thoát ra là đưa vào một **phân phối hậu nghiệm xấp xỉ (approximate posterior)** $q_\phi(z \mid x)$ và xây một chặn dưới gọi là **ELBO (Evidence Lower Bound)**.

Có hai con đường dẫn tới cùng một kết quả.

---

## 4.1. Cách 1 — bất đẳng thức Jensen

Đưa $q_\phi(z \mid x)$ bất kỳ vào tích phân rồi nhân và chia cho chính nó:

$$
\begin{aligned}
\log p_\theta(x)
&= \log \int p_\theta(x, z)\, dz
 = \log \int q_\phi(z \mid x)\, \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\, dz \\
&= \log \mathbb{E}_{q_\phi(z \mid x)}\!\left[\frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] \\
&\ge \mathbb{E}_{q_\phi(z \mid x)}\!\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] \qquad \text{(Jensen, vì } \log \text{ lõm)} \\
&= \underbrace{\mathbb{E}_{q_\phi}\big[\log p_\theta(x \mid z)\big] - D_{\mathrm{KL}}\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\textstyle \mathcal{L}_{\text{ELBO}}(\theta, \phi; x)}
\end{aligned}
$$

Bước cuối tách logarit của thương ra hai phần:

$$\log \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)} = \log p_\theta(x \mid z) - \log \frac{q_\phi(z \mid x)}{p(z)}$$

trong đó số hạng thứ hai khi lấy kỳ vọng theo $q_\phi$ chính là $D_{\mathrm{KL}}\big(q_\phi(z \mid x) \,\|\, p(z)\big)$.

Bất đẳng thức Jensen áp dụng được vì $\log$ là hàm lõm, nên $\log \mathbb{E}[\cdot] \ge \mathbb{E}[\log(\cdot)]$.

---

## 4.2. Cách 2 — phân rã chính xác (cho thấy khoảng hở)

Cách Jensen cho ta chặn dưới nhưng không nói rõ ta đã "mất" bao nhiêu. Cách thứ hai trả lời câu hỏi đó một cách chính xác.

**Mệnh đề.** Với mọi $q_\phi$:

$$\log p_\theta(x) = \mathcal{L}_{\text{ELBO}}(\theta, \phi; x) + D_{\mathrm{KL}}\big(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x)\big)$$

**Chứng minh.** Xuất phát từ KL giữa hậu nghiệm xấp xỉ và hậu nghiệm thật $p_\theta(z \mid x) = \dfrac{p_\theta(x, z)}{p_\theta(x)}$:

$$
\begin{aligned}
D_{\mathrm{KL}}\big(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x)\big)
&= \mathbb{E}_{q_\phi}\!\left[\log \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)}\right]
 = \mathbb{E}_{q_\phi}\!\left[\log \frac{q_\phi(z \mid x)\, p_\theta(x)}{p_\theta(x, z)}\right] \\
&= \mathbb{E}_{q_\phi}\!\left[\log \frac{q_\phi(z \mid x)}{p_\theta(x, z)}\right] + \log p_\theta(x) \\
&= -\mathcal{L}_{\text{ELBO}} + \log p_\theta(x)
\end{aligned}
$$

Ở bước giữa, $\log p_\theta(x)$ tách ra ngoài kỳ vọng vì nó không phụ thuộc $z$. Số hạng còn lại chính là $-\mathcal{L}_{\text{ELBO}}$ theo định nghĩa. Chuyển vế được điều phải chứng minh. $\blacksquare$

> **Hệ quả quan trọng.** Vì $D_{\mathrm{KL}} \ge 0$, ta có ngay $\log p_\theta(x) \ge \mathcal{L}_{\text{ELBO}}$, và **khoảng hở (gap) đúng bằng** KL giữa hậu nghiệm xấp xỉ và hậu nghiệm thật. Tối đa ELBO theo $\phi$ vừa kéo $q_\phi$ tiến gần hậu nghiệm thật, vừa siết chặt chặn dưới; tối đa theo $\theta$ làm tăng likelihood. ELBO chặt (gap $= 0$) khi và chỉ khi $q_\phi = p_\theta(z \mid x)$.

---

## 4.3. Diễn giải hai số hạng ELBO

Viết lại ELBO dưới dạng hai số hạng có ý nghĩa rõ ràng:

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x \mid z)\big]}_{\text{tái tạo (reconstruction)}} - \underbrace{D_{\mathrm{KL}}\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\text{điều chuẩn (regularization)}}$$

* **Số hạng tái tạo** buộc $z$ được giải mã lại đúng thành $x$ — đảm bảo mã ẩn giữ đủ thông tin.
* **Số hạng KL** kéo phân phối ẩn về gần prior $\mathcal{N}(0, I)$, giữ cho không gian ẩn liên tục và lấy mẫu được. Đây chính là thứ mà autoencoder thường thiếu.

Hai số hạng này kéo nhau theo hai hướng đối nghịch: tái tạo muốn $z$ mang nhiều thông tin về $x$, còn KL muốn $z$ trông như nhiễu chuẩn. Sự cân bằng giữa chúng tạo ra một không gian ẩn vừa hữu ích vừa lấy mẫu được.

---

# 5. KL với prior chuẩn có dạng đóng (closed form)

Số hạng KL trong ELBO có thể tính được **chính xác** khi cả hai phân phối đều là Gauss. Với $q_\phi(z \mid x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$ và $p(z) = \mathcal{N}(0, I)$ trong $d$ chiều:

$$D_{\mathrm{KL}}\big(\mathcal{N}(\mu, \operatorname{diag}(\sigma^2)) \,\|\, \mathcal{N}(0, I)\big) = \frac{1}{2}\sum_{j=1}^{d}\Big(\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2\Big)$$

Công thức dạng đóng này là lý do thực tiễn khiến VAE huấn luyện được dễ dàng: không cần ước lượng Monte Carlo cho số hạng KL.

---

## 5.1. Chứng minh (một chiều, rồi cộng theo chiều)

**Mệnh đề.** Với hai Gauss một chiều $q = \mathcal{N}(\mu, \sigma^2)$ và $p = \mathcal{N}(0, 1)$:

$$D_{\mathrm{KL}}(q \,\|\, p) = \tfrac{1}{2}\big(\mu^2 + \sigma^2 - 1 - \log \sigma^2\big)$$

**Chứng minh.** Viết KL dưới dạng kỳ vọng của hiệu hai log-mật độ:

$$
\begin{aligned}
D_{\mathrm{KL}}(q \,\|\, p)
&= \mathbb{E}_q\!\left[\log q(z) - \log p(z)\right] \\
&= \mathbb{E}_q\!\left[-\tfrac{1}{2}\log(2\pi\sigma^2) - \tfrac{(z-\mu)^2}{2\sigma^2} + \tfrac{1}{2}\log(2\pi) + \tfrac{z^2}{2}\right] \\
&= -\tfrac{1}{2}\log\sigma^2 - \tfrac{1}{2\sigma^2}\underbrace{\mathbb{E}_q[(z-\mu)^2]}_{=\,\sigma^2} + \tfrac{1}{2}\underbrace{\mathbb{E}_q[z^2]}_{=\,\mu^2 + \sigma^2} \\
&= -\tfrac{1}{2}\log\sigma^2 - \tfrac{1}{2} + \tfrac{1}{2}(\mu^2 + \sigma^2) \\
&= \tfrac{1}{2}\big(\mu^2 + \sigma^2 - 1 - \log\sigma^2\big)
\end{aligned}
$$

Ở đây ta dùng $\mathbb{E}_q[(z-\mu)^2] = \sigma^2$ (định nghĩa phương sai) và $\mathbb{E}_q[z^2] = \operatorname{Var}_q(z) + (\mathbb{E}_q[z])^2 = \sigma^2 + \mu^2$.

Vì $q_\phi$ có hiệp phương sai chéo (các chiều độc lập), KL nhiều chiều cộng tách thành tổng các KL một chiều, cho ra công thức tổng quát ở trên. $\blacksquare$

---

# 6. Mẹo tái tham số hóa (reparameterization trick)

Số hạng tái tạo $\mathbb{E}_{q_\phi(z \mid x)}[\cdots]$ phụ thuộc $\phi$ qua **phân phối lấy kỳ vọng**. Vấn đề: ta không thể lan truyền ngược (backpropagation) trực tiếp qua một phép lấy mẫu ngẫu nhiên — gradient không "chui" qua được nút ngẫu nhiên.

VAE giải quyết bằng cách viết lại phép lấy mẫu thành một hàm khả vi của một nhiễu độc lập:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

Ý tưởng: tách phần ngẫu nhiên ($\epsilon$) ra khỏi phần tham số ($\mu_\phi, \sigma_\phi$). Nhiễu $\epsilon$ không phụ thuộc $\phi$, còn phép biến đổi từ $\epsilon$ sang $z$ thì khả vi theo $\phi$.

---

## 6.1. Vì sao mẹo này hợp lệ?

**Mệnh đề.** Phép tái tham số hóa giữ nguyên phân phối của $z$ và cho phép đưa gradient vào trong kỳ vọng:

$$\nabla_\phi \mathbb{E}_{q_\phi}\big[h(z)\big] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\big[\nabla_\phi\, h(\mu_\phi + \sigma_\phi \odot \epsilon)\big]$$

**Chứng minh.** Phép biến đổi affine một Gauss chuẩn $\epsilon \sim \mathcal{N}(0, I)$ cho ra đúng $\mu_\phi + \sigma_\phi \odot \epsilon \sim \mathcal{N}(\mu_\phi, \operatorname{diag}(\sigma_\phi^2)) = q_\phi(z \mid x)$, nên phân phối của $z$ không thay đổi. Do đó ta đổi biến trong kỳ vọng:

$$
\begin{aligned}
\nabla_\phi \mathbb{E}_{q_\phi}\big[h(z)\big]
&= \nabla_\phi \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\big[h(\mu_\phi + \sigma_\phi \odot \epsilon)\big] \\
&= \mathbb{E}_{\epsilon}\big[\nabla_\phi\, h(\mu_\phi + \sigma_\phi \odot \epsilon)\big]
\end{aligned}
$$

Bước cuối hoán đổi được đạo hàm và kỳ vọng vì phân phối lấy kỳ vọng (của $\epsilon$) **không còn phụ thuộc $\phi$** — đây chính là điều mà phép tái tham số hóa đạt được. $\blacksquare$

Ước lượng Monte Carlo của vế phải có **phương sai thấp hơn nhiều** so với ước lượng score-function (REINFORCE). Đây là lý do lý thuyết khiến VAE huấn luyện hiệu quả trong thực tế.

Triển khai thực tế chỉ vài dòng:

```python
mu, log_var = encoder(x)
eps = torch.randn_like(mu)
z = mu + torch.exp(0.5 * log_var) * eps
x_hat = decoder(z)
recon = mse(x_hat, x)
kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
loss = recon + kld
```

---

# 7. Vì sao mẫu VAE bị mờ?

Đây là dẫn chứng lý thuyết cho nhược điểm chính của VAE. Giả định bộ giải mã là Gauss:

$$p_\theta(x \mid z) = \mathcal{N}(g_\theta(z), \sigma^2 I)$$

Khi đó $\log p_\theta(x \mid z)$ tỉ lệ với $-\lVert x - g_\theta(z) \rVert^2$, nên số hạng tái tạo trở thành sai số bình phương (L2).

**Mệnh đề.** Nghiệm tối ưu của bộ giải mã dưới mục tiêu L2 là **trung bình có điều kiện**:

$$g_\theta(z)^{*} = \mathbb{E}[x \mid z]$$

**Chứng minh.** Cực tiểu kỳ vọng L2 theo một hằng số dự đoán $c$ (ứng với một $z$ cố định):

$$
\begin{aligned}
\nabla_c\, \mathbb{E}\big[\lVert x - c \rVert^2 \mid z\big]
&= \mathbb{E}\big[-2(x - c) \mid z\big] = 0 \\
&\Longrightarrow\; c = \mathbb{E}[x \mid z]
\end{aligned}
$$

Vì hàm $\lVert x - c \rVert^2$ lồi theo $c$, điểm dừng này là cực tiểu toàn cục. $\blacksquare$

Hệ quả: khi một $z$ tương ứng với nhiều ảnh sắc nét hợp lý (do hậu nghiệm xấp xỉ không hoàn hảo), nghiệm tối ưu là **trung bình** của chúng — và trung bình của các ảnh sắc nét là một ảnh **mờ**.

GAN tránh được hiện tượng này vì mục tiêu đối kháng phạt thẳng vào những ảnh trung bình hóa, ép mô hình sinh ra mẫu sắc nét.

---

# 8. Ưu điểm

* **Huấn luyện ổn định** — chỉ tối ưu một hàm mục tiêu trơn (ELBO), không có động lực đối kháng (adversarial dynamics) như GAN; gradient lại có phương sai thấp nhờ reparameterization (mục 6.1).
* **Không gian ẩn diễn giải được** — số hạng KL ép $q_\phi$ về prior trơn, nên nội suy (interpolation) trong không gian $z$ cho chuyển tiếp mượt giữa các mẫu.
* **Cho ước lượng likelihood** — qua ELBO, hữu ích cho phát hiện bất thường (anomaly / OOD detection).
* **Nén thật sự** — $\dim z \ll \dim x$, khác hẳn flow phải giữ nguyên số chiều.

---

# 9. Nhược điểm

* **Mẫu mờ (blurry samples)** — đã chứng minh ở mục 7: bộ giải mã Gauss kết hợp mục tiêu L2 dẫn tới hồi quy về trung bình có điều kiện.
* **Hậu nghiệm sụp đổ (posterior collapse)** — khi bộ giải mã quá mạnh (ví dụ autoregressive), nó tự mô hình hóa $x$ mà bỏ qua $z$; quá trình tối ưu kéo $q_\phi(z \mid x) \to p(z)$ làm số hạng KL về $0$ và biến $z$ thành vô dụng.
* **Khoảng hở ELBO** — VAE chỉ tối ưu chặn dưới; nếu họ phân phối $q_\phi$ (Gauss chéo) quá nghèo so với hậu nghiệm thật, gap lớn (mục 4.2) và likelihood thực bị bỏ xa.

---

# 10. Tổng kết

VAE không phát minh ra một dạng phân phối mới, mà kết hợp hai ý tưởng kinh điển: **biến ẩn (latent variable)** và **suy diễn biến phân (variational inference)**. Toàn bộ mô hình xoay quanh một đẳng thức duy nhất:

$$\log p_\theta(x) = \mathcal{L}_{\text{ELBO}}(\theta, \phi; x) + D_{\mathrm{KL}}\big(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x)\big)$$

Vì hậu nghiệm thật bất khả tính, VAE tối ưu chặn dưới ELBO thay vì likelihood chính xác. Phép tái tham số hóa biến phép lấy mẫu ngẫu nhiên thành khả vi, cho phép huấn luyện toàn bộ bằng lan truyền ngược thông thường.

VAE đổi **độ chính xác của likelihood** lấy **một không gian ẩn nén được, trơn, và lấy mẫu nhanh một bước**. Cái giá phải trả là mẫu mờ và khoảng hở ELBO. Tư duy của nó tái xuất ở khắp nơi — đặc biệt **latent diffusion** chạy quá trình diffusion ngay trong không gian ẩn của một VAE.

> Bài tiếp theo — **GAN** — đi ngược lại hoàn toàn: bỏ hẳn likelihood và ELBO, học bằng đối kháng (adversarial learning) để đổi lấy những mẫu sắc nét mà VAE không sinh được.

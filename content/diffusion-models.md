# Mô hình khuếch tán (Diffusion)

> Mô hình khuếch tán (diffusion model) học cách **khử nhiễu dần dần (iterative denoising)**: bắt đầu từ nhiễu thuần (pure noise) và từng bước biến nó thành mẫu sắc nét. Đây là nền tảng của Stable Diffusion, DALL·E và phần lớn hệ sinh ảnh hiện đại.

## 1. Trực giác ra đời

GAN cho mẫu đẹp nhưng huấn luyện bất ổn (bài toán điểm yên ngựa) và hay sụp đổ chế độ. VAE ổn định nhưng mẫu mờ. Trực giác của diffusion:

> Sinh một bức ảnh hoàn chỉnh trong **một bước** là quá khó (đó là chỗ GAN chật vật). Nhưng **khử một chút nhiễu** thì dễ. Vậy hãy chia bài toán khó thành hàng trăm bước nhỏ dễ.

Đây là tư duy "chia để trị" trên miền xác suất: thay vì học một phép nhảy lớn từ nhiễu sang dữ liệu, ta học rất nhiều phép nhảy nhỏ, mỗi phép chỉ cần khử nhiễu nhẹ.

## 2. Khắc phục điều gì (lập luận lý thuyết)

- **So với GAN:** huấn luyện là **hồi quy bình phương (regression)**, có nghiệm tối ưu xác định và gradient luôn có ý nghĩa — không có động lực đối kháng nên không mất ổn định, không mode collapse. Mục tiêu MLE-cảm hứng ⇒ mode-covering ⇒ đa dạng cao.
- **So với VAE:** thay vì một bước giải mã (gây mờ do trung bình hóa), diffusion tinh chỉnh qua hàng trăm bước, mỗi bước thêm chi tiết ⇒ sắc nét.
- **Cái giá:** lấy mẫu chậm vì cần nhiều bước suy luận.

## 3. Hai quá trình

### 3.1. Quá trình thuận (forward / diffusion process)

Dần phá hủy dữ liệu bằng cách cộng nhiễu Gauss qua $T$ bước theo lịch nhiễu (noise schedule) $\beta_t \in (0, 1)$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left( x_t; \sqrt{1 - \beta_t}\, x_{t-1}, \; \beta_t I \right)$$

**Tính chất nhảy bước (closed-form).** Đặt $\alpha_t = 1 - \beta_t$ và $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$. Khi đó:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t; \sqrt{\bar{\alpha}_t}\, x_0, \; (1 - \bar{\alpha}_t) I\right) \;\Leftrightarrow\; x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)$$

**Chứng minh (quy nạp).** Giả sử $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\,\bar{\epsilon}$. Thay vào định nghĩa $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{\beta_t}\,\epsilon'$:

$$x_t = \sqrt{\alpha_t \bar{\alpha}_{t-1}}\, x_0 + \underbrace{\sqrt{\alpha_t (1 - \bar{\alpha}_{t-1})}\,\bar{\epsilon} + \sqrt{1 - \alpha_t}\,\epsilon'}_{\text{tổng hai Gauss độc lập}}$$

Tổng hai Gauss độc lập trung bình 0 có phương sai cộng dồn: $\alpha_t(1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t\bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t$. Vậy $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon$. $\blacksquare$

Khi $t \to T$, lịch nhiễu khiến $\bar{\alpha}_t \to 0$, nên $x_T \to \mathcal{N}(0, I)$ — nhiễu thuần, mất hết thông tin.

### 3.2. Quá trình nghịch (reverse process)

Mô hình học đảo ngược, từng bước đưa nhiễu về dữ liệu:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left( x_{t-1};\, \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t) \right)$$

**Cơ sở lý thuyết:** Feller (1949) chỉ ra rằng nếu mỗi bước thuận đủ nhỏ ($\beta_t \ll 1$), thì bước nghịch $q(x_{t-1}\mid x_t)$ **cũng xấp xỉ Gauss**. Đây là lý do giả định Gauss cho $p_\theta$ là chính đáng — và là lý do diffusion cần nhiều bước nhỏ.

## 4. Hậu nghiệm thuận có dạng đóng

Mấu chốt của việc huấn luyện: tuy $q(x_{t-1}\mid x_t)$ khó, nhưng khi **biết thêm $x_0$**, hậu nghiệm thuận lại là Gauss tường minh:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\, \tilde{\beta}_t I\right)$$

với (suy ra bằng quy tắc Bayes $q(x_{t-1}\mid x_t, x_0) \propto q(x_t\mid x_{t-1})\, q(x_{t-1}\mid x_0)$ và gom số mũ bậc hai):

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t, \qquad \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\,\beta_t$$

## 5. Lý thuyết: từ ELBO đến mục tiêu dự đoán nhiễu

### 5.1. Chặn dưới biến phân

Giống VAE, diffusion tối ưu ELBO trên $\log p_\theta(x_0)$, coi $x_{1:T}$ là biến ẩn:

$$\log p_\theta(x_0) \ge \mathbb{E}_q\!\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right] =: -\mathcal{L}$$

Khai triển và sắp xếp lại (dùng tính Markov và đưa vào $x_0$ để tạo hậu nghiệm dạng đóng ở mục 4), thu được:

$$\mathcal{L} = \mathbb{E}_q\!\Big[ \underbrace{D_{\mathrm{KL}}(q(x_T\mid x_0)\,\|\,p(x_T))}_{L_T:\ \text{không tham số}} + \sum_{t>1} \underbrace{D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}} \underbrace{-\log p_\theta(x_0\mid x_1)}_{L_0} \Big]$$

### 5.2. Mỗi số hạng KL rút về khoảng cách trung bình

$L_{t-1}$ là KL giữa **hai Gauss cùng phương sai** ($q$ dạng đóng ở mục 4, $p_\theta$ ta đặt cùng $\Sigma = \tilde{\beta}_t I$). KL giữa hai Gauss cùng hiệp phương sai chỉ còn khoảng cách bình phương giữa hai trung bình:

$$L_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\tilde{\beta}_t}\big\lVert \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\big\rVert^2\right] + C$$

### 5.3. Tham số hóa dự đoán nhiễu — bước then chốt

Từ mục 3.1, $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\big(x_t - \sqrt{1 - \bar{\alpha}_t}\,\epsilon\big)$. Thế vào $\tilde{\mu}_t$ rồi rút gọn, ta được dạng:

$$\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\epsilon\right)$$

Vì vậy ta để mạng **dự đoán nhiễu** $\epsilon_\theta(x_t, t)$ và đặt $\mu_\theta$ cùng dạng. Hiệu hai trung bình rút về hiệu hai nhiễu:

$$\big\lVert \tilde{\mu}_t - \mu_\theta \big\rVert^2 = \frac{\beta_t^2}{\alpha_t(1 - \bar{\alpha}_t)}\big\lVert \epsilon - \epsilon_\theta(x_t, t)\big\rVert^2$$

### 5.4. Mục tiêu rút gọn (DDPM, Ho et al. 2020)

Bỏ các hệ số đầu (thực nghiệm cho thấy bỏ đi còn ổn định hơn), mục tiêu trở thành:

$$\boxed{\;\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\, \epsilon,\, t} \left[ \big\lVert \epsilon - \epsilon_\theta(x_t, t) \big\rVert^2 \right], \quad x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon\;}$$

> Diễn giải đẹp: huấn luyện diffusion chỉ là **dạy một mạng đoán xem đã thêm bao nhiêu nhiễu** vào ảnh tại mức $t$ bất kỳ. Đây là hồi quy L2 thuần túy — ổn định, không đối kháng, nghiệm tối ưu xác định.

```python
t = torch.randint(0, T, (batch,))
eps = torch.randn_like(x0)
x_t = sqrt_acp[t] * x0 + sqrt_one_minus_acp[t] * eps
loss = mse(model(x_t, t), eps)
```

Mạng $\epsilon_\theta$ thường là **U-Net** với time embedding để biết đang ở mức nhiễu nào.

## 6. Liên hệ score matching (cầu nối tới EBM)

Có thể chứng minh dự đoán nhiễu tương đương ước lượng **hàm điểm (score function)** của phân phối đã làm nhiễu:

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\,x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

Tức $\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t}\,\nabla_{x_t}\log q(x_t)$. Vì score $\nabla_x \log p(x)$ làm hằng số chuẩn hóa $Z$ biến mất, đây chính là **denoising score matching** — cách huấn luyện một mô hình dựa trên năng lượng (energy-based model) mà ổn định và mở rộng được. Cách nhìn liên tục (SDE) thống nhất toàn bộ diffusion dưới một khung.

## 7. Lấy mẫu

Bắt đầu từ $x_T \sim \mathcal{N}(0, I)$, lặp bước khử nhiễu từ $t = T$ về $1$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z, \qquad z \sim \mathcal{N}(0, I)$$

- **DDPM** — lấy mẫu ngẫu nhiên ($\sigma_t > 0$), cần hàng trăm đến hàng nghìn bước.
- **DDIM** — quá trình xác định ($\sigma_t = 0$), rút còn vài chục bước nhờ bỏ tính Markov.

## 8. Sinh có điều kiện (conditional generation)

- **Classifier-free guidance:** huấn luyện đồng thời có và không điều kiện, rồi ngoại suy dự đoán nhiễu: $\tilde{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w\big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\big)$, với $w$ điều khiển mức bám prompt.
- **Latent Diffusion (Stable Diffusion):** chạy khuếch tán trong không gian ẩn nén (latent space) của một autoencoder, giảm mạnh chi phí tính toán so với khuếch tán trực tiếp trên điểm ảnh.

## 9. Ưu điểm (có dẫn chứng lý thuyết)

- **Chất lượng và đa dạng mẫu dẫn đầu** — mục tiêu mode-covering (gần MLE) nên phủ mode tốt hơn GAN; nhiều bước tinh chỉnh cho độ sắc nét cao.
- **Huấn luyện ổn định** — mục 5.4 cho thấy chỉ là hồi quy L2, không mode collapse, không điểm yên ngựa.
- **Khung lý thuyết phong phú** — thống nhất ELBO, score matching, SDE; nối liền với EBM.

## 10. Nhược điểm (có dẫn chứng lý thuyết)

- **Lấy mẫu chậm** — cần $T$ lần truyền U-Net (mục 7); đây là cái giá trực tiếp của việc chia bài toán thành nhiều bước nhỏ.
- **Tốn tính toán** ở độ phân giải cao — giảm bớt nhờ latent diffusion nhưng vẫn nặng hơn GAN khi suy luận.
- **Không cho likelihood chính xác** — chỉ tối ưu ELBO (chặn dưới), giống VAE.

## 11. Vị trí trong bức tranh chung

Diffusion hiện dẫn đầu sinh ảnh và đang lan sang âm thanh, video, sinh phân tử. Bí quyết, đúng như chứng minh ở mục 5–6: biến một bài toán sinh khó thành **chuỗi bài toán khử nhiễu dễ và ổn định**, và đó cũng là cách huấn luyện EBM hiệu quả nhất tới nay.

> Bài cuối chuyên mục — **Energy-Based Models** — trình bày khung tổng quát nhất, nơi diffusion lộ ra là một trường hợp đặc biệt được huấn luyện khéo léo.

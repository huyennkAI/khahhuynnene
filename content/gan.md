# Mạng đối kháng sinh (GAN)

> Mạng đối kháng sinh (Generative Adversarial Network - GAN) huấn luyện hai mạng cạnh tranh: bộ sinh (generator) tạo dữ liệu giả như thật, bộ phân biệt (discriminator) cố vạch trần. Cuộc đua đẩy chất lượng mẫu lên rất cao mà không cần tính mật độ.

## 1. Trực giác ra đời

VAE và normalizing flow đều cố **mô hình hóa mật độ (density)** $p(x)$ rõ ràng, và phải trả giá: VAE cho mẫu mờ (do mục tiêu L2 hồi quy về trung bình), flow bị ràng buộc khả nghịch. Ý tưởng đột phá của Goodfellow (2014):

> Để sinh ảnh đẹp, ta đâu cần biết $p(x)$ là gì. Ta chỉ cần một **giám khảo (critic)** không phân biệt được thật với giả.

Đây là chuyển từ mô hình sinh **tường minh (explicit)** sang **ngầm định (implicit)**: ta không bao giờ viết ra $p_\theta(x)$, chỉ định nghĩa cách lấy mẫu $x = G(z),\ z \sim p(z)$.

## 2. Khắc phục điều gì (lập luận lý thuyết)

- **Né hoàn toàn mật độ và hằng số chuẩn hóa** — không cần likelihood, không cần $Z_\theta$, không cần biến đổi khả nghịch.
- **Mẫu sắc nét (sharp samples):** sẽ chứng minh ở mục 3 rằng GAN cực tiểu một phân kỳ (JS divergence) thay vì L2; nó không trung bình hóa các đáp án hợp lý nên không bị mờ như VAE.
- **Lấy mẫu một bước (one-step sampling):** chỉ một lần truyền qua $G$.

## 3. Lý thuyết: mục tiêu minimax và nghiệm tối ưu

Hàm giá trị (value function) của trò chơi:

$$\min_{G} \max_{D} \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]$$

Gọi $p_g$ là phân phối do $G$ sinh ra. Đổi biến kỳ vọng thứ hai sang không gian $x$: $\mathbb{E}_{z}[\,\cdot\,] = \mathbb{E}_{x\sim p_g}[\,\cdot\,]$.

### 3.1. Bước 1 — Discriminator tối ưu (chứng minh)

**Bổ đề.** Với $G$ cố định, discriminator tối ưu là:

$$D^{*}(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

**Chứng minh.** Viết $V$ thành một tích phân theo $x$:

$$V(D, G) = \int_x \Big[ p_{\text{data}}(x) \log D(x) + p_g(x) \log\big(1 - D(x)\big) \Big] dx$$

Vì các điểm $x$ độc lập, cực đại hóa điểm-theo-điểm hàm $\varphi(D) = a\log D + b\log(1 - D)$ với $a = p_{\text{data}}(x),\ b = p_g(x)$. Lấy đạo hàm và cho bằng 0:

$$\varphi'(D) = \frac{a}{D} - \frac{b}{1 - D} = 0 \;\Rightarrow\; a(1 - D) = bD \;\Rightarrow\; D^{*} = \frac{a}{a + b}$$

Đạo hàm bậc hai $\varphi''(D) = -\frac{a}{D^2} - \frac{b}{(1-D)^2} < 0$ nên đây là cực đại. $\blacksquare$

### 3.2. Bước 2 — Thay vào, hiện ra JS divergence (chứng minh)

Thay $D^{*}$ vào $V$:

$$
\begin{aligned}
C(G) := V(D^{*}, G) &= \mathbb{E}_{p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}}{p_{\text{data}} + p_g}\right] + \mathbb{E}_{p_g}\!\left[\log \frac{p_g}{p_{\text{data}} + p_g}\right]
\end{aligned}
$$

Thêm và bớt $\log 2$ trong mỗi kỳ vọng (viết $\frac{p_{\text{data}}}{p_{\text{data}}+p_g} = \frac{1}{2}\cdot\frac{p_{\text{data}}}{(p_{\text{data}}+p_g)/2}$):

$$
\begin{aligned}
C(G) &= -\log 2 \cdot \underbrace{\mathbb{E}_{p_{\text{data}}}[1]}_{=1} + \mathbb{E}_{p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}}{(p_{\text{data}} + p_g)/2}\right] \\
&\quad -\log 2 \cdot \underbrace{\mathbb{E}_{p_g}[1]}_{=1} + \mathbb{E}_{p_g}\!\left[\log \frac{p_g}{(p_{\text{data}} + p_g)/2}\right] \\
&= -\log 4 + D_{\mathrm{KL}}\!\Big(p_{\text{data}} \,\Big\|\, \tfrac{p_{\text{data}} + p_g}{2}\Big) + D_{\mathrm{KL}}\!\Big(p_g \,\Big\|\, \tfrac{p_{\text{data}} + p_g}{2}\Big)
\end{aligned}
$$

Theo định nghĩa phân kỳ Jensen–Shannon (Jensen–Shannon divergence) $D_{\mathrm{JS}}(p\,\|\,q) = \frac{1}{2}D_{\mathrm{KL}}(p\,\|\,m) + \frac{1}{2}D_{\mathrm{KL}}(q\,\|\,m)$ với $m = \frac{p+q}{2}$:

$$\boxed{\;C(G) = -\log 4 + 2\, D_{\mathrm{JS}}\big(p_{\text{data}} \,\|\, p_g\big)\;}$$

### 3.3. Bước 3 — Nghiệm tối ưu toàn cục

Vì $D_{\mathrm{JS}} \ge 0$ và bằng 0 **khi và chỉ khi** $p_{\text{data}} = p_g$, ta có $C(G) \ge -\log 4$, đạt cực tiểu duy nhất tại:

$$p_g = p_{\text{data}}, \qquad D^{*}(x) = \tfrac{1}{2}\ \forall x$$

Tại đó giám khảo chỉ còn đoán mò. **Kết luận: GAN cực tiểu phân kỳ JS giữa phân phối sinh và phân phối thật.** $\blacksquare$

## 4. Vì sao GAN sắc nét còn VAE mờ? (so sánh lý thuyết)

VAE cực tiểu một mục tiêu liên quan forward KL với số hạng tái tạo L2 ⇒ hồi quy về trung bình ⇒ mờ. GAN cực tiểu JS divergence, **không có số hạng L2 trung bình hóa**: nếu $G$ sinh ảnh "trung bình mờ", $D$ phát hiện ngay vì ảnh mờ không thuộc $p_{\text{data}}$, đẩy $G$ tránh xa. Đây là dẫn chứng lý thuyết cho ưu thế độ sắc nét.

## 5. Huấn luyện thực tế và non-saturating loss

```text
Lặp:
  1. Cố định G, cập nhật D trên một lô mẫu thật và mẫu giả (vài bước)
  2. Cố định D, cập nhật G để D đánh giá mẫu giả là thật
```

### 5.1. Vì sao đổi mục tiêu của G?

Mục tiêu gốc $\min_G \mathbb{E}_z[\log(1 - D(G(z)))]$ có vấn đề gradient. Khi $G$ còn kém, $D(G(z)) \approx 0$, đạo hàm của $\log(1 - D)$ theo tham số $G$ gần như **biến mất (vanishing gradient)**:

$$\frac{d}{dD}\log(1 - D)\Big|_{D \to 0} = \frac{-1}{1 - D}\Big|_{D\to 0} = -1 \quad\text{(nhỏ)}, \quad\text{trong khi}\quad \frac{d}{dD}\log D\Big|_{D\to 0} = \frac{1}{D} \to \infty$$

Vì vậy dùng mục tiêu **non-saturating** $\max_G \mathbb{E}_z[\log D(G(z))]$: cùng điểm bất động nhưng gradient mạnh khi $G$ còn yếu.

## 6. Những khó khăn kinh điển (giải thích lý thuyết)

- **Sụp đổ chế độ (mode collapse):** mục tiêu chỉ yêu cầu lừa được $D$, không yêu cầu phủ hết các mode; $G$ có thể dồn khối lượng vào vài mode dễ lừa. Đây là biểu hiện của bản chất mode-seeking, đối lập với forward-KL mode-covering của MLE.
- **Gradient biến mất khi hai phân phối không chồng lấn:** nếu $p_{\text{data}}$ và $p_g$ nằm trên hai đa tạp tách rời, $D_{\mathrm{JS}}$ **hằng bằng $\log 2$** bất kể chúng cách nhau bao xa ⇒ gradient bằng 0, $G$ không biết đi hướng nào. Theo manifold hypothesis, dữ liệu ảnh nằm trên đa tạp mỏng nên tình huống này rất hay xảy ra.
- **Mất ổn định:** bài toán là tìm điểm yên ngựa (saddle point / Nash equilibrium), không phải cực tiểu hàm — các thuật toán gradient có thể dao động vòng quanh thay vì hội tụ.

## 7. WGAN — khắc phục gradient biến mất (lý thuyết)

WGAN thay JS divergence bằng khoảng cách Wasserstein-1 (Earth Mover's Distance), với dạng đối ngẫu Kantorovich–Rubinstein:

$$W(p_{\text{data}}, p_g) = \sup_{\lVert f \rVert_L \le 1} \;\mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]$$

trong đó $f$ chạy trên các hàm 1-Lipschitz. Ưu thế lý thuyết then chốt: **Wasserstein cho gradient có ý nghĩa ngay cả khi hai phân phối không chồng lấn**, vì nó đo "công vận chuyển khối lượng" — một khoảng cách hình học, không phải tỉ số mật độ. Ràng buộc Lipschitz được thực thi bằng cắt trọng số (weight clipping) hoặc phạt gradient (gradient penalty, WGAN-GP).

## 8. Các cải tiến quan trọng

| Biến thể | Đóng góp |
| --- | --- |
| **DCGAN** | Kiến trúc tích chập, ổn định huấn luyện ảnh |
| **WGAN / WGAN-GP** | Wasserstein, gradient mượt, giảm mode collapse |
| **Conditional GAN (cGAN)** | Sinh có điều kiện theo nhãn $y$ |
| **StyleGAN** | Tiêm phong cách theo từng tầng, khuôn mặt cực thực |

## 9. Ưu điểm (có dẫn chứng lý thuyết)

- **Mẫu sắc nét nhất** trong các mô hình tường minh 2014–2020 — dẫn chứng ở mục 4 (cực tiểu JS, không trung bình hóa L2).
- **Lấy mẫu nhanh** — một lần truyền qua $G$, $O(1)$.
- **Khung linh hoạt** — chỉ cần $G$ khả vi; không ràng buộc mật độ, khả nghịch hay biến ẩn.

## 10. Nhược điểm (có dẫn chứng lý thuyết)

- **Huấn luyện bất ổn** — bài toán điểm yên ngựa (mục 6), nhạy siêu tham số.
- **Mode collapse** — mục tiêu mode-seeking không phạt việc bỏ mode (mục 6); recall thấp khi đo precision/recall.
- **Không có likelihood** — không đánh giá định lượng bằng bits-per-dim được, phải dựa FID.
- **Gradient biến mất** khi hỗ trợ (support) không chồng lấn — mục 6, chính là động lực ra đời của WGAN.

## 11. Vị trí trong bức tranh chung

GAN từng là tiêu chuẩn vàng cho sinh ảnh, trước khi mô hình khuếch tán (diffusion) vươn lên dẫn đầu nhờ ổn định và đa dạng hơn.

> Bài tiếp theo — **Diffusion** — đạt chất lượng GAN nhưng huấn luyện ổn định như MLE, bằng cách chia bài toán sinh thành nhiều bước khử nhiễu nhỏ.

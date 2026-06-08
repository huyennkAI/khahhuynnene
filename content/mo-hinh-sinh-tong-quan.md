# Giới thiệu mô hình sinh sâu

> Mô hình sinh sâu (Deep Generative Models) học **phân phối xác suất của dữ liệu** $p_{\text{data}}(x)$ bằng mạng nơ-ron sâu, để từ đó tạo ra các mẫu mới giống thật: ảnh, âm thanh, văn bản, phân tử... Bài này dựng nền tảng lý thuyết chung cho toàn bộ chuyên mục, và chứng minh đầy đủ các kết quả cốt lõi.

## 1. Trực giác: vì sao cần mô hình sinh?

Một mô hình **phân biệt (discriminative model)** chỉ cần vẽ ranh giới giữa các lớp — nó trả lời "đây là mèo hay chó". Nhưng nó không hiểu *mèo trông như thế nào*. Mô hình **sinh (generative model)** đặt câu hỏi tham vọng hơn:

> Nếu tôi thực sự hiểu dữ liệu, tôi phải tự tạo ra được dữ liệu mới thuyết phục.

Richard Feynman: *"What I cannot create, I do not understand"*. Học $p(x)$ buộc mô hình nắm cấu trúc sâu của dữ liệu: bố cục khuôn mặt, ngữ pháp ngôn ngữ, quy luật vật lý của âm thanh.

### Lập luận lý thuyết: sinh khó hơn phân biệt

Theo công thức Bayes, $p(y \mid x) = \dfrac{p(x \mid y)\, p(y)}{p(x)}$. Từ mô hình sinh $p(x \mid y)$ ta **suy ra được** mô hình phân biệt $p(y \mid x)$, nhưng chiều ngược lại thì không: biết ranh giới quyết định không cho ta biết mật độ dữ liệu. Vì vậy mô hình sinh "chứa nhiều thông tin hơn" — và đó là lý do nó khó học hơn.

## 2. Bài toán cốt lõi: ước lượng mật độ (density estimation)

Cho tập dữ liệu $\mathcal{D} = \{x_1, \dots, x_n\}$ rút độc lập cùng phân phối (i.i.d.) từ phân phối thật $p_{\text{data}}$ chưa biết, ta muốn học $p_\theta$ sao cho $p_\theta \approx p_{\text{data}}$. Tiêu chí kinh điển là **hợp lý cực đại (Maximum Likelihood Estimation - MLE)**:

$$\theta^{*} = \arg\max_{\theta} \; \prod_{i=1}^{n} p_\theta(x_i) = \arg\max_{\theta} \; \sum_{i=1}^{n} \log p_\theta(x_i)$$

### 2.1. Chứng minh: MLE tương đương cực tiểu KL

**Mệnh đề.** Khi $n \to \infty$, cực đại hóa log-likelihood tương đương cực tiểu hóa phân kỳ Kullback–Leibler (KL divergence) $D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta)$.

**Chứng minh.** Theo luật số lớn (law of large numbers), trung bình mẫu hội tụ về kỳ vọng:

$$\frac{1}{n} \sum_{i=1}^{n} \log p_\theta(x_i) \xrightarrow{n \to \infty} \mathbb{E}_{x \sim p_{\text{data}}}\big[\log p_\theta(x)\big]$$

Khai triển định nghĩa KL divergence:

$$
\begin{aligned}
D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta)
&= \int p_{\text{data}}(x) \log \frac{p_{\text{data}}(x)}{p_\theta(x)}\, dx \\
&= \int p_{\text{data}}(x) \log p_{\text{data}}(x)\, dx - \int p_{\text{data}}(x) \log p_\theta(x)\, dx \\
&= \underbrace{-\,\mathcal{H}(p_{\text{data}})}_{\text{hằng số theo } \theta} - \mathbb{E}_{x \sim p_{\text{data}}}\big[\log p_\theta(x)\big]
\end{aligned}
$$

Số hạng $\mathcal{H}(p_{\text{data}})$ là entropy của dữ liệu — **không phụ thuộc $\theta$**. Do đó:

$$\arg\min_{\theta} D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta) = \arg\max_{\theta} \mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(x)\big] = \arg\max_{\theta} \sum_i \log p_\theta(x_i) \qquad \blacksquare$$

> **Hệ quả:** cực đại hợp lý chính là kéo $p_\theta$ về gần $p_{\text{data}}$ theo nghĩa KL **thuận (forward KL)**. Hướng KL này có tính chất "bao phủ mode (mode-covering)": vì khi $p_{\text{data}}(x) > 0$ mà $p_\theta(x) \to 0$ thì $\log$ phân kỳ, nên $p_\theta$ buộc phải phủ mọi nơi dữ liệu xuất hiện. Đây là lý do lý thuyết khiến mô hình MLE (VAE, flow, autoregressive) ít bị bỏ sót mode hơn GAN.

### 2.2. KL không đối xứng — dẫn chứng cho khác biệt giữa các họ

KL không đối xứng: $D_{\mathrm{KL}}(p \,\|\, q) \ne D_{\mathrm{KL}}(q \,\|\, p)$.

- **Forward KL** $D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta)$ — phạt nặng khi $p_\theta$ bỏ sót vùng có dữ liệu ⇒ **mode-covering** ⇒ mẫu đa dạng nhưng có thể mờ.
- **Reverse KL** $D_{\mathrm{KL}}(p_\theta \,\|\, p_{\text{data}})$ — phạt nặng khi $p_\theta$ đặt khối lượng vào vùng không có dữ liệu ⇒ **mode-seeking** ⇒ mẫu sắc nét nhưng dễ bỏ mode.

Sự khác biệt này giải thích bằng lý thuyết vì sao GAN (gần với mode-seeking) cho ảnh sắc nét nhưng hay sụp đổ chế độ, còn VAE (mode-covering) cho ảnh mờ nhưng đa dạng.

## 3. Vì sao khó? Lời nguyền số chiều và hằng số chuẩn hóa

Một bức ảnh $256 \times 256$ màu nằm trong không gian $\approx 200{,}000$ chiều. Hai khó khăn lý thuyết nảy sinh:

### 3.1. Giả thuyết đa tạp (manifold hypothesis)

Dữ liệu thật không lấp đầy không gian mà tập trung trên một **đa tạp (manifold)** có số chiều thấp hơn nhiều. Phân phối $p_{\text{data}}$ gần như suy biến (degenerate): khối lượng xác suất dồn vào một tập có thể tích gần 0. Mô hình phải học được hình dạng đa tạp mỏng này.

### 3.2. Hằng số chuẩn hóa (normalizing constant)

Một mô hình tổng quát viết $p_\theta(x) = \dfrac{\tilde{p}_\theta(x)}{Z_\theta}$ với $Z_\theta = \int \tilde{p}_\theta(x)\, dx$. Tích phân này trên không gian khổng lồ là **bất khả thi**. Mỗi họ mô hình là một chiến lược khác nhau để né $Z_\theta$ — đây là sợi chỉ đỏ xuyên suốt cả lĩnh vực.

## 4. Năm họ mô hình và chiến lược né $Z_\theta$

| Họ | Trực giác | Cách né $Z_\theta$ |
| --- | --- | --- |
| **Autoregressive** | Bẻ $p(x)$ thành tích xác suất có điều kiện một chiều | Mỗi thừa số tự chuẩn hóa bằng softmax |
| **Normalizing Flow** | Biến nhiễu đơn giản thành dữ liệu qua hàm khả nghịch | Tính $Z$ chính xác bằng định thức Jacobian |
| **VAE** | Đưa qua biến ẩn, tối ưu chặn dưới ELBO | Suy diễn biến phân thay cho tích phân |
| **GAN** | Bỏ hẳn likelihood, học bằng đối kháng | Không cần mật độ — chỉ cần lấy mẫu |
| **Diffusion** | Khử nhiễu dần từ nhiễu thuần | Score matching, gradient làm $Z$ biến mất |
| **Energy-Based** | Gán năng lượng tùy ý cho mọi cấu hình | Không né — xấp xỉ $Z$ bằng MCMC |

> Energy-Based Model (EBM) là họ tổng quát nhất: mọi phân phối dương đều viết được dưới dạng $p(x) \propto e^{-E(x)}$. Các họ còn lại có thể xem là những cách thông minh để **né $Z_\theta$**, mà EBM phải đối mặt trực diện.

## 5. Tam giác đánh đổi (the generative trilemma)

Mọi mô hình sinh giằng co giữa ba mục tiêu khó cùng đạt:

1. **Chất lượng & đa dạng mẫu (high-quality & diverse samples)**
2. **Lấy mẫu nhanh (fast sampling)**
3. **Likelihood/density tính được (tractable likelihood)**

Lý thuyết cho thấy mỗi họ chỉ giữ tốt hai trong ba:

| Tiêu chí | Autoregressive | Flow | VAE | GAN | Diffusion | EBM |
| --- | --- | --- | --- | --- | --- | --- |
| Likelihood chính xác | Có | Có | Chặn dưới | Không | Chặn dưới | Không ($Z$) |
| Chất lượng mẫu | Cao | Trung bình | Trung bình | Cao | Rất cao | Cao |
| Tốc độ lấy mẫu | Chậm | Nhanh | Nhanh | Nhanh | Chậm | Rất chậm |
| Ổn định huấn luyện | Cao | Cao | Cao | Thấp | Cao | Thấp |

## 6. Đánh giá mô hình sinh

### 6.1. Log-likelihood / bits-per-dim

Cho mô hình tính được likelihood (autoregressive, flow). Bits-per-dim chuẩn hóa theo số chiều để so sánh giữa các tập dữ liệu:

$$\text{bpd} = \frac{-\log_2 p_\theta(x)}{D}$$

### 6.2. FID (Fréchet Inception Distance)

So khoảng cách giữa hai phân phối đặc trưng (giả định Gauss) của ảnh thật và ảnh sinh:

$$\text{FID} = \lVert \mu_r - \mu_g \rVert^2 + \operatorname{tr}\!\Big(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\Big)$$

Càng thấp càng tốt — đo cả độ trung thực lẫn đa dạng.

### 6.3. Precision / Recall cho mô hình sinh

Tách riêng hai khía cạnh mà một con số như FID gộp chung:

- **Precision** — tỉ lệ mẫu sinh rơi vào vùng dữ liệu thật (độ trung thực, fidelity).
- **Recall** — tỉ lệ vùng dữ liệu thật được mẫu sinh phủ tới (độ bao phủ, coverage). GAN thường precision cao nhưng recall thấp (mode collapse) — đúng như dự đoán từ phân tích reverse KL ở mục 2.2.

## 7. Bản đồ chuyên mục

Các bài tiếp theo đi sâu từng họ, mỗi bài theo mạch: **trực giác ra đời → khắc phục điều gì → lý thuyết & chứng minh đầy đủ → ưu/nhược điểm có dẫn chứng lý thuyết**. Thứ tự gợi ý:

1. **Autoregressive** — phân tích quy tắc chuỗi, likelihood chính xác.
2. **Normalizing Flows** — đổi biến, likelihood chính xác + lấy mẫu nhanh.
3. **VAE** — biến ẩn và chặn dưới ELBO.
4. **GAN** — học đối kháng, bỏ likelihood.
5. **Diffusion** — khử nhiễu nhiều bước, dẫn đầu chất lượng.
6. **Energy-Based Models** — khung tổng quát nhất, đối mặt trực tiếp với $Z_\theta$.

> Một câu để nhớ cả lĩnh vực: **khó khăn cốt lõi của mô hình sinh là hằng số chuẩn hóa $Z_\theta$, và lịch sử của lĩnh vực này phần lớn là lịch sử những cách thông minh để né tránh nó.**

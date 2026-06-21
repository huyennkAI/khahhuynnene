# Tổng quan mô hình sinh sâu (Deep Generative Models)

> Mô hình sinh sâu (deep generative model) học **phân phối xác suất của dữ liệu** $p_{\text{data}}(x)$ bằng mạng nơ-ron sâu, để từ đó tạo ra các mẫu mới giống thật: ảnh, âm thanh, văn bản, phân tử. Bài này dựng nền tảng lý thuyết chung cho toàn bộ chuyên mục:
>
> $$\mathcal{D} = \{x_1, \ldots, x_n\} \;\xrightarrow{\;\text{học}\;}\; p_\theta(x) \approx p_{\text{data}}(x)$$
>
> Mọi kết quả cốt lõi đều được chứng minh đầy đủ, và mọi họ mô hình đều quy về **một câu hỏi chung**: làm sao né được hằng số chuẩn hóa $Z_\theta$.

---

# 1. Trực giác: vì sao cần mô hình sinh?

Một mô hình **phân biệt (discriminative model)** chỉ cần vẽ ranh giới giữa các lớp — nó trả lời "đây là mèo hay chó". Nhưng nó **không hiểu** *mèo trông như thế nào*.

Mô hình **sinh (generative model)** đặt câu hỏi tham vọng hơn:

> Nếu tôi thực sự hiểu dữ liệu, tôi phải tự tạo ra được dữ liệu mới thuyết phục.

Richard Feynman: *"What I cannot create, I do not understand."* Học $p(x)$ buộc mô hình nắm cấu trúc sâu của dữ liệu: bố cục khuôn mặt, ngữ pháp ngôn ngữ, quy luật vật lý của âm thanh.

---

## 1.1. Vì sao sinh khó hơn phân biệt?

Theo công thức Bayes:

$$p(y \mid x) = \frac{p(x \mid y)\, p(y)}{p(x)}$$

Từ mô hình sinh $p(x \mid y)$ ta **suy ra được** mô hình phân biệt $p(y \mid x)$. Nhưng chiều ngược lại thì **không**: biết ranh giới quyết định không cho ta biết mật độ dữ liệu.

Vì vậy mô hình sinh "chứa nhiều thông tin hơn" — và đó là lý do nó khó học hơn.

---

# 2. Bài toán cốt lõi: ước lượng mật độ

Cho tập dữ liệu $\mathcal{D} = \{x_1, \ldots, x_n\}$ rút độc lập cùng phân phối (i.i.d.) từ phân phối thật $p_{\text{data}}$ chưa biết. Ta muốn học $p_\theta$ sao cho $p_\theta \approx p_{\text{data}}$.

Đây gọi là bài toán **ước lượng mật độ (density estimation)**. Tiêu chí kinh điển là **hợp lý cực đại (Maximum Likelihood Estimation — MLE)**:

$$\theta^{*} = \arg\max_{\theta} \; \prod_{i=1}^{n} p_\theta(x_i) = \arg\max_{\theta} \; \sum_{i=1}^{n} \log p_\theta(x_i)$$

Lấy logarit để biến tích thành tổng — vừa ổn định số học, vừa biến bài toán thành tổng các số hạng độc lập.

> Trực giác: ta đi chỉnh tham số $\theta$ sao cho mô hình gán **xác suất cao** cho đúng những mẫu đã quan sát được trong dữ liệu.

---

## 2.1. MLE tương đương cực tiểu KL

**Mệnh đề.** Khi $n \to \infty$, cực đại hóa log-likelihood tương đương cực tiểu hóa phân kỳ Kullback–Leibler (KL divergence) $D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta)$.

**Chứng minh.** Theo luật số lớn (law of large numbers), trung bình mẫu hội tụ về kỳ vọng:

$$\frac{1}{n} \sum_{i=1}^{n} \log p_\theta(x_i) \xrightarrow{\,n \to \infty\,} \mathbb{E}_{x \sim p_{\text{data}}}\big[\log p_\theta(x)\big]$$

Khai triển định nghĩa KL divergence:

$$
\begin{aligned}
D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta)
&= \int p_{\text{data}}(x) \log \frac{p_{\text{data}}(x)}{p_\theta(x)}\,dx \\
&= \int p_{\text{data}}(x) \log p_{\text{data}}(x)\,dx - \int p_{\text{data}}(x) \log p_\theta(x)\,dx \\
&= \underbrace{-\,\mathcal{H}(p_{\text{data}})}_{\text{hằng số theo } \theta} - \mathbb{E}_{x \sim p_{\text{data}}}\big[\log p_\theta(x)\big]
\end{aligned}
$$

Số hạng $\mathcal{H}(p_{\text{data}})$ là entropy của dữ liệu — **không phụ thuộc $\theta$**. Do đó cực tiểu KL theo $\theta$ chỉ còn là cực đại kỳ vọng log-likelihood:

$$\arg\min_{\theta} D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta) = \arg\max_{\theta} \, \mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(x)\big] = \arg\max_{\theta} \sum_i \log p_\theta(x_i) \qquad \blacksquare$$

**Hệ quả.** Cực đại hợp lý chính là kéo $p_\theta$ về gần $p_{\text{data}}$ theo nghĩa **KL thuận (forward KL)**.

Hướng KL này có tính chất "bao phủ mode (mode-covering)": khi $p_{\text{data}}(x) > 0$ mà $p_\theta(x) \to 0$ thì $\log \frac{p_{\text{data}}}{p_\theta} \to +\infty$, nên $p_\theta$ **buộc phải phủ mọi nơi dữ liệu xuất hiện**. Đây là lý do lý thuyết khiến các mô hình MLE (VAE, flow, autoregressive) ít bị bỏ sót mode hơn GAN.

---

## 2.2. KL không đối xứng — gốc rễ khác biệt giữa các họ

Một sự thật quan trọng: KL **không đối xứng**.

$$D_{\mathrm{KL}}(p \,\|\, q) \ne D_{\mathrm{KL}}(q \,\|\, p)$$

Hai hướng dẫn tới hai hành vi trái ngược:

| Hướng KL | Phạt nặng khi | Hành vi | Hệ quả mẫu |
| --- | --- | --- | --- |
| **Forward** $D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta)$ | $p_\theta$ bỏ sót vùng có dữ liệu | Bao phủ mode (mode-covering) | Đa dạng nhưng có thể mờ |
| **Reverse** $D_{\mathrm{KL}}(p_\theta \,\|\, p_{\text{data}})$ | $p_\theta$ đặt khối lượng vào vùng không có dữ liệu | Săn mode (mode-seeking) | Sắc nét nhưng dễ bỏ mode |

Ví dụ trực quan: nếu $p_{\text{data}}$ có **hai cụm** (hai mode) mà mô hình $p_\theta$ chỉ là một Gauss đơn:

* tối ưu **forward KL** sẽ kéo Gauss đó **trải rộng phủ cả hai cụm** (kể cả vùng trống ở giữa) — ảnh sinh ra mờ;
* tối ưu **reverse KL** sẽ co Gauss đó **chụm vào đúng một cụm** — sắc nét nhưng bỏ hẳn cụm còn lại.

Sự khác biệt này giải thích bằng lý thuyết vì sao GAN (gần với mode-seeking) cho ảnh sắc nét nhưng hay sụp đổ chế độ (mode collapse), còn VAE (mode-covering) cho ảnh mờ nhưng đa dạng.

---

# 3. Vì sao khó? Lời nguyền số chiều và hằng số chuẩn hóa

Một bức ảnh $256 \times 256$ màu nằm trong không gian $\approx 200{,}000$ chiều. Hai khó khăn lý thuyết nảy sinh.

---

## 3.1. Giả thuyết đa tạp (manifold hypothesis)

Dữ liệu thật **không lấp đầy** không gian mà tập trung trên một **đa tạp (manifold)** có số chiều thấp hơn nhiều.

Ví dụ: tập hợp mọi ma trận pixel ngẫu nhiên gần như chắc chắn là nhiễu; chỉ một phần cực nhỏ trong số đó trông giống khuôn mặt thật. Những khuôn mặt hợp lệ nằm trên một "mặt cong" mỏng trong không gian khổng lồ.

Do đó $p_{\text{data}}$ gần như suy biến (degenerate): khối lượng xác suất dồn vào một tập có thể tích gần $0$. Mô hình phải học được hình dạng đa tạp mỏng này.

---

## 3.2. Hằng số chuẩn hóa (normalizing constant)

Một mô hình tổng quát viết:

$$p_\theta(x) = \frac{\tilde{p}_\theta(x)}{Z_\theta}, \qquad Z_\theta = \int \tilde{p}_\theta(x)\,dx$$

trong đó $\tilde{p}_\theta(x) \ge 0$ là "mật độ chưa chuẩn hóa" (mô hình tự do gán điểm số cho mỗi $x$), còn $Z_\theta$ ép tổng xác suất về $1$.

Tích phân $Z_\theta$ trên không gian khổng lồ là **bất khả thi**: không có công thức đóng, mà lấy mẫu Monte Carlo thì phương sai bùng nổ theo số chiều.

> Mỗi họ mô hình là một **chiến lược khác nhau để né $Z_\theta$** — đây là sợi chỉ đỏ xuyên suốt cả lĩnh vực.

---

# 4. Sáu họ mô hình và chiến lược né $Z_\theta$

Đến đây ta có thể phân loại toàn bộ lĩnh vực theo đúng **một tiêu chí**: làm gì với $Z_\theta$?

| Họ | Trực giác | Cách né $Z_\theta$ |
| --- | --- | --- |
| **Autoregressive** | Bẻ $p(x)$ thành tích xác suất có điều kiện một chiều | Mỗi thừa số tự chuẩn hóa bằng softmax |
| **Normalizing Flow** | Biến nhiễu đơn giản thành dữ liệu qua hàm khả nghịch | Tính $Z$ chính xác bằng định thức Jacobian |
| **VAE** | Đưa qua biến ẩn, tối ưu chặn dưới ELBO | Suy diễn biến phân thay cho tích phân |
| **GAN** | Bỏ hẳn likelihood, học bằng đối kháng | Không cần mật độ — chỉ cần lấy mẫu |
| **Diffusion** | Khử nhiễu dần từ nhiễu thuần | Score matching, gradient làm $Z$ biến mất |
| **Energy-Based (EBM)** | Gán năng lượng tùy ý cho mọi cấu hình | Không né — xấp xỉ $Z$ bằng MCMC |

Vì sao EBM là họ **tổng quát nhất**? Vì mọi phân phối dương đều viết được dưới dạng năng lượng:

$$p(x) = \frac{e^{-E(x)}}{Z}, \qquad Z = \int e^{-E(x)}\,dx$$

với chỉ cần đặt $E(x) = -\log \tilde p(x)$. Các họ còn lại có thể xem là những cách **thông minh để né $Z$**, mà EBM phải đối mặt trực diện.

---

# 5. Tam giác đánh đổi (the generative trilemma)

Mọi mô hình sinh giằng co giữa ba mục tiêu khó cùng đạt:

1. **Chất lượng & đa dạng mẫu (high-quality & diverse samples)**
2. **Lấy mẫu nhanh (fast sampling)**
3. **Likelihood/density tính được (tractable likelihood)**

Lý thuyết cho thấy mỗi họ thường chỉ giữ tốt **hai trong ba**:

| Tiêu chí | Autoregressive | Flow | VAE | GAN | Diffusion | EBM |
| --- | --- | --- | --- | --- | --- | --- |
| Likelihood chính xác | Có | Có | Chặn dưới | Không | Chặn dưới | Không ($Z$) |
| Chất lượng mẫu | Cao | Trung bình | Trung bình | Cao | Rất cao | Cao |
| Tốc độ lấy mẫu | Chậm | Nhanh | Nhanh | Nhanh | Chậm | Rất chậm |
| Ổn định huấn luyện | Cao | Cao | Cao | Thấp | Cao | Thấp |

Đọc bảng theo trilemma:

* **Autoregressive / Flow** giữ likelihood chính xác, nhưng phải đánh đổi (autoregressive chậm khi sinh; flow ràng buộc khả nghịch).
* **GAN** lấy mẫu nhanh và sắc nét, nhưng bỏ hẳn likelihood và khó huấn luyện.
* **Diffusion** đạt chất lượng và đa dạng cao nhất, nhưng lấy mẫu chậm (nhiều bước khử nhiễu).

---

# 6. Đánh giá mô hình sinh

Học xong $p_\theta$ rồi, làm sao biết nó **tốt**? Mỗi tiêu chí dưới đây soi một khía cạnh khác nhau.

---

## 6.1. Log-likelihood và bits-per-dim

Dùng cho mô hình tính được likelihood (autoregressive, flow). Để so sánh **giữa các tập dữ liệu** có số chiều khác nhau, ta chuẩn hóa theo số chiều:

$$\text{bpd} = \frac{-\log_2 p_\theta(x)}{D}$$

Diễn giải: số bit trung bình cần để mã hóa **một chiều** của dữ liệu. Càng thấp nghĩa là mô hình nén dữ liệu càng tốt, tức mô hình hóa $p_{\text{data}}$ càng sát.

---

## 6.2. FID (Fréchet Inception Distance)

So khoảng cách giữa hai phân phối đặc trưng (giả định Gauss) của ảnh thật và ảnh sinh:

$$\text{FID} = \lVert \mu_r - \mu_g \rVert^2 + \operatorname{tr}\!\Big(\Sigma_r + \Sigma_g - 2\,(\Sigma_r \Sigma_g)^{1/2}\Big)$$

trong đó $(\mu_r, \Sigma_r)$ là trung bình và hiệp phương sai của đặc trưng ảnh thật, $(\mu_g, \Sigma_g)$ của ảnh sinh.

Càng thấp càng tốt — FID đo **cả** độ trung thực lẫn đa dạng, nhưng gộp chung thành một con số.

---

## 6.3. Precision và Recall cho mô hình sinh

Tách riêng hai khía cạnh mà một con số như FID gộp chung:

* **Precision** — tỉ lệ mẫu sinh rơi vào vùng dữ liệu thật (độ trung thực, fidelity).
* **Recall** — tỉ lệ vùng dữ liệu thật được mẫu sinh phủ tới (độ bao phủ, coverage).

GAN thường **precision cao nhưng recall thấp** (mode collapse) — đúng như dự đoán từ phân tích reverse KL ở mục 2.2. Ngược lại, các mô hình MLE thiên về recall cao (mode-covering).

---

# 7. Tổng kết

Toàn bộ lĩnh vực mô hình sinh, dù trông rất nhiều kiến trúc, thực ra xoay quanh **một bài toán** và **một khó khăn**:

* **Bài toán:** học $p_\theta \approx p_{\text{data}}$, mà MLE chính là cực tiểu KL thuận (mục 2.1).
* **Khó khăn:** hằng số chuẩn hóa $Z_\theta$ là tích phân bất khả thi trên không gian nhiều chiều (mục 3.2).

Hướng KL ta tối ưu (forward hay reverse) quyết định mô hình **bao phủ** hay **săn mode** (mục 2.2); và mỗi họ là một chiến lược né $Z_\theta$ khác nhau (mục 4), kéo theo vị trí của nó trên tam giác đánh đổi (mục 5).

> Một câu để nhớ cả lĩnh vực: **khó khăn cốt lõi của mô hình sinh là hằng số chuẩn hóa $Z_\theta$, và lịch sử của lĩnh vực này phần lớn là lịch sử những cách thông minh để né tránh nó.**

---

# 8. Bản đồ chuyên mục

Các bài tiếp theo đi sâu từng họ, mỗi bài theo cùng một mạch: **trực giác ra đời → khắc phục điều gì → lý thuyết và chứng minh đầy đủ → ưu/nhược điểm có dẫn chứng lý thuyết**. Thứ tự gợi ý:

1. **Autoregressive** — phân rã quy tắc chuỗi, likelihood chính xác.
2. **Normalizing Flows** — đổi biến, likelihood chính xác kèm lấy mẫu nhanh.
3. **VAE** — biến ẩn và chặn dưới ELBO.
4. **GAN** — học đối kháng, bỏ likelihood.
5. **Diffusion** — khử nhiễu nhiều bước, dẫn đầu chất lượng.
6. **Energy-Based Models** — khung tổng quát nhất, đối mặt trực tiếp với $Z_\theta$.

> Bài tiếp theo, **Mô hình tự hồi quy (Autoregressive Models)**, mở đầu hành trình: nó né $Z_\theta$ một cách tinh tế nhất — bẻ $p(x)$ thành tích các xác suất có điều kiện một chiều, mỗi thừa số tự chuẩn hóa bằng softmax, nên **toàn bộ mô hình chuẩn hóa miễn phí**.

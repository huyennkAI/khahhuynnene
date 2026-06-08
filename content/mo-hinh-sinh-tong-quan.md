# Giới thiệu mô hình sinh sâu

> Mô hình sinh sâu (Deep Generative Models) học **phân phối xác suất của dữ liệu** $p_{\text{data}}(x)$ bằng mạng nơ-ron sâu, để từ đó tạo ra các mẫu mới giống thật: ảnh, âm thanh, văn bản, phân tử... Bài này dựng nền tảng lý thuyết chung cho toàn bộ chuyên mục.

## Trực giác: vì sao cần mô hình sinh?

Một mô hình **phân biệt** chỉ cần vẽ ranh giới giữa các lớp — nó trả lời "đây là mèo hay chó". Nhưng nó không hiểu *mèo trông như thế nào*. Mô hình **sinh** đặt câu hỏi tham vọng hơn:

> Nếu tôi thực sự hiểu dữ liệu, tôi phải tự tạo ra được dữ liệu mới thuyết phục.

Richard Feynman: *"What I cannot create, I do not understand"*. Học $p(x)$ buộc mô hình nắm được cấu trúc sâu của dữ liệu: bố cục khuôn mặt, ngữ pháp ngôn ngữ, quy luật vật lý của âm thanh.

## Phân biệt vs sinh — về mặt xác suất

- **Mô hình phân biệt** học trực tiếp $p(y \mid x)$ — ranh giới quyết định.
- **Mô hình sinh** học $p(x)$ hoặc phân phối hợp $p(x, y) = p(x \mid y)\,p(y)$.

Từ mô hình sinh ta suy ra được mô hình phân biệt qua Bayes, nhưng chiều ngược lại thì không. Mô hình sinh "biết nhiều hơn", và cũng khó học hơn.

## Bài toán cốt lõi: ước lượng mật độ

Cho tập dữ liệu $\mathcal{D} = \{x_1, \dots, x_n\}$ rút độc lập từ phân phối thật $p_{\text{data}}$ chưa biết, ta muốn học $p_\theta$ sao cho $p_\theta \approx p_{\text{data}}$. Tiêu chí kinh điển là **hợp lý cực đại** (Maximum Likelihood):

$$\theta^{*} = \arg\max_{\theta} \; \frac{1}{n} \sum_{i=1}^{n} \log p_\theta(x_i)$$

### Tại sao MLE tương đương cực tiểu KL?

Đây là kết quả nền tảng cho toàn bộ lĩnh vực. Khi $n \to \infty$, trung bình mẫu hội tụ về kỳ vọng:

$$\frac{1}{n} \sum_{i} \log p_\theta(x_i) \to \mathbb{E}_{x \sim p_{\text{data}}}\big[\log p_\theta(x)\big]$$

Xét phân kỳ KL giữa phân phối thật và mô hình:

$$D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log p_{\text{data}}(x)\big] - \mathbb{E}_{x \sim p_{\text{data}}}\big[\log p_\theta(x)\big]$$

Số hạng đầu là entropy của dữ liệu — **hằng số** với $\theta$. Do đó:

$$\arg\min_{\theta} D_{\mathrm{KL}}(p_{\text{data}} \,\|\, p_\theta) = \arg\max_{\theta} \mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(x)\big]$$

> **Kết luận:** cực đại hóa hợp lý chính là kéo $p_\theta$ về gần $p_{\text{data}}$ theo nghĩa KL. Mỗi họ mô hình sinh là một cách khác nhau để xấp xỉ hoặc né tránh việc tính $\log p_\theta(x)$ — vốn bất khả thi trực tiếp ở dữ liệu nhiều chiều.

## Vì sao khó? Lời nguyền số chiều

Một bức ảnh $256 \times 256$ màu nằm trong không gian $\sim 200{,}000$ chiều. Phân phối $p(x)$ trên không gian đó vừa cực kỳ phức tạp, vừa tập trung trên một **đa tạp** (manifold) mỏng. Hai khó khăn nảy sinh:

1. **Chuẩn hóa:** $p_\theta(x) = \frac{\tilde{p}_\theta(x)}{Z_\theta}$ với hằng số chuẩn hóa $Z_\theta = \int \tilde{p}_\theta(x)\,dx$ không tính nổi.
2. **Lấy mẫu:** sinh một điểm hợp lệ trên đa tạp mỏng giữa không gian khổng lồ là rất khó.

Bốn họ mô hình bên dưới là bốn chiến lược khác nhau để vượt qua hai khó khăn này.

## Bốn họ mô hình và trực giác cốt lõi

| Họ | Trực giác | Khắc phục điều gì |
| --- | --- | --- |
| **Autoregressive** | Bẻ $p(x)$ thành tích các xác suất có điều kiện theo quy tắc chuỗi | Né hằng số chuẩn hóa, có likelihood chính xác |
| **Normalizing Flow** | Biến nhiễu đơn giản thành dữ liệu qua hàm khả nghịch | Tính likelihood chính xác bằng đổi biến |
| **VAE** | Đưa qua biến ẩn, tối ưu chặn dưới ELBO | Né $Z_\theta$ bằng suy diễn biến phân |
| **GAN** | Bỏ hẳn likelihood, học bằng đối kháng | Né tính mật độ, đổi lấy mẫu sắc nét |
| **Diffusion** | Khử nhiễu dần từ nhiễu thuần | Chia bài toán khó thành nhiều bước dễ |

## Tam giác đánh đổi

Mọi mô hình sinh đều giằng co giữa ba mục tiêu khó cùng đạt:

1. **Likelihood tính được** — có ước lượng được $p_\theta(x)$ không?
2. **Chất lượng & đa dạng mẫu** — mẫu có sắc nét và phong phú không?
3. **Lấy mẫu nhanh** — sinh một mẫu tốn bao nhiêu bước?

| Tiêu chí | Autoregressive | Flow | VAE | GAN | Diffusion |
| --- | --- | --- | --- | --- | --- |
| Likelihood chính xác | Có | Có | Chặn dưới | Không | Chặn dưới |
| Chất lượng mẫu | Cao | Trung bình | Trung bình | Cao | Rất cao |
| Tốc độ lấy mẫu | Chậm | Nhanh | Nhanh | Nhanh | Chậm |
| Ổn định huấn luyện | Cao | Cao | Cao | Thấp | Cao |

## Đánh giá mô hình sinh

- **Log-likelihood / bits-per-dim** — cho mô hình tính được likelihood (autoregressive, flow).
- **FID (Fréchet Inception Distance)** — so khoảng cách đặc trưng giữa ảnh thật và ảnh sinh; càng thấp càng tốt.
- **Inception Score** — đo độ rõ ràng và đa dạng của mẫu.
- **Precision / Recall** — tách riêng độ trung thực (fidelity) và độ bao phủ (coverage).

> Các bài tiếp theo đi sâu từng họ — mỗi bài đều theo mạch: **trực giác ra đời → khắc phục điều gì → lý thuyết & chứng minh → ưu/nhược điểm**. Thứ tự gợi ý: Autoregressive → Normalizing Flow → VAE → GAN → Diffusion.

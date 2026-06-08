# Mô hình dựa trên năng lượng (Energy-Based Models)

> Mô hình dựa trên năng lượng (Energy-Based Model - EBM) gán cho mỗi cấu hình dữ liệu một **mức năng lượng (energy)** vô hướng: cấu hình hợp lý có năng lượng thấp, cấu hình vô lý có năng lượng cao. Đây là họ mô hình sinh (generative model) **linh hoạt nhất về mặt biểu diễn**, nhưng cũng khó huấn luyện và lấy mẫu nhất.

## Trực giác ra đời

Mọi họ mô hình sinh trước đó đều bị một ràng buộc trói buộc:

- **Autoregressive** phải áp một thứ tự và phân tích quy tắc chuỗi (chain rule).
- **Normalizing flow** phải giữ tính khả nghịch (invertibility) và Jacobian rẻ.
- **VAE** phải giả định cấu trúc biến ẩn (latent variable) và phân phối Gauss.
- **GAN** phải định nghĩa một generator khả vi và một trò chơi đối kháng.

Câu hỏi tham vọng nhất: **liệu có thể để mạng nơ-ron tự do gán một "điểm tin cậy" cho mọi cấu hình $x$, không ràng buộc kiến trúc gì cả?**

Trực giác đến từ vật lý thống kê (statistical physics): trong tự nhiên, một hệ ở trạng thái cân bằng nhiệt có xác suất ở cấu hình $x$ tỉ lệ nghịch theo hàm mũ với năng lượng của cấu hình đó — **phân phối Boltzmann (Boltzmann distribution)**. Trạng thái năng lượng thấp (ổn định) thì hay gặp; trạng thái năng lượng cao thì hiếm.

EBM mượn nguyên ý tưởng này: định nghĩa một **hàm năng lượng (energy function)** $E_\theta(x)$ tùy ý bằng mạng nơ-ron, rồi biến nó thành phân phối xác suất.

## Khắc phục điều gì

- **Tự do kiến trúc tối đa:** $E_\theta$ chỉ cần là một hàm vô hướng khả vi — bất kỳ mạng nào (CNN, Transformer) cũng được, không ràng buộc khả nghịch hay nhân quả.
- **Hợp nhất nhiều mô hình:** rất nhiều mô hình khác là EBM trá hình (softmax phân loại, CRF, RBM...).
- Đổi lại, EBM **không né được hằng số chuẩn hóa (normalizing constant)** $Z_\theta$ — và đây chính là nguồn gốc mọi khó khăn của nó.

## Lý thuyết: từ năng lượng đến xác suất

Định nghĩa phân phối Boltzmann–Gibbs:

$$p_\theta(x) = \frac{\exp\big(-E_\theta(x)\big)}{Z_\theta}, \qquad Z_\theta = \int \exp\big(-E_\theta(x)\big)\, dx$$

trong đó:

- $E_\theta(x) \in \mathbb{R}$ — **năng lượng** của cấu hình $x$ (càng thấp càng hợp lý).
- $Z_\theta$ — **hàm phân hoạch (partition function)**, đảm bảo $p_\theta$ tích phân về 1.

Vấn đề trung tâm: $Z_\theta$ là tích phân trên toàn không gian dữ liệu nhiều chiều — **bất khả thi** để tính hay khả vi trực tiếp. Mọi kỹ thuật huấn luyện EBM đều xoay quanh việc **né hoặc xấp xỉ** $Z_\theta$.

### Vì sao biểu diễn này mạnh?

Bất kỳ phân phối $p(x) > 0$ nào cũng viết được dưới dạng EBM bằng cách đặt $E(x) = -\log p(x) + \text{const}$. Nói cách khác, lớp EBM **bao trùm mọi phân phối dương** — đó là lý do nó linh hoạt nhất. Cái giá là phải làm việc với dạng chưa chuẩn hóa (unnormalized).

## Huấn luyện bằng hợp lý cực đại

Ta muốn cực đại log-likelihood $\log p_\theta(x)$. Viết ra:

$$\log p_\theta(x) = -E_\theta(x) - \log Z_\theta$$

### Suy ra gradient (kết quả cốt lõi)

Lấy gradient theo $\theta$:

$$\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta$$

Số hạng thứ hai khai triển:

$$
\begin{aligned}
\nabla_\theta \log Z_\theta &= \frac{1}{Z_\theta} \nabla_\theta \int \exp(-E_\theta(x))\, dx \\
&= \frac{1}{Z_\theta} \int -\nabla_\theta E_\theta(x)\, \exp(-E_\theta(x))\, dx \\
&= \int -\nabla_\theta E_\theta(x)\, \frac{\exp(-E_\theta(x))}{Z_\theta}\, dx \\
&= -\,\mathbb{E}_{x \sim p_\theta}\big[\nabla_\theta E_\theta(x)\big]
\end{aligned}
$$

Thay vào, gradient của log-likelihood (lấy kỳ vọng trên dữ liệu thật) trở thành:

$$\nabla_\theta \mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(x)\big] = -\underbrace{\mathbb{E}_{x \sim p_{\text{data}}}\big[\nabla_\theta E_\theta(x)\big]}_{\text{pha dương (positive phase)}} + \underbrace{\mathbb{E}_{x \sim p_\theta}\big[\nabla_\theta E_\theta(x)\big]}_{\text{pha âm (negative phase)}}$$

### Diễn giải hai pha

Quy tắc cập nhật này có một diễn giải vật lý rất đẹp:

- **Pha dương (positive phase):** **hạ năng lượng** của dữ liệu thật $x \sim p_{\text{data}}$ — kéo các mẫu thật xuống "thung lũng".
- **Pha âm (negative phase):** **nâng năng lượng** của mẫu do chính mô hình sinh ra $x \sim p_\theta$ — đẩy các "ảo giác" của mô hình lên cao.

> Huấn luyện EBM là cuộc giằng co: dìm dữ liệu thật xuống, nâng dữ liệu mô hình tưởng tượng lên, đến khi hai phân phối trùng nhau và lực cân bằng.

Nút thắt: pha âm cần **lấy mẫu $x \sim p_\theta$**, mà lấy mẫu từ EBM lại khó vì không có cách sinh trực tiếp.

## Lấy mẫu bằng MCMC và Langevin dynamics

Vì không lấy mẫu trực tiếp được, ta dùng chuỗi Markov Monte Carlo (Markov Chain Monte Carlo - MCMC). Phương pháp hiện đại phổ biến là **động lực học Langevin (Langevin dynamics)**, chỉ cần gradient của năng lượng:

$$x_{t+1} = x_t - \frac{\eta}{2}\, \nabla_x E_\theta(x_t) + \sqrt{\eta}\, \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, I)$$

- Số hạng $-\nabla_x E$ kéo mẫu về vùng năng lượng thấp (giống gradient descent trên không gian dữ liệu).
- Số hạng nhiễu $\sqrt{\eta}\,\epsilon$ giữ tính ngẫu nhiên để khám phá, tránh kẹt ở một điểm cực tiểu.

Khi $\eta \to 0$ và số bước $\to \infty$, phân phối của $x_t$ hội tụ về đúng $p_\theta(x)$. Lưu ý chỉ cần $\nabla_x E_\theta$ — **không cần $Z_\theta$**, vì $\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$ (hằng số $Z_\theta$ biến mất khi lấy gradient theo $x$).

```python
def langevin_sample(energy, x, steps, eta):
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        grad = torch.autograd.grad(energy(x).sum(), x)[0]
        x = x - 0.5 * eta * grad + (eta ** 0.5) * torch.randn_like(x)
    return x
```

## Contrastive Divergence (CD)

Chạy MCMC tới hội tụ ở mỗi bước huấn luyện là quá tốn. **Contrastive Divergence (CD-k)** của Hinton là mẹo xấp xỉ: chỉ chạy $k$ bước MCMC (thường $k=1$) **bắt đầu từ chính dữ liệu thật**, rồi dùng mẫu thô đó cho pha âm.

```text
Với mỗi batch dữ liệu thật x+:
  x- = MCMC_k_buoc(khoi tao = x+)
  loss = E(x+) - E(x-)
  cap nhat theta theo -grad(loss)
```

CD đánh đổi độ chính xác lấy tốc độ; nó thiên lệch (biased) nhưng thực dụng. Persistent CD (PCD) cải thiện bằng cách duy trì chuỗi MCMC liên tục giữa các batch.

## Né hằng số chuẩn hóa: các hướng khác

Vì $Z_\theta$ là gốc rễ khó khăn, nhiều phương pháp tránh nó hoàn toàn:

### Score Matching

Thay vì khớp mật độ, khớp **hàm điểm (score function)** $\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$. Vì gradient theo $x$ làm $Z_\theta$ biến mất, ta hoàn toàn né được nó. Mục tiêu score matching của Hyvärinen:

$$J(\theta) = \mathbb{E}_{p_{\text{data}}}\left[ \frac{1}{2}\big\lVert \nabla_x \log p_\theta(x) \big\rVert^2 + \operatorname{tr}\!\big(\nabla_x^2 \log p_\theta(x)\big) \right]$$

> Đây chính là cây cầu nối EBM với **mô hình khuếch tán (diffusion model)**: diffusion là một dạng score matching khử nhiễu (denoising score matching) trên nhiều mức nhiễu.

### Noise Contrastive Estimation (NCE)

Biến bài toán ước lượng mật độ thành bài toán **phân loại nhị phân**: dạy mô hình phân biệt dữ liệu thật với mẫu nhiễu từ một phân phối tham chiếu đã biết. Hằng số $Z_\theta$ trở thành một tham số học được.

## Các đại diện tiêu biểu

| Mô hình | Ghi chú |
| --- | --- |
| **Máy Boltzmann hạn chế (Restricted Boltzmann Machine - RBM)** | EBM cổ điển với biến ẩn nhị phân, huấn luyện bằng CD |
| **Deep Boltzmann Machine** | Nhiều tầng biến ẩn |
| **Deep EBM (IGEBM)** | Hàm năng lượng là CNN sâu, lấy mẫu Langevin |
| **JEM (Joint Energy Model)** | Diễn giải lại bộ phân loại softmax như một EBM |

### Bộ phân loại softmax là một EBM

Một bộ phân loại với logits $f_\theta(x)[y]$ định nghĩa $p(y\mid x) = \frac{\exp(f_\theta(x)[y])}{\sum_{y'} \exp(f_\theta(x)[y'])}$. Đặt $E_\theta(x, y) = -f_\theta(x)[y]$ thì đây đúng là một EBM trên cặp $(x, y)$. JEM khai thác chính điều này để vừa phân loại vừa sinh.

## Ưu điểm

- **Linh hoạt nhất:** không ràng buộc kiến trúc, bao trùm mọi phân phối dương.
- **Hợp nhất lý thuyết:** liên hệ tới score matching, diffusion, phân loại, vật lý thống kê.
- **Hàm năng lượng diễn giải được:** $E_\theta(x)$ trực tiếp đo mức "bất thường", hữu ích cho phát hiện ngoài phân phối (out-of-distribution detection).

## Nhược điểm

- **Hằng số chuẩn hóa $Z_\theta$ bất khả thi** — không tính được likelihood chính xác.
- **Lấy mẫu chậm và khó:** phải chạy MCMC/Langevin nhiều bước, dễ kẹt ở mode.
- **Huấn luyện bất ổn:** pha âm dựa trên mẫu MCMC thô khiến gradient nhiễu, khó hội tụ.
- **Khó mở rộng** lên ảnh độ phân giải cao so với diffusion.

## Vị trí trong bức tranh chung

EBM là họ mô hình **tổng quát nhất về mặt khái niệm**: nhiều mô hình khác có thể xem là EBM với một mẹo cụ thể để né $Z_\theta$. Diffusion thành công rực rỡ chính vì nó là cách huấn luyện EBM (qua score matching trên nhiều mức nhiễu) mà vừa ổn định vừa lấy mẫu tốt.

> EBM dạy ta một bài học sâu sắc: khó khăn cốt lõi của mô hình sinh nằm ở **hằng số chuẩn hóa**, và lịch sử lĩnh vực này phần lớn là lịch sử của những cách thông minh để né tránh nó.

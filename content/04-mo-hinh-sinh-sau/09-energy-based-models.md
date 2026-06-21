# Mô hình dựa trên năng lượng (Energy-Based Models)

> Mô hình dựa trên năng lượng (energy-based model — EBM) gán cho mỗi cấu hình dữ liệu một **mức năng lượng (energy)** vô hướng: cấu hình hợp lý có năng lượng thấp, cấu hình vô lý có năng lượng cao. Xác suất được suy ra từ năng lượng theo hàm mũ:
>
> $$p_\theta(x) \propto \exp\big(-E_\theta(x)\big)$$
>
> Đây là họ mô hình sinh (generative model) **linh hoạt nhất về mặt biểu diễn** — bao trùm mọi phân phối dương — nhưng cũng khó huấn luyện và lấy mẫu nhất.

---

# 1. Động cơ ra đời

Mọi họ mô hình sinh ta đã gặp đều phải trả giá bằng một **ràng buộc kiến trúc** để né được hằng số chuẩn hóa (normalizing constant):

| Họ mô hình | Ràng buộc phải chịu |
| --- | --- |
| **Tự hồi quy (autoregressive)** | Áp một thứ tự cố định, phân rã theo quy tắc chuỗi (chain rule) |
| **Luồng chuẩn hóa (normalizing flow)** | Phép biến đổi khả nghịch (invertible), Jacobian tính được rẻ |
| **VAE** | Giả định cấu trúc biến ẩn (latent variable) và họ phân phối hậu nghiệm |
| **GAN** | Một generator khả vi và một trò chơi đối kháng (adversarial game) |

Câu hỏi tham vọng nhất là:

> Liệu có thể để mạng nơ-ron **tự do gán một điểm tin cậy** cho mọi cấu hình $x$, mà **không ràng buộc kiến trúc gì cả**?

Trực giác đến từ vật lý thống kê (statistical physics). Trong tự nhiên, một hệ ở trạng thái cân bằng nhiệt có xác suất rơi vào cấu hình $x$ giảm theo hàm mũ với năng lượng của cấu hình đó — đây là **phân phối Boltzmann (Boltzmann distribution)**.

Trạng thái năng lượng thấp (ổn định) thì hay gặp; trạng thái năng lượng cao thì hiếm. Một hòn bi luôn có xu hướng lăn xuống đáy thung lũng, không leo lên đỉnh đồi.

EBM mượn nguyên ý tưởng này: định nghĩa một **hàm năng lượng (energy function)** $E_\theta(x)$ tùy ý bằng mạng nơ-ron, rồi biến nó thành phân phối xác suất.

---

# 2. Ý tưởng cốt lõi: từ năng lượng đến xác suất

Định nghĩa phân phối Boltzmann–Gibbs:

$$p_\theta(x) = \frac{\exp\big(-E_\theta(x)\big)}{Z_\theta}, \qquad Z_\theta = \int \exp\big(-E_\theta(x)\big)\, dx$$

trong đó:

* $E_\theta(x) \in \mathbb{R}$ — **năng lượng** của cấu hình $x$ (càng thấp càng hợp lý),
* $Z_\theta$ — **hàm phân hoạch (partition function)**, đảm bảo $p_\theta$ tích phân về $1$.

Dấu trừ trong $\exp(-E_\theta)$ là then chốt: năng lượng **thấp** cho xác suất **cao**, và ngược lại. Hàm mũ đảm bảo $p_\theta(x) > 0$ với mọi $x$, bất kể $E_\theta$ nhận giá trị gì.

Vấn đề trung tâm: $Z_\theta$ là tích phân trên **toàn bộ không gian dữ liệu nhiều chiều** — bất khả thi để tính hay khả vi trực tiếp.

> Mọi kỹ thuật huấn luyện EBM, về bản chất, đều là một cách **né hoặc xấp xỉ** $Z_\theta$.

---

## 2.1. Vì sao biểu diễn này mạnh nhất?

Một sự thật đáng kinh ngạc: EBM **bao trùm mọi phân phối xác suất dương**.

**Mệnh đề.** Với bất kỳ phân phối $p(x)$ thỏa $p(x) > 0$ với mọi $x$, tồn tại một hàm năng lượng $E$ sao cho $p$ chính là phân phối Boltzmann của $E$.

**Chứng minh.** Đặt

$$E(x) = -\log p(x).$$

Vì $p(x) > 0$ nên $\log p(x)$ xác định khắp nơi, do đó $E(x)$ là một hàm vô hướng hợp lệ. Hàm phân hoạch tương ứng là

$$Z = \int \exp\big(-E(x)\big)\, dx = \int \exp\big(\log p(x)\big)\, dx = \int p(x)\, dx = 1.$$

Khi đó

$$\frac{\exp(-E(x))}{Z} = \frac{p(x)}{1} = p(x). \qquad \blacksquare$$

Nói cách khác, lớp EBM **không bỏ sót** một phân phối dương nào — đó là lý do nó linh hoạt nhất trong các họ mô hình sinh. Cái giá phải trả là ta luôn làm việc với dạng **chưa chuẩn hóa (unnormalized)** $\exp(-E_\theta(x))$ và phải đối mặt với $Z_\theta$.

---

# 3. Mô hình thực sự học cái gì?

Khác với autoregressive (học các xác suất điều kiện) hay flow (học một phép biến đổi khả nghịch), EBM chỉ học **một hàm vô hướng duy nhất** $E_\theta(x)$.

Đầu vào là một cấu hình $x$ (một ảnh, một câu), đầu ra là **một con số thực**:

$$x \;\longmapsto\; E_\theta(x) \in \mathbb{R}$$

Đây là **tự do kiến trúc tối đa**: $E_\theta$ chỉ cần là một hàm vô hướng khả vi, nên bất kỳ mạng nào — CNN, Transformer, MLP — cũng dùng được, không ràng buộc khả nghịch hay nhân quả.

Ví dụ trực quan trên bài toán phân biệt ảnh chữ số viết tay:

| Cấu hình $x$ | $E_\theta(x)$ | Diễn giải |
| --- | --- | --- |
| Ảnh chữ số "7" rõ ràng | $-12.4$ (rất thấp) | Rất hợp lý, mô hình "tin" |
| Ảnh chữ số "7" mờ nhiễu | $-3.1$ (trung bình) | Tạm hợp lý |
| Ảnh nhiễu trắng ngẫu nhiên | $+8.7$ (rất cao) | Vô lý, mô hình "từ chối" |

Vì $E_\theta(x)$ trực tiếp đo mức "bất thường" của $x$, nó rất hữu ích cho **phát hiện ngoài phân phối (out-of-distribution detection)**.

---

# 4. Huấn luyện bằng hợp lý cực đại

Ta muốn cực đại log-likelihood $\log p_\theta(x)$ trên dữ liệu thật. Lấy logarit của phân phối Boltzmann:

$$\log p_\theta(x) = -E_\theta(x) - \log Z_\theta$$

Số hạng $-E_\theta(x)$ tính được dễ dàng bằng một forward pass. Khó khăn duy nhất nằm ở $\log Z_\theta$.

---

## 4.1. Suy ra gradient hai pha (kết quả cốt lõi)

Đây là kết quả trung tâm của toàn bộ lý thuyết EBM.

**Mệnh đề.** Gradient của log-likelihood theo $\theta$ phân tách thành hai số hạng:

$$\nabla_\theta \mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(x)\big] = -\underbrace{\mathbb{E}_{x \sim p_{\text{data}}}\big[\nabla_\theta E_\theta(x)\big]}_{\text{pha dương (positive phase)}} + \underbrace{\mathbb{E}_{x \sim p_\theta}\big[\nabla_\theta E_\theta(x)\big]}_{\text{pha âm (negative phase)}}$$

**Chứng minh.** Lấy gradient theo $\theta$ của $\log p_\theta(x) = -E_\theta(x) - \log Z_\theta$ với **một** mẫu $x$:

$$\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta$$

Số hạng $\nabla_\theta \log Z_\theta$ **không phụ thuộc** vào $x$; ta khai triển nó:

$$
\begin{aligned}
\nabla_\theta \log Z_\theta
&= \frac{1}{Z_\theta}\, \nabla_\theta \int \exp\big(-E_\theta(x)\big)\, dx \\
&= \frac{1}{Z_\theta} \int \nabla_\theta \exp\big(-E_\theta(x)\big)\, dx \\
&= \frac{1}{Z_\theta} \int -\nabla_\theta E_\theta(x)\, \exp\big(-E_\theta(x)\big)\, dx \\
&= \int -\nabla_\theta E_\theta(x)\, \underbrace{\frac{\exp\big(-E_\theta(x)\big)}{Z_\theta}}_{=\, p_\theta(x)}\, dx \\
&= -\,\mathbb{E}_{x \sim p_\theta}\big[\nabla_\theta E_\theta(x)\big]
\end{aligned}
$$

(ở bước thứ hai ta đổi thứ tự gradient và tích phân; ở bước cuối nhận ra biểu thức dưới dấu tích phân chính là $p_\theta(x)$ nên tích phân trở thành kỳ vọng).

Thay kết quả này vào, với một mẫu $x$:

$$\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) + \mathbb{E}_{x' \sim p_\theta}\big[\nabla_\theta E_\theta(x')\big]$$

Cuối cùng lấy kỳ vọng trên phân phối dữ liệu thật $p_{\text{data}}$:

$$\nabla_\theta \mathbb{E}_{p_{\text{data}}}\big[\log p_\theta(x)\big] = -\mathbb{E}_{x \sim p_{\text{data}}}\big[\nabla_\theta E_\theta(x)\big] + \mathbb{E}_{x \sim p_\theta}\big[\nabla_\theta E_\theta(x)\big]. \qquad \blacksquare$$

Điều đẹp đẽ ở đây: dù $Z_\theta$ là một tích phân bất khả thi, **gradient** của $\log Z_\theta$ lại chỉ là một **kỳ vọng** lấy theo chính phân phối $p_\theta$ — thứ ta có thể xấp xỉ bằng lấy mẫu.

---

## 4.2. Diễn giải hai pha

Quy tắc cập nhật trên có một diễn giải vật lý rất đẹp. Vì ta cực đại log-likelihood (đi **theo** gradient), tham số dịch chuyển sao cho:

* **Pha dương (positive phase):** **hạ năng lượng** của dữ liệu thật $x \sim p_{\text{data}}$ — kéo các mẫu thật xuống "thung lũng".
* **Pha âm (negative phase):** **nâng năng lượng** của mẫu do chính mô hình sinh ra $x \sim p_\theta$ — đẩy các "ảo giác" của mô hình lên cao.

> Huấn luyện EBM là một cuộc giằng co: dìm dữ liệu thật xuống, nâng dữ liệu mô hình tưởng tượng lên, cho đến khi hai phân phối trùng nhau và hai lực cân bằng.

Tại điểm hội tụ, $p_\theta = p_{\text{data}}$, nên hai kỳ vọng triệt tiêu nhau và gradient bằng $0$ — đúng như kỳ vọng của một nghiệm hợp lý cực đại.

Nút thắt: pha âm cần **lấy mẫu $x \sim p_\theta$**, mà lấy mẫu từ EBM lại khó vì không có cách sinh trực tiếp như autoregressive hay flow.

---

# 5. Lấy mẫu bằng MCMC và Langevin dynamics

Vì không lấy mẫu trực tiếp được, ta dùng **chuỗi Markov Monte Carlo (Markov Chain Monte Carlo — MCMC)**. Phương pháp hiện đại phổ biến là **động lực học Langevin (Langevin dynamics)**, chỉ cần gradient của năng lượng theo $x$:

$$x_{t+1} = x_t - \frac{\eta}{2}\, \nabla_x E_\theta(x_t) + \sqrt{\eta}\, \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, I)$$

Hai số hạng có vai trò bù trừ nhau:

* $-\nabla_x E_\theta$ kéo mẫu về vùng năng lượng thấp (giống gradient descent **trên không gian dữ liệu**, không phải không gian tham số).
* $\sqrt{\eta}\, \epsilon_t$ là nhiễu giữ tính ngẫu nhiên để khám phá, tránh kẹt cứng ở một cực tiểu.

Khi $\eta \to 0$ và số bước $\to \infty$, phân phối của $x_t$ hội tụ về đúng $p_\theta(x)$.

---

## 5.1. Vì sao Langevin không cần $Z_\theta$?

Đây là lý do MCMC khả thi với EBM.

**Mệnh đề.** Hàm điểm (score) theo $x$ không phụ thuộc hằng số chuẩn hóa:

$$\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$$

**Chứng minh.** Từ $\log p_\theta(x) = -E_\theta(x) - \log Z_\theta$, lấy gradient theo $x$:

$$\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x) - \nabla_x \log Z_\theta$$

Nhưng $Z_\theta$ là một **hằng số** đối với $x$ (nó là tích phân đã lấy hết biến $x$), nên $\nabla_x \log Z_\theta = 0$. Suy ra điều phải chứng minh. $\blacksquare$

Bước cập nhật Langevin chỉ dùng $\nabla_x E_\theta(x_t)$, mà theo mệnh đề trên đây chính là $-\nabla_x \log p_\theta(x_t)$. Vì vậy ta lấy mẫu được từ $p_\theta$ **mà không bao giờ phải đụng tới $Z_\theta$**.

```python
def langevin_sample(energy, x, steps, eta):
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        grad = torch.autograd.grad(energy(x).sum(), x)[0]
        x = x - 0.5 * eta * grad + (eta ** 0.5) * torch.randn_like(x)
    return x
```

---

# 6. Contrastive Divergence (CD)

Chạy MCMC tới hội tụ ở **mỗi** bước huấn luyện là quá tốn: mỗi cập nhật tham số sẽ cần hàng nghìn bước Langevin.

**Contrastive Divergence (CD-$k$)** của Hinton là một mẹo xấp xỉ: chỉ chạy $k$ bước MCMC (thường $k = 1$) **bắt đầu từ chính dữ liệu thật**, rồi dùng mẫu thô đó cho pha âm.

```text
Với mỗi batch dữ liệu thật x+:
  x- = MCMC_k_buoc(khoi tao = x+)
  loss = E(x+) - E(x-)
  cap nhat theta theo -grad(loss)
```

Trực giác: khởi tạo từ dữ liệu thật rồi chạy vài bước cho biết mô hình "muốn kéo" dữ liệu đi đâu. Nếu mô hình đã khớp, $x^-$ sẽ ở gần $x^+$ và gradient nhỏ; nếu chưa, $x^-$ trôi xa, tạo tín hiệu sửa lỗi.

CD đánh đổi độ chính xác lấy tốc độ: nó **thiên lệch (biased)** nhưng thực dụng. **Persistent CD (PCD)** cải thiện bằng cách **duy trì chuỗi MCMC liên tục** giữa các batch thay vì khởi tạo lại từ dữ liệu, giúp pha âm khám phá không gian rộng hơn.

---

# 7. Né hằng số chuẩn hóa: các hướng khác

Vì $Z_\theta$ là gốc rễ mọi khó khăn, nhiều phương pháp tránh nó **hoàn toàn** thay vì xấp xỉ bằng MCMC.

---

## 7.1. Score Matching

Ý tưởng: thay vì khớp **mật độ** $p_\theta$, ta khớp **hàm điểm (score function)**:

$$\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$$

Như đã chứng minh ở mục 5.1, gradient theo $x$ làm $Z_\theta$ biến mất, nên ta né được nó **một cách chính xác**, không xấp xỉ. Mục tiêu score matching của Hyvärinen:

$$J(\theta) = \mathbb{E}_{p_{\text{data}}}\left[ \frac{1}{2}\big\lVert \nabla_x \log p_\theta(x) \big\rVert^2 + \operatorname{tr}\!\big(\nabla_x^2 \log p_\theta(x)\big) \right]$$

Số hạng thứ nhất phạt score quá lớn; số hạng vết Hessian (Laplacian) ràng buộc hình dạng. Cả hai chỉ dùng đạo hàm theo $x$ — không có $Z_\theta$ ở đâu cả.

> Đây chính là cây cầu nối EBM với **mô hình khuếch tán (diffusion model)**: diffusion là một dạng score matching khử nhiễu (denoising score matching) ước lượng score trên **nhiều mức nhiễu** khác nhau. Toàn bộ mạch này — phân kỳ Fisher, định lý Hyvärinen, denoising score matching, NCSN và góc nhìn SDE — được khai triển đầy đủ ở bài **[Score-Based Models](#/score-based-models)**.

---

## 7.2. Noise Contrastive Estimation (NCE)

Ý tưởng: biến bài toán ước lượng mật độ thành bài toán **phân loại nhị phân (binary classification)**.

Ta trộn dữ liệu thật với mẫu nhiễu sinh từ một phân phối tham chiếu đã biết $p_n(x)$, rồi dạy mô hình phân biệt "thật" với "nhiễu". Điểm khéo léo: **hằng số chuẩn hóa $Z_\theta$ được coi là một tham số học được** (thay vì một tích phân phải tính).

Nếu mô hình phân biệt được hoàn hảo thật–nhiễu, có thể chứng minh phân phối ngầm của nó hội tụ về $p_{\text{data}}$, và tham số chuẩn hóa hội tụ về $\log Z_\theta$ đúng. Ta đã đổi một tích phân bất khả thi lấy một bài toán phân loại quen thuộc.

---

## 7.3. So sánh các hướng huấn luyện

| Phương pháp | Cách xử lý $Z_\theta$ | Cần MCMC? | Đánh đổi |
| --- | --- | --- | --- |
| **CD / PCD** | Xấp xỉ gradient bằng mẫu MCMC | Có ($k$ bước) | Thiên lệch, gradient nhiễu |
| **Score matching** | Né hoàn toàn (gradient theo $x$) | Không | Cần Hessian, nhạy với chiều cao |
| **NCE** | Học $Z_\theta$ như tham số | Không | Phụ thuộc chất lượng phân phối nhiễu |

---

# 8. Các đại diện tiêu biểu

Mọi mô hình dưới đây đều dùng chung phân phối Boltzmann $p_\theta(x) \propto \exp(-E_\theta(x))$; chúng chỉ khác ở **dạng của $E_\theta$** và **cách né $Z_\theta$**.

| Mô hình | Hàm năng lượng | Ghi chú |
| --- | --- | --- |
| **Máy Boltzmann hạn chế (Restricted Boltzmann Machine — RBM)** | Song tuyến tính với biến ẩn nhị phân | EBM cổ điển, huấn luyện bằng CD |
| **Deep Boltzmann Machine** | Nhiều tầng biến ẩn | Mở rộng sâu của RBM |
| **Deep EBM (IGEBM)** | CNN sâu | Lấy mẫu bằng Langevin |
| **JEM (Joint Energy Model)** | Tái diễn giải logits của bộ phân loại | Vừa phân loại vừa sinh |

---

## 8.1. Bộ phân loại softmax là một EBM

Một kết quả gây bất ngờ: **bộ phân loại softmax quen thuộc chính là một EBM trá hình**.

Một bộ phân loại với logits $f_\theta(x)[y]$ định nghĩa:

$$p_\theta(y \mid x) = \frac{\exp\big(f_\theta(x)[y]\big)}{\sum_{y'} \exp\big(f_\theta(x)[y']\big)}$$

Đặt năng lượng $E_\theta(x, y) = -f_\theta(x)[y]$. Khi đó

$$p_\theta(y \mid x) = \frac{\exp\big(-E_\theta(x, y)\big)}{\sum_{y'} \exp\big(-E_\theta(x, y')\big)}$$

đúng là một phân phối Boltzmann trên cặp $(x, y)$, với mẫu số đóng vai trò hàm phân hoạch (ở đây tính được vì $y$ rời rạc và hữu hạn). JEM khai thác chính điều này: lấy một bộ phân loại đã huấn luyện và đọc nó như một mô hình sinh, vừa dự đoán nhãn vừa sinh ảnh.

> Đây là minh chứng cho tuyên bố ở mục 2.1: rất nhiều mô hình quen thuộc thật ra là EBM với một mẹo cụ thể để khiến $Z_\theta$ tính được.

---

# 9. Ưu điểm

* **Linh hoạt nhất** — từ mục 2.1, EBM bao trùm mọi phân phối dương, không ràng buộc kiến trúc; $E_\theta$ chỉ cần khả vi.
* **Hợp nhất lý thuyết** — liên hệ trực tiếp tới score matching, diffusion, bộ phân loại softmax (mục 8.1) và vật lý thống kê.
* **Hàm năng lượng diễn giải được** — $E_\theta(x)$ đo trực tiếp mức "bất thường" của $x$, rất hữu ích cho phát hiện ngoài phân phối.
* **Tự do tái dùng kiến trúc mạnh** — bất kỳ CNN hay Transformer nào cũng dùng làm $E_\theta$ được mà không cần sửa đổi.

---

# 10. Nhược điểm

Mọi nhược điểm đều quy về cùng một gốc: **hằng số chuẩn hóa $Z_\theta$**.

**Không có likelihood chính xác.** $Z_\theta$ là tích phân bất khả thi (mục 2), nên không tính được $\log p_\theta(x)$ tường minh — khác hẳn autoregressive hay flow.

**Lấy mẫu chậm và khó.** Phải chạy MCMC/Langevin nhiều bước (mục 5), dễ kẹt ở một mode, độ trễ cao khi sinh.

**Huấn luyện bất ổn.** Pha âm dựa trên mẫu MCMC thô (mục 6) khiến gradient nhiễu và khó hội tụ; CD lại còn thiên lệch.

**Khó mở rộng.** Lên ảnh độ phân giải cao, EBM thuần kém ổn định hơn diffusion — vốn chính là một cách huấn luyện EBM khéo léo hơn.

---

# 11. Tổng kết

EBM không phát minh ra một dạng xác suất mới mà đẩy ý tưởng "gán điểm tin cậy" tới giới hạn tự do nhất:

$$p_\theta(x) = \frac{\exp\big(-E_\theta(x)\big)}{Z_\theta}$$

Toàn bộ độ khó dồn vào hằng số chuẩn hóa $Z_\theta$. Kết quả cốt lõi — gradient hai pha (mục 4.1) — cho thấy dù $Z_\theta$ bất khả thi, **gradient** của nó vẫn là một kỳ vọng lấy mẫu được. Từ đó nảy ra mọi kỹ thuật: MCMC/Langevin, CD, score matching, NCE — mỗi cách là một chiến lược khác nhau để né hoặc xấp xỉ $Z_\theta$.

EBM là họ mô hình **tổng quát nhất về mặt khái niệm**: autoregressive, flow, VAE, GAN và diffusion đều có thể nhìn như EBM kèm một mẹo riêng để khiến $Z_\theta$ trở nên xử lý được. Diffusion thành công rực rỡ chính vì nó là cách huấn luyện EBM (qua score matching trên nhiều mức nhiễu) mà vừa ổn định vừa lấy mẫu tốt.

> Khép lại nhóm bài về mô hình sinh: ta đã đi từ **autoregressive** (likelihood chính xác, lấy mẫu tuần tự), qua **normalizing flow** (khả nghịch, một bước), **VAE** (biến ẩn, ELBO), **GAN** (đối kháng, không likelihood), tới **EBM** ở đây. Bài học sâu sắc nhất mà EBM để lại là: khó khăn cốt lõi của mọi mô hình sinh nằm ở **hằng số chuẩn hóa**, và phần lớn lịch sử lĩnh vực này chính là lịch sử của những cách thông minh để né tránh nó.

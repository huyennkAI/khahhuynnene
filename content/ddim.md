# Mô hình khuếch tán ẩn (DDIM)

> DDIM (Denoising Diffusion Implicit Models, Song và cộng sự, 2020) tăng tốc lấy mẫu của diffusion **mà không huấn luyện lại**. Mấu chốt: mục tiêu huấn luyện của DDPM **chỉ phụ thuộc các phân phối biên** $q(x_t\mid x_0)$, nên ta tự do thay quá trình thuận Markov bằng một họ quá trình **không Markov (non-Markovian)** chia sẻ đúng mục tiêu đó — nhưng có quá trình nghịch **tất định (deterministic)** và **nhảy bước được**.
>
> Kết quả: lấy mẫu **nhanh hơn 10–50 lần**, một bộ sinh tất định biến $x_T$ thành một **mã ẩn (latent code)** có ý nghĩa ngữ nghĩa, cho phép nội suy ảnh mượt mà.

---

# 1. Động cơ ra đời

[Diffusion](#/diffusion-models) (DDPM) cho mẫu tuyệt đẹp nhưng có một điểm yếu chí mạng: **lấy mẫu chậm**. Quá trình nghịch là một chuỗi Markov, phải đi **tuần tự** đủ $T$ bước (thường hàng trăm đến hàng nghìn), mỗi bước một lần gọi U-Net. Không thể "nhảy" vì mỗi bước nghịch $p_\theta(x_{t-1}\mid x_t)$ chỉ định nghĩa cho hai mức **liền kề**.

Câu hỏi của DDIM:

> Có thể giữ nguyên mạng $\epsilon_\theta$ đã huấn luyện (rất tốn kém) mà **đổi cách lấy mẫu** để nhanh hơn nhiều lần không?

Câu trả lời đến từ một quan sát tinh tế về **chính xác cái gì** mà DDPM thực sự học.

> **Lưu ý ký hiệu.** Bài này dùng $\bar\alpha_t = \prod_{s=1}^t \alpha_s$ cho tích lũy, **thống nhất với [Diffusion](#/diffusion-models)**. Bài báo DDIM gốc ký hiệu đại lượng tích lũy này là $\alpha_t$ — đừng nhầm.

---

# 2. Quan sát then chốt: mục tiêu chỉ phụ thuộc phân phối biên

Nhắc lại mục tiêu rút gọn của DDPM ([Diffusion](#/diffusion-models) mục 5.4):

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\, \epsilon,\, t}\Big[\big\lVert \epsilon - \epsilon_\theta(x_t, t)\big\rVert^2\Big], \qquad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$

Hãy nhìn kỹ: vế phải **chỉ** dùng tới $x_t$ sinh từ $x_0$ qua **phân phối biên (marginal)** $q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I)$. Nó **không hề** đụng tới quá trình thuận đầy đủ $q(x_{1:T}\mid x_0)$ hay tính chất Markov từng bước $q(x_t\mid x_{t-1})$.

**Hệ quả mạnh mẽ:**

> Mọi quá trình thuận có **cùng các biên** $q(x_t\mid x_0)$ — dù Markov hay không — đều dẫn tới **đúng** mục tiêu $\mathcal{L}_{\text{simple}}$. Một mạng $\epsilon_\theta$ đã huấn luyện theo DDPM **dùng lại được** cho bất kỳ quá trình nào trong họ đó.

DDPM chỉ là **một** thành viên của họ này. DDIM đi tìm những thành viên khác có quá trình nghịch tốt hơn cho việc lấy mẫu.

---

# 3. Họ quá trình thuận không Markov

DDIM định nghĩa một họ quá trình **lập chỉ mục bởi vector $\sigma \in \mathbb{R}_{\ge 0}^T$**, qua các phân phối nghịch **có điều kiện $x_0$**:

$$q_\sigma(x_{t-1}\mid x_t, x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}},\;\; \sigma_t^2 I\right)$$

Trung bình được **cố ý chọn** sao cho họ này giữ đúng biên mong muốn:

**Mệnh đề (bất biến biên).** Nếu $q(x_t\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$ thì với $q_\sigma$ ở trên, ta cũng có $q(x_{t-1}\mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}}\,x_0, (1-\bar\alpha_{t-1})I)$.

**Ý chứng minh.** $x_{t-1}$ là tổ hợp tuyến tính của $x_0$ và $\tfrac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}}$ (vốn là một Gauss chuẩn $\mathcal{N}(0,I)$ khi biết $x_0$) cộng nhiễu $\sigma_t$. Cộng phương sai: $(1-\bar\alpha_{t-1}-\sigma_t^2) + \sigma_t^2 = 1-\bar\alpha_{t-1}$, đúng phương sai biên cần có. $\blacksquare$

Vì mọi biên trùng với DDPM, theo mục 2 mục tiêu huấn luyện không đổi. Bài báo phát biểu chính xác điều này:

**Định lý 1 (DDIM).** Với mọi $\sigma > 0$, tồn tại trọng số $\gamma$ và hằng số $C$ sao cho mục tiêu biến phân $J_\sigma = \mathcal{L}_\gamma + C$. Nói cách khác, **mọi lựa chọn $\sigma$ đều khớp cùng một họ mục tiêu** $\mathcal{L}_{\text{simple}}$ — một mạng đã huấn luyện dùng được cho tất cả.

---

# 4. Quá trình sinh

Khi lấy mẫu, ta **không biết** $x_0$. Mẹo (giống DDPM, [Diffusion](#/diffusion-models) mục 5.3): từ $x_t$ và nhiễu dự đoán $\epsilon_\theta(x_t,t)$, **ước lượng** $x_0$:

$$\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

Thay $\hat x_0$ vào trung bình của $q_\sigma$ (và để ý $\tfrac{x_t-\sqrt{\bar\alpha_t}\hat x_0}{\sqrt{1-\bar\alpha_t}} = \epsilon_\theta$), ta được **phương trình sinh DDIM** (Eq. 12 của bài báo):

$$\boxed{\;x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\underbrace{\hat{x}_0(x_t,t)}_{\text{(1) đoán ảnh sạch}} + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\epsilon_\theta(x_t,t)}_{\text{(2) hướng chỉ về }x_t} + \underbrace{\sigma_t\,\epsilon}_{\text{(3) nhiễu mới}}\;}$$

với $\epsilon\sim\mathcal{N}(0,I)$. Ba thành phần có ý nghĩa rõ ràng:

1. **Đoán ảnh sạch** $\hat x_0$ rồi co lại về mức nhiễu $t-1$ bằng $\sqrt{\bar\alpha_{t-1}}$.
2. **Hướng xác định chỉ tới $x_t$** — phần nhiễu được giữ lại một cách *có kiểm soát*, không ngẫu nhiên.
3. **Nhiễu mới ngẫu nhiên** với độ lớn $\sigma_t$ — đây là núm vặn duy nhất tách DDIM khỏi DDPM.

---

# 5. Núm vặn $\sigma$: nội suy DDPM ↔ DDIM

Tham số $\sigma_t$ điều khiển **mức ngẫu nhiên** của quá trình sinh. Bài báo chọn

$$\sigma_t(\eta) = \eta\,\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\,\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}}$$

với một hệ số $\eta \ge 0$. Hai đầu mút:

| $\eta$ | $\sigma_t$ | Quá trình sinh | Bản chất |
| --- | --- | --- | --- |
| $\eta = 1$ | $\sigma_t^2 = \tilde\beta_t$ | **DDPM** | ngẫu nhiên (Markov gốc) |
| $\eta = 0$ | $\sigma_t = 0$ | **DDIM** | **tất định** hoàn toàn |

* **$\eta=1$** cho $\sigma_t^2 = \tfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t = \tilde\beta_t$ — đúng phương sai hậu nghiệm của DDPM ([Diffusion](#/diffusion-models) mục 4), nên ta thu lại **chính xác** bộ lấy mẫu DDPM ngẫu nhiên.
* **$\eta=0$** xóa hẳn thành phần (3): mỗi bước hoàn toàn **tất định**. Một $x_T$ cố định luôn cho **đúng một** $x_0$. Đây là **DDIM**.

Giữa hai đầu, $\eta\in(0,1)$ cho một phổ liên tục đánh đổi ngẫu nhiên ↔ tất định.

---

# 6. DDIM tất định là một ODE

Khi $\sigma_t = 0$, quá trình sinh không còn nhiễu — nó trở thành phép giải số một **phương trình vi phân thường (ODE)**. Đây là cầu nối trực tiếp với probability flow ODE của [Score-Based Models](#/score-based-models).

**Suy ra.** Với $\sigma_t=0$, phương trình (12) thành

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}} + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta$$

Chia hai vế cho $\sqrt{\bar\alpha_{t-1}}$ và nhóm lại:

$$\frac{x_{t-1}}{\sqrt{\bar\alpha_{t-1}}} = \frac{x_t}{\sqrt{\bar\alpha_t}} + \left(\sqrt{\frac{1-\bar\alpha_{t-1}}{\bar\alpha_{t-1}}} - \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\right)\epsilon_\theta$$

Đặt biến **tái tham số hóa** $\sigma(t) = \sqrt{\tfrac{1-\bar\alpha_t}{\bar\alpha_t}}$ và $\bar x = \tfrac{x}{\sqrt{\bar\alpha_t}}$, đẳng thức trên gọn lại thành $\bar x_{t-1} - \bar x_t = (\sigma_{t-1}-\sigma_t)\,\epsilon_\theta$. Đây đúng là **rời rạc Euler** của ODE liên tục:

$$d\bar x = \epsilon_\theta\!\left(\frac{\bar x}{\sqrt{\sigma^2+1}},\, t\right) d\sigma$$

(dùng $\sqrt{\bar\alpha_t} = 1/\sqrt{\sigma^2+1}$, nên $x = \bar x/\sqrt{\sigma^2+1}$). Hệ quả quan trọng:

* Giải ODE bằng các **bộ giải bậc cao** (Runge–Kutta…) cho phép dùng **ít bước** mà vẫn chính xác.
* Vì là ODE tất định và khả nghịch, DDIM còn cho phép **mã hóa** ảnh thật về $x_T$ (chạy ODE xuôi) rồi tái tạo lại — một dạng luồng chuẩn hóa liên tục (continuous normalizing flow).

---

# 7. Lấy mẫu tăng tốc bằng dãy con bước thời gian

Đây là lợi ích thực dụng nhất. Vì mục tiêu chỉ phụ thuộc các biên (mục 2), quá trình nghịch **không bị buộc** phải đi qua **mọi** mức $T, T-1, \dots, 1$. Ta chọn một **dãy con (subsequence)** $\tau = (\tau_1 < \tau_2 < \dots < \tau_S)$ của $\{1, \dots, T\}$ với $S \ll T$, rồi áp dụng cùng phương trình (12) **chỉ trên các mức trong $\tau$**:

$$x_{\tau_{i-1}} = \sqrt{\bar\alpha_{\tau_{i-1}}}\,\hat x_0(x_{\tau_i}, \tau_i) + \sqrt{1-\bar\alpha_{\tau_{i-1}}-\sigma_{\tau_i}^2}\,\epsilon_\theta(x_{\tau_i}, \tau_i) + \sigma_{\tau_i}\epsilon$$

Mỗi bước có thể "nhảy" qua nhiều mức nhiễu một lúc. Với DDIM tất định ($\eta=0$), nhảy bước hầu như **không hại** chất lượng vì ta đang giải một ODE trơn. Thực nghiệm: dùng $S = 20$–$100$ bước thay vì $T = 1000$ cho mẫu chất lượng tương đương — **nhanh hơn 10–50 lần**, trong khi DDPM ngẫu nhiên xuống cấp rõ rệt khi giảm bước.

```python
# Lấy mẫu DDIM tất định trên dãy con tau (eta = 0)
x = torch.randn(...)                      # x_T ~ N(0, I)
for i in reversed(range(len(tau))):
    t, s = tau[i], tau[i-1] if i > 0 else 0
    eps = model(x, t)
    x0_hat = (x - sqrt(1 - acp[t]) * eps) / sqrt(acp[t])
    x = sqrt(acp[s]) * x0_hat + sqrt(1 - acp[s]) * eps   # sigma = 0
```

---

# 8. Tính nhất quán và nội suy ngữ nghĩa

Tính tất định của DDIM ($\eta=0$) sinh ra hai tính chất mà DDPM **không có**.

**Nhất quán (consistency).** Cùng một $x_T$ luôn cho cùng một $x_0$, bất kể số bước hay lịch dãy con $\tau$. Hơn nữa, các $x_0$ sinh từ cùng $x_T$ với số bước khác nhau có **đặc trưng mức cao giống nhau**. Điều này biến $x_T$ thành một **mã ẩn (latent code)** ổn định, đặc trưng cho ảnh — vai trò mà trong DDPM bị nhiễu ngẫu nhiên ở mỗi bước xóa nhòa.

**Nội suy ngữ nghĩa.** Vì $x_T$ là latent code có ý nghĩa, ta nội suy **trong không gian $x_T$** rồi giải DDIM để được chuỗi ảnh chuyển tiếp mượt. Do prior là Gauss nhiều chiều, nên dùng **nội suy cầu (spherical interpolation, slerp)** thay vì tuyến tính:

$$x_T^{(\lambda)} = \frac{\sin((1-\lambda)\phi)}{\sin\phi}\,x_T^{(0)} + \frac{\sin(\lambda\phi)}{\sin\phi}\,x_T^{(1)}, \qquad \phi = \arccos\!\frac{\langle x_T^{(0)}, x_T^{(1)}\rangle}{\lVert x_T^{(0)}\rVert\,\lVert x_T^{(1)}\rVert}$$

Đây là điều GAN làm tốt nhờ không gian ẩn liền mạch, mà DDPM thuần khó làm — DDIM lấy lại được khả năng đó cho diffusion.

---

# 9. Ưu điểm

* **Lấy mẫu nhanh, không huấn luyện lại** — dùng thẳng mạng $\epsilon_\theta$ của DDPM, chỉ đổi bộ lấy mẫu; nhanh hơn 10–50 lần qua dãy con $\tau$ (mục 7).
* **Tất định + latent code có ý nghĩa** — $x_T$ ổn định cho phép nội suy ngữ nghĩa và mã hóa/tái tạo ảnh (mục 6, 8).
* **Đánh đổi linh hoạt** — núm $\eta$ trượt liên tục giữa DDPM (đa dạng, ngẫu nhiên) và DDIM (nhanh, tất định); núm $S$ đổi tốc độ lấy chất lượng.
* **Cầu nối lý thuyết** — dạng ODE của DDIM là một hiện thân cụ thể của probability flow ODE, nối liền với [Score-Based Models](#/score-based-models).

---

# 10. Nhược điểm

* **Mẫu kém đa dạng hơn một chút** — bỏ nhiễu ngẫu nhiên làm giảm tính khám phá; ở số bước rất thấp, DDIM có thể mất chi tiết tần số cao.
* **Vẫn cần nhiều bước hơn mô hình một bước** — nhanh hơn DDPM nhưng vẫn chậm hơn GAN/flow; với $S$ quá nhỏ chất lượng vẫn xuống.
* **Không cải thiện huấn luyện** — DDIM thuần là kỹ thuật **suy luận (inference)**; chất lượng trần vẫn bị giới hạn bởi mạng $\epsilon_\theta$ đã có.

---

# 11. Tổng kết

DDIM không đổi một dòng nào trong huấn luyện diffusion. Toàn bộ sức mạnh đến từ một quan sát: **mục tiêu $\mathcal{L}_{\text{simple}}$ chỉ ràng buộc các phân phối biên $q(x_t\mid x_0)$**, nên ta tự do thay quá trình thuận Markov bằng cả một họ không Markov chia sẻ đúng mục tiêu đó:

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta + \sigma_t\,\epsilon$$

Núm $\sigma_t$ ($\eta=1\to0$) trượt từ **DDPM ngẫu nhiên** sang **DDIM tất định**; ở đầu tất định, mỗi bước là một bước Euler của một **ODE**, cho phép nhảy bước để lấy mẫu nhanh 10–50 lần và biến $x_T$ thành latent code nội suy được.

DDIM đổi **một chút đa dạng** lấy **tốc độ và khả năng điều khiển** — và bắc cầu thẳng tới góc nhìn liên tục.

> Bài tiếp theo — **[Score-Based Models](#/score-based-models)** — tổng quát hóa chính ODE tất định này thành **probability flow ODE**, đặt cả DDPM lẫn DDIM vào một khung phương trình vi phân ngẫu nhiên (SDE) duy nhất, nơi "dự đoán nhiễu" lộ ra chính là "khớp score".

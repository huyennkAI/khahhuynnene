# Mô hình khuếch tán (Diffusion Models)

> Mô hình khuếch tán (diffusion model) học cách **khử nhiễu dần dần (iterative denoising)**: bắt đầu từ nhiễu thuần (pure noise) rồi từng bước biến nó thành một mẫu sắc nét.
>
> $$x_T \;\rightarrow\; x_{T-1} \;\rightarrow\; \cdots \;\rightarrow\; x_1 \;\rightarrow\; x_0$$
>
> Đây là nền tảng của Stable Diffusion, DALL·E và phần lớn các hệ sinh ảnh hiện đại.

---

# 1. Động cơ ra đời

Các họ mô hình sinh trước đó đều có điểm yếu cố hữu.

GAN cho mẫu rất đẹp nhưng huấn luyện bất ổn — bản chất là một bài toán điểm yên ngựa (saddle point) giữa bộ sinh và bộ phân biệt — và hay rơi vào sụp đổ chế độ (mode collapse).

VAE thì ổn định nhưng mẫu thường **mờ**, vì giải mã chỉ trong một bước nên buộc phải trung bình hóa nhiều khả năng.

Trực giác của diffusion xuất phát từ một quan sát đơn giản:

> Sinh một bức ảnh hoàn chỉnh chỉ trong **một bước** là quá khó (đó chính là chỗ GAN chật vật). Nhưng **khử đi một chút nhiễu** thì lại dễ. Vậy hãy chia bài toán khó thành hàng trăm bước nhỏ dễ.

Đây là tư duy "chia để trị" trên miền xác suất: thay vì học một phép nhảy lớn từ nhiễu sang dữ liệu, ta học rất nhiều phép nhảy nhỏ, mỗi phép chỉ cần khử nhiễu nhẹ.

---

# 2. Diffusion khắc phục điều gì?

Trước khi đi vào toán, hãy nêu rõ diffusion sửa được những gì so với hai họ đi trước.

| Tiêu chí | GAN | VAE | Diffusion |
| --- | --- | --- | --- |
| Mục tiêu huấn luyện | đối kháng (minimax) | ELBO + tái tạo | hồi quy L2 (dự đoán nhiễu) |
| Ổn định | thấp (điểm yên ngựa) | cao | cao |
| Độ đa dạng mẫu | dễ mode collapse | tốt | rất tốt (mode-covering) |
| Độ sắc nét | cao | thấp (mờ) | cao |
| Tốc độ lấy mẫu | nhanh (1 bước) | nhanh (1 bước) | chậm ($T$ bước) |

**So với GAN.** Mục tiêu huấn luyện của diffusion là một bài toán **hồi quy bình phương (regression)**, có nghiệm tối ưu xác định và gradient luôn có ý nghĩa. Không có động lực đối kháng (adversarial dynamics) nên không mất ổn định, không mode collapse. Mục tiêu lấy cảm hứng từ MLE nên có xu hướng **phủ mode (mode-covering)**, cho độ đa dạng cao.

**So với VAE.** Thay vì một bước giải mã (gây mờ do trung bình hóa), diffusion tinh chỉnh qua hàng trăm bước, mỗi bước thêm một ít chi tiết, nên mẫu sắc nét.

**Cái giá phải trả.** Lấy mẫu chậm, vì cần rất nhiều bước suy luận (mục 9).

---

# 3. Hai quá trình của diffusion

Diffusion gồm hai quá trình ngược chiều nhau: một quá trình **thuận** phá hủy dữ liệu thành nhiễu, và một quá trình **nghịch** học cách đảo ngược.

## 3.1. Quá trình thuận (forward / diffusion process)

Quá trình thuận dần phá hủy dữ liệu bằng cách cộng nhiễu Gauss qua $T$ bước, theo một lịch nhiễu (noise schedule) $\beta_t \in (0, 1)$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left( x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I \right)$$

Mỗi bước co nhẹ tín hiệu cũ lại (nhân với $\sqrt{1-\beta_t}$) rồi cộng thêm một chút nhiễu mới. Quá trình này **cố định, không có tham số học** — nó chỉ là cách ta định nghĩa "làm hỏng" dữ liệu.

Điều đẹp đẽ là ta không cần mô phỏng từng bước một để biết phân phối tại bước $t$ bất kỳ.

---

## 3.2. Tính chất nhảy bước: $q(x_t \mid x_0)$ có dạng đóng

Đặt $\alpha_t = 1 - \beta_t$ và $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$.

**Mệnh đề (closed-form forward).** Với mọi $t$:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\right) \;\Longleftrightarrow\; x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)$$

**Chứng minh** (quy nạp theo $t$). Giả thiết quy nạp:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\,\bar{\epsilon}, \qquad \bar{\epsilon} \sim \mathcal{N}(0, I)$$

Thay vào định nghĩa một bước thuận $x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{\beta_t}\,\epsilon'$ với $\epsilon' \sim \mathcal{N}(0,I)$ độc lập với $\bar\epsilon$:

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t}\left(\sqrt{\bar{\alpha}_{t-1}}\, x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\,\bar{\epsilon}\right) + \sqrt{1-\alpha_t}\,\epsilon' \\
&= \sqrt{\alpha_t \bar{\alpha}_{t-1}}\, x_0 + \underbrace{\sqrt{\alpha_t (1 - \bar{\alpha}_{t-1})}\,\bar{\epsilon} + \sqrt{1 - \alpha_t}\,\epsilon'}_{\text{tổng hai Gauss độc lập, trung bình } 0}
\end{aligned}
$$

Tổng hai biến Gauss độc lập trung bình $0$ lại là một Gauss trung bình $0$, với phương sai cộng dồn:

$$\alpha_t(1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t\bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t$$

Do đó

$$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

đúng điều phải chứng minh. Cơ sở quy nạp $t=1$ trùng với định nghĩa một bước thuận. $\blacksquare$

Hệ quả: khi $t \to T$, lịch nhiễu khiến $\bar{\alpha}_t \to 0$, nên

$$x_T \to \mathcal{N}(0, I)$$

tức nhiễu thuần, mất hết thông tin của $x_0$. Đây là điểm xuất phát cho quá trình sinh.

---

## 3.3. Quá trình nghịch (reverse process)

Mô hình học cách **đảo ngược**, từng bước đưa nhiễu trở về dữ liệu:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left( x_{t-1};\, \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t) \right)$$

Một câu hỏi tự nhiên:

> Vì sao ta được phép giả định bước nghịch cũng là Gauss?

**Cơ sở lý thuyết (Feller, 1949).** Nếu mỗi bước thuận đủ nhỏ ($\beta_t \ll 1$), thì bước nghịch thật sự $q(x_{t-1}\mid x_t)$ **cũng xấp xỉ Gauss**. Đây chính là lý do giả định Gauss cho $p_\theta$ là chính đáng — và cũng là lý do diffusion buộc phải dùng **nhiều bước nhỏ** thay vì vài bước lớn.

---

# 4. Hậu nghiệm thuận có dạng đóng

Bước nghịch thật sự $q(x_{t-1}\mid x_t)$ là khó (nó phụ thuộc vào toàn bộ phân phối dữ liệu). Nhưng mấu chốt của huấn luyện nằm ở một quan sát:

> Khi **biết thêm $x_0$**, hậu nghiệm thuận lại là một Gauss tường minh.

**Mệnh đề.** Với $\tilde{\beta}_t$ và $\tilde\mu_t$ xác định dưới đây:

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t I\right)$$

trong đó

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t, \qquad \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\,\beta_t$$

**Chứng minh.** Dùng quy tắc Bayes và tính Markov của quá trình thuận:

$$q(x_{t-1}\mid x_t, x_0) = \frac{q(x_t \mid x_{t-1})\, q(x_{t-1}\mid x_0)}{q(x_t \mid x_0)} \;\propto\; q(x_t\mid x_{t-1})\, q(x_{t-1}\mid x_0)$$

Cả ba phân phối đều là Gauss đã biết (mục 3.1 và 3.2). Lấy logarit, giữ lại các số hạng phụ thuộc $x_{t-1}$:

$$
\begin{aligned}
\log q(x_{t-1}\mid x_t, x_0)
&= -\frac{1}{2}\left[\frac{(x_t - \sqrt{\alpha_t}\,x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\,x_0)^2}{1 - \bar{\alpha}_{t-1}}\right] + C' \\
&= -\frac{1}{2}\left[\left(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}\right)x_{t-1}^2 - 2\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}x_0\right)x_{t-1}\right] + C''
\end{aligned}
$$

Đây là một đa thức bậc hai theo $x_{t-1}$, nên hậu nghiệm là Gauss. Phương sai là nghịch đảo hệ số bậc hai:

$$\tilde{\beta}_t^{-1} = \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} = \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})} = \frac{1 - \bar{\alpha}_t}{\beta_t(1 - \bar{\alpha}_{t-1})}$$

tức $\tilde{\beta}_t = \dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\,\beta_t$. Trung bình là hệ số bậc nhất nhân $\tilde\beta_t$:

$$\tilde{\mu}_t = \tilde{\beta}_t\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}x_0\right) = \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\, x_0$$

đúng điều phải chứng minh. $\blacksquare$

Hậu nghiệm dạng đóng này chính là "đáp án mẫu" mà mạng nghịch $p_\theta$ phải bắt chước.

---

# 5. Từ ELBO đến mục tiêu dự đoán nhiễu

Phần này là cốt lõi lý thuyết: ta sẽ đi từ một chặn dưới biến phân tổng quát tới một mục tiêu hồi quy L2 đơn giản đến bất ngờ.

## 5.1. Chặn dưới biến phân (ELBO)

Giống VAE, diffusion tối ưu ELBO trên $\log p_\theta(x_0)$, coi toàn bộ chuỗi nhiễu $x_{1:T}$ là biến ẩn (latent variable):

$$\log p_\theta(x_0) \ge \mathbb{E}_q\!\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right] =: -\mathcal{L}$$

Khai triển và sắp xếp lại (dùng tính Markov của cả hai quá trình, và khéo léo đưa $x_0$ vào điều kiện để tạo ra hậu nghiệm dạng đóng ở mục 4), ta thu được:

$$\mathcal{L} = \mathbb{E}_q\!\Big[ \underbrace{D_{\mathrm{KL}}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{L_T:\ \text{không tham số}} + \sum_{t>1} \underbrace{D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}} \underbrace{-\,\log p_\theta(x_0\mid x_1)}_{L_0} \Big]$$

Số hạng $L_T$ không chứa tham số $\theta$ (đầu vào $x_T$ luôn gần $\mathcal{N}(0,I)$), nên ta tập trung vào các $L_{t-1}$.

---

## 5.2. Mỗi số hạng KL rút về khoảng cách giữa hai trung bình

$L_{t-1}$ là KL giữa **hai Gauss cùng phương sai**: $q(x_{t-1}\mid x_t, x_0)$ có dạng đóng (mục 4), còn $p_\theta$ ta cố ý đặt cùng hiệp phương sai $\Sigma = \tilde{\beta}_t I$.

KL giữa hai Gauss cùng hiệp phương sai chỉ còn lại khoảng cách bình phương giữa hai trung bình:

$$L_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\tilde{\beta}_t}\big\lVert \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\big\rVert^2\right] + C$$

với $C$ là hằng số không phụ thuộc $\theta$. Việc còn lại chỉ là cho $\mu_\theta$ khớp với $\tilde\mu_t$.

---

## 5.3. Tham số hóa dự đoán nhiễu — bước then chốt

Đây là mẹo quan trọng nhất của DDPM. Từ tính chất nhảy bước (mục 3.2), ta đảo ngược để biểu diễn $x_0$ qua $x_t$ và nhiễu:

$$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\big(x_t - \sqrt{1 - \bar{\alpha}_t}\,\epsilon\big)$$

**Mệnh đề.** Thế biểu thức $x_0$ này vào $\tilde\mu_t$ (mục 4) thì trung bình hậu nghiệm rút gọn thành:

$$\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\epsilon\right)$$

**Chứng minh.** Thay $x_0 = \tfrac{1}{\sqrt{\bar\alpha_t}}(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)$ vào số hạng chứa $x_0$ của $\tilde\mu_t$:

$$
\begin{aligned}
\tilde\mu_t
&= \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\cdot\frac{1}{\sqrt{\bar{\alpha}_t}}\big(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\big) \\
&= \left[\frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{(1 - \bar{\alpha}_t)\sqrt{\bar\alpha_t}}\right] x_t - \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}}\sqrt{1-\bar\alpha_t}\,\epsilon
\end{aligned}
$$

Dùng $\sqrt{\bar\alpha_t} = \sqrt{\alpha_t}\sqrt{\bar\alpha_{t-1}}$, hệ số của $x_t$ trở thành

$$\frac{1}{1-\bar\alpha_t}\left[\sqrt{\alpha_t}(1-\bar\alpha_{t-1}) + \frac{\beta_t}{\sqrt{\alpha_t}}\right] = \frac{1}{\sqrt{\alpha_t}\,(1-\bar\alpha_t)}\left[\alpha_t(1-\bar\alpha_{t-1}) + \beta_t\right] = \frac{1-\bar\alpha_t}{\sqrt{\alpha_t}\,(1-\bar\alpha_t)} = \frac{1}{\sqrt{\alpha_t}}$$

còn hệ số của $\epsilon$ là

$$\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{\sqrt{\bar\alpha_t}\,\sqrt{1-\bar\alpha_t}} = \frac{\beta_t}{\sqrt{\alpha_t}\,\sqrt{1-\bar\alpha_t}}$$

Gộp lại đúng bằng $\tilde\mu_t = \tfrac{1}{\sqrt{\alpha_t}}\big(x_t - \tfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\big)$. $\blacksquare$

Quan sát then chốt: trong biểu thức này, $x_t$ đã biết (là đầu vào), thứ duy nhất chưa biết là **nhiễu $\epsilon$**. Vậy thay vì cho mạng dự đoán trung bình $\mu_\theta$ trực tiếp, ta để mạng **dự đoán nhiễu** $\epsilon_\theta(x_t, t)$ và đặt $\mu_\theta$ cùng dạng:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right)$$

Khi đó hiệu hai trung bình rút gọn về hiệu hai nhiễu:

$$\big\lVert \tilde{\mu}_t - \mu_\theta \big\rVert^2 = \frac{\beta_t^2}{\alpha_t(1 - \bar{\alpha}_t)}\big\lVert \epsilon - \epsilon_\theta(x_t, t)\big\rVert^2$$

---

## 5.4. Mục tiêu rút gọn (DDPM, Ho et al. 2020)

Thế kết quả trên vào $L_{t-1}$ (mục 5.2), ta được một tổng các số hạng $\lVert \epsilon - \epsilon_\theta\rVert^2$ kèm hệ số $\tfrac{\beta_t^2}{2\tilde\beta_t\,\alpha_t(1-\bar\alpha_t)}$.

Thực nghiệm cho thấy **bỏ hẳn các hệ số đầu** này lại còn ổn định và cho mẫu đẹp hơn. Mục tiêu trở thành:

$$\boxed{\;\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\, \epsilon,\, t} \Big[\, \big\lVert \epsilon - \epsilon_\theta(x_t, t) \big\rVert^2 \,\Big], \quad x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon\;}$$

> Diễn giải đẹp: huấn luyện diffusion thực chất chỉ là **dạy một mạng đoán xem đã thêm bao nhiêu nhiễu** vào ảnh tại một mức $t$ bất kỳ. Đây là hồi quy L2 thuần túy — ổn định, không đối kháng, nghiệm tối ưu xác định.

Vòng lặp huấn luyện gọn tới mức đáng kinh ngạc:

```python
t = torch.randint(0, T, (batch,))
eps = torch.randn_like(x0)
x_t = sqrt_acp[t] * x0 + sqrt_one_minus_acp[t] * eps
loss = mse(model(x_t, t), eps)
```

Mạng $\epsilon_\theta$ thường là một **U-Net** kèm time embedding để biết đang ở mức nhiễu nào.

---

# 6. Liên hệ với score matching (cầu nối tới EBM)

Dự đoán nhiễu không chỉ là một mẹo kỹ thuật — nó tương đương với việc ước lượng **hàm điểm (score function)** của phân phối đã làm nhiễu.

**Mệnh đề.** Score của $q(x_t\mid x_0)$ tỉ lệ với nhiễu đã thêm:

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\,x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

**Chứng minh.** Từ mục 3.2, $q(x_t\mid x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$, nên

$$\log q(x_t\mid x_0) = -\frac{\lVert x_t - \sqrt{\bar\alpha_t}\,x_0\rVert^2}{2(1-\bar\alpha_t)} + \text{const}$$

Lấy gradient theo $x_t$ cho ngay $-\dfrac{x_t - \sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}$. Thay $x_t - \sqrt{\bar\alpha_t}\,x_0 = \sqrt{1-\bar\alpha_t}\,\epsilon$ được $-\dfrac{\epsilon}{\sqrt{1-\bar\alpha_t}}$. $\blacksquare$

Do đó $\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar{\alpha}_t}\,\nabla_{x_t}\log q(x_t)$: mạng dự đoán nhiễu thực chất đang **học score**.

Vì score $\nabla_x \log p(x)$ làm hằng số chuẩn hóa $Z$ **biến mất** (đạo hàm của hằng số bằng 0), đây chính là **denoising score matching** — cách huấn luyện một mô hình dựa trên năng lượng (energy-based model) một cách ổn định và mở rộng được. Góc nhìn liên tục qua phương trình vi phân ngẫu nhiên (SDE) thống nhất toàn bộ diffusion dưới một khung duy nhất.

---

# 7. Lấy mẫu sau khi huấn luyện

Sau khi học xong $\epsilon_\theta$, ta sinh mẫu bằng cách bắt đầu từ nhiễu thuần $x_T \sim \mathcal{N}(0, I)$ rồi lặp bước khử nhiễu từ $t = T$ về $1$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z, \qquad z \sim \mathcal{N}(0, I)$$

Công thức này chính là lấy mẫu từ $p_\theta(x_{t-1}\mid x_t)$ với $\mu_\theta$ ở mục 5.3.

Mô tả tuần tự:

```text
x_T  ~ N(0, I)
x_{T-1} = bước khử nhiễu(x_T,  T)
x_{T-2} = bước khử nhiễu(x_{T-1}, T-1)
...
x_0     = bước khử nhiễu(x_1, 1)
```

| Bộ lấy mẫu | Tính chất | Số bước điển hình |
| --- | --- | --- |
| **DDPM** | ngẫu nhiên ($\sigma_t > 0$) | hàng trăm đến hàng nghìn |
| **DDIM** | xác định ($\sigma_t = 0$), bỏ tính Markov | vài chục |

---

# 8. Sinh có điều kiện (conditional generation)

Sức mạnh thực tế của diffusion đến từ khả năng sinh **theo điều kiện** $c$ (ví dụ một câu mô tả).

**Classifier-free guidance.** Huấn luyện đồng thời cả phiên bản có và không điều kiện, rồi khi lấy mẫu thì ngoại suy dự đoán nhiễu:

$$\tilde{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w\big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\big)$$

trong đó $w$ điều khiển mức độ bám prompt: $w$ lớn cho ảnh bám sát mô tả hơn nhưng kém đa dạng.

**Latent Diffusion (Stable Diffusion).** Chạy khuếch tán trong không gian ẩn nén (latent space) của một autoencoder thay vì trực tiếp trên điểm ảnh, giảm mạnh chi phí tính toán.

---

# 9. Ưu điểm

* **Chất lượng và đa dạng mẫu dẫn đầu** — mục tiêu mode-covering (gần MLE) phủ mode tốt hơn GAN; nhiều bước tinh chỉnh cho độ sắc nét cao.
* **Huấn luyện ổn định** — như mục 5.4 cho thấy, bản chất chỉ là hồi quy L2, không mode collapse, không điểm yên ngựa.
* **Khung lý thuyết phong phú** — thống nhất ELBO, score matching và SDE; nối liền với energy-based model (mục 6).

---

# 10. Nhược điểm

Chính lựa chọn chia bài toán thành nhiều bước nhỏ cũng dẫn tới các hạn chế.

**Lấy mẫu chậm.** Cần $T$ lần truyền U-Net (mục 7); đây là cái giá trực tiếp của việc chia bài toán thành nhiều bước nhỏ. DDIM và các bộ giải nhanh giảm bớt nhưng không xóa hẳn.

**Tốn tính toán ở độ phân giải cao.** Latent diffusion giảm tải đáng kể nhưng suy luận vẫn nặng hơn GAN.

**Không cho likelihood chính xác.** Diffusion chỉ tối ưu ELBO (một chặn dưới), giống VAE — khác với autoregressive vốn cho likelihood chính xác.

---

# 11. Tổng kết

Nhìn từ góc độ xác suất, diffusion **không phát minh ra một dạng phân phối mới**. Bí quyết, đúng như các chứng minh ở mục 3–6, là biến một bài toán sinh khó thành **một chuỗi bài toán khử nhiễu dễ và ổn định**:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\, \epsilon,\, t} \Big[\, \big\lVert \epsilon - \epsilon_\theta(x_t, t) \big\rVert^2 \,\Big]$$

Từ một quá trình thêm nhiễu Gauss có dạng đóng, kết hợp với một mạng U-Net dự đoán nhiễu, diffusion đã trở thành nền tảng của hầu hết các hệ sinh ảnh hiện đại — và đang lan sang âm thanh, video, sinh phân tử.

Diffusion đổi **tốc độ lấy mẫu** lấy **độ ổn định và chất lượng mẫu**. Với ảnh, đây hiện là lựa chọn thống trị; cái giá duy nhất là $T$ bước suy luận.

> Bài tiếp theo — **DDIM** — giữ nguyên mạng đã huấn luyện ở đây nhưng thay bộ lấy mẫu: khai thác việc $\mathcal{L}_{\text{simple}}$ chỉ phụ thuộc các phân phối biên để dựng một quá trình nghịch **tất định, nhảy bước được**, nhanh hơn 10–50 lần. Sau đó **Score-Based Models** sẽ tổng quát cả hai vào một khung SDE duy nhất.

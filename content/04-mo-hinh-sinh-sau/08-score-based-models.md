# Mô hình dựa trên điểm số (Score-Based Generative Models)

> Thay vì học **mật độ** $p(x)$, mô hình dựa trên điểm số (score-based model) học **trường vector điểm số (score)** — gradient của log-mật độ:
>
> $$s_\theta(x) \approx \nabla_x \log p(x)$$
>
> Đây là một trường vector tại mỗi điểm chỉ về hướng **mật độ tăng nhanh nhất**. Học nó né được hằng số chuẩn hóa (normalizing constant) $Z$, lấy mẫu bằng động lực học Langevin (Langevin dynamics), và — qua góc nhìn phương trình vi phân ngẫu nhiên (SDE) — **thống nhất** chính nó với mô hình khuếch tán (diffusion) thành một khung duy nhất.

---

# 1. Động cơ ra đời

Ở [Diffusion](#/diffusion-models) mục 6, ta đã thấy việc dự đoán nhiễu thực chất là **ước lượng score**. Ở [Energy-Based Models](#/energy-based-models) mục 7.1, score matching xuất hiện như một cách **né hằng số chuẩn hóa** $Z_\theta$. Bài này gom hai mạch đó lại thành một lý thuyết hoàn chỉnh.

Nhắc lại nỗi đau chung của mọi mô hình dựa trên likelihood: để $p_\theta(x)$ là một mật độ hợp lệ, ta phải đảm bảo nó **tích phân về 1**. Cái giá là một ràng buộc:

* **Tự hồi quy / flow** né $Z$ bằng cách **ràng buộc kiến trúc** (thứ tự nhân quả, hoặc khả nghịch + Jacobian rẻ).
* **EBM** $p_\theta(x) = e^{-f_\theta(x)}/Z_\theta$ thì kiến trúc tự do, nhưng $Z_\theta = \int e^{-f_\theta(x)}\,dx$ là tích phân **bất khả thi**.

Ý tưởng giải thoát:

> Đừng mô hình hóa **mật độ**. Hãy mô hình hóa **gradient của log-mật độ** — tức hàm điểm số. Khi đó $Z$ tự động biến mất.

Lý do nằm ở một quan sát một dòng: $Z_\theta$ là **hằng số đối với $x$**, nên đạo hàm của nó theo $x$ bằng 0.

---

# 2. Hàm điểm số (score function)

**Định nghĩa.** Hàm điểm số (Stein score) của một phân phối $p$ là gradient của log-mật độ theo **dữ liệu** $x$ (không phải theo tham số):

$$\nabla_x \log p(x) \in \mathbb{R}^D$$

Đây là một **trường vector**: tại mỗi điểm $x$ nó cho một mũi tên chỉ hướng làm $\log p$ tăng nhanh nhất — tức hướng "về phía" các mode của phân phối.

**Vì sao score né được $Z$.** Với một EBM $p_\theta(x) = e^{-f_\theta(x)}/Z_\theta$:

$$s_\theta(x) = \nabla_x \log p_\theta(x) = \nabla_x\big(\!-f_\theta(x) - \log Z_\theta\big) = -\nabla_x f_\theta(x)$$

vì $\nabla_x \log Z_\theta = 0$. Hằng số chuẩn hóa **bốc hơi hoàn toàn**. Hệ quả thực tiễn rất lớn: **mạng score $s_\theta: \mathbb{R}^D \to \mathbb{R}^D$ có thể là một mạng bất kỳ** xuất ra một vector cùng chiều với đầu vào (ví dụ U-Net), không cần khả nghịch, không cần thứ tự, không cần tính tích phân nào. Ta mở rộng được họ mô hình ra rất rộng so với cách tiếp cận likelihood.

---

# 3. Score matching: khớp một trường vector

Câu hỏi: làm sao huấn luyện $s_\theta$ để nó khớp với score thật $\nabla_x \log p_{\text{data}}(x)$, khi ta **không biết** $p_{\text{data}}$?

## 3.1. Phân kỳ Fisher và score matching tường minh (Hyvärinen)

Mục tiêu tự nhiên là **phân kỳ Fisher (Fisher divergence)** — bình phương $\ell_2$ giữa hai trường vector, lấy trung bình theo dữ liệu:

$$J(\theta) = \frac{1}{2}\,\mathbb{E}_{p_{\text{data}}(x)}\Big[\big\lVert s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\big\rVert_2^2\Big]$$

Vướng mắc: ta **không có** $\nabla_x \log p_{\text{data}}(x)$. Phép màu của Hyvärinen (2005) là khử nó bằng tích phân từng phần.

**Định lý (score matching tường minh).** Với các giả thiết chính quy nhẹ ($p_{\text{data}}$ khả vi, $p_{\text{data}}(x)\,s_\theta(x) \to 0$ khi $\lVert x\rVert \to \infty$):

$$J(\theta) = \mathbb{E}_{p_{\text{data}}(x)}\Big[\,\tfrac{1}{2}\big\lVert s_\theta(x)\big\rVert_2^2 + \operatorname{tr}\!\big(\nabla_x s_\theta(x)\big)\Big] + \text{const}$$

trong đó $\nabla_x s_\theta(x)$ là Jacobian của trường score (một ma trận $D\times D$), và hằng số không phụ thuộc $\theta$.

**Chứng minh.** Khai triển bình phương trong $J(\theta)$:

$$J(\theta) = \mathbb{E}_{p_{\text{data}}}\Big[\tfrac{1}{2}\lVert s_\theta\rVert^2\Big] - \underbrace{\mathbb{E}_{p_{\text{data}}}\big[s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)\big]}_{(\ast)} + \underbrace{\mathbb{E}_{p_{\text{data}}}\big[\tfrac12\lVert \nabla_x \log p_{\text{data}}\rVert^2\big]}_{\text{const}}$$

Số hạng cuối không chứa $\theta$. Xử lý số hạng chéo $(\ast)$ bằng đẳng thức $p\,\nabla\log p = \nabla p$:

$$(\ast) = \int p_{\text{data}}(x)\, s_\theta(x)^\top \nabla_x \log p_{\text{data}}(x)\, dx = \int s_\theta(x)^\top \nabla_x p_{\text{data}}(x)\, dx$$

Tích phân từng phần theo từng chiều $i$ (số hạng biên triệt tiêu nhờ giả thiết chính quy):

$$\int s_{\theta,i}(x)\,\frac{\partial p_{\text{data}}}{\partial x_i}\, dx = -\int p_{\text{data}}(x)\,\frac{\partial s_{\theta,i}}{\partial x_i}\, dx$$

Cộng theo $i$:

$$(\ast) = -\int p_{\text{data}}(x)\sum_i \frac{\partial s_{\theta,i}}{\partial x_i}\, dx = -\,\mathbb{E}_{p_{\text{data}}}\big[\operatorname{tr}(\nabla_x s_\theta(x))\big]$$

Thay lại, dấu trừ trước $(\ast)$ đảo dấu, cho đúng điều phải chứng minh. $\blacksquare$

Vế phải **chỉ phụ thuộc $s_\theta$ và đạo hàm của nó** — ước lượng được trực tiếp trên dữ liệu bằng SGD, **không** cần biết $p_{\text{data}}$, **không** cần $Z$.

## 3.2. Sliced score matching — né vết Hessian

Vướng mắc còn lại: $\operatorname{tr}(\nabla_x s_\theta)$ cần $D$ lần lan truyền ngược (mỗi chiều một lần) — quá đắt với ảnh $D \sim 10^5$. **Sliced score matching** (Song và cộng sự, 2019) thay vết bằng **phép chiếu ngẫu nhiên** lên hướng $v$:

$$J_{\text{sliced}}(\theta) = \mathbb{E}_{v}\,\mathbb{E}_{p_{\text{data}}}\Big[\tfrac{1}{2}\big(v^\top s_\theta(x)\big)^2 + v^\top \nabla_x s_\theta(x)\, v\Big]$$

Số hạng $v^\top \nabla_x s_\theta\, v$ tính được bằng **một** phép nhân Jacobian–vector (cùng chi phí một lần backward), nhờ đẳng thức $\mathbb{E}_v[v^\top A v] = \operatorname{tr}(A)$ khi $\mathbb{E}[vv^\top] = I$.

## 3.3. Denoising score matching — phiên bản thực dụng

Cách phổ biến nhất (Vincent, 2011) tránh **mọi** đạo hàm bậc hai: làm nhiễu dữ liệu rồi học score của phân phối **đã làm nhiễu**. Lấy nhân nhiễu Gauss $q_\sigma(\tilde x \mid x) = \mathcal{N}(\tilde x;\, x,\, \sigma^2 I)$. Score của nhân này **tính được tường minh**:

$$\nabla_{\tilde x} \log q_\sigma(\tilde x \mid x) = -\frac{\tilde x - x}{\sigma^2} = \frac{x - \tilde x}{\sigma^2}$$

**Định lý (Vincent).** Cực tiểu denoising score matching tương đương (sai khác hằng số) với score matching tường minh trên dữ liệu đã làm nhiễu $p_\sigma(\tilde x) = \int q_\sigma(\tilde x\mid x)p_{\text{data}}(x)\,dx$:

$$J_{\text{DSM}}(\theta) = \frac{1}{2}\,\mathbb{E}_{p_{\text{data}}(x)}\,\mathbb{E}_{q_\sigma(\tilde x\mid x)}\Big[\big\lVert s_\theta(\tilde x) - \tfrac{x - \tilde x}{\sigma^2}\big\rVert^2\Big]$$

Diễn giải đẹp: $\tfrac{x-\tilde x}{\sigma^2}$ là **vector chỉ từ điểm nhiễu về điểm sạch**. Vậy huấn luyện score thực chất là **dạy mạng khử nhiễu** — chỉ ra hướng quay về dữ liệu thật. Đây chính là cây cầu với [Diffusion](#/diffusion-models) mục 6: mạng dự đoán nhiễu $\epsilon_\theta$ và mạng score $s_\theta$ chỉ khác nhau một hệ số tỉ lệ.

---

# 4. Lấy mẫu bằng động lực học Langevin

Có score rồi, ta sinh mẫu mà **không cần** mật độ tường minh, bằng động lực học Langevin — một MCMC chỉ truy cập $p$ qua $\nabla_x \log p$:

$$x_{i+1} = x_i + \epsilon\, \nabla_x \log p(x_i) + \sqrt{2\epsilon}\, z_i, \qquad z_i \sim \mathcal{N}(0, I)$$

Khởi tạo $x_0$ từ một prior tùy ý rồi lặp. Khi $\epsilon \to 0$ và số bước $\to \infty$, $x_K$ hội tụ về đúng $p(x)$. Hai lực: số hạng $\nabla_x \log p$ kéo mẫu **leo lên** vùng mật độ cao (về mode); số hạng nhiễu $\sqrt{2\epsilon}\,z_i$ giữ tính ngẫu nhiên để không kẹt cứng và để khớp đúng phân phối thay vì chỉ tìm cực đại. Thay $\nabla_x\log p$ bằng $s_\theta$, ta lấy mẫu được từ mô hình đã học.

> So với Langevin của [EBM](#/energy-based-models) mục 5: ở đó score là $-\nabla_x E_\theta$ (đạo hàm của năng lượng); ở đây score được **học trực tiếp** bằng $s_\theta$, khỏi qua trung gian hàm năng lượng.

---

# 5. Vì sao score matching "ngây thơ" thất bại

Ghép score matching (mục 3) với Langevin (mục 4) một cách trực tiếp lại **cho mẫu rất tệ**. Có ba lý do, đều quy về cùng một gốc: **score học sai ở vùng mật độ thấp**.

**Trọng số của phân kỳ Fisher.** Mục tiêu $J(\theta) = \tfrac12\mathbb{E}_{p_{\text{data}}}[\lVert s_\theta - \nabla\log p_{\text{data}}\rVert^2]$ lấy trung bình **theo $p_{\text{data}}$**. Ở những vùng hiếm dữ liệu, sai số gần như **không bị phạt**, nên $s_\theta$ ở đó học bừa. Trớ trêu là chuỗi Langevin lại **khởi đầu** từ vùng mật độ thấp (một điểm ngẫu nhiên trong không gian rộng), nơi score sai nhất — nên ngay những bước đầu đã đi lạc.

**Giả thuyết đa tạp (manifold hypothesis).** Dữ liệu thật (ảnh) thường nằm trên một **đa tạp số chiều thấp** chìm trong không gian nhiều chiều. Ngoài đa tạp, $\log p_{\text{data}}$ không xác định (mật độ bằng 0), nên $\nabla_x \log p_{\text{data}}$ vô nghĩa — score không có mục tiêu để học.

**Trộn mode chậm.** Giữa các mode tách rời bởi vùng mật độ rất thấp, Langevin khó vượt qua, và **tỉ lệ khối lượng giữa các mode** bị ước lượng sai.

Tất cả gợi ý một giải pháp: **làm nhiễu dữ liệu** để lấp đầy vùng mật độ thấp và "thổi phồng" đa tạp ra cả không gian.

---

# 6. Nhiều mức nhiễu: Noise Conditional Score Network

Giải pháp của Song & Ermon (2019): nhiễu **nhiều mức** thay vì một mức.

Chọn một dãy độ lệch chuẩn tăng dần $\sigma_1 < \sigma_2 < \cdots < \sigma_L$. Làm nhiễu dữ liệu ở mọi mức $p_{\sigma_i}(x) = \int \mathcal{N}(x; x', \sigma_i^2 I)\,p_{\text{data}}(x')\,dx'$:

* $\sigma_L$ **lớn** làm nhiễu nặng, trải mật độ ra **khắp** không gian — score được học tốt cả ở vùng xa, và các mode được nối liền (Langevin trộn dễ).
* $\sigma_1$ **nhỏ** giữ dữ liệu gần như nguyên vẹn — score sắc nét, bám sát phân phối thật.

Huấn luyện **một** mạng score **có điều kiện theo mức nhiễu**, $s_\theta(x, i)$ — gọi là **Noise Conditional Score Network (NCSN)** — bằng tổng có trọng số các mục tiêu denoising score matching (mục 3.3):

$$\mathcal{L}(\theta) = \sum_{i=1}^{L} \lambda(i)\,\mathbb{E}_{p_{\sigma_i}(x)}\Big[\big\lVert s_\theta(x, i) - \nabla_x \log p_{\sigma_i}(x)\big\rVert_2^2\Big]$$

với trọng số chọn $\lambda(i) = \sigma_i^2$ để cân bằng độ lớn các số hạng giữa các mức.

**Annealed Langevin dynamics.** Khi sinh, chạy Langevin **tuần tự từ nhiễu cao xuống thấp** $i = L, L-1, \dots, 1$: dùng score ở mức $\sigma_L$ để định vị thô (nhờ trường mượt phủ khắp không gian), rồi giảm dần mức nhiễu để tinh chỉnh, mỗi mức khởi tạo từ kết quả mức trước. Đây là chìa khóa vá cả ba thất bại ở mục 5.

---

# 7. Góc nhìn liên tục: phương trình vi phân ngẫu nhiên (SDE)

Đẩy số mức nhiễu $L \to \infty$, dãy rời rạc $\{\sigma_i\}$ trở thành một **quá trình liên tục theo thời gian**, và toàn bộ lý thuyết gói gọn trong ngôn ngữ phương trình vi phân ngẫu nhiên (stochastic differential equation — SDE).

**SDE thuận (làm nhiễu).** Một quá trình khuếch tán biến dữ liệu thành nhiễu:

$$dx = f(x, t)\, dt + g(t)\, dw$$

với $f$ là hệ số trôi (drift), $g$ là hệ số khuếch tán (diffusion), $w$ là chuyển động Brown. Tại $t=0$, $p_0 = p_{\text{data}}$; tới $t=T$, $p_T$ tiến về một prior đơn giản $\pi$ (ví dụ Gauss).

**SDE nghịch (sinh mẫu).** Kết quả nền tảng (Anderson, 1982): mọi SDE thuận có một **SDE nghịch** chạy ngược thời gian, và nó **chỉ phụ thuộc score của các phân phối biên** $\nabla_x \log p_t(x)$:

$$dx = \big[f(x, t) - g(t)^2\, \nabla_x \log p_t(x)\big]\, dt + g(t)\, d\bar w$$

Ta học một mạng score **phụ thuộc thời gian** $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ bằng denoising score matching trên mọi $t$ (đúng tinh thần mục 6, chỉ là liên tục), rồi giải ngược SDE từ $x_T \sim \pi$ về $x_0$ để sinh.

**Probability flow ODE.** Mỗi SDE còn có một **ODE tất định** với **cùng phân phối biên** $p_t$ tại mọi thời điểm:

$$dx = \Big[f(x, t) - \tfrac{1}{2} g(t)^2\, \nabla_x \log p_t(x)\Big]\, dt$$

Đây là một **luồng chuẩn hóa liên tục (continuous normalizing flow)** — xem [Normalizing Flows](#/normalizing-flows). Vì tất định và khả nghịch, nó cho phép **tính likelihood chính xác** qua công thức đổi biến (tích phân vết Jacobian theo thời gian). Trên CIFAR-10, hướng này đạt **2.99 bits/dim**, vượt cả flow và autoregressive thuần.

---

# 8. Quan hệ với mô hình khuếch tán

Score-based model và diffusion là **hai góc nhìn của cùng một họ mô hình**, chỉ khác cách huấn luyện và cách rời rạc hóa SDE:

| | Khung huấn luyện | Là rời rạc hóa của |
| --- | --- | --- |
| **NCSN / score-based** | Denoising score matching nhiều mức (mục 6) | **VE-SDE** (variance exploding) |
| **DDPM / diffusion** | ELBO → dự đoán nhiễu (xem [Diffusion](#/diffusion-models) mục 5) | **VP-SDE** (variance preserving) |

Sự tương đương đã được chứng minh cụ thể ở [Diffusion](#/diffusion-models) mục 6: mạng dự đoán nhiễu $\epsilon_\theta(x_t, t)$ tỉ lệ với score $\nabla_{x_t}\log q(x_t)$, và ELBO của DDPM **chính là** một tổ hợp có trọng số của các mục tiêu score matching. Nói cách khác, "dự đoán nhiễu" và "khớp score" là **cùng một việc** nhìn từ hai phía.

Khung SDE còn cho các bộ lấy mẫu lai mạnh hơn, ví dụ **Predictor–Corrector** (xen kẽ một bước giải số SDE nghịch với vài bước Langevin hiệu chỉnh), đạt FID $= 2.20$ trên CIFAR-10 — mức dẫn đầu khi công bố.

---

# 9. Ưu điểm

* **Tự do kiến trúc + né $Z$** — chỉ cần một mạng xuất vector cùng chiều dữ liệu; hằng số chuẩn hóa biến mất theo gradient (mục 2), không như flow/autoregressive phải ràng buộc.
* **Huấn luyện ổn định** — là hồi quy $\ell_2$ trên trường vector (denoising score matching), không đối kháng như GAN, không cần MCMC trong vòng lặp như EBM cổ điển.
* **Chất lượng mẫu dẫn đầu** — NCSN/SDE cho FID ngang hoặc vượt GAN tốt nhất, đồng thời phủ mode tốt.
* **Khung thống nhất và likelihood chính xác** — góc nhìn SDE gom score-based và diffusion làm một; probability flow ODE còn cho likelihood chính xác (mục 7).

---

# 10. Nhược điểm

* **Lấy mẫu chậm** — annealed Langevin hoặc giải SDE nghịch cần rất nhiều bước đánh giá mạng (giống diffusion), chậm hơn nhiều so với GAN/flow một bước.
* **Nhạy với siêu tham số nhiễu** — dải $\{\sigma_i\}$, lịch nhiễu, bước Langevin phải tinh chỉnh cẩn thận; chọn sai là mẫu hỏng (mục 5–6).
* **Không cho mật độ tường minh trực tiếp** — bản thân $s_\theta$ chỉ là gradient; muốn likelihood phải đi qua probability flow ODE, tốn thêm chi phí giải ODE.

---

# 11. Tổng kết

Bài học cốt lõi: **đừng học mật độ, hãy học gradient của log-mật độ.** Chỉ một thay đổi quan điểm đó đã gỡ được hằng số chuẩn hóa $Z$ — gốc rễ khó khăn của mọi mô hình sinh dựa trên likelihood:

$$s_\theta(x) \approx \nabla_x \log p(x), \qquad \nabla_x \log Z = 0$$

Phần còn lại là một chuỗi vá lỗi tinh tế: score matching tường minh (Hyvärinen) khử được $\nabla\log p_{\text{data}}$ chưa biết; denoising score matching biến nó thành hồi quy khử nhiễu; nhiễu **nhiều mức** (NCSN) vá chỗ score sai ở vùng mật độ thấp; và giới hạn liên tục **SDE** thống nhất tất cả, đồng thời sinh ra probability flow ODE cho likelihood chính xác.

Đây cũng là lời giải thích sâu nhất cho thành công của diffusion: nó là một **trường hợp riêng** của khung score-based, được rời rạc hóa và tham số hóa khéo léo. Hai cái tên — "dự đoán nhiễu" và "khớp score" — chỉ là hai cách gọi của cùng một ý tưởng.

> Bài cuối chuyên mục — **[Energy-Based Models](#/energy-based-models)** — lùi lại một bước để trình bày khung tổng quát nhất $p_\theta \propto e^{-f_\theta}$, nơi cả score matching ở đây lẫn diffusion đều lộ ra chỉ là những chiến lược khác nhau để xử lý cùng một hằng số chuẩn hóa $Z_\theta$.

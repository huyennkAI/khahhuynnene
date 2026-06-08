# Mạng đối kháng sinh (Generative Adversarial Network)

> Mạng đối kháng sinh (Generative Adversarial Network — GAN) huấn luyện **hai mạng cạnh tranh nhau**: bộ sinh (generator) cố tạo dữ liệu giả như thật, bộ phân biệt (discriminator) cố vạch trần đâu là giả. Hai mạng đẩy nhau lên trong một trò chơi:
>
> $$z \sim p(z) \;\xrightarrow{\;G\;}\; x_{\text{giả}} \qquad\text{vs.}\qquad x_{\text{thật}} \sim p_{\text{data}}$$
>
> Cuộc đua này đẩy chất lượng mẫu lên rất cao **mà không cần tính mật độ** $p(x)$ — một sự thay đổi tư duy căn bản so với VAE hay normalizing flow.

---

# 1. Động cơ ra đời

Các mô hình sinh trước đó — VAE và normalizing flow — đều cố **mô hình hóa mật độ (density)** $p(x)$ một cách tường minh, và phải trả giá cho điều đó:

* **VAE** cho mẫu **mờ**, vì mục tiêu tái tạo kiểu L2 khiến mô hình hồi quy về trung bình của các đáp án hợp lý.
* **Normalizing flow** bị ràng buộc bởi yêu cầu **khả nghịch (invertibility)** của mọi phép biến đổi.

Câu hỏi đặt ra là: để sinh được ảnh đẹp, ta có **thật sự cần** biết $p(x)$ là gì không?

Ý tưởng đột phá của Goodfellow (2014) trả lời rằng: **không cần**.

> Để sinh ảnh đẹp, ta đâu cần viết ra công thức của $p(x)$. Ta chỉ cần một **giám khảo (critic)** đủ tinh, đến mức nó **không phân biệt nổi** đâu là thật, đâu là giả.

Đây là bước chuyển từ mô hình sinh **tường minh (explicit)** sang **ngầm định (implicit)**: ta không bao giờ viết ra $p_\theta(x)$, mà chỉ định nghĩa **cách lấy mẫu**:

$$x = G(z), \qquad z \sim p(z)$$

trong đó $z$ là vector nhiễu (thường là Gaussian), còn $G$ là một mạng nơ-ron khả vi biến nhiễu thành dữ liệu.

---

# 2. GAN khắc phục được điều gì?

So với các họ mô hình tường minh, lập luận lý thuyết cho thấy GAN né được nhiều ràng buộc khó:

* **Né hoàn toàn mật độ và hằng số chuẩn hóa.** Không cần likelihood, không cần hằng số chuẩn hóa (normalizing constant) $Z_\theta$, không cần biến đổi khả nghịch.
* **Mẫu sắc nét (sharp samples).** Ở mục 4 ta sẽ chứng minh GAN cực tiểu một phân kỳ (Jensen–Shannon divergence) chứ không phải sai số L2; nó không "trung bình hóa" các đáp án hợp lý nên không bị mờ như VAE.
* **Lấy mẫu một bước (one-step sampling).** Sinh một mẫu chỉ tốn **một lần truyền xuôi** qua $G$, độ phức tạp $O(1)$ — khác hẳn autoregressive ($O(D)$) hay diffusion (nhiều bước khử nhiễu).

---

# 3. Trò chơi đối kháng: trực giác

Hãy hình dung một ví dụ kinh điển: **kẻ làm tiền giả** đấu với **cảnh sát**.

* Bộ sinh $G$ là **kẻ làm tiền giả**: cố in ra những tờ tiền giống tiền thật nhất.
* Bộ phân biệt $D$ là **cảnh sát**: cố phân biệt tiền thật với tiền giả.

Lúc đầu kẻ làm giả vụng về, cảnh sát dễ dàng bắt bài. Nhưng mỗi lần bị bắt, kẻ làm giả lại học được điểm sơ hở và cải thiện. Đáp lại, cảnh sát cũng tinh hơn. Cuộc đua leo thang này tiếp diễn cho tới khi:

> Tiền giả **giống hệt** tiền thật, và cảnh sát chỉ còn cách **tung đồng xu** để đoán.

Trạng thái cân bằng đó — cảnh sát đoán đúng đúng 50% — chính là điểm tối ưu mà ta sẽ chứng minh chặt chẽ ở mục dưới. Cụ thể, $D$ xuất ra một số trong $[0, 1]$: $D(x)$ là **xác suất $D$ tin rằng $x$ là thật**.

| Vai trò | Mạng | Mục tiêu |
| --- | --- | --- |
| Kẻ làm tiền giả | $G$ | Làm cho $D(G(z))$ **lớn** (lừa được cảnh sát) |
| Cảnh sát | $D$ | Cho $D(x_{\text{thật}}) \to 1$ và $D(G(z)) \to 0$ |

---

# 4. Lý thuyết: mục tiêu minimax và nghiệm tối ưu

Trò chơi đối kháng được viết gọn thành một bài toán **cực tiểu–cực đại (minimax)** trên một **hàm giá trị (value function)** $V(D, G)$:

$$\min_{G} \max_{D} \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]$$

Đọc công thức này theo trực giác:

* $D$ muốn **cực đại** $V$: làm $\log D(x)$ lớn trên dữ liệu thật (số hạng đầu) và $\log(1 - D(G(z)))$ lớn trên dữ liệu giả (số hạng sau) — tức là gán xác suất thật cao cho mẫu thật, thấp cho mẫu giả.
* $G$ muốn **cực tiểu** $V$: làm cho $D(G(z))$ lớn để $\log(1 - D(G(z)))$ nhỏ — tức là lừa được $D$.

Gọi $p_g$ là phân phối do $G$ sinh ra. Vì $x = G(z)$, ta đổi biến kỳ vọng thứ hai từ không gian nhiễu sang không gian dữ liệu:

$$\mathbb{E}_{z \sim p(z)}\big[\,\cdot\,\big] = \mathbb{E}_{x \sim p_g}\big[\,\cdot\,\big]$$

---

## 4.1. Bước 1 — Bộ phân biệt tối ưu

**Mệnh đề.** Với $G$ (do đó $p_g$) cố định, bộ phân biệt tối ưu là:

$$D^{*}(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

**Chứng minh.** Viết $V$ thành một tích phân duy nhất theo $x$, dùng phép đổi biến ở trên:

$$V(D, G) = \int_x \Big[\, p_{\text{data}}(x) \log D(x) + p_g(x) \log\big(1 - D(x)\big) \,\Big]\,dx$$

Vì giá trị của $D$ tại mỗi điểm $x$ **độc lập** với các điểm khác, ta cực đại hóa **điểm-theo-điểm**. Xét hàm một biến:

$$\varphi(D) = a \log D + b \log(1 - D), \qquad a = p_{\text{data}}(x),\quad b = p_g(x)$$

Lấy đạo hàm và cho bằng $0$:

$$
\begin{aligned}
\varphi'(D) = \frac{a}{D} - \frac{b}{1 - D} = 0
&\;\Rightarrow\; a(1 - D) = b D \\
&\;\Rightarrow\; a = (a + b) D \\
&\;\Rightarrow\; D^{*} = \frac{a}{a + b}
\end{aligned}
$$

Đây đúng là cực đại vì đạo hàm bậc hai âm trên toàn miền $(0, 1)$:

$$\varphi''(D) = -\frac{a}{D^2} - \frac{b}{(1 - D)^2} < 0$$

Thay lại $a = p_{\text{data}}(x),\ b = p_g(x)$ ta được $D^{*}(x) = \dfrac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$. $\blacksquare$

> Trực giác: tại mỗi điểm $x$, giám khảo tối ưu chỉ cần so **tỉ lệ** giữa mật độ thật và mật độ giả. Nếu $p_{\text{data}}(x) \gg p_g(x)$ thì $D^{*}(x) \to 1$; nếu hai bên bằng nhau thì $D^{*}(x) = \tfrac{1}{2}$.

---

## 4.2. Bước 2 — Thay vào, hiện ra Jensen–Shannon divergence

Bây giờ cố định $D = D^{*}$ và xét hàm chi phí mà $G$ phải tối thiểu:

$$
\begin{aligned}
C(G) := V(D^{*}, G)
&= \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}}{p_{\text{data}} + p_g}\right]
 + \mathbb{E}_{x \sim p_g}\!\left[\log \frac{p_g}{p_{\text{data}} + p_g}\right]
\end{aligned}
$$

Mẹo then chốt: thêm và bớt $\log 2$ trong mỗi kỳ vọng, bằng cách viết mẫu số dưới dạng trung bình $m = \tfrac{p_{\text{data}} + p_g}{2}$:

$$\frac{p_{\text{data}}}{p_{\text{data}} + p_g} = \frac{1}{2} \cdot \frac{p_{\text{data}}}{(p_{\text{data}} + p_g)/2}$$

Áp dụng cho cả hai số hạng:

$$
\begin{aligned}
C(G)
&= -\log 2 \cdot \underbrace{\mathbb{E}_{p_{\text{data}}}[1]}_{=\,1}
   + \mathbb{E}_{p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}}{(p_{\text{data}} + p_g)/2}\right] \\
&\quad -\log 2 \cdot \underbrace{\mathbb{E}_{p_g}[1]}_{=\,1}
   + \mathbb{E}_{p_g}\!\left[\log \frac{p_g}{(p_{\text{data}} + p_g)/2}\right] \\
&= -\log 4
   + D_{\mathrm{KL}}\!\Big(p_{\text{data}} \,\Big\|\, \tfrac{p_{\text{data}} + p_g}{2}\Big)
   + D_{\mathrm{KL}}\!\Big(p_g \,\Big\|\, \tfrac{p_{\text{data}} + p_g}{2}\Big)
\end{aligned}
$$

Theo định nghĩa phân kỳ Jensen–Shannon (Jensen–Shannon divergence):

$$D_{\mathrm{JS}}(p \,\|\, q) = \tfrac{1}{2} D_{\mathrm{KL}}(p \,\|\, m) + \tfrac{1}{2} D_{\mathrm{KL}}(q \,\|\, m), \qquad m = \tfrac{p + q}{2}$$

Hai số hạng KL ở trên đúng bằng $2 D_{\mathrm{JS}}(p_{\text{data}} \,\|\, p_g)$, nên:

$$\boxed{\; C(G) = -\log 4 + 2\, D_{\mathrm{JS}}\big(p_{\text{data}} \,\|\, p_g\big) \;}$$

---

## 4.3. Bước 3 — Nghiệm tối ưu toàn cục

**Mệnh đề.** Cực tiểu toàn cục của $C(G)$ đạt được **khi và chỉ khi** $p_g = p_{\text{data}}$, và khi đó $D^{*}(x) = \tfrac{1}{2}$ với mọi $x$.

**Chứng minh.** Phân kỳ Jensen–Shannon luôn không âm, $D_{\mathrm{JS}}(p_{\text{data}} \,\|\, p_g) \ge 0$, và **bằng $0$ khi và chỉ khi** $p_{\text{data}} = p_g$. Do đó:

$$C(G) = -\log 4 + 2\, D_{\mathrm{JS}}(p_{\text{data}} \,\|\, p_g) \;\ge\; -\log 4$$

với dấu bằng đạt **duy nhất** tại $p_g = p_{\text{data}}$. Thay $p_g = p_{\text{data}}$ vào công thức bộ phân biệt tối ưu:

$$D^{*}(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_{\text{data}}(x)} = \frac{1}{2} \quad \forall x$$

Tại điểm này giám khảo chỉ còn **đoán mò** — đúng như trực giác kẻ làm tiền giả ở mục 3. $\blacksquare$

> **Kết luận cốt lõi:** GAN, ở mức lý thuyết, **cực tiểu phân kỳ Jensen–Shannon** giữa phân phối sinh $p_g$ và phân phối dữ liệu thật $p_{\text{data}}$.

---

# 5. Vì sao GAN sắc nét còn VAE mờ?

Đây là một hệ quả lý thuyết quan trọng của việc GAN cực tiểu JS divergence.

VAE cực tiểu một mục tiêu liên quan tới **forward KL** với số hạng tái tạo kiểu **L2**. Số hạng L2 khiến mô hình hồi quy về **trung bình** của tất cả các đáp án hợp lý — và trung bình của nhiều ảnh sắc nét lại là một ảnh **mờ**.

GAN thì khác. Nó cực tiểu JS divergence, **không hề có số hạng L2 trung bình hóa**. Nếu $G$ sinh ra một "ảnh trung bình mờ" không thuộc $p_{\text{data}}$, thì $D$ phát hiện ngay lập tức (vì ảnh mờ hiếm khi xuất hiện trong dữ liệu thật), và đẩy $G$ tránh xa khỏi vùng đó.

> Đây chính là dẫn chứng lý thuyết cho ưu thế **độ sắc nét** của GAN: thước đo JS không thưởng cho việc "đoán an toàn ở giữa".

---

# 6. Huấn luyện thực tế và non-saturating loss

Trên thực tế, ta luân phiên cập nhật hai mạng:

```text
Lặp:
  1. Cố định G, cập nhật D trên một lô mẫu thật và mẫu giả (vài bước)
  2. Cố định D, cập nhật G để D đánh giá mẫu giả là thật
```

---

## 6.1. Vì sao phải đổi mục tiêu của G?

Mục tiêu gốc của $G$ là $\min_G \mathbb{E}_z\big[\log(1 - D(G(z)))\big]$. Mục tiêu này có một vấn đề nghiêm trọng về gradient **đúng lúc cần nó nhất**.

Khi $G$ còn kém (giai đoạn đầu huấn luyện), $D$ dễ dàng nhận ra mẫu giả nên $D(G(z)) \approx 0$. Lúc đó đạo hàm của số hạng $\log(1 - D)$ gần như **biến mất (vanishing gradient)**:

$$\frac{d}{dD}\log(1 - D)\Big|_{D \to 0} = \frac{-1}{1 - D}\Big|_{D \to 0} = -1 \quad (\text{rất nhỏ})$$

Trong khi nếu thay bằng mục tiêu $\log D$ thì gradient lại **rất mạnh** đúng ở vùng này:

$$\frac{d}{dD}\log D\Big|_{D \to 0} = \frac{1}{D} \to \infty$$

Vì vậy thực tế dùng mục tiêu **non-saturating**:

$$\max_G \; \mathbb{E}_z\big[\log D(G(z))\big]$$

Mục tiêu này có **cùng điểm bất động** với mục tiêu gốc (vẫn đẩy $D(G(z)) \to 1$) nhưng **gradient mạnh** khi $G$ còn yếu, giúp tối ưu khởi động được.

---

# 7. Những khó khăn kinh điển

Dù lý thuyết đẹp, GAN nổi tiếng khó huấn luyện. Mỗi khó khăn dưới đây đều có gốc rễ lý thuyết.

**Sụp đổ chế độ (mode collapse).** Mục tiêu chỉ yêu cầu $G$ **lừa được $D$**, chứ không yêu cầu phủ hết mọi chế độ (mode) của dữ liệu. Vì vậy $G$ có thể dồn toàn bộ khối lượng vào **một vài mode dễ lừa** (ví dụ chỉ sinh một kiểu chữ số duy nhất). Đây là biểu hiện của bản chất **mode-seeking**, đối lập với tính **mode-covering** của forward-KL trong MLE.

**Gradient biến mất khi hai phân phối không chồng lấn.** Nếu $p_{\text{data}}$ và $p_g$ nằm trên hai đa tạp (manifold) tách rời nhau, thì:

$$D_{\mathrm{JS}}(p_{\text{data}} \,\|\, p_g) = \log 2 \quad (\text{hằng số})$$

bất kể hai phân phối cách nhau **bao xa**. Phân kỳ hằng nghĩa là gradient bằng $0$ — $G$ không biết nên đi về hướng nào. Theo giả thuyết đa tạp (manifold hypothesis), dữ liệu ảnh nằm trên đa tạp mỏng trong không gian nhiều chiều, nên tình huống không chồng lấn này **rất hay xảy ra**.

**Mất ổn định (instability).** Bài toán cốt lõi là tìm một **điểm yên ngựa (saddle point / Nash equilibrium)** của trò chơi minimax, chứ không phải cực tiểu một hàm. Các thuật toán dựa trên gradient có thể **dao động vòng quanh** điểm cân bằng thay vì hội tụ vào nó.

---

# 8. WGAN — khắc phục gradient biến mất

Nguồn gốc của bệnh gradient biến mất là JS divergence "vô cảm" với khoảng cách khi hai phân phối không chồng lấn. WGAN thay JS divergence bằng **khoảng cách Wasserstein-1** (Earth Mover's Distance), với dạng đối ngẫu Kantorovich–Rubinstein:

$$W(p_{\text{data}}, p_g) = \sup_{\lVert f \rVert_L \le 1} \; \mathbb{E}_{x \sim p_{\text{data}}}\big[f(x)\big] - \mathbb{E}_{x \sim p_g}\big[f(x)\big]$$

trong đó $f$ chạy trên các hàm **1-Lipschitz**.

Ưu thế lý thuyết then chốt: **Wasserstein cho gradient có ý nghĩa ngay cả khi hai phân phối không chồng lấn**. Lý do là nó đo "công vận chuyển khối lượng" từ phân phối này sang phân phối kia — một khoảng cách **hình học** phản ánh hai phân phối cách nhau bao xa, chứ không phải một tỉ số mật độ chỉ biết "có chồng lấn hay không".

Ràng buộc 1-Lipschitz được thực thi trong thực tế bằng **cắt trọng số (weight clipping)** hoặc **phạt gradient (gradient penalty)** trong biến thể WGAN-GP.

---

# 9. Các cải tiến quan trọng

Toàn bộ lý thuyết trên chỉ nói về mục tiêu, chưa nói về kiến trúc. Nhiều biến thể nổi bật đã ra đời để ổn định và mở rộng GAN:

| Biến thể | Đóng góp chính |
| --- | --- |
| **DCGAN** | Kiến trúc tích chập (convolution), ổn định huấn luyện ảnh |
| **WGAN / WGAN-GP** | Dùng khoảng cách Wasserstein, gradient mượt, giảm mode collapse |
| **Conditional GAN (cGAN)** | Sinh có điều kiện theo nhãn $y$: $G(z, y)$ |
| **StyleGAN** | Tiêm phong cách theo từng tầng, sinh khuôn mặt cực kỳ chân thực |

Điểm chung của tất cả là vẫn giữ nguyên cấu trúc trò chơi đối kháng giữa $G$ và $D$; khác biệt nằm ở **hàm mục tiêu** hoặc **cách biểu diễn** hai mạng.

---

# 10. Ưu điểm

* **Mẫu sắc nét nhất** trong các họ mô hình sinh giai đoạn 2014–2020 — dẫn chứng ở mục 5 (cực tiểu JS divergence, không trung bình hóa L2 như VAE).
* **Lấy mẫu nhanh** — sinh một mẫu chỉ tốn một lần truyền xuôi qua $G$, độ phức tạp $O(1)$ (so với $O(D)$ của autoregressive).
* **Khung linh hoạt** — chỉ cần $G$ khả vi; **không** ràng buộc về mật độ, khả nghịch, hay biến ẩn cụ thể.

---

# 11. Nhược điểm

Chính lựa chọn "né mật độ và dùng đối kháng" cũng kéo theo những hạn chế cơ bản.

**Huấn luyện bất ổn.** Bài toán là tìm điểm yên ngựa (mục 7), rất nhạy với siêu tham số và lựa chọn kiến trúc.

**Mode collapse.** Mục tiêu mode-seeking không phạt việc bỏ sót mode (mục 7), nên độ phủ (recall) thường thấp khi đo bằng precision/recall.

**Không có likelihood.** Vì là mô hình ngầm định, GAN không cho $p_\theta(x)$ nên **không** đánh giá định lượng được bằng bits-per-dim; phải dựa vào các chỉ số gián tiếp như FID.

**Gradient biến mất** khi hỗ trợ (support) của hai phân phối không chồng lấn (mục 7) — đây chính là động lực lý thuyết dẫn tới sự ra đời của WGAN (mục 8).

---

# 12. Tổng kết

Nhìn từ góc độ lý thuyết, GAN **không mô hình hóa mật độ**. Thay vào đó, nó dựng lên một trò chơi minimax:

$$\min_{G} \max_{D} \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p(z)}\big[\log\big(1 - D(G(z))\big)\big]$$

và ta đã chứng minh rằng bài toán này tương đương với việc **cực tiểu phân kỳ Jensen–Shannon** $D_{\mathrm{JS}}(p_{\text{data}} \,\|\, p_g)$, với nghiệm tối ưu toàn cục duy nhất $p_g = p_{\text{data}}$ và $D^{*} \equiv \tfrac{1}{2}$.

GAN đổi **sự ổn định và likelihood** lấy **độ sắc nét và tốc độ lấy mẫu**. Nó từng là tiêu chuẩn vàng cho sinh ảnh, trước khi mô hình khuếch tán (diffusion) vươn lên dẫn đầu nhờ huấn luyện ổn định hơn và đa dạng mẫu hơn.

> Bài tiếp theo — **Diffusion** — đạt chất lượng sánh ngang GAN nhưng huấn luyện ổn định như MLE, bằng cách chia bài toán sinh thành **nhiều bước khử nhiễu nhỏ** thay vì một trò chơi đối kháng một-bước đầy bất ổn.

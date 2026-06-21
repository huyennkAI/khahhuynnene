# Luồng chuẩn hóa (Normalizing Flows)

> Luồng chuẩn hóa (normalizing flow) biến một phân phối **đơn giản** (thường là Gauss) thành phân phối dữ liệu **phức tạp** qua một chuỗi **hàm khả nghịch (invertible function)**:
>
> $$z \sim \mathcal{N}(0, I) \;\xrightarrow{\; f_1 \;}\; z_1 \;\xrightarrow{\; f_2 \;}\; z_2 \;\xrightarrow{\;\cdots\;}\; x$$
>
> Nhờ công thức đổi biến (change of variables), nó cho likelihood **chính xác** mà vẫn lấy mẫu chỉ trong **một bước**. Đây là cầu nối giữa hai thế giới: sự chính xác của mô hình tự hồi quy (autoregressive) và tốc độ lấy mẫu của các mô hình một bước.

---

# 1. Động cơ ra đời

Ở bài trước ta đã thấy một sự đánh đổi cốt lõi giữa các họ mô hình sinh (generative model):

* **Mô hình tự hồi quy (autoregressive)** $p_\theta(x) = \prod_{i=1}^n p_\theta(x_i \mid x_{<i})$ cho likelihood **chính xác**, nhưng lấy mẫu **chậm** $O(D)$ vì sinh tuần tự từng phần tử và **không** có cơ chế học biểu diễn đặc trưng (feature) trực tiếp.
* **VAE** $p_\theta(x) = \int p_\theta(x, z)\, dz$ học được **biểu diễn ẩn** $z$ và lấy mẫu **nhanh** (một lần truyền qua decoder), nhưng marginal likelihood **bất khả thi** nên chỉ cho **chặn dưới** (ELBO).

Câu hỏi trung tâm của bài này:

> Liệu có thể thiết kế một **mô hình biến ẩn** vừa học được biểu diễn, vừa cho likelihood **chính xác** và lấy mẫu nhanh không? **Có.**

Ý tưởng then chốt nằm ở một quan sát hình học. Giả sử phép biến đổi từ nhiễu sang dữ liệu là **song ánh khả vi (diffeomorphism)** — vừa một-một, vừa trơn, vừa có hàm ngược trơn. Khác với VAE (nơi $p(x\mid z)$ là một phân phối, mỗi $x$ có thể sinh ra từ **nhiều** $z$), ở đây $x = f_\theta(z)$ là **xác định và khả nghịch**: với mỗi $x$ có **đúng một** $z$ tương ứng, không cần liệt kê.

Nếu ta biết phép biến đổi này "kéo giãn" hay "nén" thể tích bao nhiêu tại mỗi điểm, thì có thể **truy ngược mật độ** một cách chính xác bằng giải tích. Công cụ làm điều đó chính là **công thức đổi biến**.

---

# 2. Khắc phục điều gì?

Trước khi đi vào toán, hãy xem flow giải quyết các điểm yếu của VAE và autoregressive như thế nào.

**So với VAE.** VAE phải tính tích phân $p(x) = \int p(x \mid z)\, p(z)\, dz$ — **bất khả thi** với dữ liệu nhiều chiều, nên buộc phải xấp xỉ bằng ELBO. Flow đặt $z$ và $x$ **cùng số chiều** và ánh xạ giữa chúng là **song ánh**. Khi đó tích phân biến mất: với mỗi $x$ chỉ có **đúng một** $z = f^{-1}(x)$. Likelihood trở nên **chính xác**, không còn khoảng hở. Hơn nữa, biểu diễn ẩn suy ra trực tiếp bằng $z = f_\theta^{-1}(x)$ — **không cần mạng suy diễn (inference network)** như VAE.

**So với autoregressive.** Lấy mẫu chỉ cần một lần truyền thẳng $x = f(z)$, **không** sinh tuần tự từng chiều (với các kiến trúc coupling).

**Cái giá phải trả.** Để giữ song ánh và để định thức Jacobian tính được rẻ, kiến trúc bị **ràng buộc rất mạnh** (xem mục 5).

| Tiêu chí | Autoregressive | VAE | Normalizing Flow |
| --- | --- | --- | --- |
| Likelihood | Chính xác | Chặn dưới (ELBO) | **Chính xác** |
| Tốc độ lấy mẫu | Chậm $O(D)$ | Nhanh | **Nhanh $O(K)$** (coupling) |
| Biểu diễn ẩn | Không có | Mạng encoder | **$z=f^{-1}(x)$, không cần encoder** |
| Số chiều ẩn | — | Nén được | Bằng số chiều dữ liệu |
| Ràng buộc kiến trúc | Nhẹ | Nhẹ | **Nặng (khả nghịch)** |

---

# 3. Lý thuyết: công thức đổi biến

Cho biến ẩn $z \sim p_z(z)$ (thường là $\mathcal{N}(0, I)$) và một hàm khả nghịch khả vi $x = f(z)$. Đặt hàm ngược:

$$z = f^{-1}(x) = g(x)$$

Khi đó mật độ của $x$ được cho bởi **công thức đổi biến**:

$$p_x(x) = p_z\big(g(x)\big)\, \left| \det J_g(x) \right|, \qquad J_g(x) = \frac{\partial g(x)}{\partial x}$$

Trong đó $J_g(x)$ là **ma trận Jacobian** của $g$ — ma trận chứa mọi đạo hàm riêng. Lưu ý: $x, z$ phải **liên tục** và **cùng số chiều** ($x \in \mathbb{R}^n \Rightarrow z \in \mathbb{R}^n$).

Lấy logarit để dùng trong huấn luyện (biến tích thành tổng, ổn định số học):

$$\boxed{\;\log p_x(x) = \log p_z\big(g(x)\big) + \log \left| \det J_g(x) \right|\;}$$

Công thức này là **trái tim** của normalizing flow. Hai số hạng có ý nghĩa rõ ràng:

* $\log p_z\big(g(x)\big)$ — xác suất của điểm ảnh ngược trong không gian nhiễu đơn giản.
* $\log \left| \det J_g(x) \right|$ — hiệu chỉnh do phép biến đổi làm **co giãn thể tích**.

---

## 3.1. Một cảnh báo: vì sao cần định thức Jacobian

Một sai lầm tự nhiên là nghĩ rằng đổi biến chỉ là "thay $z$ bởi $g(x)$ trong mật độ". Ví dụ kinh điển bác bỏ điều đó: cho $Z \sim \mathcal{U}[0, 2]$, nên $p_Z(1) = \tfrac12$. Đặt $X = 4Z$. Suy luận sai:

$$p_X(4) \overset{?}{=} p_Z(1) = \tfrac12$$

Nhưng rõ ràng $X \sim \mathcal{U}[0, 8]$, nên $p_X(4) = \tfrac18$. Sai số đến từ việc bỏ quên hệ số co giãn. Với $g(x) = x/4$, $g'(x) = 1/4$:

$$p_X(4) = p_Z(1)\,|g'(4)| = \tfrac12 \cdot \tfrac14 = \tfrac18 \;\checkmark$$

Một ví dụ thú vị hơn: $X = \exp(Z)$ với $Z \sim \mathcal{U}[0, 2]$, thì $g(x) = \ln x$ và $p_X(x) = p_Z(\ln x)\,\tfrac1x = \tfrac{1}{2x}$ trên $[1, e^2]$. **Hình dạng** của $p_X$ đã phức tạp hơn hẳn phân phối đều ban đầu — đó chính là cách flow tạo ra phân phối phức tạp từ phân phối đơn giản.

---

## 3.2. Chứng minh trường hợp một chiều

**Mệnh đề.** Với $D = 1$, hàm khả nghịch khả vi $x = f(z)$, $z = g(x)$, ta có $p_x(x) = p_z\big(g(x)\big)\, |g'(x)|$.

**Chứng minh.** Giả sử $f$ tăng ngặt. Xuất phát từ hàm phân phối tích lũy (CDF):

$$F_X(x) = \Pr[X \le x] = \Pr[f(Z) \le x] = \Pr[Z \le g(x)] = F_Z\big(g(x)\big)$$

Lấy đạo hàm hai vế theo $x$ (dùng quy tắc dây chuyền):

$$p_x(x) = \frac{dF_X(x)}{dx} = \frac{dF_Z(g(x))}{dx} = p_z\big(g(x)\big)\, g'(x)$$

Với $f$ giảm thì $g'(x) < 0$, và bù trừ dấu cho ta **trị tuyệt đối** $|g'(x)|$ — hợp lý vì mật độ luôn không âm. $\blacksquare$

---

## 3.3. Chứng minh trường hợp nhiều chiều

**Định lý đổi biến (multivariate change of variables).** Cho $g: \mathbb{R}^D \to \mathbb{R}^D$ là song ánh khả vi. Khi đó:

$$p_x(x) = p_z\big(g(x)\big)\, \left| \det J_g(x) \right|$$

**Chứng minh.** Xuất phát từ bảo toàn xác suất trên một tập đo được $A$ bất kỳ. Vì $g$ là song ánh, sự kiện $\{x \in A\}$ tương đương với $\{z \in g(A)\}$:

$$\Pr[x \in A] = \Pr[z \in g(A)] = \int_{g(A)} p_z(z)\, dz$$

Đổi biến tích phân với phép thế $z = g(x)$. Theo **định lý đổi biến trong giải tích nhiều biến**, phần tử thể tích biến đổi theo trị tuyệt đối định thức Jacobian:

$$dz = \left| \det J_g(x) \right|\, dx$$

Thay vào, đổi miền lấy tích phân từ $g(A)$ về $A$:

$$\int_{g(A)} p_z(z)\, dz = \int_{A} p_z\big(g(x)\big)\, \left| \det J_g(x) \right|\, dx$$

Mặt khác, theo định nghĩa mật độ, $\Pr[x \in A] = \int_A p_x(x)\, dx$. Vì đẳng thức

$$\int_A p_x(x)\, dx = \int_A p_z\big(g(x)\big)\, \left| \det J_g(x) \right|\, dx$$

đúng với **mọi** tập đo được $A$, hai hàm dưới dấu tích phân phải bằng nhau hầu khắp nơi:

$$p_x(x) = p_z\big(g(x)\big)\, \left| \det J_g(x) \right| \qquad \blacksquare$$

> **Trực giác hình học.** $\left| \det J_g \right|$ đo **tỉ lệ thay đổi thể tích cục bộ** của phép biến đổi. Nơi $g$ nén không gian lại (thể tích co), mật độ **tăng**; nơi $g$ giãn không gian ra (thể tích nở), mật độ **giảm**. Định thức Jacobian chính là "hệ số nén thể tích" của đại số tuyến tính, áp dụng cục bộ tại từng điểm thông qua đạo hàm.

---

## 3.4. Vì sao định thức = thể tích: trường hợp tuyến tính

Trực giác trên có gốc rễ đại số tuyến tính. Cho $Z$ phân bố đều trên khối lập phương đơn vị $[0, 1]^D$ và $X = AZ$ với $A$ là ma trận khả nghịch. Hình học: $A$ ánh xạ khối lập phương đơn vị thành một **khối hộp xiên (parallelotope)** — tổng quát của hình bình hành lên nhiều chiều.

Trong 2D, với $A = \begin{psmallmatrix} a & c \\ b & d \end{psmallmatrix}$, hình bình hành ảnh có **diện tích** đúng bằng $|\det A| = |ad - bc|$. Tổng quát, thể tích khối hộp xiên bằng $|\det A|$.

Vì $X$ phân bố đều trên khối hộp diện tích $|\det A|$, mật độ là hằng số $1/|\det A|$ bên trong đó. Đặt $W = A^{-1}$:

$$p_X(x) = \frac{p_Z(Wx)}{|\det A|} = p_Z(Wx)\,|\det W|$$

vì $\det(A^{-1}) = 1/\det(A)$. Đây chính là công thức đổi biến cho ánh xạ tuyến tính — và phép biến đổi **phi tuyến** chỉ là tuyến tính hóa cục bộ qua Jacobian: thể tích thay đổi tại mỗi điểm theo $|\det J|$ thay vì một hằng số $|\det A|$ chung.

---

## 3.5. Quan hệ giữa Jacobian thuận và nghịch

Vì $f$ và $g = f^{-1}$ là hai hàm nghịch nhau, theo **quy tắc đạo hàm hàm hợp (chain rule)** áp dụng cho $g(f(z)) = z$:

$$J_g(x) = \big(J_f(z)\big)^{-1}$$

Lấy định thức rồi logarit (nhớ $\det(M^{-1}) = 1/\det(M)$):

$$\log \left| \det J_g(x) \right| = -\log \left| \det J_f(z) \right|$$

Hệ quả này rất tiện: ta có thể tính log-det theo **chiều nào dễ hơn** — thuận hay nghịch — tùy thiết kế của tầng biến đổi. Tương đương, công thức đổi biến cũng viết được theo chiều thuận:

$$p_X(x) = p_Z(z)\,\left| \det \frac{\partial f(z)}{\partial z} \right|^{-1}$$

---

# 4. Học và suy diễn

Ba thao tác cốt lõi, mỗi thao tác dùng một chiều biến đổi:

* **Huấn luyện (MLE)** — cần $g = f^{-1}$ để tính chính xác $\log p_x(x)$ từ dữ liệu (đẩy dữ liệu **về** nhiễu):

$$\max_\theta \sum_{x \in \mathcal{D}} \log p_z\big(f_\theta^{-1}(x)\big) + \log \left| \det J_{f_\theta^{-1}}(x) \right|$$

* **Sinh mẫu** — cần $f$ để biến nhiễu thành dữ liệu: $z \sim p_z(z),\; x = f_\theta(z)$.
* **Suy diễn biểu diễn ẩn** — chỉ cần $z = f_\theta^{-1}(x)$, **không cần mạng suy diễn riêng**.

Thiếu một trong hai chiều $f, g$ là mất một chức năng. Đây là lý do tính khả nghịch không phải tùy chọn mà là **bắt buộc**.

---

# 5. Hai ràng buộc thiết kế (và vì sao bắt buộc)

Để khả thi, mỗi phép biến đổi $f$ phải đồng thời thỏa mãn:

1. **Khả nghịch (invertible) với cả hai chiều tính được hiệu quả.** Tính likelihood cần ánh xạ $x \mapsto z$; lấy mẫu cần ánh xạ $z \mapsto x$.
2. **Định thức Jacobian dễ tính.** Định thức tổng quát của ma trận $D \times D$ tốn $O(D^3)$. Với ảnh $D \sim 10^5$, chi phí này **bất khả thi** trong vòng lặp huấn luyện. Phải thiết kế để Jacobian có **cấu trúc đặc biệt**.

Và prior $p_z(z)$ phải **đơn giản** (lấy mẫu nhanh, likelihood tường minh) — thường là $\mathcal{N}(0, I)$.

Mọi kiến trúc flow đều là các cách khéo léo để thỏa mãn **cả ba** điều kiện cùng lúc.

---

## 5.1. Ý tưởng then chốt: Jacobian tam giác

**Ý tưởng cốt lõi của cả ngành:** chọn phép biến đổi sao cho Jacobian là ma trận **tam giác**. Khi đó định thức chỉ là **tích các phần tử đường chéo** — chi phí $O(D)$ thay vì $O(D^3)$.

Viết $x = (x_1, \dots, x_n) = f(z) = (f_1(z), \dots, f_n(z))$. Nếu mỗi $x_i = f_i(z)$ **chỉ phụ thuộc** $z_{\le i}$, thì $\partial f_i / \partial z_j = 0$ với mọi $j > i$, nên Jacobian là **tam giác dưới**:

$$J = \frac{\partial f}{\partial z} = \begin{pmatrix} \frac{\partial f_1}{\partial z_1} & \cdots & 0 \\ \vdots & \ddots & \vdots \\ \frac{\partial f_n}{\partial z_1} & \cdots & \frac{\partial f_n}{\partial z_n} \end{pmatrix}, \qquad \det J = \prod_{i=1}^n \frac{\partial f_i}{\partial z_i}$$

(Tương tự, $J$ tam giác trên nếu $x_i$ chỉ phụ thuộc $z_{\ge i}$.) Đây là nguyên lý chung đứng sau coupling layer (5.2), flow tự hồi quy (mục 7), và MintNet (6.4).

---

## 5.2. Coupling layer (RealNVP) — chứng minh Jacobian tam giác

Mẹo kinh điển của RealNVP: chia $x$ thành hai nửa $x_a, x_b$. Giữ **nguyên** nửa đầu, biến đổi affine nửa sau theo cách **phụ thuộc nửa đầu**:

$$y_a = x_a, \qquad y_b = x_b \odot \exp\!\big(s(x_a)\big) + t(x_a)$$

trong đó $s, t$ là các mạng nơ-ron tùy ý, $\odot$ là nhân theo từng phần tử.

**Khả nghịch (giải ngược tường minh).** Điều tinh tế: hàm ngược **không** cần nghịch đảo các mạng $s, t$ — chỉ cần đánh giá chúng tại $y_a$:

$$x_a = y_a, \qquad x_b = \big(y_b - t(y_a)\big) \odot \exp\!\big(-s(y_a)\big)$$

**Jacobian tam giác dưới.** Tính ma trận đạo hàm $\frac{\partial y}{\partial x}$ theo từng khối:

$$J = \frac{\partial(y_a, y_b)}{\partial(x_a, x_b)} = \begin{pmatrix} \dfrac{\partial y_a}{\partial x_a} & \dfrac{\partial y_a}{\partial x_b} \\[2mm] \dfrac{\partial y_b}{\partial x_a} & \dfrac{\partial y_b}{\partial x_b} \end{pmatrix} = \begin{pmatrix} I & 0 \\[1mm] \dfrac{\partial y_b}{\partial x_a} & \operatorname{diag}\!\big(\exp(s(x_a))\big) \end{pmatrix}$$

Vì $y_a = x_a$ nên khối trên-trái là $I$ và khối trên-phải là $0$. Định thức của ma trận **tam giác khối** bằng tích định thức các khối trên đường chéo:

$$\det J = \det(I) \cdot \det\!\big(\operatorname{diag}(\exp(s(x_a)))\big) = \prod_j \exp\big(s(x_a)_j\big)$$

Suy ra log-det **cực rẻ**, và quan trọng là **không phụ thuộc** vào khối phức tạp $\frac{\partial y_b}{\partial x_a}$:

$$\log \left| \det J \right| = \sum_j s(x_a)_j \qquad \blacksquare$$

> **Điểm tuyệt đẹp.** Các mạng $s, t$ có thể **tùy ý phức tạp** (CNN sâu) mà Jacobian vẫn rẻ, vì cấu trúc tam giác triệt tiêu hoàn toàn khối khó. Để mọi chiều đều được biến đổi qua thời gian, ta **hoán đổi vai trò** $a, b$ giữa các tầng liên tiếp.

---

# 6. Xếp chồng và các họ kiến trúc "coupling"

## 6.1. Xếp chồng nhiều phép biến đổi

Một phép biến đổi đơn lẻ thường quá yếu. Sức mạnh của flow đến từ việc **ghép chuỗi** (đây là chữ *flow* — một **dòng** biến đổi).

**Mệnh đề.** Hợp của $K$ phép khả nghịch $f = f_K \circ \cdots \circ f_1$ vẫn khả nghịch, và log-det **cộng dồn** qua các tầng.

**Chứng minh.** Tính khả nghịch của hợp các song ánh suy ra bằng quy nạp. Với log-det, theo quy tắc đạo hàm hàm hợp, Jacobian của hợp là tích các Jacobian $J_f = J_{f_K} \cdots J_{f_1}$. Lấy định thức ($\det(AB) = \det(A)\det(B)$) rồi logarit của trị tuyệt đối:

$$\log \left| \det J_f \right| = \sum_{k=1}^{K} \log \left| \det J_{f_k} \right| \qquad \blacksquare$$

Đặt $z_K = x$ và $z_0 = f^{-1}(x)$, log-likelihood toàn cục là:

$$\log p_x(x) = \log p_z(z_0) + \sum_{k=1}^{K} \log \left| \det \frac{\partial z_{k-1}}{\partial z_k} \right|$$

Càng nhiều tầng, dòng chảy càng nắn được phân phối phức tạp hơn (planar flow ở 6.3 minh họa: chỉ 10 tầng đã biến Gauss/đều thành phân phối đa mode).

```python
log_det = 0
z = x
for layer in reversed(flow):
    z, ld = layer.inverse(z)
    log_det = log_det + ld
log_px = normal_log_prob(z) + log_det
loss = -log_px.mean()
```

Toàn bộ lý thuyết trên không ràng buộc kiến trúc cụ thể — điều duy nhất cần là một tầng vừa khả nghịch vừa có Jacobian rẻ. Các mô hình nổi bật khác nhau ở **cách đạt được hai tính chất đó**.

| Mô hình | Đóng góp chính |
| --- | --- |
| **NICE** | Coupling cộng (additive) + tầng rescaling; bảo toàn thể tích |
| **RealNVP** | Coupling affine (có scale), kiến trúc đa tỉ lệ (multi-scale) |
| **Glow** | Tích chập khả nghịch $1 \times 1$ thay phép hoán vị cố định |
| **Planar flow** | Biến đổi hạng-một, log-det qua bổ đề định thức ma trận |
| **MintNet** | CNN khả nghịch bằng tích chập có mặt nạ (masked convolution) |
| **MAF / IAF** | Flow tự hồi quy, đánh đổi tốc độ huấn luyện ↔ tốc độ lấy mẫu |

---

## 6.2. NICE — coupling cộng và tầng rescaling

NICE (Nonlinear Independent Components Estimation, Dinh và cộng sự, 2014) là tiền thân của RealNVP, ghép hai loại tầng.

**Tầng coupling cộng (additive).** Là trường hợp riêng của 5.2 khi **bỏ scale** ($s \equiv 0$):

$$x_{1:d} = z_{1:d}, \qquad x_{d+1:n} = z_{d+1:n} + m_\theta(z_{1:d})$$

Hàm ngược chỉ là phép trừ: $z_{d+1:n} = x_{d+1:n} - m_\theta(x_{1:d})$. Jacobian tam giác với **toàn bộ đường chéo bằng 1**, nên

$$\det J = 1$$

— **biến đổi bảo toàn thể tích (volume preserving)**. Đây là điểm yếu: nếu mọi tầng đều bảo toàn thể tích, mô hình không thể tập trung/trải mật độ giữa các vùng.

**Tầng rescaling.** Để vá điểm yếu trên, NICE thêm một tầng tỉ lệ cuối cùng theo từng chiều:

$$x_i = s_i\, z_i, \qquad z_i = x_i / s_i, \qquad \det J = \prod_{i=1}^n s_i$$

Tầng này (với $\det J \ne 1$) cho phép mô hình co/giãn từng trục — chính là điều RealNVP sau đó tổng quát hóa bằng cách đưa scale $\exp(s(x_a))$ trực tiếp vào mỗi coupling layer (**non-volume preserving**, định thức có thể $>1$ hoặc $<1$).

---

## 6.3. Planar flow — log-det qua bổ đề định thức ma trận

Planar flow (Rezende & Mohamed, 2016) dùng một phép biến đổi **hạng một (rank-one)**:

$$x = f_\theta(z) = z + u\, h\big(w^\top z + b\big)$$

với $\theta = (w, u, b)$ và $h$ là phi tuyến (ví dụ $\tanh$). Trực giác: nó "ép/kéo" không gian dọc theo hướng $u$, quanh siêu phẳng $w^\top z + b = 0$.

Jacobian là $I$ cộng một ma trận hạng một:

$$\frac{\partial f}{\partial z} = I + h'(w^\top z + b)\, u\, w^\top$$

Định thức của $I$ cộng ma trận hạng một tính được trong $O(D)$ nhờ **bổ đề định thức ma trận (matrix determinant lemma)** $\det(I + uv^\top) = 1 + v^\top u$:

$$\left| \det \frac{\partial f}{\partial z} \right| = \left| 1 + h'(w^\top z + b)\, u^\top w \right|$$

Để khả nghịch cần ràng buộc tham số (ví dụ với $h = \tanh$, yêu cầu $h'(w^\top z + b)\,u^\top w \ge -1$). Hạn chế: mỗi tầng chỉ "nắn" theo một hướng nên cần **rất nhiều** tầng — đó là lý do coupling/autoregressive thực dụng hơn cho ảnh.

---

## 6.4. Glow — tích chập khả nghịch $1 \times 1$

RealNVP hoán đổi kênh bằng một phép **hoán vị cố định**. Glow (Kingma & Dhariwal, 2018) tổng quát hóa: thay nó bằng một ma trận **học được** $W$ kích thước $c \times c$ ($c$ là số kênh), áp dụng tại từng vị trí điểm ảnh — chính là một tích chập $1 \times 1$.

**Log-det cho ảnh $h \times w$.** Vì cùng một $W$ tác động độc lập lên $h \cdot w$ vị trí, log-det cộng dồn:

$$\log \left| \det J \right| = h \cdot w \cdot \log \left| \det W \right|$$

Để tính $\det W$ rẻ, Glow dùng **phân rã LU** $W = PLU$, khi đó $\det W$ chỉ là tích đường chéo của $U$. Chi phí $O(c^3)$ — **nhỏ** vì $c$ là số kênh (cỡ vài chục), **không phải** số điểm ảnh (cỡ hàng chục nghìn).

---

## 6.5. MintNet — CNN khả nghịch bằng tích chập có mặt nạ

CNN thông thường rất mạnh nhưng **không khả nghịch** và định thức Jacobian đắt. MintNet (Song và cộng sự, 2019) làm CNN trở thành flow bằng cách dùng **tích chập có mặt nạ (masked convolution)** để áp đặt một thứ tự lên điểm ảnh — đúng tinh thần PixelCNN.

Nhờ thứ tự đó, Jacobian là **tam giác** ⇒ định thức tính trong $O(D)$. Và nếu mọi phần tử đường chéo của Jacobian là **dương ngặt**, phép biến đổi **khả nghịch**. MintNet cho thấy ý tưởng "thứ tự ⇒ Jacobian tam giác" của 5.1 mở rộng được sang tích chập, không chỉ tầng kết nối đầy đủ.

---

# 7. Flow tự hồi quy: MAF và IAF

Đây là họ flow quan trọng nhất ngoài coupling, và là chỗ thể hiện rõ nhất **sự đánh đổi tốc độ huấn luyện ↔ lấy mẫu**.

## 7.1. Mô hình tự hồi quy Gauss chính là một flow

Xét mô hình tự hồi quy với mỗi điều kiện là Gauss:

$$p(x) = \prod_{i=1}^n p(x_i \mid x_{<i}), \qquad p(x_i \mid x_{<i}) = \mathcal{N}\!\big(\mu_i(x_{<i}),\, \exp(\alpha_i(x_{<i}))^2\big)$$

Bộ lấy mẫu của nó dùng mẹo tái tham số hóa (reparameterization): lấy $z_i \sim \mathcal{N}(0,1)$ rồi đặt

$$x_i = \exp(\alpha_i)\, z_i + \mu_i$$

trong đó $\mu_i, \alpha_i$ là hàm của $x_{<i}$. Đây **chính là** một phép biến đổi khả nghịch từ nhiễu $(z_1, \dots, z_n)$ sang dữ liệu $(x_1, \dots, x_n)$. Vậy mọi mô hình AR Gauss đều có thể đọc như một normalizing flow — chỉ khác nhau ở việc ta dùng chiều nào (thuận hay nghịch) cho việc gì.

## 7.2. Masked Autoregressive Flow (MAF)

MAF (Papamakarios và cộng sự, 2017) định nghĩa **chiều thuận $z \mapsto x$** đúng như bộ lấy mẫu AR ở trên:

$$x_i = \exp(\alpha_i(x_{<i}))\, z_i + \mu_i(x_{<i})$$

* **Lấy mẫu (thuận): tuần tự, $O(n)$.** Phải có $x_{<i}$ mới tính được $\mu_i, \alpha_i$ rồi mới ra $x_i$.
* **Tính likelihood (nghịch $x \mapsto z$): song song, $O(1)$ độ sâu.** Cho trước **toàn bộ** $x$, mọi $\mu_i, \alpha_i$ tính được cùng lúc (bằng mạng kiểu MADE), rồi $z_i = (x_i - \mu_i)/\exp(\alpha_i)$.

Jacobian tam giác dưới ⇒ log-det là $-\sum_i \alpha_i$. **Kết luận: MAF tính likelihood nhanh, lấy mẫu chậm** ⇒ hợp với huấn luyện MLE và ước lượng mật độ.

## 7.3. Inverse Autoregressive Flow (IAF)

IAF (Kingma và cộng sự, 2016) **đổi vai trò** $z$ và $x$. Ở đây $\mu_i, \alpha_i$ phụ thuộc $z_{<i}$ thay vì $x_{<i}$:

$$x_i = \exp(\alpha_i(z_{<i}))\, z_i + \mu_i(z_{<i})$$

* **Lấy mẫu (thuận $z \mapsto x$): song song, nhanh.** Có toàn bộ $z$ thì mọi $\mu_i, \alpha_i$ tính cùng lúc.
* **Tính likelihood của dữ liệu ngoài (nghịch $x \mapsto z$): tuần tự, chậm.** Phải có $z_{<i}$ mới tính được $z_i$.

**Kết luận: IAF lấy mẫu nhanh, tính likelihood (dữ liệu ngoài) chậm** ⇒ hợp với sinh thời gian thực.

## 7.4. IAF là nghịch đảo của MAF

Hai mô hình là hai mặt của cùng một đồng xu: **hoán đổi $z \leftrightarrow x$ trong phép nghịch của MAF cho ra phép thuận của IAF**, và ngược lại. Đó là lý do bảng tốc độ của chúng đối xứng nhau:

| | Tính likelihood (huấn luyện) | Lấy mẫu | Hợp với |
| --- | --- | --- | --- |
| **MAF** | **Nhanh** (song song) | Chậm ($O(n)$) | MLE, ước lượng mật độ |
| **IAF** | Chậm ($O(n)$, dữ liệu ngoài) | **Nhanh** (song song) | Sinh thời gian thực |

Một quan sát then chốt cho mục sau: **IAF tính được likelihood của chính những mẫu nó sinh ra một cách nhanh chóng** — vì lúc sinh nó đã có sẵn các $z_1, \dots, z_n$ (chỉ cần lưu lại/cache), khỏi phải đảo ngược tuần tự.

---

# 8. Parallel WaveNet — chưng cất mật độ xác suất

Câu hỏi: **có lấy được cả hai mặt tốt** (huấn luyện nhanh của MAF + lấy mẫu nhanh của IAF) không? Parallel WaveNet trả lời bằng kiến trúc **thầy–trò (teacher–student)** và kỹ thuật **chưng cất mật độ xác suất (probability density distillation)**.

1. **Thầy = MAF.** Huấn luyện thầy bằng MLE — nhanh và ổn định.
2. **Trò = IAF.** Khởi tạo trò, huấn luyện để bắt chước thầy. Trò lấy mẫu nhanh; tuy không tính được likelihood dữ liệu ngoài hiệu quả, nó **tính được likelihood mẫu do chính nó sinh ra** (quan sát ở 7.4).

Trò được huấn luyện để cực tiểu KL giữa phân phối trò $s$ và thầy $t$:

$$D_{\mathrm{KL}}(s \,\|\, t) = \mathbb{E}_{x \sim s}\big[\log s(x) - \log t(x)\big]$$

Ước lượng Monte Carlo của mục tiêu này cần: (i) mẫu $x$ từ trò (IAF — nhanh), (ii) mật độ $s(x)$ do trò gán (nhanh, vì là mẫu của chính nó), (iii) mật độ $t(x)$ do thầy gán (MAF — nhanh khi tính likelihood). **Cả ba đều hiệu quả.**

Khi triển khai: chỉ dùng **trò** để sinh. Kết quả: tăng tốc lấy mẫu so với WaveNet tự hồi quy gốc tới **~1000 lần**.

---

# 9. Gaussianization flows — đảo ngược trực giác

Gaussianization flow (Meng và cộng sự, 2020) tiếp cận từ chiều ngược: thay vì hỏi "biến Gauss thành dữ liệu thế nào", nó hỏi **"biến dữ liệu thành Gauss thế nào"** — vì cực tiểu $D_{\mathrm{KL}}(p_\text{data} \,\|\, p_\theta)$ tương đương với việc đẩy dữ liệu qua $f_\theta^{-1}$ ra đúng phân phối $\mathcal{N}(0, I)$.

**Bước 1 — Gauss hóa theo từng chiều (mẹo CDF nghịch).** Với một chiều, gọi $F_\text{data}$ là CDF của dữ liệu. Khi đó $U = F_\text{data}(\tilde X)$ phân bố **đều** trên $[0,1]$, và $\Phi^{-1}(U)$ (với $\Phi$ là CDF của Gauss chuẩn) phân bố **Gauss**. Vậy phép biến đổi

$$\Phi^{-1} \circ F_\text{data}$$

Gauss hóa dữ liệu một chiều. Áp dụng độc lập cho từng chiều ⇒ Jacobian đường chéo, định thức tính dễ. Lưu ý: sau bước này mỗi chiều **Gauss biên (marginally Gaussian)** nhưng **chưa Gauss đồng thời (jointly)** — xấp xỉ dừng ở đây chính là mô hình **copula** nông (Sklar, 1959).

**Bước 2 — Phép quay.** Áp một ma trận **trực giao** (rotation) lên dữ liệu. Jacobian trực giao có $|\det| = 1$, và $\mathcal{N}(0, I)$ **bất biến qua phép quay**, nên bước này không phá vỡ tính Gauss biên mà lại trộn các chiều để tiến tới Gauss đồng thời.

**Lặp lại Bước 1 + Bước 2** (xếp chồng các copula học được) sẽ dần đưa phân phối về Gauss thật sự — một flow đẹp về lý thuyết với mọi tầng đều có Jacobian tường minh.

---

# 10. Ưu điểm

* **Likelihood chính xác và tối ưu trực tiếp** — từ mục 3, không qua chặn dưới như VAE; huấn luyện đúng bằng cực đại hợp lý (MLE).
* **Lấy mẫu nhanh** — với coupling chỉ một lần truyền thẳng $x = f(z)$, chi phí $O(K)$ theo số tầng, **không** tuần tự theo số chiều như autoregressive.
* **Không gian ẩn diễn giải được, không cần encoder** — vì là song ánh, mỗi $x$ ứng với đúng một $z = f^{-1}(x)$; nội suy (interpolation) và thao tác thuộc tính trong không gian $z$ rất sạch.
* **Linh hoạt về đánh đổi tính toán** — họ MAF/IAF cho phép chọn chiều nhanh tùy mục đích (huấn luyện hay sinh), và Parallel WaveNet ghép được cả hai.

---

# 11. Nhược điểm

* **Ràng buộc khả nghịch giới hạn biểu diễn** — từ mục 5, mỗi tầng phải song ánh và có Jacobian rẻ; lớp hàm này **hẹp hơn** mạng tự do, nên cần **rất nhiều tầng** để bù lại sức biểu diễn.
* **Bảo toàn số chiều tốn bộ nhớ** — vì $z$ và $x$ **cùng kích thước** ở mọi tầng (không nén chiều được như VAE), mô hình lớn và ngốn bộ nhớ.
* **Chất lượng mẫu kém GAN/Diffusion** ở cùng quy mô — hệ quả trực tiếp của hai hạn chế trên.
* **Bất tương thích tô-pô (topology mismatch)** — một song ánh liên tục không thể biến một Gauss đơn mode thành phân phối **đa mode tách rời** mà không phải kéo giãn vô hạn ở vùng giữa. Đây là một giới hạn lý thuyết của flow đơn giản.

---

# 12. Tổng kết

Nhìn từ góc độ xác suất, normalizing flow **không né tránh** bài toán tính likelihood như GAN, cũng **không xấp xỉ** nó như VAE. Thay vào đó nó tính **chính xác** bằng một công cụ giải tích cổ điển — công thức đổi biến:

$$\log p_x(x) = \log p_z\big(g(x)\big) + \log \left| \det J_g(x) \right|$$

Toàn bộ nghệ thuật thiết kế flow nằm ở chỗ làm cho phép biến đổi vừa **khả nghịch** vừa có **Jacobian rẻ** — và mọi mẹo (coupling, tích chập $1\times1$, Jacobian tam giác của flow tự hồi quy, tích chập có mặt nạ của MintNet, mẹo CDF nghịch của Gaussianization) đều là lời giải tài tình cho ràng buộc kép đó.

Flow đổi **sức biểu diễn tự do** lấy **likelihood chính xác cộng lấy mẫu nhanh**. Nó là lựa chọn **đẹp về lý thuyết**, nhưng cái giá khả nghịch khiến nó khó cạnh tranh về độ sắc nét với GAN/Diffusion. Dù vậy, ý tưởng đổi biến và Jacobian tam giác của nó **tái xuất** trong nhiều mô hình lai (flow trên không gian ẩn của VAE, hay diffusion liên tục dưới dạng continuous-time flow).

> Bài tiếp theo — **Autoencoder & VAE** — nới lỏng ràng buộc khả nghịch bằng cách cho decoder **tự do** và cho phép **nén số chiều**, đổi lại chỉ tối ưu được chặn dưới ELBO thay vì likelihood chính xác.

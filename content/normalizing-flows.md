# Luồng chuẩn hóa (Normalizing Flows)

> Luồng chuẩn hóa (normalizing flow) biến một phân phối **đơn giản** (thường là Gauss) thành phân phối dữ liệu **phức tạp** qua một chuỗi **hàm khả nghịch (invertible function)**:
>
> $$z \sim \mathcal{N}(0, I) \;\xrightarrow{\; f_1 \;}\; z_1 \;\xrightarrow{\; f_2 \;}\; z_2 \;\xrightarrow{\;\cdots\;}\; x$$
>
> Nhờ công thức đổi biến (change of variables), nó cho likelihood **chính xác** mà vẫn lấy mẫu chỉ trong **một bước**. Đây là cầu nối giữa hai thế giới: sự chính xác của mô hình tự hồi quy (autoregressive) và tốc độ lấy mẫu của các mô hình một bước.

---

# 1. Động cơ ra đời

Ở bài trước ta đã thấy một sự đánh đổi cốt lõi giữa các họ mô hình sinh (generative model):

* **Mô hình tự hồi quy (autoregressive)** cho likelihood **chính xác**, nhưng lấy mẫu **chậm** $O(D)$ vì sinh tuần tự từng phần tử.
* **VAE** lấy mẫu **nhanh** (một lần truyền qua decoder), nhưng chỉ cho **chặn dưới** likelihood (ELBO), không phải giá trị chính xác.

Câu hỏi trung tâm:

> Liệu có cách nào vừa lấy mẫu nhanh, vừa tính được likelihood **chính xác**?

Ý tưởng then chốt nằm ở một quan sát hình học. Giả sử ta có một phép biến đổi từ nhiễu sang dữ liệu là **song ánh khả vi (diffeomorphism)** — vừa một-một, vừa trơn, vừa có hàm ngược trơn.

Nếu ta biết phép biến đổi này "kéo giãn" hay "nén" thể tích bao nhiêu tại mỗi điểm, thì có thể **truy ngược mật độ** một cách chính xác bằng giải tích. Công cụ làm điều đó chính là **công thức đổi biến**.

---

# 2. Khắc phục điều gì?

Trước khi đi vào toán, hãy xem flow giải quyết các điểm yếu của VAE và autoregressive như thế nào.

**So với VAE.** VAE đưa vào một biến ẩn (latent variable) $z$ rồi phải tính tích phân:

$$p(x) = \int p(x \mid z)\, p(z)\, dz$$

Tích phân này **bất khả thi** với dữ liệu nhiều chiều, nên VAE buộc phải xấp xỉ bằng ELBO. Flow đặt $z$ và $x$ **cùng số chiều** và ánh xạ giữa chúng là **song ánh**. Khi đó tích phân biến mất: với mỗi $x$ chỉ có **đúng một** $z = f^{-1}(x)$. Likelihood trở nên **chính xác**, không còn khoảng hở.

**So với autoregressive.** Lấy mẫu chỉ cần một lần truyền thẳng $x = f(z)$, **không** sinh tuần tự từng chiều.

**Cái giá phải trả.** Để giữ song ánh và để định thức Jacobian tính được rẻ, kiến trúc bị **ràng buộc rất mạnh** (xem mục 5).

| Tiêu chí | Autoregressive | VAE | Normalizing Flow |
| --- | --- | --- | --- |
| Likelihood | Chính xác | Chặn dưới (ELBO) | **Chính xác** |
| Tốc độ lấy mẫu | Chậm $O(D)$ | Nhanh | **Nhanh $O(K)$** |
| Số chiều ẩn | — | Nén được | Bằng số chiều dữ liệu |
| Ràng buộc kiến trúc | Nhẹ | Nhẹ | **Nặng (khả nghịch)** |

---

# 3. Lý thuyết: công thức đổi biến

Cho biến ẩn $z \sim p_z(z)$ (thường là $\mathcal{N}(0, I)$) và một hàm khả nghịch khả vi $x = f(z)$. Đặt hàm ngược:

$$z = f^{-1}(x) = g(x)$$

Khi đó mật độ của $x$ được cho bởi **công thức đổi biến**:

$$p_x(x) = p_z\big(g(x)\big)\, \left| \det J_g(x) \right|, \qquad J_g(x) = \frac{\partial g(x)}{\partial x}$$

Trong đó $J_g(x)$ là **ma trận Jacobian** của $g$ — ma trận chứa mọi đạo hàm riêng.

Lấy logarit để dùng trong huấn luyện (biến tích thành tổng, ổn định số học):

$$\boxed{\;\log p_x(x) = \log p_z\big(g(x)\big) + \log \left| \det J_g(x) \right|\;}$$

Công thức này là **trái tim** của normalizing flow. Hai số hạng có ý nghĩa rõ ràng:

* $\log p_z\big(g(x)\big)$ — xác suất của điểm ảnh ngược trong không gian nhiễu đơn giản.
* $\log \left| \det J_g(x) \right|$ — hiệu chỉnh do phép biến đổi làm **co giãn thể tích**.

---

## 3.1. Chứng minh trường hợp một chiều

**Mệnh đề.** Với $D = 1$, hàm khả nghịch khả vi $x = f(z)$, $z = g(x)$, ta có $p_x(x) = p_z\big(g(x)\big)\, |g'(x)|$.

**Chứng minh.** Nguyên lý nền tảng là **bảo toàn xác suất**: khối lượng xác suất trên một khoảng vi phân không đổi dù ta mô tả nó bằng biến nào,

$$p_x(x)\, |dx| = p_z(z)\, |dz|$$

Chia hai vế cho $|dx|$:

$$p_x(x) = p_z(z)\left| \frac{dz}{dx} \right| = p_z\big(g(x)\big)\, |g'(x)|$$

Ta dùng **trị tuyệt đối** vì mật độ luôn không âm, trong khi $\frac{dz}{dx}$ có thể âm nếu $f$ là hàm giảm. $\blacksquare$

---

## 3.2. Chứng minh trường hợp nhiều chiều

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

## 3.3. Ví dụ trực quan: kéo giãn một chiều

Lấy $z \sim \mathcal{N}(0, 1)$ và phép biến đổi tuyến tính $x = f(z) = a z$ với $a > 0$. Khi đó $g(x) = x/a$ và $g'(x) = 1/a$.

Theo công thức:

$$p_x(x) = p_z\!\left(\frac{x}{a}\right) \cdot \frac{1}{a}$$

* Nếu $a > 1$ (kéo giãn), mật độ bị "trải mỏng" ra: chia cho $a$ làm đỉnh thấp xuống.
* Nếu $a < 1$ (nén lại), mật độ bị "dồn cao" lên: chia cho $a < 1$ làm đỉnh cao lên.

Đây chính là trực giác hình học ở mục 3.2 hiện ra dưới dạng số cụ thể.

---

## 3.4. Quan hệ giữa Jacobian thuận và nghịch

Vì $f$ và $g = f^{-1}$ là hai hàm nghịch nhau, theo **quy tắc đạo hàm hàm hợp (chain rule)** áp dụng cho $g(f(z)) = z$:

$$J_g(x) = \big(J_f(z)\big)^{-1}$$

Lấy định thức rồi logarit (nhớ $\det(M^{-1}) = 1/\det(M)$):

$$\log \left| \det J_g(x) \right| = -\log \left| \det J_f(z) \right|$$

Hệ quả này rất tiện: ta có thể tính log-det theo **chiều nào dễ hơn** — thuận hay nghịch — tùy thiết kế của tầng biến đổi.

---

# 4. Hai chức năng, hai chiều biến đổi

Để hiểu vì sao flow cần đồng thời cả $f$ lẫn $g$, hãy tách bạch hai pha hoạt động:

* **Huấn luyện** — cần $g = f^{-1}$ để tính $\log p_x(x)$ từ dữ liệu $x$ (đẩy dữ liệu **về** nhiễu).
* **Sinh mẫu** — cần $f$ để biến nhiễu $z$ **thành** dữ liệu $x = f(z)$.

Thiếu một trong hai chiều là mất một chức năng. Đây là lý do tính khả nghịch không phải tùy chọn mà là **bắt buộc**.

---

# 5. Hai ràng buộc thiết kế (và vì sao bắt buộc)

Để khả thi, mỗi phép biến đổi $f$ phải đồng thời thỏa mãn:

1. **Khả nghịch (invertible).** Như mục 4 — thiếu chiều nào mất chức năng đó.
2. **Định thức Jacobian dễ tính.** Định thức tổng quát của ma trận $D \times D$ tốn $O(D^3)$. Với ảnh $D \sim 10^5$, chi phí này **bất khả thi**. Phải thiết kế để Jacobian có **cấu trúc đặc biệt** (ví dụ tam giác) sao cho định thức tính được rẻ.

Mọi kiến trúc flow đều là các cách khéo léo để thỏa mãn **cả hai** ràng buộc cùng lúc.

---

## 5.1. Coupling layer (RealNVP) — chứng minh Jacobian tam giác

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

# 6. Xếp chồng nhiều phép biến đổi

Một phép biến đổi đơn lẻ thường quá yếu. Sức mạnh của flow đến từ việc **ghép chuỗi**.

**Mệnh đề.** Hợp của $K$ phép khả nghịch $f = f_K \circ \cdots \circ f_1$ vẫn khả nghịch, và log-det **cộng dồn** qua các tầng.

**Chứng minh.** Tính khả nghịch của hợp hai song ánh là song ánh suy ra hiển nhiên bằng quy nạp. Với log-det, theo quy tắc đạo hàm hàm hợp, Jacobian của hợp là tích các Jacobian:

$$J_f = J_{f_K} \cdots J_{f_1}$$

Lấy định thức (nhớ $\det(AB) = \det(A)\det(B)$), rồi logarit của trị tuyệt đối biến tích thành tổng:

$$\log \left| \det J_f \right| = \sum_{k=1}^{K} \log \left| \det J_{f_k} \right| \qquad \blacksquare$$

Đặt $z_K = x$ và $z_0 = f^{-1}(x)$, log-likelihood toàn cục là:

$$\log p_x(x) = \log p_z(z_0) + \sum_{k=1}^{K} \log \left| \det \frac{\partial z_{k-1}}{\partial z_k} \right|$$

Càng nhiều tầng, "dòng chảy" (flow) càng nắn được phân phối phức tạp hơn. Tên gọi *normalizing flow* đến từ đây: một **dòng** các phép biến đổi đưa dữ liệu **về** phân phối chuẩn hóa $\mathcal{N}(0, I)$.

```python
log_det = 0
z = x
for layer in reversed(flow):
    z, ld = layer.inverse(z)
    log_det = log_det + ld
log_px = normal_log_prob(z) + log_det
loss = -log_px.mean()
```

---

# 7. Các biến thể quan trọng

Toàn bộ lý thuyết trên không ràng buộc kiến trúc cụ thể — điều duy nhất cần là một tầng vừa khả nghịch vừa có Jacobian rẻ. Các mô hình nổi bật khác nhau ở **cách đạt được hai tính chất đó**.

| Mô hình | Đóng góp chính |
| --- | --- |
| **NICE** | Coupling cộng (additive), Jacobian $= 1$ (log-det $= 0$) |
| **RealNVP** | Coupling affine, kiến trúc đa tỉ lệ (multi-scale) |
| **Glow** | Tích chập khả nghịch $1 \times 1$ thay phép hoán vị cố định |
| **MAF / IAF** | Flow tự hồi quy, đánh đổi tốc độ huấn luyện ↔ tốc độ lấy mẫu |

---

## 7.1. Tích chập khả nghịch $1 \times 1$ của Glow

RealNVP hoán đổi kênh bằng một phép **hoán vị cố định**. Glow tổng quát hóa ý này: thay nó bằng một ma trận **học được** $W$ kích thước $c \times c$ (với $c$ là số kênh), áp dụng tại từng vị trí điểm ảnh — đây chính là một tích chập $1 \times 1$.

**Log-det cho ảnh $h \times w$.** Vì cùng một $W$ tác động độc lập lên $h \cdot w$ vị trí, log-det cộng dồn qua mọi vị trí:

$$\log \left| \det J \right| = h \cdot w \cdot \log \left| \det W \right|$$

Để tính $\det W$ rẻ, Glow dùng **phân rã LU**: $W = PLU$, khi đó $\det W$ chỉ là tích đường chéo của $U$. Chi phí là $O(c^3)$ — **nhỏ** vì $c$ là số kênh (cỡ vài chục), **không phải** số điểm ảnh $h \cdot w$ (cỡ hàng chục nghìn).

---

# 8. Ưu điểm

* **Likelihood chính xác và tối ưu trực tiếp** — từ mục 3, không qua chặn dưới như VAE; huấn luyện đúng bằng cực đại hợp lý (MLE).
* **Lấy mẫu nhanh** — chỉ một lần truyền thẳng $x = f(z)$, chi phí $O(K)$ theo số tầng, **không** tuần tự theo số chiều dữ liệu như autoregressive.
* **Không gian ẩn diễn giải được** — vì là song ánh, mỗi $x$ ứng với đúng một $z$; nội suy (interpolation) và thao tác thuộc tính trong không gian $z$ rất sạch.

---

# 9. Nhược điểm

* **Ràng buộc khả nghịch giới hạn biểu diễn** — từ mục 5, mỗi tầng phải song ánh và có Jacobian rẻ; lớp hàm này **hẹp hơn** mạng tự do, nên cần **rất nhiều tầng** để bù lại sức biểu diễn.
* **Bảo toàn số chiều tốn bộ nhớ** — vì $z$ và $x$ **cùng kích thước** ở mọi tầng (không nén chiều được như VAE), mô hình lớn và ngốn bộ nhớ.
* **Chất lượng mẫu kém GAN/Diffusion** ở cùng quy mô — hệ quả trực tiếp của hai hạn chế trên: ràng buộc kiến trúc khiến flow khó đạt độ sắc nét tương đương.
* **Bất tương thích tô-pô (topology mismatch)** — một song ánh liên tục không thể biến một Gauss đơn mode thành phân phối **đa mode tách rời** mà không phải kéo giãn vô hạn ở vùng giữa. Đây là một giới hạn lý thuyết của flow đơn giản.

---

# 10. Tổng kết

Nhìn từ góc độ xác suất, normalizing flow **không né tránh** bài toán tính likelihood như GAN, cũng **không xấp xỉ** nó như VAE. Thay vào đó nó tính **chính xác** bằng một công cụ giải tích cổ điển — công thức đổi biến:

$$\log p_x(x) = \log p_z\big(g(x)\big) + \log \left| \det J_g(x) \right|$$

Toàn bộ nghệ thuật thiết kế flow nằm ở chỗ làm cho phép biến đổi vừa **khả nghịch** vừa có **Jacobian rẻ** — và các mẹo như coupling layer hay tích chập $1 \times 1$ chính là lời giải tài tình cho ràng buộc kép đó.

Flow đổi **sức biểu diễn tự do** lấy **likelihood chính xác cộng lấy mẫu nhanh**. Nó là lựa chọn **đẹp về lý thuyết**, nhưng cái giá khả nghịch khiến nó khó cạnh tranh về độ sắc nét với GAN/Diffusion. Dù vậy, ý tưởng đổi biến và Jacobian tam giác của nó **tái xuất** trong nhiều mô hình lai (ví dụ flow trên không gian ẩn của VAE hay diffusion).

> Bài tiếp theo — **Autoencoder & VAE** — nới lỏng ràng buộc khả nghịch bằng cách cho decoder **tự do** và cho phép **nén số chiều**, đổi lại chỉ tối ưu được chặn dưới ELBO thay vì likelihood chính xác.

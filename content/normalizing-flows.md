# Normalizing Flows

> Normalizing Flow biến một phân phối đơn giản (như Gauss) thành phân phối dữ liệu phức tạp qua một chuỗi **hàm khả nghịch (invertible function)**. Nhờ công thức đổi biến (change of variables), nó cho likelihood **chính xác** mà vẫn lấy mẫu một bước.

## 1. Trực giác ra đời

Autoregressive cho likelihood chính xác nhưng lấy mẫu chậm $O(D)$ vì tuần tự. VAE lấy mẫu nhanh nhưng chỉ cho *chặn dưới* likelihood (ELBO). Câu hỏi:

> Có cách nào vừa lấy mẫu nhanh, vừa tính likelihood **chính xác**?

Ý tưởng then chốt: nếu phép biến đổi từ nhiễu sang dữ liệu là **song ánh khả vi (diffeomorphism)** và ta biết nó "kéo giãn/nén" thể tích bao nhiêu tại mỗi điểm, thì có thể truy ngược mật độ một cách chính xác bằng giải tích — đó chính là công thức đổi biến.

## 2. Khắc phục điều gì (lập luận lý thuyết)

- **So với VAE:** VAE phải đưa vào biến ẩn $z$ và tích phân $p(x) = \int p(x\mid z) p(z)\, dz$ — bất khả thi, nên phải dùng ELBO. Flow đặt $z$ và $x$ **cùng số chiều** và ánh xạ **song ánh**, nên tích phân biến mất: với mỗi $x$ chỉ có đúng một $z = f^{-1}(x)$. Likelihood thành **chính xác**, không còn khoảng hở.
- **So với Autoregressive:** lấy mẫu chỉ một lần truyền thẳng $x = f(z)$, không tuần tự.
- **Cái giá:** để song ánh và Jacobian rẻ, kiến trúc bị **ràng buộc mạnh**.

## 3. Lý thuyết: công thức đổi biến

Cho biến ẩn $z \sim p_z(z)$ (thường $\mathcal{N}(0, I)$) và hàm khả nghịch khả vi $x = f(z)$, đặt $z = f^{-1}(x) = g(x)$. Khi đó:

$$p_x(x) = p_z\big(g(x)\big)\, \left| \det J_g(x) \right|, \qquad J_g(x) = \frac{\partial g(x)}{\partial x}$$

Lấy log để dùng cho huấn luyện:

$$\boxed{\;\log p_x(x) = \log p_z\big(g(x)\big) + \log \left| \det J_g(x) \right|\;}$$

### 3.1. Chứng minh trường hợp một chiều

Bảo toàn xác suất: khối lượng xác suất trên một khoảng vi phân phải bằng nhau dù đổi biến,

$$p_x(x)\, |dx| = p_z(z)\, |dz| \;\Rightarrow\; p_x(x) = p_z(z)\left| \frac{dz}{dx} \right| = p_z\big(g(x)\big)\, |g'(x)|$$

Trị tuyệt đối vì mật độ luôn không âm, còn $\frac{dz}{dx}$ có thể âm nếu $f$ giảm. $\blacksquare$

### 3.2. Chứng minh trường hợp nhiều chiều

**Định lý đổi biến (multivariate).** Cho $g: \mathbb{R}^D \to \mathbb{R}^D$ song ánh khả vi. Với mọi tập đo được $A$:

$$\Pr[x \in A] = \Pr[z \in g(A)] = \int_{g(A)} p_z(z)\, dz$$

Đổi biến tích phân $z = g(x)$, theo định lý đổi biến trong giải tích nhiều biến, phần tử thể tích biến đổi theo trị tuyệt đối định thức Jacobian:

$$dz = \left| \det J_g(x) \right|\, dx$$

Do đó:

$$\int_{g(A)} p_z(z)\, dz = \int_{A} p_z\big(g(x)\big)\, \left| \det J_g(x) \right|\, dx$$

Vì $\Pr[x \in A] = \int_A p_x(x)\, dx$ đúng với **mọi** $A$, hai hàm dưới dấu tích phân phải bằng nhau:

$$p_x(x) = p_z\big(g(x)\big)\, \left| \det J_g(x) \right| \qquad \blacksquare$$

> **Trực giác hình học:** $\left|\det J_g\right|$ đo tỉ lệ thay đổi thể tích cục bộ của phép biến đổi. Nơi $g$ nén không gian lại (thể tích co), mật độ tăng; nơi $g$ giãn không gian ra, mật độ giảm. Định thức Jacobian chính là "hệ số nén thể tích" của đại số tuyến tính, áp dụng cục bộ qua đạo hàm.

### 3.3. Quan hệ giữa Jacobian thuận và nghịch

Vì $f$ và $g = f^{-1}$ nghịch nhau, theo quy tắc đạo hàm hàm hợp $J_g(x) = \big(J_f(z)\big)^{-1}$, nên:

$$\log\left|\det J_g(x)\right| = -\log\left|\det J_f(z)\right|$$

Cho phép tính log-det theo chiều nào tiện hơn.

## 4. Hai ràng buộc thiết kế (và vì sao bắt buộc)

Để khả thi, mỗi phép biến đổi $f$ phải đồng thời:

1. **Khả nghịch** — vì huấn luyện cần $g = f^{-1}$ (để tính $\log p_x$), còn sinh mẫu cần $f$. Thiếu một chiều là mất một chức năng.
2. **Định thức Jacobian dễ tính** — định thức tổng quát của ma trận $D \times D$ tốn $O(D^3)$, bất khả thi khi $D \sim 10^5$. Phải thiết kế để Jacobian có cấu trúc đặc biệt.

### 4.1. Coupling layer (RealNVP) — chứng minh Jacobian tam giác

Mẹo kinh điển: chia $x$ thành hai nửa $x_a, x_b$, giữ nguyên nửa đầu, biến đổi affine nửa sau **phụ thuộc nửa đầu**:

$$y_a = x_a, \qquad y_b = x_b \odot \exp\!\big(s(x_a)\big) + t(x_a)$$

**Khả nghịch** (giải ngược tường minh, không cần nghịch đảo mạng $s, t$):

$$x_a = y_a, \qquad x_b = \big(y_b - t(y_a)\big) \odot \exp\!\big(-s(y_a)\big)$$

**Jacobian tam giác dưới.** Tính ma trận đạo hàm $\frac{\partial y}{\partial x}$ theo khối:

$$J = \frac{\partial(y_a, y_b)}{\partial(x_a, x_b)} = \begin{pmatrix} \dfrac{\partial y_a}{\partial x_a} & \dfrac{\partial y_a}{\partial x_b} \\[2mm] \dfrac{\partial y_b}{\partial x_a} & \dfrac{\partial y_b}{\partial x_b} \end{pmatrix} = \begin{pmatrix} I & 0 \\[1mm] \dfrac{\partial y_b}{\partial x_a} & \operatorname{diag}\!\big(\exp(s(x_a))\big) \end{pmatrix}$$

Vì $y_a = x_a$ nên khối trên-trái là $I$ và trên-phải là $0$. Định thức ma trận tam giác khối bằng tích định thức các khối đường chéo:

$$\det J = \det(I) \cdot \det\!\big(\operatorname{diag}(\exp(s(x_a)))\big) = \prod_j \exp\big(s(x_a)_j\big)$$

Suy ra log-det cực rẻ, **không phụ thuộc** vào khối phức tạp $\frac{\partial y_b}{\partial x_a}$:

$$\log\left|\det J\right| = \sum_j s(x_a)_j \qquad \blacksquare$$

> Điểm tuyệt đẹp: mạng $s, t$ có thể **tùy ý phức tạp** (CNN sâu) mà Jacobian vẫn rẻ, vì cấu trúc tam giác triệt tiêu khối khó. Để mọi chiều đều được biến đổi, ta hoán đổi vai trò $a, b$ qua các tầng.

## 5. Xếp chồng nhiều phép biến đổi

Ghép $K$ phép khả nghịch $f = f_K \circ \cdots \circ f_1$ vẫn khả nghịch. Theo quy tắc đạo hàm hàm hợp, định thức Jacobian là tích, nên log-det **cộng dồn**:

$$\log p_x(x) = \log p_z(z_0) + \sum_{k=1}^{K} \log \left| \det \frac{\partial z_{k-1}}{\partial z_k} \right|, \qquad z_K = x,\; z_0 = f^{-1}(x)$$

Càng nhiều tầng, "dòng chảy" (flow) càng nắn được phân phối phức tạp hơn. Tên *normalizing flow*: dòng các phép biến đổi đưa dữ liệu về phân phối chuẩn hóa $\mathcal{N}(0, I)$.

```python
log_det = 0
z = x
for layer in reversed(flow):
    z, ld = layer.inverse(z)
    log_det = log_det + ld
log_px = normal_log_prob(z) + log_det
loss = -log_px.mean()
```

## 6. Các biến thể quan trọng

| Mô hình | Đóng góp |
| --- | --- |
| **NICE** | Coupling cộng (additive), Jacobian = 1 |
| **RealNVP** | Coupling affine, đa tỉ lệ (multi-scale) |
| **Glow** | Tích chập khả nghịch $1\times 1$ thay phép hoán vị cố định |
| **MAF / IAF** | Flow tự hồi quy, đánh đổi tốc độ huấn luyện ↔ tốc độ lấy mẫu |

### 6.1. Tích chập khả nghịch $1\times 1$ của Glow

Glow thay phép hoán vị kênh cố định bằng ma trận học được $W$ ($c \times c$). Log-det cho ảnh $h \times w$:

$$\log\left|\det J\right| = h \cdot w \cdot \log\left|\det W\right|$$

Phân rã LU giúp tính $\det W$ với chi phí $O(c^3)$ nhỏ (vì $c$ là số kênh, không phải số điểm ảnh).

## 7. Ưu điểm (có dẫn chứng lý thuyết)

- **Likelihood chính xác và tối ưu trực tiếp** — từ mục 3, không qua chặn dưới như VAE; huấn luyện đúng bằng MLE.
- **Lấy mẫu nhanh** — một lần truyền thẳng $x = f(z)$, $O(K)$ tầng, không tuần tự theo chiều dữ liệu.
- **Không gian ẩn diễn giải được** — vì song ánh, mỗi $x$ ứng đúng một $z$; nội suy và thao tác thuộc tính trong không gian $z$ rất sạch.

## 8. Nhược điểm (có dẫn chứng lý thuyết)

- **Ràng buộc khả nghịch giới hạn biểu diễn** — từ mục 4, mỗi tầng phải song ánh và Jacobian rẻ; lớp hàm này hẹp hơn mạng tự do, nên cần **rất nhiều tầng** để bù.
- **Bảo toàn số chiều tốn bộ nhớ** — vì $z$ và $x$ cùng kích thước ở mọi tầng (không nén được chiều như VAE), mô hình lớn và ngốn bộ nhớ.
- **Chất lượng mẫu kém GAN/Diffusion** ở cùng quy mô — hệ quả của hai hạn chế trên: ràng buộc kiến trúc khiến flow khó đạt độ sắc nét tương đương.
- **Bài toán tô-pô (topology mismatch):** một song ánh liên tục không thể biến một Gauss đơn mode thành phân phối đa mode tách rời mà không kéo giãn vô hạn — một giới hạn lý thuyết của flow đơn giản.

## 9. Vị trí trong bức tranh chung

Flow là lựa chọn **đẹp về lý thuyết** khi cần likelihood chính xác và lấy mẫu nhanh, nhưng cái giá khả nghịch khiến nó khó cạnh tranh về độ sắc nét. Các ý tưởng đổi biến và Jacobian tam giác của nó tái xuất trong nhiều mô hình lai (ví dụ flow trên không gian ẩn của VAE/diffusion).

> Bài tiếp theo — **VAE** — nới ràng buộc khả nghịch bằng cách cho decoder tự do, đổi lại chỉ tối ưu được chặn dưới ELBO.

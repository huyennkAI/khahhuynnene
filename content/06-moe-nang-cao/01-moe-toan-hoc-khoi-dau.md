# Bài 1: Mô hình hóa Toán học — Cấu trúc Cụm và Giới hạn của Expert Đơn

> **Nguồn:** Chen et al. (2022), *"Towards Understanding Mixture of Experts in Deep Learning"*, arXiv:2208.02813.
>
> Câu hỏi trung tâm: **Tại sao MoE lại hoạt động?** Bài này thiết lập mô hình toán học chặt chẽ chứng minh rằng khi dữ liệu có **cấu trúc cụm** (cluster structure), một expert đơn có giới hạn lý thuyết không thể vượt qua, còn MoE có thể đạt test error $o(1)$.
>
> $$\underbrace{\text{Expert đơn (bất kỳ kiến trúc nào)}}_{\text{test error} \geq 1/8 = 12.5\%} \quad \longleftrightarrow \quad \underbrace{\text{MoE}}_{\text{test error} = o(1)}$$

---

# 1. Bài toán phân loại với cấu trúc cụm

## 1.1. Dữ liệu dạng đa patch

Paper xét bài toán **phân loại nhị phân** trên dữ liệu dạng đa patch (multi-patch input) — trừu tượng hóa dữ liệu thực tế như ảnh (nhiều vùng không gian) hay văn bản (nhiều token):

$$\mathbf{x} = \big(\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(P)}\big) \in (\mathbb{R}^d)^P$$

Mỗi $\mathbf{x}^{(p)} \in \mathbb{R}^d$ là một patch. Nhãn $y \in \{+1, -1\}$. Mục tiêu: học hàm phân loại $F: (\mathbb{R}^d)^P \to \mathbb{R}$ sao cho $\text{sign}(F(\mathbf{x})) = y$ với xác suất cao.

## 1.2. Cấu trúc cụm — trái tim của lý thuyết

Dữ liệu có $K$ cụm. Mỗi cụm $k \in [K] := \{1, \ldots, K\}$ được đặc trưng bởi hai vector đơn vị:

- **Tín hiệu nhãn** $\mathbf{v}_k \in \mathbb{R}^d$: vector mang thông tin để phân loại
- **Tín hiệu tâm cụm** $\mathbf{c}_k \in \mathbb{R}^d$: định danh "cụm này là cụm nào"

Các vector này **trực giao hoàn toàn với nhau** (orthogonal across all clusters):

$$\|\mathbf{v}_k\|_2 = \|\mathbf{c}_k\|_2 = 1 \quad \forall k, \qquad \langle \mathbf{v}_k, \mathbf{v}_{k'}\rangle = \langle \mathbf{c}_k, \mathbf{c}_{k'}\rangle = \langle \mathbf{v}_k, \mathbf{c}_{k'}\rangle = 0 \quad \forall k \neq k'$$

Điều kiện trực giao đảm bảo các cụm thực sự "khác nhau" trong không gian đặc trưng, không có thông tin chồng lấn.

## 1.3. Định nghĩa 3.1 — Phân phối sinh dữ liệu

**Định nghĩa (Distribution $\mathcal{D}$):** Một mẫu $(\mathbf{x}, y)$ được sinh như sau:

**Bước 1:** Rút ngẫu nhiên một cặp cụm $(k, k')$ với $k \neq k'$ đồng đều từ tất cả các cặp trong $[K]$.

**Bước 2:** Rút nhãn $y \sim \text{Uniform}\{+1, -1\}$ và nhiễu Rademacher $\epsilon \sim \text{Uniform}\{+1, -1\}$, độc lập với nhau.

**Bước 3:** Rút các hệ số biên độ $\alpha \sim \mathcal{D}_\alpha$, $\beta \sim \mathcal{D}_\beta$, $\gamma \sim \mathcal{D}_\gamma$ với điều kiện bị chặn:
$$0 < C_1 \leq \alpha, \beta, \gamma \leq C_2$$

**Bước 4:** Xây dựng $P$ patch của $\mathbf{x}$ theo bảng sau:

| Patch | Công thức | Vai trò |
|---|---|---|
| Patch 1 | $y\alpha \mathbf{v}_k$ | **Feature signal**: tín hiệu nhãn từ cụm $k$ |
| Patch 2 | $\beta \mathbf{c}_k$ | **Cluster-center signal**: định danh cụm $k$ |
| Patch 3 | $\epsilon\gamma \mathbf{v}_{k'}$ | **Feature noise**: tín hiệu nhãn từ cụm *khác*, dấu ngẫu nhiên |
| Patch 4..P | $\sim \mathcal{N}\!\left(0, \frac{\sigma_p^2}{d}\mathbf{I}_d\right)$ | **Gaussian noise**: nhiễu thuần túy |

**Định nghĩa cụm:** Điểm $(x, y) \in \Omega_k$ khi và chỉ khi patch 2 chứa tín hiệu tâm cụm $\mathbf{c}_k$.

## 1.4. Tại sao bài toán này khó?

Độ khó nằm ở **Patch 3**: $\epsilon\gamma\mathbf{v}_{k'}$ là tín hiệu nhãn của cụm *khác* nhưng có dấu Rademacher $\epsilon$ **ngẫu nhiên và độc lập với $y$** — tức là nó **không mang thông tin nào về nhãn thật**. Nhưng về độ lớn, nó tương đương với patch 1 (khi $\alpha \approx \gamma$).

Vấn đề: một expert đơn nhìn vào tất cả patches và phải "đoán" cái nào là tín hiệu thật. Không biết điểm thuộc cụm nào, nó không thể phân biệt $y\alpha\mathbf{v}_k$ (patch 1 — thật) và $\epsilon\gamma\mathbf{v}_{k'}$ (patch 3 — nhiễu).

---

# 2. Định lý 4.1 — Giới hạn tuyệt đối của Expert Đơn

## 2.1. Phát biểu định lý

**Định lý 4.1 (Failure of Single Expert):** Giả sử $\mathcal{D}_\alpha = \mathcal{D}_\gamma$ (biên độ tín hiệu thật và nhiễu bằng nhau). Khi đó, *bất kỳ* hàm số nào có dạng:

$$F(\mathbf{x}) = \sum_{p=1}^P f\!\left(\mathbf{x}^{(p)}\right)$$

dù $f: \mathbb{R}^d \to \mathbb{R}$ là **bất kỳ hàm nào** (kể cả mạng nơ-ron nhiều lớp, bất kỳ hàm kích hoạt, bất kỳ số tham số), đều có test error:

$$\Pr_{(\mathbf{x},y)\sim\mathcal{D}}\!\left[y \cdot F(\mathbf{x}) \leq 0\right] \geq \frac{1}{8}$$

**Ý nghĩa:** Expert đơn không thể đạt accuracy quá **87.5%**, bất kể kiến trúc hay số tham số.

## 2.2. Phác thảo chứng minh

Xét trường hợp $K = 2$ để đơn giản hóa. Đặt $\mathcal{D}_\alpha = \mathcal{D}_\gamma = \delta_\alpha$ (cố định biên độ bằng $\alpha$).

**Ký hiệu:** Đặt bộ tứ ngẫu nhiên $(k, k', y, \epsilon)$ trong quá trình sinh. Có $2K(K-1)$ tổ hợp $(k, k', y, \epsilon)$ đồng xác suất.

**Quan sát chính:** Với $F(\mathbf{x}) = \sum_p f(\mathbf{x}^{(p)})$, ta tách đóng góp:

$$F(\mathbf{x}) = f(y\alpha\mathbf{v}_k) + f(\beta\mathbf{c}_k) + f(\epsilon\gamma\mathbf{v}_{k'}) + \sum_{p \geq 4} f(\text{noise}_p)$$

Xét **bốn trường hợp** với cùng $\mathbf{c}_k$ nhưng thay đổi $(y, \epsilon)$:

| Trường hợp | $y$ | $\epsilon$ | Patch 1 | Patch 3 |
|---|---|---|---|---|
| A | $+1$ | $+1$ | $+\alpha\mathbf{v}_k$ | $+\alpha\mathbf{v}_{k'}$ |
| B | $+1$ | $-1$ | $+\alpha\mathbf{v}_k$ | $-\alpha\mathbf{v}_{k'}$ |
| C | $-1$ | $+1$ | $-\alpha\mathbf{v}_k$ | $+\alpha\mathbf{v}_{k'}$ |
| D | $-1$ | $-1$ | $-\alpha\mathbf{v}_k$ | $-\alpha\mathbf{v}_{k'}$ |

Trong trường hợp A và C: $F_A - F_C = f(+\alpha\mathbf{v}_k) - f(-\alpha\mathbf{v}_k)$ (không phụ thuộc $\epsilon$). Tương tự B và D.

Vì $f$ dùng chung trọng số cho cả patch 1 và patch 3, và $\alpha = \gamma$, **giá trị của $f(\pm\alpha\mathbf{v}_{k'})$ gây nhiễu ngẫu nhiên** vào kết quả. Phân tích xác suất cho thấy tồn tại ít nhất 1/4 × 1/2 = 1/8 tổ hợp $(y, \epsilon)$ mà $\text{sign}(F(\mathbf{x})) \neq y$. $\square$

## 2.3. Điều kiện $\mathcal{D}_\alpha = \mathcal{D}_\gamma$ có nghĩa gì?

Đây là điều kiện "tín hiệu thật và nhiễu cùng cường độ". Trong thực tế, nếu $\alpha \gg \gamma$ thì expert đơn vẫn có thể hoạt động tốt vì tín hiệu thật mạnh hơn nhiễu. Nhưng trong dữ liệu thực tế — ví dụ ngôn ngữ tự nhiên nơi các pattern từ nhiều chủ đề xuất hiện đan xen — điều kiện này thường gần đúng.

---

# 3. Kiến trúc MoE — Giải pháp

## 3.1. Ý tưởng then chốt

MoE thêm một thành phần: **router (bộ định tuyến)** học nhận ra điểm thuộc cụm nào **qua Patch 2** ($\beta\mathbf{c}_k$), rồi giao cho expert chuyên trách cụm đó xử lý. Mỗi expert chỉ cần học phân loại trong **một cụm đơn** — bài toán đơn giản hơn nhiều.

## 3.2. Router (Gating Network)

Router có tham số $\boldsymbol{\Theta} = [\boldsymbol{\theta}_1, \ldots, \boldsymbol{\theta}_M] \in \mathbb{R}^{d \times M}$ với $M$ expert.

Logit của expert $m$ (cộng tổng đóng góp của tất cả patches):

$$h_m(\mathbf{x}; \boldsymbol{\Theta}) = \sum_{p=1}^P \boldsymbol{\theta}_m^\top \mathbf{x}^{(p)}$$

Trọng số định tuyến qua softmax:

$$\pi_m(\mathbf{x}; \boldsymbol{\Theta}) = \frac{\exp\!\left(h_m(\mathbf{x}; \boldsymbol{\Theta})\right)}{\sum_{m'=1}^M \exp\!\left(h_{m'}(\mathbf{x}; \boldsymbol{\Theta})\right)}$$

## 3.3. Expert: CNN 2 lớp với kích hoạt phi tuyến

Expert thứ $m$ với $J$ filter, trọng số $\mathbf{W}_m = [\mathbf{w}_{m,1}, \ldots, \mathbf{w}_{m,J}] \in \mathbb{R}^{d \times J}$:

$$f_m(\mathbf{x}; \mathbf{W}_m) = \sum_{j=1}^J \sum_{p=1}^P \sigma\!\left(\langle \mathbf{w}_{m,j},\, \mathbf{x}^{(p)} \rangle\right)$$

Hàm kích hoạt: $\sigma(z) = z^3$ (lập phương). **Tính phi tuyến là cốt lõi** — Bài 3 sẽ chứng minh rằng MoE tuyến tính ($\sigma(z) = z$) thất bại.

## 3.4. Output MoE với Top-1 Routing

$$F(\mathbf{x}; \boldsymbol{\Theta}, \mathbf{W}) = \sum_{m \in \mathcal{T}_x} \pi_m(\mathbf{x}; \boldsymbol{\Theta}) \cdot f_m(\mathbf{x}; \mathbf{W}_m)$$

trong đó $\mathcal{T}_x = \{\arg\max_m h_m(\mathbf{x}; \boldsymbol{\Theta})\}$ — chỉ **một expert** được kích hoạt cho mỗi input.

## 3.5. Hàm loss

$$\mathcal{L}(\boldsymbol{\Theta}, \mathbf{W}) = \frac{1}{n}\sum_{i=1}^n \ell\!\left(y_i F(\mathbf{x}_i; \boldsymbol{\Theta}, \mathbf{W})\right), \quad \ell(z) = \log(1 + e^{-z})$$

---

# 4. Cấu trúc cụm — Tại sao MoE có thể thành công

## 4.1. Patch 2 là chìa khóa mà Expert đơn bỏ lỡ

Trong thiết kế phân phối $\mathcal{D}$:
- **Patch 1** chứa tín hiệu nhãn → cần học để phân loại
- **Patch 2** chứa tâm cụm → **không mang thông tin nhãn nhưng định danh cụm**
- **Patch 3** là nhiễu → cần bỏ qua

Expert đơn ($F = \sum_p f(\mathbf{x}^{(p)})$) buộc phải dùng **cùng $f$ cho mọi patch** → không thể xử lý patch 2 và patch 1 khác nhau → không thể dùng patch 2 để routing.

MoE **tách biệt** vai trò:
- **Router** đọc patch 2 ($\beta\mathbf{c}_k$) → học $\boldsymbol{\theta}_m \to \mathbf{c}_k$ → nhận ra "đây là cụm $k$"
- **Expert $m$** được router giao dữ liệu cụm $k$ → học $\mathbf{w}_{m,j} \to \mathbf{v}_k$ → phân loại chính xác

## 4.2. Giải thích trực quan với $K = 2$

Hãy tưởng tượng dữ liệu về hai chủ đề: **Thể thao** và **Khoa học**. Văn bản có thể chứa:
- Từ định danh chủ đề (patch 2): "bóng đá", "vật lý" — không nói lên sentiment
- Từ sentiment (patch 1): "tuyệt vời", "thất bại" — phân loại tích cực/tiêu cực
- Nhiễu từ chủ đề khác (patch 3): "vũ trụ" xuất hiện trong bài thể thao, gây confusion

Expert đơn không phân biệt sentiment của "tuyệt vời trong bóng đá" vs "tuyệt vời trong vật lý". Router nhận ra chủ đề (từ patch 2) rồi giao cho đúng expert.

## 4.3. Decomposition bài toán khó thành bài toán đơn giản

| Bài toán | Độ phức tạp | Error lý thuyết |
|---|---|---|
| Expert đơn ($K$ cụm) | Phải phân biệt $K$ pattern tín hiệu + $K$ pattern nhiễu | $\geq 1/8$ |
| Expert $m$ (sau routing) | Chỉ phân biệt $\pm\mathbf{v}_k$ trong cụm $k$ | $o(1)$ |
| Router | Chỉ cần nhận ra $K$ tâm cụm $\mathbf{c}_k$ | Học được qua gradient |

---

# 5. Bảng tóm tắt

| | Expert đơn | MoE (paper 2208.02813) |
|---|---|---|
| **Kiến trúc** | $F = \sum_p f(\mathbf{x}^{(p)})$ | Router $\boldsymbol{\Theta}$ + $M$ expert $\mathbf{W}_m$ |
| **Hàm kích hoạt** | Bất kỳ | $\sigma(z) = z^3$ (lập phương) |
| **Routing** | Không | Top-1 softmax + noise perturbation |
| **Định lý giới hạn** | Test error $\geq 1/8$ (Thm 4.1) | Test error $= o(1)$ (Thm 4.2) |
| **Tín hiệu học được** | Chỉ $\mathbf{v}_k$ (trung bình mờ) | Router: $\mathbf{c}_k$, Expert: $\mathbf{v}_k$ |

---

# 6. Tổng kết

Bài 1 thiết lập hai kết quả nền tảng:

1. **Kết quả âm** — Định lý 4.1: Bất kỳ kiến trúc expert đơn nào, dù mạnh đến đâu, cũng bị chặn về test error $\geq 1/8$ khi dữ liệu có cấu trúc cụm với nhiễu chéo cụm ngang bằng tín hiệu thật.

2. **Kiến trúc MoE** có khả năng vượt qua giới hạn này nhờ router học tín hiệu tâm cụm $\mathbf{c}_k$ trong khi expert học tín hiệu nhãn $\mathbf{v}_k$.

**Bài 2** sẽ phân tích chi tiết thuật toán huấn luyện và cơ chế học của router: tại sao gradient descent đẩy $\boldsymbol{\theta}_m$ hội tụ về $\mathbf{c}_k$ và hai giai đoạn (exploration + router learning) diễn ra như thế nào.

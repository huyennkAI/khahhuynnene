# Bài 3: Định lý 4.2 — Chuyên môn hóa Hoàn toàn và Vai trò của Phi tuyến

> **Nguồn:** Chen et al. (2022), arXiv:2208.02813 — Định lý 4.2 và Mục 6 (Role of Nonlinearity).
>
> Định lý trung tâm của paper: MoE với kích hoạt **lập phương** ($\sigma(z) = z^3$) đạt đồng thời (1) training error bằng 0, (2) test error $o(1)$, và (3) **chuyên môn hóa hoàn toàn** — các expert phân nhóm $[M] = \bigsqcup_k \mathcal{M}_k$, mỗi nhóm chuyên về đúng một cụm. Ngược lại, MoE **tuyến tính thất bại** dù có thể biểu diễn đúng hàm mục tiêu.
>
> $$[M] = \mathcal{M}_1 \sqcup \mathcal{M}_2 \sqcup \cdots \sqcup \mathcal{M}_K, \quad \Pr\!\left[yf_m(\mathbf{x}) \leq 0 \;\middle|\; (\mathbf{x},y) \in \Omega_k, m \in \mathcal{M}_k\right] = o(1)$$

---

# 1. Định lý 4.2 — Phát biểu đầy đủ

## 1.1. Điều kiện

Dưới các điều kiện của Bài 2 (hyperparameter, khởi tạo, phân phối dữ liệu), thuật toán Algorithm 1 sau $T = \tilde{O}(\eta^{-1})$ bước đạt với xác suất $1 - o(1)$:

## 1.2. Bốn kết quả đồng thời

**Kết quả 1 — Training Perfection:**
$$y_i F(\mathbf{x}_i; \boldsymbol{\Theta}^{(T)}, \mathbf{W}^{(T)}) > 0 \quad \forall i \in [n]$$

Toàn bộ tập huấn luyện được phân loại đúng.

**Kết quả 2 — Generalization:**
$$\Pr_{(\mathbf{x},y)\sim\mathcal{D}}\!\left[yF(\mathbf{x}; \boldsymbol{\Theta}^{(T)}, \mathbf{W}^{(T)}) \leq 0\right] = o(1)$$

Test error hội tụ về 0 — vượt qua giới hạn 1/8 của expert đơn.

**Kết quả 3 — Expert Specialization:**

Tồn tại phân hoạch $[M] = \mathcal{M}_1 \sqcup \mathcal{M}_2 \sqcup \cdots \sqcup \mathcal{M}_K$ sao cho với mỗi $k \in [K]$ và mỗi $m \in \mathcal{M}_k$:

$$\Pr_{(\mathbf{x},y)\sim\mathcal{D}}\!\left[yf_m(\mathbf{x}; \mathbf{W}_m^{(T)}) \leq 0 \;\middle|\; (\mathbf{x},y) \in \Omega_k\right] = o(1)$$

Expert $m \in \mathcal{M}_k$ phân loại đúng *gần như tất cả* điểm thuộc cụm $k$.

Và với cụm khác $k' \neq k$, expert $m$ *hoạt động kém*:

$$\Pr_{(\mathbf{x},y)\sim\mathcal{D}}\!\left[yf_m(\mathbf{x}; \mathbf{W}_m^{(T)}) \leq 0 \;\middle|\; (\mathbf{x},y) \in \Omega_{k'}\right] = \Omega\!\left(\frac{1}{K}\right)$$

**Kết quả 4 — Correct Routing:**

Router phân loại đúng cụm với xác suất cao:

$$\Pr_{(\mathbf{x},y)\sim\mathcal{D}}\!\left[\arg\max_m h_m(\mathbf{x}; \boldsymbol{\Theta}^{(T)}) \in \mathcal{M}_k \;\middle|\; (\mathbf{x},y) \in \Omega_k\right] = 1 - o(1)$$

## 1.3. Ý nghĩa tổng hợp

Định lý 4.2 cho thấy training phát hiện **cấu trúc ẩn trong dữ liệu** mà không cần biết trước $K$ hay $\{\mathbf{c}_k\}$ hay $\{\mathbf{v}_k\}$. Quá trình gradient descent **tự tổ chức** thành $K$ nhóm expert, mỗi nhóm tương ứng với một cụm dữ liệu.

---

# 2. Phân tích Expert Specialization

## 2.1. Phân nhóm dựa vào khởi tạo

Phân nhóm $\mathcal{M}_k$ được định nghĩa dựa trên **khởi tạo** (Bài 2, mục 2.3):

$$\mathcal{M}_k = \left\{m \;\middle|\; \underset{k' \in [K],\, j \in [J]}{\arg\max} \langle \mathbf{v}_{k'}, \mathbf{w}_{m,j}^{(0)} \rangle = k\right\}$$

Expert $m$ thuộc nhóm $\mathcal{M}_k$ nếu filter nào đó của nó có inner product lớn nhất với $\mathbf{v}_k$ (tín hiệu nhãn của cụm $k$) trong tất cả $K$ cụm và $J$ filter.

## 2.2. Xác suất phân nhóm đồng đều

Với $M = \Theta(K \log K \log\log d)$ expert và $J = \Theta(\log M \log\log d)$ filter, có thể chứng minh:

$$\Pr\!\left[|\mathcal{M}_k| \geq 1 \quad \forall k \in [K]\right] = 1 - o(1)$$

Mỗi cụm có ít nhất một expert với xác suất cao. Kích thước $J$ đủ lớn để với xác suất cao, tồn tại filter $j$ trong expert $m$ align với mỗi $\mathbf{v}_k$.

## 2.3. Tốc độ học trong giai đoạn khám phá

Tại sao expert $m \in \mathcal{M}_k$ học phân loại cụm $k$ nhanh hơn các cụm khác?

**Gradient của filter $j$ của expert $m$ theo $\mathbf{w}_{m,j}$** (với $\sigma(z) = z^3$):

$$\nabla_{\mathbf{w}_{m,j}} f_m(\mathbf{x}) = 3\sum_{p=1}^P \langle \mathbf{w}_{m,j}, \mathbf{x}^{(p)} \rangle^2 \cdot \mathbf{x}^{(p)}$$

Xét patch 1 của một điểm cụm $k$: $\mathbf{x}^{(1)} = y\alpha\mathbf{v}_k$. Đóng góp vào gradient:

$$3\langle \mathbf{w}_{m,j}, y\alpha\mathbf{v}_k\rangle^2 \cdot y\alpha\mathbf{v}_k = 3\alpha^2 \langle \mathbf{w}_{m,j}, \mathbf{v}_k\rangle^2 \cdot y\alpha\mathbf{v}_k$$

**Thừa số khuếch đại:** $\langle \mathbf{w}_{m,j}, \mathbf{v}_k\rangle^2$ — đây chính là bình phương của alignment ban đầu.

- Nếu $m \in \mathcal{M}_k$: $\langle \mathbf{w}_{m,j}, \mathbf{v}_k\rangle$ lớn nhất → thừa số khuếch đại lớn nhất → học nhanh nhất
- Nếu $m \notin \mathcal{M}_k$: $\langle \mathbf{w}_{m,j}, \mathbf{v}_k\rangle$ nhỏ → học chậm, dễ bị lấn át bởi nhiễu

**Đây chính là lý do phi tuyến $z^3$ (hoặc bất kỳ $z^{2l+1}$) tạo ra phân hóa**: gradient tỉ lệ với lũy thừa alignment → **khuếch đại sự khác biệt nhỏ từ khởi tạo**.

---

# 3. Vai trò của Phi tuyến — Tại sao MoE Tuyến tính Thất bại

## 3.1. Định lý thất bại của MoE tuyến tính

Paper chứng minh (Section 6):

> **Mặc dù MoE tuyến tính có thể biểu diễn đúng hàm mục tiêu** (mixture of linear classifiers có thể đạt accuracy 100% về lý thuyết), thuật toán gradient descent **không học được** MoE tuyến tính hiệu quả.

Thực nghiệm trong paper:
- **MoE phi tuyến** ($\sigma(z) = z^3$): test accuracy ~99.5%, dispatch entropy → ~0.1 (routing rất chắc chắn)
- **MoE tuyến tính** ($\sigma(z) = z$): test accuracy chỉ ~93%, dispatch entropy → ~1.3 (routing gần đồng đều, không có chuyên môn hóa)

## 3.2. Tại sao tuyến tính thất bại — phân tích toán học

Với expert tuyến tính $f_m(\mathbf{x}) = \sum_{j,p} \langle \mathbf{w}_{m,j}, \mathbf{x}^{(p)}\rangle$, gradient:

$$\nabla_{\mathbf{w}_{m,j}} f_m(\mathbf{x}) = \sum_p \mathbf{x}^{(p)}$$

**Thừa số khuếch đại = 1** (hằng số) — không phụ thuộc vào alignment hiện tại của $\mathbf{w}_{m,j}$. Điều này có nghĩa:

1. Mỗi filter trong mỗi expert nhận **cùng tốc độ cập nhật** về hướng $\sum_p \mathbf{x}^{(p)}$
2. **Không có lý do nào** khiến filter của expert $m$ hội tụ về $\mathbf{v}_k$ nhanh hơn $\mathbf{v}_{k'}$ (ngoài nhiễu ngẫu nhiên nhỏ từ khởi tạo)
3. Tất cả expert học **cùng một thứ** → không có phân hóa → router không có tín hiệu → dispatch entropy cao

## 3.3. Đo lường chuyên môn hóa: Dispatch Entropy

Dispatch entropy đo mức độ tập trung của routing:

$$H_{\text{dispatch}} = -\sum_{m=1}^M \bar{\pi}_m \log \bar{\pi}_m, \quad \bar{\pi}_m = \frac{1}{n}\sum_{i=1}^n \Pr(m_{i} = m)$$

| MoE loại | $H_{\text{dispatch}}$ khi hội tụ | Ý nghĩa |
|---|---|---|
| **Phi tuyến** ($z^3$) | ~0.1 (gần 0) | Routing rất chắc chắn, mỗi expert chuyên về cụm riêng |
| **Tuyến tính** ($z$) | ~1.3 (gần $\log M$) | Routing gần đồng đều, expert không phân hóa |

Entropy gần 0 nghĩa là với mỗi input, router chọn **một expert với xác suất rất cao** → routing deterministic → chuyên môn hóa rõ ràng.

## 3.4. Điều kiện cần thiết về kích hoạt

Paper cho thấy **bất kỳ hàm kích hoạt phi tuyến** $\sigma$ thỏa $\sigma''(0) \neq 0$ đều có thể tạo ra phân hóa. Lý do: gradient $\sigma'(\langle \mathbf{w}, \mathbf{x}\rangle) \cdot \mathbf{x}$ khi $\sigma' \neq \text{const}$ sẽ phụ thuộc vào alignment $\langle \mathbf{w}, \mathbf{x}\rangle$ → khuếch đại.

Cụ thể với $\sigma(z) = z^3$: $\sigma'(z) = 3z^2$ → gradient $\propto \langle \mathbf{w}, \mathbf{x}\rangle^2 \cdot \mathbf{x}$ — đây là thừa số khuếch đại bậc 2.

---

# 4. Cơ chế Chống Sụp đổ (Non-Collapse Mechanism)

## 4.1. Khi nào collapse xảy ra?

**Collapse** = tất cả input được route đến một expert duy nhất. Điều này xảy ra khi:
1. Router bắt đầu cho một expert "điểm cao hơn" một chút
2. Expert đó nhận nhiều dữ liệu hơn → loss thấp hơn → router càng ưu tiên expert đó
3. Các expert còn lại không nhận dữ liệu → không cập nhật → càng yếu → router càng không chọn

Đây là **vòng lặp tự tăng cường tiêu cực** — một khi bắt đầu, rất khó thoát.

## 4.2. Ba cơ chế chống collapse trong Algorithm 1

**Cơ chế 1: Noise Perturbation**

Nhiễu $r_{m,i}^{(t)} \sim U[0,1]$ đảm bảo rằng khi router chưa có tín hiệu mạnh (giai đoạn khám phá), mỗi expert vẫn được chọn với xác suất $\approx 1/M$. Không expert nào bị "chết đói" trong giai đoạn đầu.

**Cơ chế 2: Normalized Gradient cho Expert**

$$\mathbf{W}_m^{(t+1)} = \mathbf{W}_m^{(t)} - \eta \frac{\nabla_{\mathbf{W}_m}\mathcal{L}}{\|\nabla_{\mathbf{W}_m}\mathcal{L}\|_F}$$

Dù expert $m$ nhận ít dữ liệu (load thấp), tốc độ học của nó vẫn được chuẩn hóa. Expert $m' \in \mathcal{M}_k$ nhận nhiều dữ liệu không học *nhanh hơn tương đối* so với expert nhận ít dữ liệu.

**Cơ chế 3: Khởi tạo router bằng 0**

$\boldsymbol{\Theta}^{(0)} = \mathbf{0}$ nghĩa là ban đầu tất cả expert có logit bằng nhau, routing hoàn toàn do nhiễu quyết định. Router "bắt đầu không thiên vị".

## 4.3. So sánh với các giải pháp thực tế

| Giải pháp | Paper 2208.02813 | Thực tế (Shazeer 2017, Switch 2021) |
|---|---|---|
| **Chống collapse** | Noise $U[0,1]$ + normalized gradient | Expert capacity + auxiliary loss |
| **Routing** | Soft top-1 với noise | Hard top-K + capacity buffer |
| **Khởi tạo** | $\boldsymbol{\Theta}^{(0)} = \mathbf{0}$ | Random nhỏ |
| **Phân tích lý thuyết** | Chứng minh chặt chẽ | Heuristic / empirical |

---

# 5. Phác thảo Chứng minh Định lý 4.2

## 5.1. Cấu trúc chứng minh

Chứng minh gồm hai phần chính, mỗi phần xử lý một giai đoạn:

### Phần A: Phân tích giai đoạn khám phá ($t \leq T_1$)

**Mục tiêu:** Chứng minh expert phân hóa, i.e., Bổ đề 5.2.

**Bước 1:** Chứng minh router gần bằng 0: $\|h_m(\mathbf{x}; \boldsymbol{\Theta}^{(t)})\|_\infty = O(\sigma_0^{1.5})$ với $t \leq T_1$.

Lý do: Gradient của router nhỏ trong giai đoạn đầu (expert chưa phân hóa → loss feedback yếu). Cụ thể $\|\nabla_{\boldsymbol{\Theta}}\mathcal{L}^{(t)}\| = O(\sigma_0^2)$, và $T_1 \cdot \eta_r \cdot O(\sigma_0^2) = O(\sigma_0^{1.5})$ (với $\eta_r = O(M^2\eta)$ và $T_1 = O(\eta^{-1}\sigma_0^{0.5})$).

**Bước 2:** Vì router gần 0, routing gần đồng đều. Phân tích gradient expert với kích hoạt $z^3$:

$$\frac{d}{dt}\langle \mathbf{w}_{m,j}^{(t)}, \mathbf{v}_k\rangle \approx \frac{\eta}{\|\nabla\|} \cdot 3\langle \mathbf{w}_{m,j}^{(t)}, \mathbf{v}_k\rangle^2 \cdot S_k$$

trong đó $S_k$ là lượng tín hiệu cụm $k$ được đưa đến expert $m$.

**Bước 3:** Giải ODE xấp xỉ, chứng minh $\langle \mathbf{w}_{m,j}^{(T_1)}, \mathbf{v}_k\rangle$ tăng đến $\Omega(1)$ với $m \in \mathcal{M}_k$, nhỏ với $m \notin \mathcal{M}_k$.

### Phần B: Phân tích giai đoạn router learning ($t \in [T_1, T]$)

**Mục tiêu:** Chứng minh routing hội tụ, tức là với $m \in \mathcal{M}_k$, xác suất $\Pr[m_{i,t} = m | (\mathbf{x}_i, y_i) \in \Omega_k] \to 1$.

**Bước 1:** Gradient của router phân rã thành phần đóng góp từ $\mathbf{c}_k$:

$$\nabla_{\boldsymbol{\theta}_m}\mathcal{L} = -\frac{\beta}{n}\sum_{i \in \text{routed to } m} \ell'(y_i F_i) \cdot y_i f_m(\mathbf{x}_i) \cdot \mathbf{c}_{k_i} + \text{small terms}$$

**Bước 2:** Vì $m \in \mathcal{M}_k$ phân loại tốt $\Omega_k$ (từ Phần A), $y_i f_m(\mathbf{x}_i) > 0$ với $(\mathbf{x}_i, y_i) \in \Omega_k$ → $\ell'(y_i F_i) < 0$ (loss giảm) → gradient cho $\boldsymbol{\theta}_m$ là $+\beta\mathbf{c}_k$.

**Bước 3:** $\langle \boldsymbol{\theta}_m, \mathbf{c}_k\rangle$ tăng đơn điệu; dùng Bổ đề 5.1 (stability) để chứng minh routing hội tụ → expert chỉ nhận đúng cụm → chuyên môn hóa hoàn toàn.

## 5.2. Tóm tắt phụ thuộc logic

```
Lemma 5.1 (Stability)
    ↓
Expert Divergence (Lemma 5.2) ← Nonlinearity (σ = z³)
    ↓
Router Signal (∇θ ∝ c_k)
    ↓
Routing Convergence
    ↓
Theorem 4.2: Specialization + Generalization
```

---

# 6. Kết quả thực nghiệm trong Paper

## 6.1. So sánh MoE phi tuyến vs tuyến tính

| Metric | MoE Phi tuyến ($z^3$) | MoE Tuyến tính ($z$) | Expert đơn |
|---|---|---|---|
| Test accuracy | ~99.5% | ~93% | ~87.5% (chặn trên lý thuyết) |
| Dispatch entropy | ~0.1 (chuyên hóa) | ~1.3 (không chuyên) | N/A |
| $\langle\boldsymbol{\theta}_m, \mathbf{c}_k\rangle$ cuối | $\approx 1$ | $\approx 0$ | N/A |
| Expert error trong cụm | $o(1)$ | $\Omega(1)$ | $\Omega(1/8)$ |

## 6.2. Kiểm tra giả thiết

Paper kiểm tra tính robust:
- **Số cụm $K$**: Tăng $K$ từ 2 đến 16, MoE phi tuyến vẫn đạt accuracy cao
- **Nhiễu $\sigma_p$**: Tăng nhiễu Gaussian, MoE phi tuyến vẫn phân hóa, MoE tuyến tính suy giảm
- **Số expert $M$**: Kết quả ổn định với $M \in [K, 4K]$ (không cần $M \gg K$)

---

# 7. Tổng kết

Bài 3 trình bày kết quả trung tâm của paper 2208.02813:

**Định lý 4.2** chứng minh rằng, dưới các điều kiện về phân phối dữ liệu và hyperparameter, Algorithm 1 đạt đồng thời training perfection, generalization ($o(1)$ test error) và expert specialization — các expert tự phân nhóm theo cụm dữ liệu.

**Bài học chính:**
1. **Cấu trúc cụm** trong dữ liệu là điều kiện đủ để MoE có lợi thế
2. **Phi tuyến bậc cao** (như $z^3$) là cần thiết để phân hóa expert qua khuếch đại alignment
3. **Hai giai đoạn** (exploration → router learning) là cơ chế cốt lõi, không phải heuristic
4. **MoE tuyến tính** thất bại không phải vì thiếu biểu diễn mà vì gradient không tạo ra phân hóa

**Bài 4** sẽ kết nối lý thuyết với thực tế: các kiến trúc MoE hiện đại (DeepSeek-MoE, Mixtral, Switch Transformer) hiện thực hóa và mở rộng những nguyên tắc này như thế nào.

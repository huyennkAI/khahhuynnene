# Bài 2: Cơ chế Học của Router — Hai giai đoạn hội tụ về tâm cụm

> **Nguồn:** Chen et al. (2022), arXiv:2208.02813 — Phần 5 (Training Analysis).
>
> Router không "biết trước" cụm nào là gì. Quá trình học diễn ra qua **hai giai đoạn**: giai đoạn *khám phá* (exploration) cho phép các expert phân kỳ mà không bị router lock-in sớm; giai đoạn *học router* (router learning) router dần học tâm cụm $\mathbf{c}_k$. Thuật toán sử dụng **nhiễu ngẫu nhiên** và **normalized gradient** để ngăn collapse.
>
> $$\boldsymbol{\theta}_m^{(t)} \;\xrightarrow{t \to T_1}\; \text{gần} \; \mathbf{0} \quad \Rightarrow \quad \boldsymbol{\theta}_m^{(t)} \;\xrightarrow{t \to T}\; \propto \mathbf{c}_k \;\; \text{với } m \in \mathcal{M}_k$$

---

# 1. Thuật toán huấn luyện (Algorithm 1)

## 1.1. Perturbed Routing — Thêm nhiễu để khám phá

Expert đơn thuần top-1 routing có vấn đề: nếu router không có tín hiệu rõ ngay từ đầu, tất cả input sẽ đổ về một expert, các expert còn lại không được cập nhật → collapse.

Giải pháp: **Perturbed top-1 routing** — thêm nhiễu đều vào logit trước khi chọn expert:

$$m_{i,t} = \arg\max_{m \in [M]}\!\left\{h_m(\mathbf{x}_i; \boldsymbol{\Theta}^{(t)}) + r_{m,i}^{(t)}\right\}$$

trong đó $r_{m,i}^{(t)} \sim \text{Uniform}[0, 1]$ i.i.d. cho mỗi expert $m$, mẫu $i$, và bước $t$.

**Tại sao uniform?** Nhiễu Uniform đảm bảo rằng khi logit chênh lệch nhỏ (router chưa học được), xác suất chọn mỗi expert gần bằng $1/M$. Khi router học được tín hiệu mạnh ($h_m \gg h_{m'}$ với $m' \neq m$), nhiễu $\in [0,1]$ không đủ mạnh để lật kết quả.

## 1.2. Expert Update — Normalized Gradient Descent

Gradient của loss theo expert $m$:

$$\nabla_{\mathbf{W}_m}\mathcal{L}^{(t)} = \frac{1}{n}\sum_{i=1}^n \ell'\!\left(y_i F(\mathbf{x}_i)\right) \cdot y_i \cdot \pi_m(\mathbf{x}_i) \cdot \nabla_{\mathbf{W}_m} f_m(\mathbf{x}_i)$$

**Update với normalized gradient:**

$$\mathbf{W}_m^{(t+1)} = \mathbf{W}_m^{(t)} - \eta \cdot \frac{\nabla_{\mathbf{W}_m}\mathcal{L}^{(t)}}{\|\nabla_{\mathbf{W}_m}\mathcal{L}^{(t)}\|_F}$$

**Tại sao normalize?** Trong giai đoạn đầu, các expert nhận lượng dữ liệu không đều nhau (do routing ngẫu nhiên). Normalize đảm bảo **tốc độ học đồng đều** giữa các expert bất kể load, tránh expert được nhiều dữ liệu học quá nhanh trong khi expert ít dữ liệu gần như không cập nhật.

## 1.3. Router Update — Standard Gradient

$$\boldsymbol{\theta}_m^{(t+1)} = \boldsymbol{\theta}_m^{(t)} - \eta_r \cdot \nabla_{\boldsymbol{\theta}_m}\mathcal{L}^{(t)}$$

Router dùng gradient chuẩn (không normalize) với learning rate $\eta_r = \Theta(M^2)\eta$ — **lớn hơn nhiều** so với expert learning rate. Điều này có chủ đích: router phải học *nhanh hơn* expert trong giai đoạn 2.

## 1.4. Khởi tạo

- Router: $\boldsymbol{\Theta}^{(0)} = \mathbf{0}$ (zero initialization)
- Expert: mỗi entry của $\mathbf{W}^{(0)} \sim \mathcal{N}(0, \sigma_0^2)$ với $\sigma_0 \in [d^{-1/3}, d^{-0.01}]$

**Khởi tạo router bằng 0** có nghĩa là ban đầu tất cả logit $h_m = 0$, nên $\pi_m = 1/M$ → routing hoàn toàn ngẫu nhiên. Chỉ có nhiễu $r^{(t)}$ quyết định expert nào được chọn trong giai đoạn đầu.

---

# 2. Giai đoạn 1 — Exploration Stage ($t \in [0, T_1]$)

## 2.1. Định nghĩa

**Thời điểm kết thúc giai đoạn 1:**

$$T_1 = \left\lfloor \eta^{-1} \sigma_0^{0.5} \right\rfloor \ll T$$

Giai đoạn này rất ngắn so với tổng số bước $T$.

## 2.2. Router hầu như không thay đổi

Trong giai đoạn khám phá, router cập nhật chậm vì tín hiệu gradient yếu. Có thể chứng minh:

$$\left\|h_m(\mathbf{x}; \boldsymbol{\Theta}^{(t)}) - h_m(\mathbf{x}; \boldsymbol{\Theta}^{(0)})\right\|_\infty = O(\sigma_0^{1.5})$$

Vì $\boldsymbol{\Theta}^{(0)} = \mathbf{0}$ nên $h_m(\mathbf{x}; \boldsymbol{\Theta}^{(0)}) = 0$, suy ra logit của router gần bằng 0 trong suốt giai đoạn này. Khi đó nhiễu $r_{m,i}^{(t)} \sim U[0,1]$ **áp đảo** logit → routing gần như đồng đều.

**Hệ quả:** Trong giai đoạn khám phá, **xác suất routing** $\pi_m \approx 1/M$ — mỗi expert nhận xấp xỉ đều $n/M$ mẫu.

## 2.3. Phân rã chuyên gia (Expert Divergence)

Đây là kết quả mấu chốt: trong khi router gần như "ngủ", các **expert tự phân hóa** dựa vào **khởi tạo ngẫu nhiên** $\mathbf{W}^{(0)}$.

**Cơ chế:** Expert $m$ được khởi tạo với các filter $\mathbf{w}_{m,j}^{(0)} \sim \mathcal{N}(0, \sigma_0^2 \mathbf{I})$. Định nghĩa phân nhóm:

$$\mathcal{M}_k := \left\{m \;\middle|\; \arg\max_{k' \in [K], j \in [J]} \langle \mathbf{v}_{k'}, \mathbf{w}_{m,j}^{(0)} \rangle = k \right\}$$

Tức là expert $m$ thuộc nhóm $\mathcal{M}_k$ nếu filter nào đó của nó có correlation cao nhất với $\mathbf{v}_k$ (tín hiệu nhãn của cụm $k$).

**Bổ đề 5.2 (Expert Divergence):** Sau $T_1$ bước, với xác suất $1 - o(1)$:

Với mọi $m \in \mathcal{M}_k$:
$$\Pr_{(\mathbf{x},y)\sim\mathcal{D}}\!\left[y \cdot f_m(\mathbf{x}; \mathbf{W}^{(T_1)}) \leq 0 \;\middle|\; (\mathbf{x},y) \in \Omega_k\right] = o(1)$$

Expert $m$ **phân loại tốt** (error $o(1)$) trên cụm của mình $\Omega_k$.

Nhưng với cụm khác $k' \neq k$:
$$\Pr_{(\mathbf{x},y)\sim\mathcal{D}}\!\left[y \cdot f_m(\mathbf{x}; \mathbf{W}^{(T_1)}) \leq 0 \;\middle|\; (\mathbf{x},y) \in \Omega_{k'}\right] = \Omega(1/K)$$

Expert $m$ **hoạt động kém** (error $\Omega(1/K)$) trên cụm không phải của mình.

## 2.4. Tại sao kích hoạt lập phương $\sigma(z) = z^3$ là cần thiết?

Với $\sigma(z) = z^3$:

$$f_m(\mathbf{x}; \mathbf{W}_m) = \sum_{j,p} \langle \mathbf{w}_{m,j}, \mathbf{x}^{(p)}\rangle^3$$

Khi $\mathbf{w}_{m,j}$ có chút alignment với $\mathbf{v}_k$ (từ khởi tạo), gradient dạng $\nabla \langle \mathbf{w}_{m,j}, \mathbf{v}_k\rangle^3 = 3\langle \mathbf{w}_{m,j}, \mathbf{v}_k\rangle^2 \cdot \mathbf{v}_k$ **khuếch đại** sự khác biệt: filter nào đã align thêm một chút thì sẽ được kéo mạnh hơn về phía $\mathbf{v}_k$.

Với $\sigma(z) = z$ (tuyến tính): gradient là $\mathbf{v}_k$ — hằng số, không khuếch đại sự khác biệt khởi tạo → tất cả expert hội tụ về cùng một hướng → không có phân hóa.

---

# 3. Giai đoạn 2 — Router Learning Stage ($t \in [T_1, T]$)

## 3.1. Sau giai đoạn khám phá: expert đã chuyên hóa, router chưa

Sau $T_1$ bước:
- Expert $m \in \mathcal{M}_k$: phân loại tốt cụm $k$, kém cụm khác
- Router: $\boldsymbol{\Theta}^{(T_1)} \approx \mathbf{0}$ — vẫn gần như không có tín hiệu

Bây giờ router **có thể học**: nó nhận feedback qua loss $\mathcal{L}$, và expert nào phân loại sai thì gradient lớn hơn → router học push input sai cụm sang expert đúng.

## 3.2. Gradient router hội tụ về tâm cụm $\mathbf{c}_k$

Xét gradient của loss theo $\boldsymbol{\theta}_m$. Vì top-1 routing, gradient "chảy qua" qua $\pi_m$:

$$\frac{\partial}{\partial \boldsymbol{\theta}_m} \mathcal{L} = \frac{1}{n}\sum_{i=1}^n \ell'(y_i F(\mathbf{x}_i)) \cdot y_i \cdot f_m(\mathbf{x}_i) \cdot \frac{\partial \pi_m(\mathbf{x}_i)}{\partial \boldsymbol{\theta}_m}$$

Với softmax: $\frac{\partial \pi_m}{\partial \boldsymbol{\theta}_m} = \pi_m(1 - \pi_m) \cdot \sum_p \mathbf{x}^{(p)}$

Đóng góp của mỗi patch vào $\sum_p \mathbf{x}^{(p)}$:
- Patch 1: $y\alpha\mathbf{v}_k$ — tương quan với $y$, trung bình $\approx 0$ khi marginalize $y$
- **Patch 2**: $\beta\mathbf{c}_k$ — **không tương quan với $y$ hay $\epsilon$** → đóng góp ổn định $\beta\mathbf{c}_k$
- Patch 3: $\epsilon\gamma\mathbf{v}_{k'}$ — trung bình $\approx 0$ (vì $\epsilon$ Rademacher)
- Patch 4..P: nhiễu Gaussian — trung bình $\approx 0$

**Kết quả:** Phần đóng góp ổn định vào gradient theo $\boldsymbol{\theta}_m$ khi lấy kỳ vọng qua phân phối $\mathcal{D}$ là **tỉ lệ với $\mathbf{c}_k$** — tâm cụm của các điểm được routing đến expert $m$.

Nếu $m \in \mathcal{M}_k$ và dần dần nhận nhiều điểm cụm $k$ hơn (vì expert nó giỏi cụm $k$):

$$\nabla_{\boldsymbol{\theta}_m}\mathcal{L} \approx -\lambda_k \beta \mathbf{c}_k + \text{noise terms}$$

Khi đó $\boldsymbol{\theta}_m^{(t)}$ được đẩy theo hướng $\mathbf{c}_k$.

## 3.3. Vòng lặp tự tăng cường

Quá trình học tạo ra **vòng lặp tự tăng cường (positive feedback loop)**:

```
Expert m ∈ M_k
  giỏi cụm k (từ giai đoạn 1)
    → nhận nhiều điểm cụm k hơn
      → gradient router θ_m ∝ c_k
        → θ_m hội tụ về c_k
          → router chọn m cho x ∈ Ω_k nhiều hơn
            → expert m nhận nhiều điểm cụm k hơn → ...
```

Đây là cơ chế **co-adaptation** giữa router và expert.

## 3.4. Lemma 5.1 — Ổn định routing qua nhiễu

Một lo ngại: nếu router thay đổi đột ngột, expert đang học tốt có thể bị routing nhầm → mất ổn định.

**Bổ đề 5.1 (Stability via Noise Perturbation):** Với routing có nhiễu Uniform, xác suất routing thỏa:

$$\|\mathbf{p} - \hat{\mathbf{p}}\|_\infty \leq M^2 \|\mathbf{h} - \hat{\mathbf{h}}\|_\infty$$

Khi router thay đổi một lượng nhỏ $\|\mathbf{h} - \hat{\mathbf{h}}\|_\infty = \delta$, xác suất routing chỉ thay đổi tối đa $M^2\delta$ — **thay đổi trơn** (smooth), không đột ngột. Điều này đảm bảo stability trong quá trình học.

---

# 4. Điều kiện đủ để Định lý 4.2 thành lập

## 4.1. Các hyperparameter cần thỏa

**Định lý 4.2** đòi hỏi các điều kiện sau trên hyperparameter:

| Tham số | Điều kiện | Ý nghĩa |
|---|---|---|
| Số mẫu $n$ | $n = \Omega(d)$ | Đủ dữ liệu cho từng chiều |
| Số expert $M$ | $M = \Theta(K \log K \log\log d)$ | Đủ expert cho $K$ cụm với dự phòng |
| Số filter $J$ | $J = \Theta(\log M \log\log d)$ | Đủ filter để một expert chiếm $\mathcal{M}_k$ |
| Khởi tạo $\sigma_0$ | $\sigma_0 \in [d^{-1/3}, d^{-0.01}]$ | Đủ nhỏ để giai đoạn khám phá tồn tại |
| Expert LR $\eta$ | $\eta = \tilde{O}(\sigma_0)$ | Học chậm để không vượt qua $T_1$ |
| Router LR $\eta_r$ | $\eta_r = \Theta(M^2)\eta$ | Router học nhanh hơn expert trong giai đoạn 2 |
| Tổng bước $T$ | $T = \tilde{O}(\eta^{-1})$ | Đủ để cả hai giai đoạn hoàn thành |

## 4.2. Load của expert

**Định nghĩa:** Load của expert $m$ tại bước $t$:

$$\text{Load}_m^{(t)} = \sum_{i=1}^n \Pr(m_{i,t} = m)$$

Tức là tổng xác suất expert $m$ được chọn trên toàn bộ training set. Do routing có nhiễu, đây không phải số nguyên.

Trong giai đoạn khám phá: $\text{Load}_m^{(t)} \approx n/M$ cho mọi $m$ — **cân bằng tải**.

Trong giai đoạn router learning: expert $m \in \mathcal{M}_k$ dần nhận nhiều điểm từ $\Omega_k$, ít điểm từ $\Omega_{k'}$ → load không cân bằng, nhưng đây là điều **mong muốn** (chuyên môn hóa).

---

# 5. Minh họa hội tụ hai giai đoạn

```
Thời gian:  0 ─────────── T1 ──────────────────── T
                │                      │
           Exploration            Router Learning
                │                      │
Router:    θ ≈ 0                  θ_m → c_k
Expert:    phân hóa               ổn định, tinh chỉnh
Routing:   ~ đồng đều             x ∈ Ω_k → m ∈ M_k
Load:      ~ n/M mỗi expert       chuyên môn hóa
```

**Giai đoạn 1 (Exploration):** $T_1 = \eta^{-1}\sigma_0^{0.5}$ bước
- Router "ngủ" ($\boldsymbol{\Theta} \approx \mathbf{0}$)
- Routing do nhiễu Uniform quyết định → gần đều
- Expert dùng normalized gradient học từ mẫu ngẫu nhiên → phân hóa theo khởi tạo

**Giai đoạn 2 (Router Learning):** $T - T_1$ bước
- Expert đã chuyên hóa → routing dần có ý nghĩa
- Router nhận tín hiệu → $\boldsymbol{\theta}_m \to \mathbf{c}_k$
- Vòng lặp tự tăng cường đẩy hệ thống đến chuyên môn hóa hoàn toàn

---

# 6. Tổng kết

Router học tâm cụm $\mathbf{c}_k$ qua một cơ chế tinh tế:

1. **Nhiễu Uniform** cho phép khám phá trong giai đoạn đầu mà không lock-in sớm
2. **Normalized gradient** cho expert đảm bảo phân hóa đồng đều không phụ thuộc load
3. **Phi tuyến lập phương** khuếch đại sự khác biệt nhỏ trong khởi tạo → phân hóa expert
4. **Vòng lặp tự tăng cường** giữa router và expert: expert tốt → nhận nhiều dữ liệu đúng cụm → router học đúng → expert nhận thêm dữ liệu đúng cụm

**Bài 3** sẽ phân tích kết quả cuối cùng: Định lý 4.2 (full statement) về sự chuyên môn hóa hoàn toàn, và tại sao MoE tuyến tính thất bại.

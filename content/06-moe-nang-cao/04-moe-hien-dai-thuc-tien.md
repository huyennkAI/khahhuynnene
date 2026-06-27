# Bài 4: Từ Lý thuyết 2208.02813 đến DeepSeek-MoE — Kết nối và Mở rộng

> Lý thuyết của Chen et al. (2022) đặt nền tảng toán học cho MoE: **cấu trúc cụm + phi tuyến → chuyên môn hóa**. Các hệ thống MoE hiện đại (Switch Transformer, Mixtral, DeepSeek-MoE) không phát minh lại — họ **kỹ thuật hóa** các nguyên tắc đó ở quy mô hàng tỉ tham số và hàng trăm expert, đồng thời giải quyết các thách thức thực tiễn mà lý thuyết chưa xét: load balancing ở quy mô phần cứng, communication overhead, và fine-grained routing.
>
> $$\underbrace{F(\mathbf{x}) = \sum_{m \in \mathcal{T}_x} \pi_m(\mathbf{x}) f_m(\mathbf{x})}_{\text{Lý thuyết: 2208.02813}} \quad \longrightarrow \quad \underbrace{\hat{y} = \sum_{i=1}^{K_s} E_i^{(\text{shared})}(x) + \sum_{i \in \text{top-}K'} G_i E_i^{(\text{routed})}(x)}_{\text{Thực tế: DeepSeek-MoE}}$$

---

# 1. Kết nối lý thuyết — Thực tiễn

## 1.1. Bản đồ các khái niệm

| Lý thuyết (2208.02813) | Thực tiễn (LLM MoE) |
|---|---|
| Cụm $\Omega_k$ với tâm $\mathbf{c}_k$ | Token thuộc "loại ngữ nghĩa" nhất định (syntax, entity, task type...) |
| Router $\boldsymbol{\Theta}$: học $\boldsymbol{\theta}_m \to \mathbf{c}_k$ | Router head (linear layer trên embedding) |
| Expert $f_m$: CNN 2 lớp | FFN của Transformer: $\text{FFN}(x) = \text{ReLU}(xW_1)W_2$ |
| Kích hoạt $z^3$, phi tuyến cần thiết | GELU/SwiGLU đóng vai trò tương đương |
| Top-1 routing + noise perturbation | Top-K routing + auxiliary loss |
| Exploration stage | Warm-up training với router LR nhỏ |
| Non-collapse qua normalized gradient | Expert capacity + load balancing loss |

## 1.2. Điều lý thuyết dự đoán đúng

1. **Phi tuyến là cần thiết**: FFN của Transformer với GELU đã phi tuyến sâu → expert tự chuyên hóa
2. **Tâm cụm học được**: Phân tích attention patterns trong Mixtral 8×7B cho thấy các expert chuyên về domain ngôn ngữ học (punctuation, số, code...) — đúng như lý thuyết dự đoán
3. **Top-1 vs top-K**: Paper chứng minh top-1 đủ mạnh; thực tế top-2 phổ biến vì cho phép mỗi token "tham khảo" hai expert

## 1.3. Điều lý thuyết chưa xét — thách thức thực tiễn

- **Phần cứng**: expert phải phân tán qua GPU, communication overhead cao
- **Discrete routing**: top-K không khả vi, cần auxiliary loss thay vì noise perturbation
- **Động học ngôn ngữ**: tokens có phân phối Zipf, không có cấu trúc cụm "sạch" như lý thuyết
- **Fine-grained granularity**: nhiều expert nhỏ hơn thay vì ít expert lớn

---

# 2. Hành trình từ GShard đến DeepSeek-MoE

## 2.1. GShard (2020) — MoE Transformer đầu tiên quy mô lớn

Lepikhin et al. (2020) thêm MoE vào Transformer encoder: mỗi lớp FFN thay bằng $E$ expert, top-2 routing.

**Đặc điểm kỹ thuật:**
- $E = 2048$ expert phân trên nhiều TPU
- Local dispatch: mỗi device chứa một số expert, token được gửi đến đúng device
- **Expert capacity $C$**: mỗi expert xử lý tối đa $C$ token/batch — token vượt quá bị drop
- **Auxiliary loss** lần đầu xuất hiện: penalize khi expert bị overloaded

$$\mathcal{L}_{\text{aux,GShard}} = \frac{1}{E}\sum_{e=1}^E \left(\frac{\text{load}_e}{\text{load}_{\text{mean}}}\right)^2$$

## 2.2. Switch Transformer (2021) — Top-1 Simplification

Fedus et al. (2021) đơn giản hóa: top-2 → **top-1** (mỗi token chỉ vào một expert).

**Lý do top-1:**
- Giảm 50% communication cost
- Paper chứng minh top-1 vẫn đạt performance tốt (consistent với lý thuyết 2208.02813 — top-1 đủ để chuyên hóa)
- Dễ phân tích hơn

**Auxiliary loss của Switch:**

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^E f_e \cdot P_e$$

- $f_e = \frac{1}{T}\sum_{\text{tokens}} \mathbf{1}[\text{token → expert }e]$: fraction thực sự được route
- $P_e = \frac{1}{T}\sum_{\text{tokens}} \text{softmax}(h_e(\mathbf{x}))$: mean routing probability
- Tích $f_e \cdot P_e$ là differentiable (vì $P_e$ khả vi) nhưng phản ánh load thực ($f_e$)

## 2.3. Mixtral 8×7B (2023) — MoE cho LLM Open Source

Mistral AI phát hành Mixtral 8×7B: 8 expert, top-2 routing, mỗi FFN trong Transformer thay bằng MoE layer.

**Thống kê:**
- 46.7B tham số tổng
- ~12.9B tham số active/token (top-2 × 7B/expert ≈ 2 × 6.7B)
- Performance tương đương Llama 2 70B với chi phí inference ~2.5× thấp hơn

**Phân tích chuyên hóa thực tế** (Mixtral paper):
- Expert 1: punctuation và structural tokens
- Expert 2: numerical và code tokens  
- Expert 3: named entities
- Expert 4: verbs và predicates
- v.v.

Điều này **xác nhận thực nghiệm** lý thuyết 2208.02813: cụm trong dữ liệu ngôn ngữ tương ứng với loại ngữ nghĩa, và router học nhận dạng cụm.

---

# 3. DeepSeek-MoE — Kiến trúc Nâng cao

## 3.1. Hai đổi mới cốt lõi

**DeepSeek-MoE (2024)** giới thiệu hai ý tưởng mở rộng trực tiếp từ phân tích lý thuyết:

### Đổi mới 1: Fine-grained Expert Decomposition

**Vấn đề với MoE thông thường:** $K$ expert, top-2 routing → mỗi token trung bình kết hợp thông tin của 2 "kỹ năng". Nhưng nếu cụm dữ liệu có **nhiều sub-cluster**, 2 expert thô chưa đủ fine-grained.

**Giải pháp:** Chia $N$ expert thành $mN$ expert *nhỏ hơn* (cùng tổng số tham số), routing top-$mK$:

$$\hat{y} = \sum_{i \in \text{top-}mK} G_i(\mathbf{x}) \cdot E_i(\mathbf{x})$$

Với $m = 4$ và $K = 2$ (top-2 thông thường): thay vì 8 expert top-2, dùng 32 expert nhỏ hơn top-8. **Cùng số tham số active** (tổng FFN size như nhau) nhưng **nhiều kết hợp hơn**.

**Phân tích toán học — Error bound:**

Với $N$ expert thông thường (top-2):

$$\epsilon_{\text{approx}} = \min_{\text{routing}} \left\|f^* - \sum_{m \in \text{top-2}} \pi_m f_m\right\|$$

Với $mN$ expert nhỏ hơn (top-$2m$), cùng compute budget:

$$\epsilon_{\text{fine-grained}} \leq \epsilon_{\text{approx}} \cdot m^{-\alpha}$$

cho hằng số $\alpha > 0$ phụ thuộc vào độ smooth của $f^*$. Granularity cao hơn → approximation error thấp hơn.

### Đổi mới 2: Shared Experts

**Vấn đề:** Trong lý thuyết 2208.02813, "thông tin chung" được học qua normalized gradient (tất cả expert đều nhận một ít từ mỗi cụm). Trong thực tế với top-K routing cứng, nhiều token *không được gửi đến expert xử lý kiến thức chung* → kiến thức này phải được duplicate trong nhiều expert → lãng phí capacity.

**Giải pháp:** $K_s$ **shared experts** luôn được kích hoạt cho mọi token:

$$\hat{y} = \underbrace{\sum_{i=1}^{K_s} E_i^{(\text{shared})}(\mathbf{x})}_{\text{luôn kích hoạt}} + \underbrace{\sum_{i \in \text{top-}K'} G_i(\mathbf{x}) E_i^{(\text{routed})}(\mathbf{x})}_{\text{conditional}}$$

**Phân tích lý thuyết** (kết nối 2208.02813): Trong paper gốc, expert $m \in \mathcal{M}_k$ học cả tín hiệu đặc trưng cụm $k$ (**từ Patch 1**) lẫn một phần thông tin chung (từ Patch 4..P). Shared expert trong DeepSeek-MoE giải phóng routed expert khỏi việc học thông tin chung → routed expert chuyên hóa sâu hơn.

## 3.2. Load Balancing Loss Nâng cao

### Expert-level Balance Loss

$$\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i \cdot P_i$$

Giống Switch Transformer nhưng trên $N' = mN$ expert nhỏ.

### Device-level Balance Loss (đổi mới của DeepSeek)

Phân expert thành $D$ nhóm theo device. Với nhóm $j$:

$$f_j^{(\text{dev})} = \sum_{i \in \text{group}_j} f_i, \quad P_j^{(\text{dev})} = \sum_{i \in \text{group}_j} P_i$$

$$\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{j=1}^D f_j^{(\text{dev})} \cdot P_j^{(\text{dev})}$$

**Tại sao cần device-level?** Expert-level balance tốt không đảm bảo device-level balance tốt: nếu nhiều expert "nặng" cùng nằm trên một device, device đó sẽ là bottleneck. Device-level loss trực tiếp penalize điều này.

### Tổng loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{ExpBal}} + \mathcal{L}_{\text{DevBal}}$$

## 3.3. Hiệu quả thực nghiệm

| Mô hình | Params total | Params active | Tương đương về quality |
|---|---|---|---|
| LLaMA2-7B (Dense) | 7B | 7B | Baseline |
| DeepSeek-MoE 16B | 16.4B | 2.8B | Tương đương LLaMA2-7B |
| Mixtral 8×7B | 46.7B | 12.9B | Tương đương LLaMA2-70B |

DeepSeek-MoE 16B sử dụng chỉ **40% compute** của LLaMA2-7B khi inference nhưng đạt quality tương đương — minh chứng cho lợi thế lý thuyết của phân rã bài toán thành sub-problems.

---

# 4. Expert Parallelism — Kỹ thuật phân tán

## 4.1. Chiến lược phân tán

Trong MoE quy mô lớn, không thể đặt tất cả expert trên một device. **Expert Parallelism (EP)** phân tán expert qua các device:

- Device $d$ chứa một tập expert $\mathcal{E}_d \subset [M]$
- Token $\mathbf{x}$ được route đến expert $m^* \in \mathcal{E}_d$ → phải gửi đến device $d$

## 4.2. All-to-All Communication

Quá trình forward pass với EP:

```
Bước 1: Mỗi device tính router score cho tokens của mình
Bước 2: All-to-All dispatch — token được gửi đến đúng device có expert
Bước 3: Mỗi device tính expert output cho tokens nhận được
Bước 4: All-to-All combine — kết quả gửi về device gốc
```

**Chi phí communication:** $O(T_{\text{batch}} \times d_{\text{model}} \times 2)$ per layer (gửi đi + nhận về) — đây là bottleneck chính của EP.

## 4.3. Kết hợp với Tensor Parallelism (TP)

| Phương pháp | Phân tán theo | Overhead |
|---|---|---|
| Tensor Parallelism | Cột/hàng ma trận trọng số | All-reduce mỗi FFN |
| Expert Parallelism | Expert | All-to-All mỗi MoE layer |
| TP + EP | Cả hai | All-reduce + All-to-All |

DeepSeek sử dụng EP=8 (8 device/nhóm EP) kết hợp với TP=1 trong các group này để tối ưu băng thông.

---

# 5. Scaling Laws của MoE

## 5.1. Kaplan et al. vs MoE

Scaling law thông thường (Kaplan 2020): $\mathcal{L} \propto C^{-\alpha}$ với $C$ = compute, $\alpha \approx 0.05$.

Với MoE: mỗi token chỉ dùng $K/E$ fraction của tham số → **compute hiệu quả** cao hơn. Scaling law của MoE:

$$\mathcal{L}(C, E) \propto \left(\frac{C}{E^{\beta}}\right)^{-\alpha}$$

với $\beta \approx 0.5$–0.8 phụ thuộc vào routing scheme. Nghĩa là: với cùng $C$, tăng số expert $E$ giảm loss thêm $E^{\alpha\beta}$ lần.

## 5.2. Kết nối với lý thuyết 2208.02813

Scaling law MoE xác nhận lý thuyết: tăng $E$ (nhiều cụm hơn được phủ) → loss giảm. Nhưng có điểm bão hòa khi $E$ lớn hơn số "cụm ngữ nghĩa" thực sự trong dữ liệu — lúc đó các expert bắt đầu overlap.

---

# 6. Hướng nghiên cứu tương lai

## 6.1. Mixture of Depths (MoD)

Thay vì "mỗi token chọn expert nào", MoD hỏi "layer nào token cần đi qua":

$$\text{output} = x + \begin{cases} f_{\text{layer}}(x) & \text{nếu token được chọn} \\ 0 & \text{skip layer} \end{cases}$$

Kết nối lý thuyết: bài toán tương tự 2208.02813 nhưng expert = "có xử lý không" thay vì "xử lý bằng cách nào".

## 6.2. Dynamic MoE

Router thay đổi theo **context** (không chỉ theo token hiện tại):

$$G_k(\mathbf{x}_t | \mathbf{x}_{1:t-1})$$

Lý thuyết 2208.02813 xét routing stateless (chỉ dựa vào $\mathbf{x}$); dynamic routing là bước mở rộng tự nhiên.

## 6.3. Mở rộng lý thuyết — câu hỏi mở

1. **Lý thuyết cho $\sigma = \text{GELU/SwiGLU}$**: Paper dùng $z^3$ cho phân tích sạch. Liệu kết quả có giữ nguyên với activation thực tế?

2. **Cluster structure trong ngôn ngữ tự nhiên**: Phân phối token có thực sự thỏa giả định trong Definition 3.1? Cần phân tích empirical.

3. **Top-K với $K > 1$**: Bài toán mở — paper chứng minh top-1, nhưng top-2 phổ biến hơn trong thực tế.

4. **Shared expert từ góc lý thuyết**: Làm thế nào shared expert thay đổi dynamics của chuyên môn hóa?

---

# 7. Tổng kết — Bản đồ kết nối

```
Chen et al. 2208.02813 (2022)
├─ Cluster structure → MoE lý do tồn tại
├─ Nonlinearity → expert divergence
├─ Two-phase training → exploration + specialization
└─ Theorem 4.2 → generalization guarantee
        │
        ▼ Kỹ thuật hóa
        
GShard (2020): EP + capacity
    → Switch (2021): top-1 + auxiliary loss
        → Mixtral (2023): open-source LLM scale
            → DeepSeek-MoE (2024): fine-grained + shared experts
                + device-level balance
```

Hành trình từ lý thuyết đến thực tiễn là câu chuyện của việc **giữ nguyên nguyên lý** (phi tuyến, phân cụm, routing) trong khi **thích nghi với ràng buộc phần cứng** (capacity, communication, load balancing). Lý thuyết dự đoán *tại sao* MoE hoạt động; kỹ thuật quyết định *làm sao* để nó hoạt động ở quy mô tỉ tham số.

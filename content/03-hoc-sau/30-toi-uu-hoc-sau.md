# Thuật toán tối ưu cho học sâu

> Huấn luyện một mạng nơ-ron sâu thực chất là **đi xuống một mặt phẳng mất mát (loss landscape)** gồ ghề và nhiều chiều. [Backprop](#/backprop) lo phần tính gradient; còn **thuật toán tối ưu (optimizer)** quyết định ta dùng gradient đó để **bước đi** như thế nào.
>
> $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\, \cdot\, (\text{một phép biến đổi thông minh của } \mathbf{g}_t)$$
>
> Bài này đi từ Gradient Descent thô sơ tới Adam — bộ tối ưu mặc định của hầu hết mô hình hiện đại.

---

# 1. Phân công lao động: backprop và optimizer

Trước khi vào chi tiết, cần tách bạch hai việc thường bị gộp làm một.

[Backprop](#/backprop) trả lời câu hỏi **"đạo hàm của mất mát theo từng tham số là bao nhiêu?"** — nó cho ta vector gradient

$$\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)$$

Optimizer trả lời câu hỏi tiếp theo: **"có gradient rồi thì cập nhật tham số ra sao?"**. Đây là hai khâu độc lập — ta có thể giữ nguyên backprop nhưng thay optimizer (SGD → Adam) mà không đụng gì tới mạng. Toàn bộ bài này nói về khâu thứ hai.

Mục tiêu vẫn là bài toán cực tiểu hóa quen thuộc từ [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu):

$$\boldsymbol{\theta}^{*} = \arg\min_{\boldsymbol{\theta}}\; L(\boldsymbol{\theta}), \qquad L(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^{n} \ell_i(\boldsymbol{\theta})$$

Điểm khác biệt cốt lõi của học sâu: $L$ **không lồi (non-convex)** (xem [Hàm lồi](#/convex-functions) để biết vì sao tính lồi lại quý), nên ta không kỳ vọng tìm được cực tiểu toàn cục, chỉ cần một cực tiểu **đủ tốt**.

---

# 2. GD, SGD và minibatch SGD

## 2.1. Đánh đổi giữa nhiễu và tốc độ

Gradient của toàn bộ tập dữ liệu là trung bình của $n$ gradient thành phần:

$$\mathbf{g}_t = \frac{1}{n}\sum_{i=1}^{n}\nabla_{\boldsymbol{\theta}}\,\ell_i(\boldsymbol{\theta}_t)$$

**Gradient Descent (GD)** dùng đúng công thức này — cộng cả $n$ mẫu cho mỗi bước cập nhật. Hướng đi chính xác nhưng mỗi bước **cực kỳ tốn**: với $n$ hàng triệu mẫu, một bước thôi đã phải quét toàn bộ dữ liệu.

**Stochastic Gradient Descent (SGD)** đi tới thái cực ngược lại: mỗi bước lấy **một mẫu** $i$ ngẫu nhiên,

$$\mathbf{g}_t \approx \nabla_{\boldsymbol{\theta}}\,\ell_i(\boldsymbol{\theta}_t)$$

Đây là **ước lượng không chệch (unbiased estimate)** của gradient thật, vì $\mathbb{E}_i[\nabla \ell_i] = \nabla L$. Rẻ gấp $n$ lần nhưng **nhiễu** — mỗi bước đi lệch hướng so với gradient thật.

**Minibatch SGD** là dung hòa: lấy một lô (batch) $\mathcal{B}$ gồm $b$ mẫu,

$$\mathbf{g}_t = \frac{1}{b}\sum_{i\in\mathcal{B}}\nabla_{\boldsymbol{\theta}}\,\ell_i(\boldsymbol{\theta}_t)$$

## 2.2. Vì sao minibatch là lựa chọn mặc định

Phương sai của ước lượng gradient giảm theo kích thước lô:

$$\operatorname{Var}[\mathbf{g}_t] \;\propto\; \frac{1}{b}$$

Tăng $b$ lên 4 lần chỉ giảm độ lệch chuẩn của nhiễu đi 2 lần ($\sqrt{b}$) — lợi ích giảm dần, trong khi chi phí tính toán tăng tuyến tính. Ngoài ra minibatch tận dụng được **song song hóa trên GPU** (nhân ma trận theo lô), điều mà SGD một-mẫu lãng phí.

| Biến thể | Mẫu mỗi bước | Nhiễu gradient | Chi phí/bước | Tận dụng GPU |
| --- | --- | --- | --- | --- |
| Batch GD | toàn bộ $n$ | thấp nhất | cao nhất | tốt nhưng ít bước |
| SGD | 1 | cao nhất | thấp nhất | kém |
| Minibatch SGD | $b$ (32–512) | trung bình | trung bình | tốt |

> Một điểm tinh tế: nhiễu của minibatch **không phải lúc nào cũng xấu**. Nó giúp thuật toán "rung lắc" thoát khỏi các cực tiểu cục bộ nông và điểm yên ngựa — điều ta bàn ngay dưới đây.

---

# 3. Thách thức của mặt phẳng mất mát

Vì $L$ không lồi, đường đi xuống gặp ba loại địa hình khó.

**Cực tiểu cục bộ (local minima).** Điểm thấp hơn mọi lân cận nhưng cao hơn cực tiểu toàn cục. Tại đó $\nabla L = \mathbf{0}$ nên GD đứng yên. Tin tốt: trong không gian rất nhiều chiều, cực tiểu cục bộ "thật sự xấu" lại hiếm — phần lớn điểm dừng là loại tiếp theo.

**Điểm yên ngựa (saddle point).** Điểm mà $\nabla L = \mathbf{0}$ nhưng là cực tiểu theo một số hướng và cực đại theo số hướng khác (ma trận Hessian có cả trị riêng dương lẫn âm). Trong không gian $d$ chiều, để là cực tiểu thật cần **cả $d$ hướng** đều cong lên — xác suất này giảm theo cấp số nhân với $d$. Vì vậy ở học sâu, **điểm yên ngựa mới là chướng ngại chính**, không phải cực tiểu cục bộ.

**Cao nguyên (plateau).** Vùng gần phẳng, gradient gần $\mathbf{0}$ trên một khoảng rộng. GD bò cực chậm vì bước đi $\eta\,\mathbf{g}_t$ tỉ lệ thuận với độ lớn gradient.

Cả ba đều có chung một bệnh: **gradient nhỏ hoặc triệt tiêu khiến bước đi đình trệ**. Các optimizer dưới đây ra đời để chữa đúng bệnh này — bằng quán tính, bằng tỉ lệ học thích nghi, hoặc cả hai.

---

# 4. Momentum: tích lũy quán tính

## 4.1. Trực giác hòn bi lăn dốc

SGD thuần giống một người đi bộ, mỗi bước chỉ nhìn độ dốc **ngay tại chân mình**. Momentum biến nó thành một **hòn bi lăn**: nó nhớ vận tốc cũ và cộng dồn, nên có quán tính.

Ta duy trì một vector vận tốc $\mathbf{v}_t$ là **trung bình trượt (running average)** của các gradient:

$$\mathbf{v}_t = \beta\,\mathbf{v}_{t-1} + \mathbf{g}_t, \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\,\mathbf{v}_t$$

với $\beta \in [0,1)$ (thường $0.9$) là hệ số quán tính.

## 4.2. Vì sao Momentum vượt khe hẹp nhanh hơn

Triển khai đệ quy công thức trên cho thấy $\mathbf{v}_t$ là tổng có trọng số giảm dần của **toàn bộ lịch sử** gradient:

$$\mathbf{v}_t = \sum_{k=0}^{t} \beta^{\,k}\,\mathbf{g}_{t-k}$$

Hãy xét một "khe hẹp" (ravine) — vùng mà mặt mất mát dốc đứng theo một hướng nhưng thoai thoải theo hướng kia (rất phổ biến quanh cực tiểu).

* **Hướng dốc đứng:** gradient liên tục **đổi dấu** (lúc trái lúc phải qua đáy khe). Khi cộng dồn theo $\mathbf{v}_t$, các số hạng triệt tiêu nhau → dao động bị **dập tắt**.
* **Hướng thoai thoải:** gradient luôn **cùng dấu**. Cộng dồn lại chúng **cộng hưởng**, vận tốc lớn dần.

Trong trường hợp gradient ổn định bằng $\mathbf{g}$ theo một hướng, vận tốc hội tụ về

$$\mathbf{v}_\infty = \mathbf{g}\sum_{k=0}^{\infty}\beta^{\,k} = \frac{\mathbf{g}}{1-\beta}$$

Với $\beta = 0.9$, bước đi hiệu dụng được **khuếch đại $\tfrac{1}{1-\beta} = 10$ lần** so với SGD theo hướng nhất quán đó. Đó chính là cơ chế giúp Momentum băng qua plateau và lướt nhanh dọc đáy khe — đồng thời dập dao động ngang.

---

# 5. AdaGrad: tỉ lệ học riêng cho từng tham số

## 5.1. Động cơ

SGD và Momentum dùng **một** tỉ lệ học $\eta$ cho mọi tham số. Nhưng các tham số rất khác nhau: với đặc trưng thưa (sparse features), một số trọng số hiếm khi nhận gradient khác $0$ và cần bước **lớn** khi có dịp; số khác nhận gradient liên tục và cần bước **nhỏ** để khỏi dao động.

AdaGrad cho mỗi tham số một tỉ lệ học riêng, bằng cách **tích lũy bình phương gradient** trong suốt quá trình:

$$\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t \odot \mathbf{g}_t, \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon}\odot \mathbf{g}_t$$

trong đó $\odot$ là nhân theo từng phần tử (element-wise) và $\epsilon \approx 10^{-8}$ tránh chia cho $0$.

## 5.2. Cơ chế và nhược điểm chí mạng

Diễn giải: tham số nào đã nhận **tổng bình phương gradient lớn** (hướng dốc, hay biến động) sẽ bị chia cho $\sqrt{\mathbf{s}_t}$ lớn → bước **nhỏ lại**; tham số ít được cập nhật giữ được bước lớn. Tỉ lệ học tự cân bằng theo từng chiều.

Nhưng $\mathbf{s}_t$ **chỉ tăng, không bao giờ giảm** (cộng dồn toàn bộ lịch sử). Theo thời gian $\sqrt{\mathbf{s}_t}\to\infty$, nên tỉ lệ học hiệu dụng

$$\frac{\eta}{\sqrt{\mathbf{s}_t}+\epsilon}\;\longrightarrow\;0$$

**tắt dần (vanishing learning rate)**. Với bài toán lồi điều này có thể ổn, nhưng học sâu cần huấn luyện rất lâu — AdaGrad thường **dừng học quá sớm** khi còn chưa tới đáy. RMSProp ra đời để sửa đúng lỗi này.

---

# 6. RMSProp: quên bớt quá khứ

Mấu chốt vấn đề của AdaGrad là **cộng dồn vô hạn**. RMSProp thay tổng tích lũy bằng một **trung bình trượt có suy giảm theo cấp số nhân (exponential moving average)** của bình phương gradient:

$$\mathbf{s}_t = \gamma\,\mathbf{s}_{t-1} + (1-\gamma)\,\mathbf{g}_t \odot \mathbf{g}_t, \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{s}_t}+\epsilon}\odot \mathbf{g}_t$$

với $\gamma$ (thường $0.9$) là hệ số phân rã.

Khác biệt tuy nhỏ mà quyết định: vì có hệ số $\gamma < 1$, các gradient cũ **bị lãng quên dần**. $\mathbf{s}_t$ giờ ước lượng độ lớn **gần đây** của gradient chứ không phải toàn bộ lịch sử, nên nó **không tăng vô hạn** — tỉ lệ học hiệu dụng giữ được ở mức hợp lý suốt quá trình. RMSProp thực chất là một bộ ước lượng kỳ vọng động:

$$\mathbf{s}_t \approx \mathbb{E}\big[\mathbf{g}^2\big]_{\text{gần đây}}$$

Đây là một trong những optimizer thích nghi đầu tiên thực sự dùng tốt cho mạng sâu và mạng hồi tiếp (RNN).

---

# 7. Adam: kết hợp Momentum và RMSProp

Adam (Adaptive Moment Estimation) là sự hợp nhất tự nhiên: lấy **quán tính** của Momentum và **tỉ lệ học thích nghi** của RMSProp. Nó duy trì hai trung bình trượt — mômen bậc nhất (trung bình gradient) và mômen bậc hai (trung bình bình phương gradient):

$$\mathbf{m}_t = \beta_1\,\mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t \qquad (\text{như Momentum})$$

$$\mathbf{v}_t = \beta_2\,\mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t \odot \mathbf{g}_t \qquad (\text{như RMSProp})$$

mặc định $\beta_1 = 0.9,\ \beta_2 = 0.999$.

## 7.1. Vì sao cần hiệu chỉnh độ chệch (bias correction)

Khởi tạo $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$. Ở những bước đầu, các trung bình trượt bị **kéo lệch về $0$** — chúng là ước lượng **chệch (biased)**. Hãy tính kỳ vọng của $\mathbf{m}_t$ giả sử gradient có kỳ vọng ổn định $\mathbb{E}[\mathbf{g}]$. Khai triển đệ quy:

$$\mathbf{m}_t = (1-\beta_1)\sum_{k=1}^{t}\beta_1^{\,t-k}\,\mathbf{g}_k \;\Longrightarrow\; \mathbb{E}[\mathbf{m}_t] = \mathbb{E}[\mathbf{g}]\,(1-\beta_1)\sum_{k=1}^{t}\beta_1^{\,t-k} = \mathbb{E}[\mathbf{g}]\,(1-\beta_1^{\,t})$$

(dùng tổng cấp số nhân $\sum_{k=1}^{t}\beta_1^{t-k} = \tfrac{1-\beta_1^{t}}{1-\beta_1}$). Vậy $\mathbf{m}_t$ nhỏ hơn giá trị thật đúng một thừa số $(1-\beta_1^{\,t})$. Để khử, ta **chia lại**:

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^{\,t}}, \qquad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^{\,t}}$$

Khi $t$ lớn, $\beta^{\,t}\to 0$ nên hiệu chỉnh tan biến; nó chỉ quan trọng ở **giai đoạn đầu**, giúp Adam không bước rón rén lúc khởi động.

## 7.2. Bước cập nhật

$$\boxed{\;\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}\odot \hat{\mathbf{m}}_t\;}$$

Đọc công thức này như một câu: **đi theo hướng trung bình gradient $\hat{\mathbf{m}}_t$ (quán tính), với độ dài bước được chuẩn hóa theo từng chiều bởi $\sqrt{\hat{\mathbf{v}}_t}$ (thích nghi)**. Tử số cho Adam tính bền vững của Momentum; mẫu số cho nó khả năng tự điều chỉnh tỉ lệ học của RMSProp.

```python
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g * g
m_hat = m / (1 - beta1 ** t)            # hiệu chỉnh độ chệch
v_hat = v / (1 - beta2 ** t)
theta -= lr * m_hat / (v_hat ** 0.5 + eps)
```

## 7.3. AdamW — tách rời suy giảm trọng số (decoupled weight decay)

Đây là biến thể được dùng nhiều nhất hiện nay, gần như **mặc định cho Transformer** (xem [Transformer](#/transformer), [LLM](#/llm-pretraining)). Vấn đề nó sửa rất tinh tế.

Cách "ngây thơ" để thêm điều chuẩn $L_2$ là cộng $\lambda \boldsymbol{\theta}$ vào gradient: $\mathbf{g}_t \leftarrow \mathbf{g}_t + \lambda\boldsymbol{\theta}_t$. Nhưng trong Adam, gradient này lại **bị chia cho $\sqrt{\hat{\mathbf{v}}_t}$**. Hậu quả: những tham số có gradient lớn (nên $\hat v$ lớn) bị suy giảm **yếu đi**, còn tham số gradient nhỏ bị suy giảm mạnh — tức lượng phạt méo mó theo từng chiều, **không còn là weight decay đúng nghĩa**. Đây là lý do "L2 trong Adam" tổng quát hóa kém hơn kỳ vọng.

**AdamW** (Loshchilov & Hutter, 2019) **tách** suy giảm trọng số ra khỏi bước thích nghi, áp thẳng lên tham số:

$$\boxed{\;\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda\,\boldsymbol{\theta}_t\right)\;}$$

Số hạng $\lambda\boldsymbol{\theta}_t$ **không** đi qua $\sqrt{\hat{\mathbf{v}}_t}$, nên mọi tham số bị co về 0 với cùng tỉ lệ $\eta\lambda$ — đúng bản chất weight decay (link [Overfitting & Điều chuẩn](#/overfitting)).

```python
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g * g
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
theta -= lr * (m_hat / (v_hat ** 0.5 + eps) + wd * theta)   # wd tách rời
```

> Quy tắc thực hành: với mạng sâu hiện đại, dùng **AdamW** thay vì "Adam + L2". Thường **không** áp weight decay cho bias và tham số của LayerNorm/BatchNorm. Các biến thể mới hơn (Nadam thêm Nesterov, LAMB cho batch cực lớn, Lion dùng dấu gradient) đều xoay quanh cùng hai trục: **quán tính** và **chuẩn hóa bước thích nghi**.

---

# 8. Lịch tỉ lệ học (learning rate schedule)

Một $\eta$ cố định ít khi tối ưu suốt quá trình: lúc đầu cần bước lớn để đi nhanh, về sau cần bước nhỏ để tinh chỉnh quanh cực tiểu. Vì vậy ta cho $\eta_t$ thay đổi theo thời gian.

## 8.1. Warmup

Ngay khi bắt đầu, các ước lượng mômen của Adam còn nhiễu và tham số chưa ổn định; một $\eta$ lớn dễ làm huấn luyện "nổ". **Warmup** tăng $\eta$ tuyến tính từ $0$ tới $\eta_{\max}$ trong $T_w$ bước đầu:

$$\eta_t = \eta_{\max}\cdot\frac{t}{T_w}, \qquad t \le T_w$$

Điều này đặc biệt quan trọng với Transformer, nơi thiếu warmup thường khiến mô hình phân kỳ ngay.

## 8.2. Cosine decay

Sau warmup, ta giảm $\eta$ mượt mà theo hình nửa chu kỳ cosin về gần $0$ ở bước cuối $T$:

$$\eta_t = \eta_{\min} + \frac{1}{2}\big(\eta_{\max}-\eta_{\min}\big)\left(1 + \cos\!\frac{\pi\,(t - T_w)}{T - T_w}\right)$$

Cosin giảm **chậm lúc đầu** (giữ bước lớn để khai phá), **nhanh ở giữa**, rồi lại **chậm lúc cuối** (bước nhỏ dần để hội tụ êm). Sự kết hợp **warmup + cosine decay** là công thức kinh điển cho việc huấn luyện các mô hình lớn ngày nay.

---

# 9. Chọn optimizer nào

| Optimizer | Thích nghi/tham số | Có quán tính | Điểm mạnh | Lưu ý |
| --- | --- | --- | --- | --- |
| SGD | không | không | đơn giản, tổng quát hóa tốt | nhạy với $\eta$, chậm ở khe hẹp |
| SGD + Momentum | không | có | vượt khe hẹp, dập dao động | vẫn cần chỉnh $\eta$ |
| AdaGrad | có | không | tốt cho đặc trưng thưa | tỉ lệ học tắt dần |
| RMSProp | có | không | sửa tắt dần của AdaGrad | thiếu quán tính |
| Adam / AdamW | có | có | mặc định, hội tụ nhanh, ít chỉnh | đôi khi tổng quát hóa kém SGD |

Định hướng thực dụng:

* **Mặc định khởi đầu:** Adam (hoặc AdamW) với $\eta \approx 3\times10^{-4}$, kèm warmup + cosine decay. Hội tụ nhanh, ít nhạy siêu tham số.
* **Khi cần điểm số tổng quát hóa cao nhất** (đặc biệt thị giác máy tính, mạng CNN): **SGD + Momentum** được chỉnh kỹ thường nhỉnh hơn Adam.
* **Transformer / mô hình ngôn ngữ:** AdamW + warmup gần như là chuẩn.
* **Đặc trưng rất thưa:** các phương pháp thích nghi (Adam, AdaGrad) là lựa chọn tự nhiên.

---

# 10. Tổng kết

Mọi optimizer trong bài đều là biến tấu của một khuôn duy nhất: **biến đổi gradient thô thành một bước đi tốt hơn**. Backprop ([Lan truyền ngược](#/backprop)) lo việc tính $\mathbf{g}_t$; optimizer lo việc dùng nó.

Mạch tiến hóa rất gọn:

$$\text{SGD} \xrightarrow{\text{thêm quán tính}} \text{Momentum}, \qquad \text{SGD} \xrightarrow{\text{thích nghi theo chiều}} \text{AdaGrad} \xrightarrow{\text{quên quá khứ}} \text{RMSProp}$$

$$\text{Momentum} + \text{RMSProp} \;\xrightarrow{\text{+ hiệu chỉnh độ chệch}}\; \text{Adam}$$

Mỗi bước trên giải đúng một điểm yếu của bước trước: Momentum chữa sự chậm chạp ở khe hẹp và plateau; AdaGrad mang lại tỉ lệ học theo từng tham số; RMSProp chữa bệnh tắt dần của AdaGrad; và Adam thu cả hai ưu điểm vào một, cộng thêm hiệu chỉnh độ chệch để khởi động trơn tru. Phủ lên trên tất cả là **lịch tỉ lệ học** — warmup để khởi động an toàn, cosine decay để hội tụ êm.

> Hiểu optimizer là hiểu **vì sao** một mô hình hội tụ hay phân kỳ, vì sao learning rate là siêu tham số quan trọng nhất, và vì sao "chỉ cần đổi từ SGD sang AdamW" đôi khi cứu cả một lần huấn luyện. Đây là cây cầu nối giữa lý thuyết gradient và thực hành huấn luyện mạng sâu.

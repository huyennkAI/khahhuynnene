# Ổn định số & Khởi tạo trọng số (Numerical Stability and Initialization)

> Một mạng sâu là một **chuỗi dài** các phép biến đổi. Khi gradient đi ngược qua chuỗi đó, nó bị **nhân liên tiếp** với các ma trận trọng số và đạo hàm kích hoạt. Tích của rất nhiều thừa số có thể **co về 0** (vanishing) hoặc **bùng nổ** (exploding).
>
> $$\frac{\partial L}{\partial \mathbf{h}^{(1)}} \;=\; \mathbf{J}^{(L)} \mathbf{J}^{(L-1)} \cdots \mathbf{J}^{(2)} \cdot \frac{\partial L}{\partial \mathbf{h}^{(L)}}$$
>
> Cách ta **khởi tạo trọng số** quyết định ngay từ đầu chuỗi tích này ổn định hay đổ vỡ. Đây là lý do một mạng "đúng kiến trúc" vẫn có thể không học được gì.

---

# 1. Vì sao độ sâu sinh ra bất ổn

Mạng nông (một, hai tầng) hầu như không gặp vấn đề này. Bất ổn số chỉ lộ ra khi ta **xếp chồng nhiều tầng**, bởi vì gradient của tầng đầu phụ thuộc vào **tích** các đại lượng của mọi tầng sau nó.

Trực giác cốt lõi rất đơn giản: nhân một số với $0.8$ một trăm lần thì ra $\approx 2\times10^{-10}$ (gần như biến mất); nhân với $1.2$ một trăm lần thì ra $\approx 8\times10^{7}$ (bùng nổ). Một mạng 100 tầng chính là một phép nhân lặp như vậy, chỉ khác là thừa số là **ma trận** chứ không phải số.

Để thấy rõ, ta cần viết ra gradient của một mạng sâu dưới dạng tích.

## 1.1. Thiết lập một mạng truyền thẳng (MLP)

Xét mạng $L$ tầng. Ký hiệu $\mathbf{h}^{(0)} = \mathbf{x}$ là đầu vào, và mỗi tầng $l$:

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)}, \qquad \mathbf{h}^{(l)} = \sigma\!\big(\mathbf{z}^{(l)}\big)$$

trong đó $\mathbf{W}^{(l)}$ là ma trận trọng số và $\sigma$ là hàm kích hoạt (activation) áp dụng theo từng phần tử. Đầu ra $\mathbf{h}^{(L)}$ đi vào hàm mất mát $L$ (xem [Hàm mất mát & Tối ưu hóa](#/toi-uu-hoc-sau) để nhắc lại mục tiêu huấn luyện).

Mục tiêu của ta là tính $\dfrac{\partial L}{\partial \mathbf{W}^{(l)}}$ cho mọi $l$, đặc biệt là các tầng **gần đầu vào**.

---

# 2. Gradient là tích các Jacobian

Đây là suy dẫn trung tâm của cả bài. Toàn bộ hiện tượng vanishing/exploding đều chui ra từ một phép nhân chuỗi.

## 2.1. Suy dẫn bằng quy tắc chuỗi

Áp dụng quy tắc chuỗi (chain rule, xem [Lan truyền ngược](#/backprop)) cho đạo hàm của mất mát theo kích hoạt tầng $l-1$, ta phải đi qua mọi tầng trung gian:

$$\frac{\partial L}{\partial \mathbf{h}^{(l-1)}} = \left(\prod_{k=l}^{L} \frac{\partial \mathbf{h}^{(k)}}{\partial \mathbf{h}^{(k-1)}}\right)^{\!\top} \frac{\partial L}{\partial \mathbf{h}^{(L)}}$$

Mỗi thừa số trong tích là một **ma trận Jacobian**. Vì $\mathbf{h}^{(k)} = \sigma(\mathbf{W}^{(k)}\mathbf{h}^{(k-1)})$, ta khai triển từng Jacobian:

$$\frac{\partial \mathbf{h}^{(k)}}{\partial \mathbf{h}^{(k-1)}} = \underbrace{\operatorname{diag}\!\big(\sigma'(\mathbf{z}^{(k)})\big)}_{\text{đạo hàm kích hoạt}} \; \underbrace{\mathbf{W}^{(k)}}_{\text{trọng số}}$$

Ghép lại, gradient theo trọng số tầng $l$ tỉ lệ với một **tích dài** đan xen giữa ma trận trọng số và ma trận đường chéo của đạo hàm kích hoạt:

$$\frac{\partial L}{\partial \mathbf{W}^{(l)}} \;\propto\; \Big[\,\operatorname{diag}\!\big(\sigma'(\mathbf{z}^{(L)})\big)\,\mathbf{W}^{(L)} \cdots \operatorname{diag}\!\big(\sigma'(\mathbf{z}^{(l+1)})\big)\,\mathbf{W}^{(l+1)}\,\Big]^{\!\top}$$

## 2.2. Tích dài thì co hoặc bùng nổ

Để cảm nhận độ lớn, hãy nhìn vào **chuẩn** (norm) của gradient. Theo bất đẳng thức nhân chuẩn:

$$\left\lVert \frac{\partial L}{\partial \mathbf{h}^{(l-1)}} \right\rVert \;\le\; \left(\prod_{k=l}^{L} \big\lVert \operatorname{diag}(\sigma'(\mathbf{z}^{(k)})) \big\rVert \cdot \big\lVert \mathbf{W}^{(k)} \big\rVert\right) \left\lVert \frac{\partial L}{\partial \mathbf{h}^{(L)}} \right\rVert$$

Gọi $\gamma_k = \lVert \operatorname{diag}(\sigma'(\mathbf{z}^{(k)})) \rVert \cdot \lVert \mathbf{W}^{(k)} \rVert$ là "hệ số khuếch đại" của tầng $k$. Khi đó độ lớn gradient bị chi phối bởi $\prod_k \gamma_k$:

- Nếu mọi $\gamma_k < 1$ → tích $\to 0$ theo cấp số nhân: **vanishing gradient**. Tầng gần đầu vào gần như không nhận tín hiệu, không học được.
- Nếu mọi $\gamma_k > 1$ → tích $\to \infty$: **exploding gradient**. Cập nhật nhảy loạn, mất mát thành `NaN`.

Điểm mấu chốt: bất ổn không nằm ở một con số nào, mà nằm ở việc **nhân lặp**. Ta cần giữ $\gamma_k \approx 1$ ở mọi tầng. Có hai cần điều khiển: **đạo hàm kích hoạt** $\sigma'$ và **ma trận trọng số** $\mathbf{W}$. Mục 3 lo cái thứ nhất, mục 5 lo cái thứ hai.

---

# 3. Hàm kích hoạt: sigmoid bão hòa vs ReLU

Thừa số $\sigma'(\mathbf{z})$ trong mỗi Jacobian phụ thuộc trực tiếp vào việc ta chọn hàm kích hoạt nào.

## 3.1. Sigmoid và sự bão hòa

Hàm sigmoid $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ có đạo hàm:

$$\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$

Cực đại của $\sigma'$ chỉ là $0.25$ tại $z=0$. Khi $|z|$ lớn (đầu vào lệch khỏi 0), $\sigma(z) \to 0$ hoặc $\to 1$, nên $\sigma'(z) \to 0$ — ta nói sigmoid **bão hòa (saturate)**.

Hậu quả với tích chuỗi rất nghiêm trọng: ngay cả khi mọi tầng ở vùng tốt nhất, mỗi thừa số đã $\le 0.25$. Qua $L$ tầng:

$$\prod_{k=1}^{L} \sigma'(\mathbf{z}^{(k)}) \;\le\; 0.25^{L} \xrightarrow[L \to \infty]{} 0$$

Với $L=10$ đã là $0.25^{10} \approx 10^{-6}$. Đây chính là lý do lịch sử khiến mạng sâu dùng sigmoid gần như không huấn luyện được — gradient tắt ngấm trước khi về tới tầng đầu.

## 3.2. ReLU đỡ hơn ở đâu

Hàm ReLU $\sigma(z) = \max(0, z)$ có đạo hàm cực kỳ đơn giản:

$$\sigma'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$$

Với những đơn vị **đang hoạt động** ($z>0$), đạo hàm đúng bằng $1$ — không co nhỏ gradient. Tích các thừa số kích hoạt vì thế không tự động suy giảm theo độ sâu như sigmoid; gánh nặng giữ ổn định dồn về phía ma trận trọng số (mục 5). Đây là một trong những lý do chính ReLU thay thế sigmoid trong các mạng sâu hiện đại.

Cái giá là **dying ReLU**: nếu một nơ-ron luôn nhận $z<0$ thì $\sigma'=0$ vĩnh viễn, nó "chết" và không bao giờ cập nhật. Các biến thể như Leaky ReLU ($\sigma'=\alpha$ nhỏ khi $z<0$) ra đời để vá lỗ hổng này.

| Hàm kích hoạt | $\max \sigma'$ | Bão hòa? | Hệ quả với gradient sâu |
| --- | --- | --- | --- |
| Sigmoid | $0.25$ | Có (hai phía) | Vanishing rất mạnh |
| Tanh | $1.0$ | Có (hai phía) | Đỡ hơn sigmoid, vẫn bão hòa |
| ReLU | $1.0$ | Một phía ($z<0$) | Không co khi hoạt động; nguy cơ dying |
| Leaky ReLU | $1.0$ | Không hẳn | Tránh được dying ReLU |

---

# 4. Vì sao khởi tạo trọng số lại quan trọng

Hàm kích hoạt mới chỉ lo nửa phần thừa số. Nửa còn lại — ma trận trọng số $\mathbf{W}$ — do **khởi tạo** quyết định, và nó quan trọng vì hai lý do tách biệt.

## 4.1. Khởi tạo điều khiển độ lớn của tín hiệu

Nhìn lại $\gamma_k = \lVert \sigma' \rVert \cdot \lVert \mathbf{W}^{(k)} \rVert$: nếu khởi tạo $\mathbf{W}$ quá lớn, $\gamma_k > 1$ và gradient bùng nổ; quá nhỏ thì $\gamma_k < 1$ và gradient biến mất. Phương sai của trọng số ở thời điểm khởi tạo trực tiếp đặt điểm xuất phát cho cả forward (độ lớn kích hoạt) lẫn backward (độ lớn gradient). Mục 5 sẽ tìm chính xác phương sai "vừa đúng".

## 4.2. Phá vỡ đối xứng (symmetry breaking)

Có một lý do thứ hai, tinh tế hơn. Giả sử ta khởi tạo **mọi trọng số bằng nhau** (chẳng hạn tất cả bằng 0, hoặc cùng một hằng số). Xét hai nơ-ron $i, j$ trong cùng một tầng ẩn của một [mạng nơ-ron](#/mang-no-ron).

Vì chúng nhận cùng đầu vào và có cùng trọng số ban đầu, đầu ra của chúng **bằng nhau** ở forward. Ở backward, gradient mà chúng nhận cũng **bằng nhau** (do đối xứng hoàn toàn). Bước cập nhật vì thế giống hệt nhau, nên sau cập nhật chúng **vẫn bằng nhau**. Bằng quy nạp, mọi nơ-ron trong một tầng mãi mãi tính cùng một hàm:

$$w_i^{(t)} = w_j^{(t)} \quad \forall t \;\Longrightarrow\; \text{tầng } n \text{ nơ-ron chỉ học được như } 1 \text{ nơ-ron}$$

Đây gọi là **đối xứng (symmetry)**, và ta phải **phá vỡ** nó. Giải pháp duy nhất là khởi tạo trọng số **ngẫu nhiên**, để mỗi nơ-ron bắt đầu khác nhau và đi theo hướng khác nhau. Lưu ý hệ quả: khởi tạo toàn 0 là vô dụng cho trọng số (không phá đối xứng), nhưng **bias** thì khởi tạo 0 lại hoàn toàn ổn, vì trọng số ngẫu nhiên đã đủ phá đối xứng.

---

# 5. Khởi tạo Xavier/Glorot

Giờ ta đi tìm phương sai khởi tạo "vừa đúng" để giữ $\gamma_k \approx 1$. Ý tưởng của Glorot & Bengio (2010): chọn phương sai sao cho **phương sai của tín hiệu được bảo toàn** khi đi qua từng tầng — cả chiều forward lẫn chiều backward.

## 5.1. Suy dẫn cho chiều forward

Xét một nơ-ron $z = \sum_{i=1}^{n_{in}} w_i\, x_i$ với $n_{in}$ đầu vào. Giả thiết: các $w_i$ độc lập cùng phân phối, trung bình 0, phương sai $\operatorname{Var}(w)$; các đầu vào $x_i$ cũng độc lập, trung bình 0, phương sai $\operatorname{Var}(x)$; và $w$ độc lập với $x$. Khi đó:

$$\operatorname{Var}(z) = \sum_{i=1}^{n_{in}} \operatorname{Var}(w_i x_i) = \sum_{i=1}^{n_{in}} \operatorname{Var}(w)\operatorname{Var}(x) = n_{in}\,\operatorname{Var}(w)\,\operatorname{Var}(x)$$

(dùng tính chất: với hai biến độc lập trung bình 0, $\operatorname{Var}(w x) = \operatorname{Var}(w)\operatorname{Var}(x)$). Để **đầu ra cùng độ lớn với đầu vào**, tức $\operatorname{Var}(z) = \operatorname{Var}(x)$, ta cần:

$$n_{in}\,\operatorname{Var}(w) = 1 \;\Longrightarrow\; \operatorname{Var}(w) = \frac{1}{n_{in}}$$

## 5.2. Suy dẫn cho chiều backward

Lập luận hoàn toàn đối xứng cho gradient đi ngược. Gradient tại một nơ-ron là tổng của $n_{out}$ gradient từ tầng sau, mỗi cái nhân với một trọng số. Lặp lại y hệt phép tính phương sai, để **gradient không bị co/giãn** khi đi ngược ta cần:

$$n_{out}\,\operatorname{Var}(w) = 1 \;\Longrightarrow\; \operatorname{Var}(w) = \frac{1}{n_{out}}$$

## 5.3. Dung hòa hai điều kiện

Hai yêu cầu $\operatorname{Var}(w) = 1/n_{in}$ (forward) và $\operatorname{Var}(w) = 1/n_{out}$ (backward) nói chung **không thể thỏa mãn đồng thời** khi $n_{in} \ne n_{out}$. Glorot đề xuất lấy **trung bình điều hòa** — thực chất là dung hòa hai mẫu số:

$$\boxed{\;\operatorname{Var}(w) = \frac{2}{n_{in} + n_{out}}\;}$$

Đây là **khởi tạo Xavier/Glorot**. Trên thực tế ta lấy mẫu trọng số từ một trong hai phân phối có đúng phương sai trên:

- Gauss: $w \sim \mathcal{N}\!\left(0,\, \dfrac{2}{n_{in}+n_{out}}\right)$.
- Đều: $w \sim \mathcal{U}\!\left(-a,\, a\right)$ với $a = \sqrt{\dfrac{6}{n_{in}+n_{out}}}$, vì phân phối đều $\mathcal{U}(-a,a)$ có phương sai $a^2/3$, và $\dfrac{(\sqrt{6/(n_{in}+n_{out})})^2}{3} = \dfrac{2}{n_{in}+n_{out}}$.

---

# 6. He init cho ReLU

Suy dẫn Xavier ngầm giả định kích hoạt **đối xứng quanh 0** (đúng cho tanh ở gần gốc). Nhưng ReLU thì **không**: nó cắt bỏ một nửa âm, nên trung bình một nửa số đơn vị bị đặt về 0.

He et al. (2015) chỉ ra rằng ReLU làm phương sai của tín hiệu đi qua **giảm đi đúng một nửa** (vì $\mathbb{E}[\sigma'(z)] = 1/2$ khi $z$ đối xứng quanh 0). Để bù lại, ta phải **nhân đôi** phương sai trọng số so với điều kiện forward của Xavier:

$$\operatorname{Var}(w) = \frac{2}{n_{in}}$$

Đây là **He initialization** (còn gọi Kaiming init), lựa chọn mặc định cho mọi mạng dùng ReLU và biến thể. Lấy mẫu: $w \sim \mathcal{N}\!\left(0, \dfrac{2}{n_{in}}\right)$.

Cách nhớ gọn: hệ số $2$ ở tử số chính là "đền bù" cho việc ReLU vứt đi một nửa năng lượng tín hiệu.

```python
# PyTorch
import torch.nn as nn
nn.init.xavier_uniform_(layer.weight)              # tanh / sigmoid
nn.init.kaiming_normal_(layer.weight,
                        nonlinearity='relu')        # ReLU
nn.init.zeros_(layer.bias)                          # bias = 0 vẫn ổn
```

---

# 7. Batch Normalization như một giải pháp bổ trợ

Khởi tạo tốt chỉ đảm bảo phương sai cân bằng **ở thời điểm bắt đầu**. Khi huấn luyện chạy, trọng số thay đổi và phân phối kích hoạt ở các tầng sâu lại trôi đi (hiện tượng internal covariate shift), kéo $\gamma_k$ lệch khỏi 1.

**Batch Normalization** vá đúng chỗ này bằng cách **chuẩn hóa lại** kích hoạt ngay trong lúc huấn luyện. Với mỗi lô (batch), nó chuẩn hóa về trung bình 0 phương sai 1 rồi co giãn lại bằng hai tham số học được $\gamma, \beta$:

$$\hat{z} = \frac{z - \mu_{\text{batch}}}{\sqrt{\operatorname{Var}_{\text{batch}}(z) + \varepsilon}}, \qquad \text{output} = \gamma\,\hat{z} + \beta$$

Vì kích hoạt được kéo về phương sai chuẩn ở **mọi bước**, batch-norm chủ động giữ $\gamma_k$ gần 1 thay vì chỉ trông cậy vào khởi tạo. Nó làm mạng bớt nhạy với cách khởi tạo và cho phép dùng learning rate lớn hơn. Chi tiết suy dẫn và cách hoạt động lúc suy luận xem [Batch Normalization](#/batch-norm). Lưu ý: batch-norm **bổ trợ** chứ không thay thế khởi tạo tốt — kết hợp cả hai mới là thực hành chuẩn.

---

# 8. Khuyến nghị thực hành

- **Không bao giờ** khởi tạo trọng số bằng 0 hay bằng hằng số chung — sẽ không phá vỡ đối xứng, cả tầng học như một nơ-ron. Bias = 0 thì chấp nhận được.
- Mạng dùng **ReLU/Leaky ReLU** → dùng **He init** ($\operatorname{Var}(w)=2/n_{in}$).
- Mạng dùng **tanh/sigmoid** → dùng **Xavier/Glorot** ($\operatorname{Var}(w)=2/(n_{in}+n_{out})$).
- Tránh sigmoid/tanh ở **các tầng ẩn sâu** vì bão hòa gây vanishing; ưu tiên ReLU và họ của nó.
- Thêm **Batch Normalization** (hoặc Layer Norm cho Transformer/RNN) để ổn định trong suốt quá trình huấn luyện, không chỉ lúc khởi tạo.
- Nếu gradient vẫn bùng nổ (thường gặp ở RNN), dùng **gradient clipping**: cắt chuẩn gradient về một ngưỡng trần.
- Quan sát triệu chứng: mất mát thành `NaN` → exploding; mất mát đứng yên, tầng đầu không đổi → vanishing.

---

# 9. Tổng kết

Bất ổn số của mạng sâu **không phải** một lỗi lập trình, mà là hệ quả toán học tất yếu của việc **nhân chuỗi** nhiều Jacobian:

$$\frac{\partial L}{\partial \mathbf{h}^{(l-1)}} \propto \prod_{k=l}^{L} \operatorname{diag}\!\big(\sigma'(\mathbf{z}^{(k)})\big)\,\mathbf{W}^{(k)}$$

Tích này co về 0 (vanishing) hay bùng nổ (exploding) tùy theo hai thừa số: **đạo hàm kích hoạt** và **độ lớn trọng số**. Ta điều khiển cái thứ nhất bằng cách chọn **ReLU** thay sigmoid bão hòa, và cái thứ hai bằng **khởi tạo có kiểm soát phương sai**.

Suy dẫn bảo toàn phương sai cho ra hai công thức cốt lõi: **Xavier** $\operatorname{Var}(w)=\frac{2}{n_{in}+n_{out}}$ cho kích hoạt đối xứng, và **He** $\operatorname{Var}(w)=\frac{2}{n_{in}}$ cho ReLU. Cùng với việc khởi tạo **ngẫu nhiên để phá vỡ đối xứng**, và **Batch Normalization** giữ ổn định trong suốt huấn luyện, đây là bộ công cụ cho phép ta huấn luyện những mạng hàng trăm tầng — điều bất khả thi chỉ một thập kỷ trước.

> Bài tiếp theo bàn về **điều chuẩn (regularization)** trong học sâu: khi mạng đã huấn luyện ổn định, làm sao để nó **tổng quát hóa** tốt thay vì học vẹt dữ liệu huấn luyện. Sau đó, [Tối ưu hóa trong học sâu](#/toi-uu-hoc-sau) sẽ xem xét các bộ tối ưu (Momentum, Adam) khai thác cấu trúc gradient để hội tụ nhanh và bền hơn.

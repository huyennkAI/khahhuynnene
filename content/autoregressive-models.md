# Mô hình tự hồi quy (Autoregressive)

> Mô hình tự hồi quy (autoregressive model) sinh dữ liệu **từng phần tử một**, mỗi phần tử dựa trên tất cả những gì đã sinh trước đó. Đây là họ mô hình sinh (generative model) đứng sau GPT, PixelCNN và WaveNet — và là họ duy nhất vừa cho likelihood chính xác vừa mở rộng tới hàng tỉ tham số.

## 1. Trực giác ra đời

Khó khăn trung tâm của mọi mô hình sinh là hằng số chuẩn hóa (normalizing constant) $Z_\theta = \int \tilde{p}_\theta(x)\, dx$ không tính nổi. Câu hỏi đặt ra:

> Liệu có cách phân tích $p(x)$ thành các mảnh, mỗi mảnh **tự chuẩn hóa được**?

Câu trả lời đến từ một sự thật xác suất **luôn đúng** — quy tắc chuỗi (chain rule). Bất kỳ phân phối hợp nào cũng tách được thành tích các xác suất có điều kiện một chiều (one-dimensional conditionals). Mỗi xác suất một chiều thì dễ chuẩn hóa (chỉ cần softmax trên một biến). Bài toán $D$ chiều khổng lồ biến thành $D$ bài toán một chiều.

## 2. Khắc phục điều gì (lập luận lý thuyết)

- **Né hằng số chuẩn hóa $Z_\theta$:** với biến rời rạc, mỗi $p_\theta(x_i \mid x_{<i})$ là softmax trên $K$ giá trị, hằng số chuẩn hóa chỉ là tổng $\sum_{k} e^{f_k}$ — **hữu hạn và tính được**. Tích của các phân phối đã chuẩn hóa vẫn chuẩn hóa, nên $p_\theta(x)$ tự động là mật độ hợp lệ.
- **Cho likelihood chính xác:** không xấp xỉ như VAE (chỉ ELBO), không né tránh như GAN (không có likelihood).
- **Tận dụng phụ thuộc dài (long-range dependency)** khi ghép với kiến trúc mạnh như Transformer.

## 3. Lý thuyết: phân tích theo quy tắc chuỗi

Với dữ liệu $x = (x_1, x_2, \dots, x_D)$, quy tắc chuỗi xác suất cho:

$$p(x) = p(x_1)\, p(x_2 \mid x_1)\, p(x_3 \mid x_1, x_2) \cdots = \prod_{i=1}^{D} p(x_i \mid x_{<i})$$

trong đó ký hiệu $x_{<i} = (x_1, \dots, x_{i-1})$.

### 3.1. Chứng minh quy tắc chuỗi

Xuất phát từ định nghĩa xác suất có điều kiện $p(a, b) = p(a)\, p(b \mid a)$, áp dụng đệ quy:

$$
\begin{aligned}
p(x_1, \dots, x_D)
&= p(x_1)\, p(x_2, \dots, x_D \mid x_1) \\
&= p(x_1)\, p(x_2 \mid x_1)\, p(x_3, \dots, x_D \mid x_1, x_2) \\
&= p(x_1)\, p(x_2 \mid x_1)\, p(x_3 \mid x_1, x_2)\, p(x_4, \dots, x_D \mid x_{\le 3}) \\
&\;\;\vdots \\
&= \prod_{i=1}^{D} p(x_i \mid x_1, \dots, x_{i-1}) \qquad \blacksquare
\end{aligned}
$$

Phân tích này **không cần bất kỳ giả định nào** về dữ liệu — nó luôn đúng. Điểm tinh tế: ta chỉ *mô hình hóa* mỗi điều kiện bằng một mạng nơ-ron có tham số chung $\theta$:

$$p_\theta(x_i \mid x_{<i}) = \text{softmax}\big(f_\theta(x_{<i})\big)$$

### 3.2. Vì sao chuẩn hóa "miễn phí"? (chứng minh)

**Mệnh đề.** Nếu mỗi $p_\theta(x_i \mid x_{<i})$ là phân phối hợp lệ (tích phân/tổng bằng 1), thì $p_\theta(x) = \prod_i p_\theta(x_i \mid x_{<i})$ cũng là phân phối hợp lệ — không cần $Z_\theta$.

**Chứng minh** (trường hợp tổng quát, tích phân lùi từ biến cuối):

$$
\begin{aligned}
\int p_\theta(x)\, dx
&= \int \cdots \int \prod_{i=1}^{D} p_\theta(x_i \mid x_{<i})\, dx_D \cdots dx_1 \\
&= \int \cdots \int \left[\prod_{i=1}^{D-1} p_\theta(x_i \mid x_{<i})\right] \underbrace{\left(\int p_\theta(x_D \mid x_{<D})\, dx_D\right)}_{=\,1}\, dx_{D-1} \cdots dx_1 \\
&= \int \cdots \int \prod_{i=1}^{D-1} p_\theta(x_i \mid x_{<i})\, dx_{D-1} \cdots dx_1
\end{aligned}
$$

Lặp lại tích phân lùi cho $x_{D-1}, \dots, x_1$, mỗi lần một thừa số tích ra 1, cuối cùng còn đúng 1. $\blacksquare$

### 3.3. Hàm mất mát: cực đại likelihood = cross-entropy

Cực đại log-likelihood trên tập dữ liệu tương đương cực tiểu negative log-likelihood:

$$\mathcal{L}(\theta) = -\sum_{i=1}^{D} \log p_\theta(x_i \mid x_{<i})$$

Đây **chính xác** là mục tiêu "dự đoán phần tử kế tiếp (next-token prediction)" của các mô hình ngôn ngữ. Với biến rời rạc, mỗi số hạng là cross-entropy giữa phân phối dự đoán và one-hot của giá trị thật.

```python
logits = model(x[:, :-1])
loss = cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
```

### 3.4. Liên hệ trực tiếp với bits-per-dim

Vì likelihood chính xác, ta đo được chất lượng nén:

$$\text{bpd} = \frac{\mathcal{L}(\theta)}{D \log 2} = \frac{-\sum_i \log_2 p_\theta(x_i \mid x_{<i})}{D}$$

Theo lý thuyết mã hóa của Shannon, đây đúng bằng số bit trung bình tối thiểu để mã hóa mỗi chiều dữ liệu — một dẫn chứng cho việc "mô hình sinh tốt = bộ nén tốt".

## 4. Vấn đề che mặt (masking)

Khi huấn luyện, ta đưa cả chuỗi vào song song để tận dụng phần cứng, nhưng phải đảm bảo vị trí $i$ **không nhìn thấy** $x_{\ge i}$ — nếu không mô hình "gian lận" bằng cách copy đáp án.

- **PixelCNN** — masked convolution: mặt nạ trên kernel chỉ cho nhìn các điểm ảnh đã quét qua.
- **Transformer (GPT)** — causal mask trong self-attention. Cụ thể, ma trận attention bị cộng một mặt nạ tam giác trên bằng $-\infty$ trước softmax:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V, \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

Vì $e^{-\infty} = 0$, mọi trọng số chú ý tới tương lai bị triệt tiêu, đảm bảo tính nhân quả (causality) đúng theo phân tích quy tắc chuỗi.

## 5. Sinh mẫu (sampling)

Lấy mẫu là **tuần tự**: sinh $x_1$, đưa lại vào để sinh $x_2$, cứ thế tới $x_D$.

```text
x1 ~ p(x1)
x2 ~ p(x2 | x1)
x3 ~ p(x3 | x1, x2)
...
xD ~ p(xD | x_<D)
```

Đây là nút thắt cổ chai về tốc độ: độ trễ $O(D)$ lần truyền mạng, không song song hóa được khi suy luận (mỗi bước cần kết quả bước trước).

## 6. Các kiến trúc tham số hóa điều kiện

| Mô hình | Miền | Cách nắm bối cảnh $x_{<i}$ |
| --- | --- | --- |
| **PixelRNN** | Ảnh | LSTM quét điểm ảnh |
| **PixelCNN** | Ảnh | Tích chập có mặt nạ |
| **WaveNet** | Âm thanh | Tích chập giãn (dilated convolution) |
| **GPT** | Văn bản | Self-attention nhân quả |

Lưu ý lý thuyết: chất lượng mô hình autoregressive phụ thuộc **kiến trúc nắm bối cảnh**, không phụ thuộc phân tích quy tắc chuỗi (vốn luôn đúng). Transformer thắng vì self-attention cho đường đi gradient ngắn $O(1)$ giữa hai vị trí bất kỳ, so với $O(D)$ của RNN — đó là lý do nó nắm phụ thuộc dài tốt hơn.

## 7. Trật tự (ordering) — một điểm yếu lý thuyết

Quy tắc chuỗi đúng với **mọi** thứ tự hoán vị của các biến. Nhưng mô hình phải **chọn một thứ tự cố định**. Với văn bản, thứ tự trái-sang-phải là tự nhiên. Với ảnh, không có thứ tự "đúng" nào — buộc phải áp đặt (ví dụ quét raster trên-xuống, trái-phải), và lựa chọn này ảnh hưởng chất lượng. Đây là một thiên lệch quy nạp (inductive bias) nhân tạo mà các họ khác (như diffusion) không mắc phải.

## 8. Ưu điểm (có dẫn chứng lý thuyết)

- **Likelihood chính xác** — từ mục 3.2, không cần xấp xỉ; cho phép so sánh mô hình minh bạch qua bits-per-dim.
- **Huấn luyện ổn định** — mục tiêu là cross-entropy lồi theo logits, không có động lực đối kháng (adversarial dynamics) như GAN; gradient có ý nghĩa ở mọi bước.
- **Mở rộng tuyệt vời** — định luật tỉ lệ (scaling laws) cho thấy loss giảm đều theo lũy thừa của số tham số và dữ liệu; đây là nền của các mô hình ngôn ngữ lớn (LLM).

## 9. Nhược điểm (có dẫn chứng lý thuyết)

- **Lấy mẫu chậm $O(D)$** — từ mục 5, sinh tuần tự không song song hóa được; với ảnh độ phân giải cao $D$ rất lớn.
- **Áp đặt thứ tự nhân tạo** — từ mục 7, gây thiên lệch với dữ liệu vốn không có thứ tự tự nhiên.
- **Sai lệch phơi bày (exposure bias)** — khi huấn luyện, điều kiện $x_{<i}$ là dữ liệu thật (teacher forcing); khi sinh, điều kiện là mẫu do chính mô hình tạo ra. Phân phối điều kiện lúc huấn luyện và lúc suy luận lệch nhau, lỗi tích lũy dọc chuỗi.

## 10. Vị trí trong bức tranh chung

Autoregressive đổi **tốc độ lấy mẫu** lấy **sự đơn giản và likelihood chính xác**. Với văn bản — vốn tuần tự — đây là lựa chọn tự nhiên và đang thống trị (GPT). Với ảnh, nó nhường ngôi cho diffusion vì điểm yếu tốc độ và thứ tự.

> Bài tiếp theo, **Normalizing Flows**, giữ likelihood chính xác như autoregressive nhưng lấy mẫu **một bước** — đánh đổi lại bằng ràng buộc khả nghịch.

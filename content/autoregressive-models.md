# Mô hình tự hồi quy (Autoregressive Models)

> Mô hình tự hồi quy (autoregressive model) là họ mô hình sinh dữ liệu bằng cách dự đoán **từng phần tử kế tiếp** từ các phần tử đã xuất hiện trước đó. Thay vì sinh toàn bộ dữ liệu trong một lần, mô hình xây dựng mẫu **từng bước một**:
>
> $$x_1 \rightarrow x_2 \rightarrow x_3 \rightarrow \cdots \rightarrow x_D$$
>
> Đây là nền tảng của các mô hình ngôn ngữ lớn như GPT, các mô hình sinh ảnh như PixelCNN, và các mô hình sinh âm thanh như WaveNet.

---

# 1. Động cơ ra đời

Mục tiêu của mô hình sinh (generative model) là học phân phối xác suất của dữ liệu:

$$p(x)$$

Nếu học được phân phối này, ta có thể:

* sinh dữ liệu mới,
* đánh giá xác suất của một mẫu,
* nén dữ liệu,
* hoặc suy luận về cấu trúc của dữ liệu.

Tuy nhiên, dữ liệu thực tế thường có **số chiều rất lớn**. Ví dụ:

* một câu văn có thể chứa hàng trăm tới hàng nghìn token,
* một ảnh RGB $256 \times 256$ chứa gần $200{,}000$ giá trị pixel,
* một đoạn âm thanh vài giây chứa hàng chục nghìn mẫu tín hiệu.

Nếu cố gắng mô hình hóa **trực tiếp** $p(x)$, ta thường vấp phải bài toán chuẩn hóa (normalization):

$$p_\theta(x) = \frac{\tilde{p}_\theta(x)}{Z_\theta}, \qquad Z_\theta = \int \tilde{p}_\theta(x)\, dx$$

Với dữ liệu nhiều chiều, hằng số chuẩn hóa (normalizing constant) $Z_\theta$ là một tích phân khổng lồ **không thể tính được**.

Do đó, câu hỏi trung tâm là:

> Liệu có thể biến bài toán mô hình hóa một phân phối nhiều chiều thành **nhiều bài toán đơn giản hơn** mà vẫn giữ nguyên ý nghĩa xác suất?

Câu trả lời đến từ một định lý cơ bản của xác suất: **quy tắc chuỗi (chain rule)**.

---

# 2. Ý tưởng cốt lõi: phân rã bằng quy tắc chuỗi

Một sự thật quan trọng là **bất kỳ phân phối hợp (joint distribution) nào cũng phân rã được thành tích các xác suất có điều kiện**.

Với $x = (x_1, x_2, \ldots, x_D)$, ta luôn có:

$$p(x) = \prod_{i=1}^{D} p(x_i \mid x_{<i})$$

trong đó ký hiệu

$$x_{<i} = (x_1, \ldots, x_{i-1})$$

Viết tường minh hơn:

$$p(x) = p(x_1)\, p(x_2 \mid x_1)\, p(x_3 \mid x_1, x_2) \cdots p(x_D \mid x_{<D})$$

Đây **không phải** giả định hay xấp xỉ — mà là một đẳng thức **luôn đúng** với mọi phân phối xác suất.

Điều đó có nghĩa: thay vì học trực tiếp $p(x)$, ta chỉ cần học các phân phối điều kiện $p(x_i \mid x_{<i})$. Nói đơn giản hơn:

> Học cách **dự đoán phần tử tiếp theo** từ tất cả những gì đã xuất hiện trước đó.

Đây chính là tư tưởng cốt lõi đứng sau GPT.

---

## 2.1. Chứng minh quy tắc chuỗi

Xuất phát từ định nghĩa xác suất có điều kiện $p(a, b) = p(a)\, p(b \mid a)$, áp dụng đệ quy:

$$
\begin{aligned}
p(x_1, \ldots, x_D)
&= p(x_1)\, p(x_2, \ldots, x_D \mid x_1) \\
&= p(x_1)\, p(x_2 \mid x_1)\, p(x_3, \ldots, x_D \mid x_1, x_2) \\
&= p(x_1)\, p(x_2 \mid x_1)\, p(x_3 \mid x_1, x_2)\, p(x_4, \ldots, x_D \mid x_{\le 3}) \\
&\;\;\vdots \\
&= \prod_{i=1}^{D} p(x_i \mid x_{<i}) \qquad \blacksquare
\end{aligned}
$$

Phép phân rã này đúng với **mọi dữ liệu** và **mọi thứ tự** các biến. Điểm duy nhất cần học là các xác suất có điều kiện.

---

# 3. Mô hình thực sự học cái gì?

Đến đây ta mới chỉ có một công thức xác suất. Để biến nó thành một mô hình học máy, ta cần một cách **biểu diễn** các xác suất $p(x_i \mid x_{<i})$.

Trong thực tế, ta dùng một mạng nơ-ron (neural network) với tham số $\theta$. Mạng nhận vào toàn bộ lịch sử $x_{<i}$ và xuất ra một vector logits:

$$h_i = f_\theta(x_{<i})$$

Sau đó áp dụng softmax để biến logits thành phân phối xác suất trên từ vựng:

$$p_\theta(x_i \mid x_{<i}) = \text{softmax}(W h_i)$$

Ví dụ, nếu từ vựng gồm bốn từ, mô hình có thể xuất ra:

| Token | Xác suất |
| --- | --- |
| trí | 0.72 |
| toán | 0.11 |
| học | 0.09 |
| máy | 0.08 |

tức là mô hình tin rằng "trí" là token tiếp theo có khả năng cao nhất.

Toàn bộ GPT, về bản chất, chỉ là một hàm $f_\theta$ rất lớn dùng để ước lượng các xác suất này.

---

# 4. Ví dụ trực quan với văn bản

Giả sử câu:

> Tôi thích học trí tuệ nhân tạo

Sau khi tách token (tokenize):

```text
[Tôi] [thích] [học] [trí] [tuệ] [nhân] [tạo]
```

Theo quy tắc chuỗi:

$$
\begin{aligned}
p(x) = \; & p(\text{Tôi}) \\
\cdot\; & p(\text{thích} \mid \text{Tôi}) \\
\cdot\; & p(\text{học} \mid \text{Tôi thích}) \\
\cdot\; & p(\text{trí} \mid \text{Tôi thích học}) \\
\cdot\; & \cdots
\end{aligned}
$$

Khi nhìn thấy `Tôi thích học`, mô hình phải dự đoán `trí`. Sau đó với `Tôi thích học trí`, nó dự đoán `tuệ`, và cứ tiếp tục như vậy.

---

# 5. Vì sao chuẩn hóa được "miễn phí"?

Một ưu điểm lớn của autoregressive model là **không cần hằng số chuẩn hóa toàn cục** $Z_\theta$.

**Mệnh đề.** Nếu mỗi điều kiện $p_\theta(x_i \mid x_{<i})$ đã là phân phối hợp lệ (tổng/tích phân bằng 1):

$$\sum_{x_i} p_\theta(x_i \mid x_{<i}) = 1 \quad \text{(hoặc } \textstyle\int p_\theta(x_i \mid x_{<i})\, dx_i = 1\text{)}$$

thì tích của chúng, $p_\theta(x) = \prod_i p_\theta(x_i \mid x_{<i})$, **cũng là phân phối hợp lệ** trên toàn không gian — không cần $Z_\theta$.

**Chứng minh** (trường hợp liên tục, tích phân lùi từ biến cuối):

$$
\begin{aligned}
\int p_\theta(x)\, dx
&= \int \cdots \int \prod_{i=1}^{D} p_\theta(x_i \mid x_{<i})\, dx_D \cdots dx_1 \\
&= \int \cdots \int \left[\prod_{i=1}^{D-1} p_\theta(x_i \mid x_{<i})\right] \underbrace{\left(\int p_\theta(x_D \mid x_{<D})\, dx_D\right)}_{=\,1}\, dx_{D-1} \cdots dx_1 \\
&= \int \cdots \int \prod_{i=1}^{D-1} p_\theta(x_i \mid x_{<i})\, dx_{D-1} \cdots dx_1
\end{aligned}
$$

Lặp lại tích phân lùi cho $x_{D-1}, \ldots, x_1$ — mỗi lần một thừa số tích ra $1$ — cuối cùng còn đúng $1$. $\blacksquare$

Do đó:

> Chỉ cần **từng bước dự đoán** được chuẩn hóa, **toàn bộ mô hình tự động** được chuẩn hóa.

Đây là lý do autoregressive cho **likelihood chính xác** mà không phải xấp xỉ như VAE (chỉ có ELBO) hay né tránh như GAN (không có likelihood).

---

# 6. Huấn luyện diễn ra như thế nào?

Sau khi đã có cách biểu diễn các xác suất điều kiện, ta cần tìm tham số $\theta$ sao cho dự đoán ngày càng gần dữ liệu thật.

Likelihood của một chuỗi là:

$$p_\theta(x) = \prod_i p_\theta(x_i \mid x_{<i})$$

Lấy logarit để biến tích thành tổng (ổn định số học và dễ tối ưu):

$$\log p_\theta(x) = \sum_i \log p_\theta(x_i \mid x_{<i})$$

Vì vậy cực đại log-likelihood tương đương cực tiểu **negative log-likelihood (NLL)**:

$$\mathcal{L}(\theta) = -\sum_i \log p_\theta(x_i \mid x_{<i})$$

Với dữ liệu rời rạc, mỗi số hạng NLL chính là **cross-entropy** giữa phân phối dự đoán và one-hot của token thật. Đây **chính xác** là mục tiêu "dự đoán token kế tiếp (next-token prediction)" của các mô hình ngôn ngữ:

```python
logits = model(x[:, :-1])
loss = cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
```

---

## 6.1. Chuẩn bị dữ liệu

Giả sử câu `Tôi thích học AI`. Ta dịch chuyển chuỗi một vị trí để tạo cặp input – target:

Input:

```text
[Tôi] [thích] [học] [AI]
```

Target:

```text
[thích] [học] [AI] [EOS]
```

Mô hình phải học mọi vị trí cùng lúc:

| Ngữ cảnh | Token đúng |
| --- | --- |
| Tôi | thích |
| Tôi thích | học |
| Tôi thích học | AI |
| Tôi thích học AI | EOS |

Như vậy **một câu duy nhất** tạo ra **nhiều ví dụ huấn luyện**.

---

## 6.2. Teacher Forcing

Một câu hỏi tự nhiên:

> Khi dự đoán token thứ ba, mô hình có dùng các token mà chính nó sinh ra không?

Câu trả lời là **không**. Trong huấn luyện, mô hình luôn được cung cấp **token thật** từ dữ liệu. Khi dự đoán `AI`, mô hình được thấy ngữ cảnh thật `Tôi thích học`, chứ không phải các token nó tự sinh.

Kỹ thuật này gọi là **teacher forcing**. Nó giúp tối ưu ổn định hơn rất nhiều vì ngữ cảnh luôn đúng — nhưng cũng chính là nguồn gốc của *exposure bias* (mục 9).

---

# 7. Là mô hình tuần tự, tại sao GPT huấn luyện song song được?

Thoạt nhìn autoregressive có vẻ bắt buộc xử lý tuần tự `x1 → x2 → x3 → x4`. Tuy nhiên, **trong huấn luyện toàn bộ chuỗi đã có sẵn** — ta không cần sinh token mới, chỉ cần dự đoán token kế tiếp ở **mọi vị trí cùng một lúc**.

Cả câu `Tôi thích học AI` được đưa vào trong **một lần truyền xuôi (forward pass)**. Vấn đề còn lại là phải **ngăn mô hình nhìn thấy tương lai** — nếu không, vị trí $i$ sẽ "gian lận" bằng cách copy đáp án $x_{\ge i}$.

---

## 7.1. Causal Mask

GPT dùng một **mặt nạ nhân quả (causal mask)** — ma trận tam giác trên bằng $-\infty$:

$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

cộng vào điểm attention **trước** softmax:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

Vì $e^{-\infty} = 0$, mọi trọng số chú ý tới tương lai bị triệt tiêu. Nhờ đó:

* token 1 chỉ thấy token 1,
* token 2 chỉ thấy token 1–2,
* token 3 chỉ thấy token 1–3,
* ...

Toàn bộ chuỗi vẫn được xử lý trong một forward pass, đảm bảo đúng tính nhân quả (causality) theo phân rã quy tắc chuỗi. Đây là lý do Transformer huấn luyện rất hiệu quả trên GPU.

---

# 8. Sinh dữ liệu sau khi huấn luyện

Sau khi học xong, mô hình đã biết ước lượng $p(x_i \mid x_{<i})$. Khác với huấn luyện, lúc sinh ta **không còn biết token tương lai** — mô hình phải dùng chính các dự đoán của mình.

Ví dụ với prompt `Tôi thích`, mô hình dự đoán:

| Token | Xác suất |
| --- | --- |
| học | 0.65 |
| ăn | 0.20 |
| chơi | 0.15 |

Lấy mẫu được `học`, chuỗi trở thành `Tôi thích học`, rồi tiếp tục dự đoán token kế tiếp. Quá trình lặp lại cho tới khi gặp token kết thúc (EOS).

Mô tả tuần tự:

```text
x1 ~ p(x1)
x2 ~ p(x2 | x1)
x3 ~ p(x3 | x1, x2)
...
xD ~ p(xD | x_<D)
```

Đây là **nút thắt cổ chai về tốc độ**: độ trễ $O(D)$ lần truyền mạng, không song song hóa được khi suy luận vì mỗi bước cần kết quả của bước trước.

---

# 9. Vấn đề Exposure Bias

Một hệ quả quan trọng nảy sinh từ sự khác biệt giữa huấn luyện và suy luận:

* **Huấn luyện** — đầu vào là token **thật** (teacher forcing).
* **Suy luận** — đầu vào là token **do chính mô hình sinh ra**.

Nếu mô hình mắc một lỗi nhỏ ở bước đầu, lỗi đó đi vào ngữ cảnh của bước sau, khiến dự đoán càng lệch:

```text
Sai → Sai hơn → Sai rất nhiều
```

Phân phối điều kiện lúc huấn luyện và lúc suy luận lệch nhau, lỗi **tích lũy dọc chuỗi**. Hiện tượng này gọi là **exposure bias** — một hạn chế lý thuyết quan trọng của autoregressive learning.

---

# 10. Các kiến trúc autoregressive nổi bật

Đến đây, toàn bộ lập luận chỉ dựa trên xác suất; ta chưa nói gì về kiến trúc cụ thể. Điều duy nhất mô hình cần là một cơ chế đủ mạnh để **biểu diễn ngữ cảnh** $x_{<i}$.

| Mô hình | Dữ liệu | Cơ chế nắm bối cảnh |
| --- | --- | --- |
| **PixelRNN** | Ảnh | LSTM quét điểm ảnh |
| **PixelCNN** | Ảnh | Tích chập có mặt nạ (masked convolution) |
| **WaveNet** | Âm thanh | Tích chập giãn (dilated convolution) |
| **GPT** | Văn bản | Self-attention nhân quả |

Điểm chung của tất cả là cùng một phân rã:

$$p(x) = \prod_i p(x_i \mid x_{<i})$$

Khác biệt **duy nhất** là cách biểu diễn lịch sử. Transformer thắng thế vì self-attention cho đường đi gradient ngắn $O(1)$ giữa hai vị trí bất kỳ (so với $O(D)$ của RNN), nên nắm phụ thuộc dài (long-range dependency) tốt hơn hẳn.

---

# 11. Ưu điểm

* **Likelihood chính xác** — từ mục 5, không cần xấp xỉ; cho phép so sánh mô hình minh bạch qua bits-per-dim.
* **Không cần xấp xỉ** như ELBO của VAE, cũng không có **động lực đối kháng (adversarial dynamics)** như GAN.
* **Huấn luyện đơn giản và ổn định** bằng cross-entropy — mục tiêu lồi theo logits, gradient có ý nghĩa ở mọi bước.
* **Mở rộng tuyệt vời** — định luật tỉ lệ (scaling laws) cho thấy loss giảm đều theo lũy thừa của số tham số và dữ liệu; đây là nền của các mô hình ngôn ngữ lớn (LLM).
* Đặc biệt hiệu quả trên **dữ liệu tuần tự** như văn bản — lý do toàn bộ họ GPT đều là autoregressive.

---

# 12. Nhược điểm

Chính lựa chọn sinh tuần tự cũng dẫn tới những hạn chế cơ bản.

**Sinh mẫu chậm $O(D)$.** Muốn sinh $x_i$ phải biết trước $x_{<i}$, nên suy luận không song song hóa được; với ảnh độ phân giải cao, $D$ rất lớn.

**Exposure bias.** Huấn luyện và suy luận dùng hai loại ngữ cảnh khác nhau (mục 9), gây tích lũy lỗi.

**Phụ thuộc thứ tự.** Quy tắc chuỗi đúng với mọi hoán vị, nhưng mô hình buộc phải **chọn một thứ tự cố định**. Với văn bản, trái-sang-phải là tự nhiên; với ảnh thì không tồn tại thứ tự "đúng" — việc chọn cách quét pixel (ví dụ raster trên-xuống, trái-phải) là một **thiên lệch quy nạp (inductive bias)** nhân tạo mà các họ khác như diffusion không mắc phải.

---

# 13. Tổng kết

Nhìn từ góc độ xác suất, autoregressive model **không phát minh ra một dạng phân phối mới**. Ý tưởng cốt lõi chỉ là tận dụng quy tắc chuỗi:

$$p(x) = \prod_i p(x_i \mid x_{<i})$$

để biến bài toán mô hình hóa một phân phối nhiều chiều thành bài toán **dự đoán phần tử tiếp theo**. Từ một nguyên lý xác suất rất đơn giản, kết hợp với các kiến trúc biểu diễn mạnh như Transformer, autoregressive đã trở thành nền tảng của hầu hết các mô hình ngôn ngữ hiện đại.

Autoregressive đổi **tốc độ lấy mẫu** lấy **sự đơn giản và likelihood chính xác**. Với văn bản — vốn tuần tự — đây là lựa chọn tự nhiên và đang thống trị (GPT). Với ảnh, nó nhường ngôi cho diffusion vì điểm yếu tốc độ và thứ tự.

> Bài tiếp theo, **Normalizing Flows**, vẫn giữ likelihood chính xác như autoregressive nhưng lấy mẫu **một bước** — đánh đổi lại bằng ràng buộc khả nghịch (invertibility) của phép biến đổi.

# Mô hình tự hồi quy (Autoregressive)

> Mô hình tự hồi quy sinh dữ liệu **từng phần tử một**, mỗi phần tử dựa trên tất cả những gì đã sinh trước đó. Đây là họ mô hình sinh đứng sau GPT, PixelCNN và WaveNet.

## Trực giác ra đời

Mọi mô hình sinh đều vướng hằng số chuẩn hóa $Z_\theta = \int \tilde{p}_\theta(x)\,dx$ không tính nổi. Câu hỏi: **liệu có cách phân tích $p(x)$ thành các mảnh, mỗi mảnh tự chuẩn hóa được không?**

Câu trả lời đến từ một sự thật xác suất luôn đúng — **quy tắc chuỗi**. Bất kỳ phân phối hợp nào cũng tách được thành tích các xác suất có điều kiện một chiều. Mỗi xác suất một chiều thì dễ chuẩn hóa (chỉ cần softmax trên một biến). Bài toán nhiều chiều khổng lồ biến thành chuỗi bài toán một chiều.

## Khắc phục điều gì

- **Né hằng số chuẩn hóa $Z_\theta$:** mỗi thừa số là một phân phối một chiều, chuẩn hóa tầm thường bằng softmax.
- **Cho likelihood chính xác:** không cần xấp xỉ như VAE hay né tránh như GAN.
- **Tận dụng được phụ thuộc dài** khi kết hợp với kiến trúc mạnh như Transformer.

## Lý thuyết: phân tích theo quy tắc chuỗi

Với dữ liệu $x = (x_1, x_2, \dots, x_n)$, quy tắc chuỗi xác suất cho:

$$p(x) = p(x_1)\, p(x_2 \mid x_1)\, p(x_3 \mid x_1, x_2) \cdots = \prod_{i=1}^{n} p(x_i \mid x_{<i})$$

### Chứng minh

Quy tắc chuỗi suy ra trực tiếp từ định nghĩa xác suất có điều kiện $p(a, b) = p(a)\,p(b \mid a)$. Áp dụng đệ quy:

$$
\begin{aligned}
p(x_1, \dots, x_n) &= p(x_1)\, p(x_2, \dots, x_n \mid x_1) \\
&= p(x_1)\, p(x_2 \mid x_1)\, p(x_3, \dots, x_n \mid x_1, x_2) \\
&= \prod_{i=1}^{n} p(x_i \mid x_1, \dots, x_{i-1})
\end{aligned}
$$

Phân tích này **luôn đúng, không cần giả định gì**. Mô hình hóa bằng cách cho một mạng nơ-ron $f_\theta$ dự đoán phân phối của $x_i$ từ bối cảnh $x_{<i}$:

$$p_\theta(x_i \mid x_{<i}) = \text{softmax}\big(f_\theta(x_{<i})\big)$$

### Hàm mất mát

Cực đại hóa log-likelihood tương đương cực tiểu hóa cross-entropy dự đoán phần tử kế tiếp:

$$\mathcal{L}(\theta) = -\sum_{i=1}^{n} \log p_\theta(x_i \mid x_{<i})$$

Đây chính xác là mục tiêu "dự đoán token kế tiếp" của các mô hình ngôn ngữ.

```python
logits = model(x[:, :-1])
loss = cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
```

## Vấn đề che mặt (masking)

Khi huấn luyện, ta cho cả chuỗi vào song song nhưng phải đảm bảo vị trí $i$ **không nhìn thấy** $x_{\ge i}$, nếu không mô hình "gian lận". Giải pháp:

- **PixelCNN** — masked convolution chỉ nhìn các điểm ảnh đã quét qua.
- **Transformer (GPT)** — causal mask trong self-attention, chặn attention tới tương lai.

## Sinh mẫu

Lấy mẫu là **tuần tự**: sinh $x_1$, đưa lại vào để sinh $x_2$, cứ thế đến $x_n$.

```text
x1 ~ p(x1)
x2 ~ p(x2 | x1)
x3 ~ p(x3 | x1, x2)
...
```

Đây cũng chính là nút thắt cổ chai về tốc độ.

## Các đại diện tiêu biểu

| Mô hình | Miền | Đơn vị sinh |
| --- | --- | --- |
| **PixelRNN / PixelCNN** | Ảnh | từng điểm ảnh |
| **WaveNet** | Âm thanh | từng mẫu sóng |
| **GPT** | Văn bản | từng token |

## Ưu điểm

- **Likelihood chính xác** — đánh giá và so sánh mô hình minh bạch.
- **Huấn luyện ổn định** — chỉ là cực tiểu cross-entropy, không có động lực đối kháng.
- **Chất lượng cao** và khả năng mở rộng tuyệt vời (chính là nền của LLM).

## Nhược điểm

- **Lấy mẫu chậm:** sinh tuần tự, độ trễ tỉ lệ với độ dài $O(n)$, không song song hóa được khi suy luận.
- **Áp đặt thứ tự nhân tạo:** ảnh không có thứ tự "trái-phải, trên-dưới" tự nhiên, nhưng buộc phải chọn một.
- **Bối cảnh hữu hạn:** chất lượng phụ thuộc vào khả năng nắm phụ thuộc dài của kiến trúc nền.

> Autoregressive đổi tốc độ lấy mẫu lấy sự đơn giản và likelihood chính xác. Với văn bản — vốn vốn dĩ tuần tự — đây là lựa chọn tự nhiên và đang thống trị. Bài tiếp theo, **Normalizing Flows**, giữ likelihood chính xác nhưng lấy mẫu một bước.

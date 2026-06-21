# Mạng nơ-ron nhân tạo (Multilayer Perceptron)

> Mô hình tuyến tính chỉ vẽ được những **ranh giới phẳng**, nên có những bài toán đơn giản như XOR mà nó vĩnh viễn không giải được. Mạng nơ-ron nhiều tầng (Multilayer Perceptron - MLP) khắc phục điều này bằng một ý tưởng duy nhất: chèn thêm **tầng ẩn (hidden layer)** kèm một **hàm phi tuyến (nonlinearity)** vào giữa.
>
> $$\mathbf{H} = \sigma(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \qquad \mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$
>
> Đây là viên gạch nền của toàn bộ học sâu (deep learning).

---

# 1. Vì sao mô hình tuyến tính không đủ

Trước khi thêm bất cứ thứ gì phức tạp, ta cần hiểu rõ giới hạn của những gì đã có.

## 1.1. Giả định tuyến tính quá mạnh

Một mô hình tuyến tính như [perceptron](#/perceptron) hay [hồi quy logistic](#/logistic-regression) đưa ra dự đoán dựa trên một tổ hợp tuyến tính của đặc trưng:

$$o = \mathbf{w}^\top \mathbf{x} + b$$

Ngầm bên trong là một giả định rất mạnh: **tăng một đặc trưng luôn kéo đầu ra theo cùng một hướng**, với cường độ cố định là trọng số tương ứng. Trong nhiều bài toán thực tế, giả định này sai. Ví dụ, khả năng được duyệt vay tăng theo thu nhập — nhưng tăng nhanh ở mức thu nhập thấp và chững lại ở mức cao; quan hệ không hề tuyến tính.

Hệ quả hình học còn nghiêm trọng hơn: với phân loại nhị phân, một mô hình tuyến tính chỉ có thể tách hai lớp bằng một **siêu phẳng (hyperplane)**. Nếu dữ liệu không **tách được tuyến tính (linearly separable)**, mô hình bó tay.

## 1.2. Phản ví dụ kinh điển: XOR

Hãy xét hàm XOR trên hai biến nhị phân. Đầu ra là $1$ khi đúng một trong hai đầu vào bằng $1$:

| $x_1$ | $x_2$ | XOR |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Vẽ bốn điểm này lên mặt phẳng: hai điểm lớp $1$ nằm ở hai góc đối nhau, hai điểm lớp $0$ ở hai góc còn lại. **Không tồn tại đường thẳng nào** chia mặt phẳng sao cho mỗi nửa chỉ chứa một lớp. Đây chính là lý do perceptron một tầng thất bại với XOR — một sự thật từng làm khựng lại cả ngành vào cuối thập niên 1960.

Trực giác lối thoát: nếu ta **biến đổi dữ liệu** sang một biểu diễn mới mà ở đó hai lớp lại tách được tuyến tính, thì mọi chuyện trở nên dễ. Câu hỏi là: ai học ra phép biến đổi đó? Câu trả lời của MLP là **để chính mạng tự học**, qua một tầng ẩn.

---

# 2. Thêm tầng ẩn

Ý tưởng cốt lõi của MLP: xếp chồng nhiều **tầng kết nối đầy đủ (fully-connected layer)**, đan xen với hàm phi tuyến, để mạng tự khám phá biểu diễn hữu ích.

## 2.1. Cấu trúc một tầng ẩn

Xét một mạng có một tầng ẩn. Đầu vào là một lô (mini-batch) gồm $n$ mẫu, mỗi mẫu có $d$ đặc trưng, gom thành ma trận $\mathbf{X} \in \mathbb{R}^{n \times d}$. Tầng ẩn có $h$ đơn vị (hidden unit).

- Trọng số và độ chệch của tầng ẩn: $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$, $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$.
- Trọng số và độ chệch của tầng ra (với $q$ đầu ra): $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$, $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$.

Ta tính **biểu diễn ẩn (hidden representation)** $\mathbf{H} \in \mathbb{R}^{n \times h}$ rồi tới đầu ra $\mathbf{O} \in \mathbb{R}^{n \times q}$:

$$
\begin{aligned}
\mathbf{H} &= \sigma\!\left(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}\right) \\
\mathbf{O} &= \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}
\end{aligned}
$$

trong đó $\sigma$ là **hàm kích hoạt (activation function)** áp dụng theo từng phần tử (element-wise). Độ chệch được cộng theo cơ chế lan truyền (broadcasting) lên mọi hàng. Mỗi đơn vị ẩn nhìn thấy toàn bộ đầu vào và học một đặc trưng phi tuyến riêng; tầng ra dùng các đặc trưng này như đầu vào mới.

## 2.2. Vì sao BẮT BUỘC phải có phi tuyến

Một câu hỏi tự nhiên:

> Nếu chỉ xếp chồng các tầng tuyến tính mà bỏ $\sigma$ đi thì sao?

**Mệnh đề.** Hợp của hai phép biến đổi affine vẫn là một phép affine. Do đó một MLP không có hàm kích hoạt, dù sâu đến đâu, cũng chỉ tương đương một mô hình tuyến tính duy nhất.

**Chứng minh.** Bỏ $\sigma$, ta có $\mathbf{H} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$. Thế vào tầng ra:

$$
\begin{aligned}
\mathbf{O} &= \left(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}\right)\mathbf{W}^{(2)} + \mathbf{b}^{(2)} \\
&= \mathbf{X}\underbrace{\mathbf{W}^{(1)}\mathbf{W}^{(2)}}_{=\;\mathbf{W}} + \underbrace{\mathbf{b}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}}_{=\;\mathbf{b}} = \mathbf{X}\mathbf{W} + \mathbf{b}
\end{aligned}
$$

Đặt $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)} \in \mathbb{R}^{d \times q}$ và $\mathbf{b} = \mathbf{b}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$, ta thu lại đúng một tầng tuyến tính. Quy nạp cho thấy điều này đúng với số tầng tùy ý. $\blacksquare$

Kết luận: **phi tuyến là thứ duy nhất khiến chiều sâu có giá trị**. Chỉ khi chèn một $\sigma$ phi tuyến giữa hai tầng affine thì mạng mới biểu diễn được các hàm vượt ngoài lớp tuyến tính — và lúc đó nó giải được XOR.

---

# 3. Hàm kích hoạt

Hàm kích hoạt quyết định một đơn vị có "kích hoạt" hay không và biến tín hiệu đầu vào thành đầu ra. Chúng phải khả vi (gần như khắp nơi) để [lan truyền ngược](#/backprop) hoạt động được.

## 3.1. ReLU — lựa chọn mặc định

Hàm **ReLU (Rectified Linear Unit)** đơn giản nhưng hiệu quả bậc nhất:

$$\operatorname{ReLU}(x) = \max(x, 0)$$

Đạo hàm của nó là một hàm bậc thang (tại $x=0$ ta quy ước lấy $0$):

$$\frac{d}{dx}\operatorname{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$

Sức mạnh của ReLU nằm ở đạo hàm: khi đầu vào dương, gradient luôn bằng $1$, **không bị co lại** khi truyền ngược qua nhiều tầng. Điều này giảm mạnh vấn đề **tiêu biến gradient (vanishing gradient)** vốn cản trở việc huấn luyện mạng sâu. Nhược điểm là hiện tượng "ReLU chết" (dying ReLU): nếu một đơn vị luôn nhận đầu vào âm, gradient bằng $0$ và nó ngừng học. Biến thể **Leaky ReLU** $\max(\alpha x, x)$ với $\alpha$ nhỏ khắc phục bằng cách cho một độ dốc nhỏ ở phần âm.

## 3.2. Sigmoid và tanh — hàm bão hòa

Hàm **sigmoid** ép giá trị về khoảng $(0, 1)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \qquad \frac{d}{dx}\sigma(x) = \sigma(x)\big(1 - \sigma(x)\big)$$

Hàm **tanh** ép về $(-1, 1)$ và đối xứng quanh gốc:

$$\tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}, \qquad \frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

Cả hai đều **bão hòa (saturate)**: khi $|x|$ lớn, đường cong nằm ngang nên đạo hàm tiến về $0$. Với sigmoid, đạo hàm cực đại chỉ là $0.25$ (tại $x=0$). Khi nhân chuỗi nhiều đạo hàm nhỏ hơn $1$ qua nhiều tầng, gradient teo dần theo cấp số nhân — chính là vanishing gradient. Đây là lý do sigmoid/tanh dần nhường chỗ cho ReLU ở các tầng ẩn của mạng sâu. Tuy vậy sigmoid vẫn rất hữu ích ở **tầng ra** khi cần một xác suất, và tanh thường tốt hơn sigmoid ở tầng ẩn vì đầu ra **lấy tâm tại 0 (zero-centered)**, giúp gradient ổn định hơn.

## 3.3. So sánh nhanh

| Hàm | Miền giá trị | Đạo hàm | Bão hòa | Vai trò tiêu biểu |
| --- | --- | --- | --- | --- |
| ReLU | $[0, \infty)$ | $0$ hoặc $1$ | không (phía dương) | tầng ẩn mạng sâu |
| sigmoid | $(0, 1)$ | $\le 0.25$ | hai phía | tầng ra nhị phân |
| tanh | $(-1, 1)$ | $\le 1$ | hai phía | tầng ẩn (RNN cổ điển) |

---

# 4. Vì sao MLP biểu diễn được mọi thứ

Việc thêm phi tuyến không chỉ giúp giải XOR — nó mở ra một khả năng biểu diễn mạnh đến mức gây kinh ngạc.

## 4.1. Định lý xấp xỉ phổ quát

**Định lý xấp xỉ phổ quát (Universal Approximation Theorem).** Một mạng nơ-ron với **một tầng ẩn duy nhất** chứa đủ nhiều đơn vị (và một hàm kích hoạt phi tuyến hợp lý) có thể xấp xỉ **bất kỳ hàm liên tục nào** trên một miền đóng bị chặn, với sai số nhỏ tùy ý.

Phát biểu trực giác: hãy hình dung mỗi đơn vị ẩn dùng hàm kích hoạt để tạo ra một "bậc thang" hoặc "đụn" (bump) cục bộ. Cộng đủ nhiều đụn có chiều cao và vị trí phù hợp lại, ta có thể vẽ xấp xỉ bất kỳ đường cong nào — giống như lát gạch để dựng nên một bề mặt cong. Tầng ra chỉ việc lấy tổ hợp tuyến tính các đụn này.

Điều này lý giải vì sao MLP "mạnh": về lý thuyết, lớp hàm mà nó biểu diễn được là cực rộng.

## 4.2. Sâu (deep) đấu với rộng (wide)

Định lý trên có một cái bẫy: nó chỉ nói tồn tại một mạng đủ rộng, **không nói tầng ẩn đó cần bao nhiêu đơn vị**. Trên thực tế, số đơn vị có thể phải lớn theo cấp số nhân — bất khả thi.

Đây là lúc **chiều sâu** lên tiếng. Kinh nghiệm và lý thuyết đều cho thấy: với nhiều hàm, một mạng **sâu và hẹp** đạt cùng độ chính xác bằng số tham số ít hơn hẳn so với một mạng **nông và rộng**. Lý do là mạng sâu xây dựng biểu diễn **phân cấp (hierarchical)**: tầng đầu học đặc trưng thô (cạnh, góc), các tầng sau ghép chúng thành khái niệm phức tạp dần (bộ phận, vật thể). Cấu trúc phân cấp này tái sử dụng các đặc trưng cấp thấp, nên hiệu quả hơn nhiều so với việc "nhồi" tất cả vào một tầng. Đó là tinh thần cốt lõi của **học sâu**.

---

# 5. Quan hệ với softmax và cách huấn luyện

MLP không thay thế những gì đã học trước đó — nó **bọc một lớp trích xuất đặc trưng** quanh chúng.

## 5.1. Tầng ra cho phân loại

Với bài toán phân loại $q$ lớp, ta giữ nguyên cơ chế của [hồi quy softmax](#/softmax-regression) ở tầng cuối: lấy đầu ra thô $\mathbf{O}$ (gọi là logit) rồi đưa qua softmax để được phân phối xác suất:

$$\hat{\mathbf{y}} = \operatorname{softmax}(\mathbf{O}), \qquad \hat{y}_j = \frac{\exp(o_j)}{\sum_{k=1}^{q} \exp(o_k)}$$

Cách nhìn gọn: **hồi quy softmax chính là một MLP không có tầng ẩn**. Khi thêm tầng ẩn phi tuyến, ta cho mô hình tự học một biểu diễn $\mathbf{H}$ tốt hơn $\mathbf{X}$ trước khi phân loại tuyến tính lên nó. Hàm mất mát vẫn là **entropy chéo (cross-entropy)** như đã trình bày trong bài [hàm mất mát & tối ưu hóa](#/ham-mat-mat-toi-uu).

Một lưu ý kỹ thuật quan trọng: trong cài đặt thực tế, ta không tính softmax rồi mới lấy log, mà gộp softmax và cross-entropy thành một bước để tránh tràn số — chi tiết ở bài [ổn định số học](#/on-dinh-so).

## 5.2. Huấn luyện bằng lan truyền ngược

Tham số của MLP gồm $\{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \mathbf{W}^{(2)}, \mathbf{b}^{(2)}\}$. Ta vẫn huấn luyện bằng hạ gradient: tính gradient của hàm mất mát theo từng tham số rồi cập nhật ngược hướng gradient. Vì mạng là hợp của nhiều tầng, gradient được tính hiệu quả nhờ **quy tắc chuỗi (chain rule)**, lan ngược từ tầng ra về tầng đầu — thuật toán [lan truyền ngược (backpropagation)](#/backprop), sẽ được trình bày chi tiết ở bài tiếp theo.

Dưới đây là một MLP một tầng ẩn viết bằng PyTorch để thấy ánh xạ trực tiếp với công thức ở mục 2:

```python
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(d, h),   # X W^(1) + b^(1)
    nn.ReLU(),         # sigma phi tuyến
    nn.Linear(h, q),   # H W^(2) + b^(2)
)
# Mất mát đã gộp softmax + cross-entropy (ổn định số học)
loss_fn = nn.CrossEntropyLoss()
```

---

# 6. Ưu điểm

* **Phá vỡ rào cản tuyến tính** — chỉ với một tầng ẩn và một phi tuyến, MLP giải được những bài toán như XOR mà mô hình tuyến tính bất lực.
* **Khả năng biểu diễn phổ quát** — theo định lý xấp xỉ phổ quát, MLP xấp xỉ được mọi hàm liên tục; chiều sâu cho phép làm điều đó tiết kiệm tham số nhờ biểu diễn phân cấp.
* **Tự học đặc trưng** — không cần thiết kế đặc trưng thủ công; mạng tự khám phá biểu diễn hữu ích từ dữ liệu thô.
* **Là viên gạch nền** — tầng kết nối đầy đủ + phi tuyến là thành phần xuất hiện trong gần như mọi kiến trúc hiện đại (CNN, Transformer).

---

# 7. Hạn chế

* **Dễ quá khớp (overfitting)** — sức biểu diễn lớn đi kèm rủi ro học thuộc nhiễu; cần các kỹ thuật điều chuẩn như dropout, weight decay.
* **Khó tối ưu** — mặt mất mát không lồi, có thể vướng điểm yên ngựa; cần chọn cẩn thận hàm kích hoạt và khởi tạo để tránh tiêu biến gradient.
* **Bỏ qua cấu trúc dữ liệu** — tầng kết nối đầy đủ không khai thác cấu trúc không gian (ảnh) hay tuần tự (văn bản), nên kém hiệu quả tham số so với CNN/RNN trên các miền đó.
* **Đòi hỏi nhiều dữ liệu và tính toán** — số tham số lớn cần lượng dữ liệu tương xứng để học tốt.

---

# 8. Tổng kết

Mạng nơ-ron nhiều tầng ra đời từ một nhận xét đơn giản: mô hình tuyến tính chỉ vẽ được ranh giới phẳng, nên ta **chèn một tầng ẩn kèm phi tuyến** để mạng tự học một biểu diễn mới, nơi bài toán trở nên dễ:

$$\mathbf{H} = \sigma(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \qquad \mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$

Hai sự thật cần khắc cốt. Thứ nhất, **phi tuyến là bắt buộc** — bỏ nó đi thì mọi chiều sâu sụp về một tầng tuyến tính duy nhất (mục 2.2). Thứ hai, **chiều sâu đáng giá hơn chiều rộng** — định lý xấp xỉ phổ quát hứa hẹn sức biểu diễn, nhưng chính việc xếp tầng theo cấu trúc phân cấp mới làm điều đó khả thi về tham số.

Về bản chất, MLP bọc một bộ trích xuất đặc trưng phi tuyến quanh một bộ phân loại quen thuộc như [softmax](#/softmax-regression). Tất cả được huấn luyện bằng hạ gradient, với gradient tính qua quy tắc chuỗi.

> Bài tiếp theo — **Lan truyền ngược ([backprop](#/backprop))** — sẽ mở "hộp đen" huấn luyện: dựng đồ thị tính toán (computational graph), đi xuôi để tính mất mát, rồi đi ngược để tính chính xác mọi gradient cần thiết cho các tham số của MLP.

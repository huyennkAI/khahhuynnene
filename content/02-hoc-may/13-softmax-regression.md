# Hồi quy Softmax (Softmax Regression)

> Hồi quy logistic (logistic regression) cho ta một bộ phân loại nhị phân (binary classifier) tuyệt đẹp: lấy điểm số tuyến tính, ép qua sigmoid thành xác suất. Nhưng thực tế phần lớn là **đa lớp** — nhận dạng chữ số có 10 lớp, phân loại ảnh có hàng nghìn lớp. **Hồi quy softmax** (softmax regression), còn gọi là **hồi quy logistic đa thức (multinomial logistic regression)**, tổng quát hóa ý tưởng đó lên $C$ lớp:
>
> $$\mathbf{z} = \mathbf{W}^\top \mathbf{x} \;\xrightarrow{\;\operatorname{softmax}\;}\; \mathbf{a} = \big(P(y{=}1\mid \mathbf{x}), \dots, P(y{=}C\mid \mathbf{x})\big)$$
>
> Nó biến một vector điểm số tùy ý thành một **phân phối xác suất** hợp lệ, huấn luyện bằng **cross-entropy**, và cho một gradient đẹp đến kinh ngạc: $\nabla_{\mathbf{z}} = \mathbf{a} - \mathbf{y}$. Vì lẽ đó nó là **lớp đầu ra chuẩn** của hầu hết mọi mạng phân loại hiện đại.

---

# 1. Bài toán phân loại đa lớp

Ta có dữ liệu $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ với $\mathbf{x}_i \in \mathbb{R}^d$ và nhãn $y_i \in \{1, 2, \dots, C\}$. Mục tiêu: với một điểm $\mathbf{x}$ mới, dự đoán nó thuộc lớp nào.

Với $C = 2$, [hồi quy logistic](logistic-regression) đã giải quyết bằng một điểm số duy nhất $z = \mathbf{w}^\top \mathbf{x}$ rồi $\sigma(z) \in (0,1)$ làm xác suất lớp dương. Ý tưởng tự nhiên khi có $C$ lớp: **mỗi lớp một bộ trọng số riêng**, cho ra $C$ điểm số.

Gom các vector trọng số thành ma trận $\mathbf{W} \in \mathbb{R}^{d \times C}$ (cột thứ $i$ là $\mathbf{w}_i$, bộ phát hiện của lớp $i$). Vector điểm số (score, hay logit):

$$\mathbf{z} = \mathbf{W}^\top \mathbf{x} \in \mathbb{R}^C, \qquad z_i = \mathbf{w}_i^\top \mathbf{x}$$

$z_i$ lớn nghĩa là mẫu "giống" lớp $i$ nhiều hơn. Nhưng $z_i$ là số thực bất kỳ — có thể âm, có thể lớn tùy ý — nên **không** dùng trực tiếp làm xác suất được. Ta cần một phép ánh xạ biến $\mathbf{z} \in \mathbb{R}^C$ thành một phân phối xác suất $\mathbf{a}$, tức thỏa $a_i \ge 0$ và $\sum_i a_i = 1$.

---

# 2. Hàm softmax

Phép ánh xạ đó là **hàm softmax**:

$$a_i = \operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

Hai tính chất cần kiểm tra để $\mathbf{a}$ là phân phối xác suất hợp lệ:

* **Không âm:** $e^{z_i} > 0$ luôn đúng, nên $a_i > 0$.
* **Tổng bằng 1:** $\sum_i a_i = \dfrac{\sum_i e^{z_i}}{\sum_j e^{z_j}} = 1$ do tử và mẫu giống hệt nhau.

Vì sao lại là hàm mũ $e^{z_i}$ chứ không phải, ví dụ, chuẩn hóa thẳng $z_i / \sum_j z_j$? Có ba lý do trực giác:

* **Bảo toàn thứ tự, khuếch đại khác biệt.** Hàm mũ tăng ngặt, nên $z_i$ lớn nhất kéo theo $a_i$ lớn nhất. Hơn nữa nó **khuếch đại** chênh lệch: khoảng cách $z_i - z_j$ biến thành **tỉ số** $a_i/a_j = e^{z_i - z_j}$, nên một logit nhỉnh hơn chút đã cho xác suất trội hẳn.
* **Luôn dương,** nên xử lý được mọi logit âm (chuẩn hóa thẳng sẽ vỡ khi có $z_i < 0$).
* **"Soft" của argmax.** Đẩy một logit lên rất cao thì $a_i \to 1$ — softmax tiến tới "chọn cứng" lớp lớn nhất (one-hot), nhưng vẫn **trơn và khả vi** mọi nơi, điều kiện tiên quyết để dùng [gradient descent](ham-mat-mat-toi-uu).

> **Trực giác.** softmax là một "argmax mềm": nó không trả về lớp thắng cuộc một cách dứt khoát mà trả về **mức độ tự tin** vào từng lớp dưới dạng xác suất. Nhiệt độ logit càng cao, phân phối càng nhọn về phía lớp thắng.

---

## 2.1. Tính bất biến khi cộng hằng số

Một tính chất quan trọng cả về lý thuyết lẫn cài đặt: **softmax không đổi khi cộng cùng một hằng số $c$ vào mọi logit.**

**Chứng minh.** Với mọi $c \in \mathbb{R}$:

$$\operatorname{softmax}(\mathbf{z} + c)_i = \frac{e^{z_i + c}}{\sum_j e^{z_j + c}} = \frac{e^{c}\, e^{z_i}}{e^{c} \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}} = \operatorname{softmax}(\mathbf{z})_i \qquad \blacksquare$$

Thừa số $e^c$ chung triệt tiêu giữa tử và mẫu. Hệ quả: chỉ **hiệu** giữa các logit mới mang thông tin, còn mức tuyệt đối thì không. (Đây cũng là lý do softmax "dư một bậc tự do", và vì sao logistic regression chỉ cần **một** điểm số cho hai lớp — xem mục 6.)

**Ổn định số học (numerical stability).** Tính bất biến này cho một mẹo cài đặt thiết yếu. Nếu các $z_i$ lớn, $e^{z_i}$ dễ **tràn số (overflow)** thành vô cực. Chọn $c = -\max_j z_j$ trước khi lấy mũ thì mọi số mũ đều $\le 0$, nên $e^{z_i - \max_j z_j} \in (0, 1]$ — an toàn tuyệt đối, mà kết quả **không đổi**:

$$a_i = \frac{e^{z_i - \max_j z_j}}{\sum_k e^{z_k - \max_j z_j}}$$

```python
def softmax_stable(z):  # z: vector logit
    z = z - z.max()      # trừ max: bất biến, tránh overflow
    e = np.exp(z)
    return e / e.sum()
```

Đây là cách mọi thư viện học sâu cài softmax trong thực tế.

---

# 3. Mã hóa nhãn one-hot

Để viết hàm mất mát gọn gàng, ta mã hóa nhãn $y \in \{1, \dots, C\}$ thành vector **one-hot** $\mathbf{y} \in \{0,1\}^C$: thành phần thứ $y$ bằng $1$, còn lại bằng $0$. Ví dụ với $C = 3$ và nhãn thật là lớp $2$:

$$y = 2 \;\Longrightarrow\; \mathbf{y} = (0, 1, 0)$$

Cách nhìn xác suất: $\mathbf{y}$ chính là **phân phối thật** — toàn bộ khối xác suất dồn vào đúng lớp thật. Còn $\mathbf{a}$ là **phân phối dự đoán** của mô hình. Huấn luyện chính là kéo $\mathbf{a}$ về gần $\mathbf{y}$. Việc đo "gần" giữa hai phân phối dẫn ta tới cross-entropy.

---

# 4. Hàm mất mát cross-entropy từ MLE

## 4.1. Suy ra từ hợp lý cực đại

Vì $\mathbf{a}$ là tham số của một phân phối Categorical (phân phối hạng mục) trên $C$ lớp, ta dùng [hợp lý cực đại (MLE)](mle-map). Xác suất mô hình gán cho **đúng nhãn thật** của một mẫu là:

$$P(y \mid \mathbf{x}; \mathbf{W}) = \prod_{i=1}^{C} a_i^{\,y_i}$$

Cách viết tích này gọn vì $\mathbf{y}$ là one-hot: mọi thừa số có $y_i = 0$ thành $a_i^0 = 1$, chỉ còn lại đúng $a_{y}$ — xác suất của lớp thật. Hợp lý (likelihood) trên toàn bộ $N$ mẫu độc lập là tích các xác suất này. Lấy **âm log** (biến tích thành tổng, biến cực đại thành cực tiểu) cho ra hàm mất mát của một mẫu:

$$L = -\log \prod_{i=1}^{C} a_i^{\,y_i} = -\sum_{i=1}^{C} y_i \log a_i$$

Đây chính là **cross-entropy** (entropy chéo) giữa phân phối thật $\mathbf{y}$ và phân phối dự đoán $\mathbf{a}$. Vì $\mathbf{y}$ one-hot, tổng rút lại còn đúng **một** số hạng:

$$L = -\log a_{y}$$

tức **âm log xác suất mà mô hình gán cho lớp đúng**. Trực giác cực kỳ rõ: mô hình tự tin đúng ($a_y \to 1$) thì $L \to 0$; mô hình gán xác suất bé cho lớp đúng ($a_y \to 0$) thì $L \to +\infty$ — bị phạt rất nặng. Hàm mất mát trên toàn tập là trung bình:

$$J(\mathbf{W}) = -\frac{1}{N}\sum_{n=1}^{N} \sum_{i=1}^{C} y_{n,i} \log a_{n,i}$$

So với [hàm mất mát tổng quát](ham-mat-mat-toi-uu), đây đúng là dạng cross-entropy đa lớp đã nêu ở đó.

---

# 5. Gradient của cross-entropy qua softmax

Đây là phần kỳ diệu của softmax regression: dù hàm hợp $L(\mathbf{a}(\mathbf{z}))$ trông rối, gradient theo logit lại **cực kỳ gọn**.

## 5.1. Đạo hàm của softmax

Trước hết cần đạo hàm $\partial a_i / \partial z_j$. Đặt $S = \sum_k e^{z_k}$, nên $a_i = e^{z_i}/S$. Xét hai trường hợp.

**Trường hợp $i = j$** (dùng quy tắc thương):

$$\frac{\partial a_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i}\cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S}\cdot\frac{S - e^{z_i}}{S} = a_i (1 - a_i)$$

**Trường hợp $i \ne j$** (tử $e^{z_i}$ không chứa $z_j$, chỉ mẫu $S$ chứa):

$$\frac{\partial a_i}{\partial z_j} = \frac{0\cdot S - e^{z_i}\cdot e^{z_j}}{S^2} = -\frac{e^{z_i}}{S}\cdot\frac{e^{z_j}}{S} = -a_i a_j$$

Gộp hai trường hợp bằng **delta Kronecker** $\delta_{ij}$ (bằng $1$ nếu $i=j$, ngược lại $0$):

$$\boxed{\;\frac{\partial a_i}{\partial z_j} = a_i(\delta_{ij} - a_j)\;}$$

## 5.2. Gradient theo logit: $\nabla_{\mathbf{z}} = \mathbf{a} - \mathbf{y}$

Áp **quy tắc chuỗi (chain rule)**. Vì $L = -\sum_i y_i \log a_i$, đạo hàm theo một logit $z_j$ là:

$$\frac{\partial L}{\partial z_j} = \sum_{i=1}^{C} \frac{\partial L}{\partial a_i}\frac{\partial a_i}{\partial z_j} = \sum_{i=1}^{C} \left(-\frac{y_i}{a_i}\right) a_i(\delta_{ij} - a_j)$$

Rút gọn $a_i$ ở tử với $a_i$ ở mẫu:

$$\frac{\partial L}{\partial z_j} = -\sum_{i=1}^{C} y_i(\delta_{ij} - a_j) = -\sum_i y_i \delta_{ij} + a_j \sum_i y_i$$

Hai tổng đơn giản đáng kinh ngạc: $\sum_i y_i \delta_{ij} = y_j$ (chỉ số hạng $i=j$ sống sót), và $\sum_i y_i = 1$ (vì $\mathbf{y}$ là phân phối xác suất one-hot). Vậy:

$$\frac{\partial L}{\partial z_j} = -y_j + a_j \cdot 1 = a_j - y_j$$

Viết gọn dạng vector:

$$\boxed{\;\nabla_{\mathbf{z}} L = \mathbf{a} - \mathbf{y}\;} \qquad \blacksquare$$

> **Vẻ đẹp.** Gradient theo logit chỉ là **sai số dự đoán** — chênh lệch giữa phân phối dự đoán và phân phối thật. Không còn $e^{z}$, không còn mẫu số, không còn delta. Đây là phiên bản đa lớp của kết quả $\sigma(z) - y$ trong logistic regression, và là lý do cặp đôi softmax + cross-entropy được ghép với nhau như mặc định.

## 5.3. Gradient theo trọng số

Vì $z_j = \mathbf{w}_j^\top \mathbf{x}$ nên $\partial z_j / \partial \mathbf{w}_j = \mathbf{x}$. Lại quy tắc chuỗi:

$$\frac{\partial L}{\partial \mathbf{w}_j} = (a_j - y_j)\,\mathbf{x}$$

Gộp toàn bộ ma trận $\mathbf{W}$ thành một biểu thức ngoại tích (outer product) duy nhất:

$$\nabla_{\mathbf{W}} L = \mathbf{x}\,(\mathbf{a} - \mathbf{y})^\top$$

Đây là gradient cho **một** mẫu; trên cả tập chỉ việc lấy trung bình. Toàn bộ thuật toán huấn luyện gói gọn trong công thức này.

---

# 6. Logistic regression là trường hợp riêng $C = 2$

Khẳng định: với $C = 2$, softmax regression **chính là** [hồi quy logistic](logistic-regression). Hãy chứng minh.

Với hai lớp, xác suất lớp $1$ là:

$$a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}}$$

Theo tính bất biến cộng hằng số (mục 2.1), ta được phép trừ $z_2$ khỏi cả hai logit mà không đổi kết quả. Chia cả tử và mẫu cho $e^{z_2}$ (tương đương việc đó):

$$a_1 = \frac{e^{z_1 - z_2}}{e^{z_1 - z_2} + 1} = \frac{1}{1 + e^{-(z_1 - z_2)}} = \sigma(z_1 - z_2)$$

Đặt $z = z_1 - z_2 = (\mathbf{w}_1 - \mathbf{w}_2)^\top \mathbf{x}$, ta thu được **đúng** hàm sigmoid của một điểm số tuyến tính duy nhất — chính là logistic regression. Quan sát này khớp với nhận xét "softmax dư một bậc tự do": hai lớp chỉ cần **một** vector trọng số hiệu dụng $\mathbf{w} = \mathbf{w}_1 - \mathbf{w}_2$, không phải hai. $\blacksquare$

---

# 7. Huấn luyện bằng gradient descent

Hàm mất mát $J(\mathbf{W})$ là **lồi (convex)** theo $\mathbf{W}$ (cross-entropy của softmax tuyến tính), nên gradient descent hội tụ về cực tiểu toàn cục. Quy tắc cập nhật chuẩn:

$$\mathbf{W} \leftarrow \mathbf{W} - \eta \, \nabla_{\mathbf{W}} J$$

với $\eta$ là [tốc độ học](ham-mat-mat-toi-uu) (learning rate). Nhờ gradient gọn ở mục 5, một vòng lặp (mini-batch) chỉ gồm: tính logit, softmax, rồi cập nhật theo sai số $\mathbf{a} - \mathbf{y}$.

```python
for epoch in range(num_epochs):
    Z = X @ W                       # (N, C) logit
    A = softmax_rows(Z)             # softmax theo từng hàng
    grad = X.T @ (A - Y) / N        # Y: nhãn one-hot (N, C)
    W -= lr * grad
```

Thực tế hay thêm **chính quy hóa (regularization)** $\tfrac{\lambda}{2}\|\mathbf{W}\|^2$ (cộng $\lambda \mathbf{W}$ vào gradient) vừa để tránh quá khớp, vừa để **khử bậc tự do dư** đã nói ở mục 2.1 — nó cố định nghiệm thay vì để $\mathbf{W}$ trôi tự do theo hướng cộng hằng số.

---

# 8. Lớp đầu ra chuẩn của mạng phân loại

Cho đến giờ logit là **tuyến tính** $\mathbf{z} = \mathbf{W}^\top\mathbf{x}$. Nhưng không có gì bắt buộc như vậy. Trong một [mạng nơ-ron](mang-no-ron), ta chỉ việc thay $\mathbf{x}$ bằng đặc trưng (feature) $\mathbf{h}$ học được từ các tầng ẩn:

$$\mathbf{z} = \mathbf{W}^\top \mathbf{h}, \qquad \mathbf{a} = \operatorname{softmax}(\mathbf{z})$$

Vì vậy softmax + cross-entropy là **đầu ra mặc định** của gần như mọi mạng phân loại. Và gradient đẹp $\nabla_{\mathbf{z}} = \mathbf{a} - \mathbf{y}$ chính là **tín hiệu sai số khởi đầu** cho [lan truyền ngược (backpropagation)](ham-mat-mat-toi-uu): nó được đẩy ngược qua các tầng để cập nhật toàn bộ trọng số. Nói cách khác, softmax regression là **tầng cuối** mà mọi bộ trích xuất đặc trưng sâu đều cắm vào.

---

# 9. Ưu điểm

* **Đầu ra là xác suất hợp lệ** — softmax bảo đảm $a_i \ge 0$, $\sum a_i = 1$, nên dự đoán có thể diễn giải như độ tin cậy, dùng được cho ngưỡng quyết định và hiệu chỉnh (calibration).
* **Gradient cực gọn** $\mathbf{a} - \mathbf{y}$ — rẻ để tính, ổn định, và là tín hiệu khởi đầu tự nhiên cho backpropagation.
* **Hàm mất mát lồi** (ở dạng tuyến tính) — gradient descent hội tụ về cực tiểu toàn cục, không kẹt cực tiểu địa phương.
* **Tổng quát hóa nhất quán** — chứa logistic regression như trường hợp riêng $C=2$ và cắm thẳng vào mọi mạng sâu làm tầng đầu ra.

---

# 10. Nhược điểm

* **Ranh giới quyết định tuyến tính** (khi logit tuyến tính) — không tách được các lớp phi tuyến nếu không có đặc trưng phi tuyến đi kèm; sức mạnh thật đến từ các tầng trước nó trong mạng.
* **Giả định loại trừ lẫn nhau (mutually exclusive)** — softmax buộc các lớp cạnh tranh nhau ($\sum a_i = 1$), nên **không** hợp với bài toán đa nhãn (multi-label, một mẫu thuộc nhiều lớp); khi đó dùng nhiều sigmoid độc lập.
* **Nhạy với mất cân bằng lớp và tràn số** — cần mẹo trừ max khi cài đặt, và cần chú ý chính quy hóa do bậc tự do dư.

---

# 11. Tổng kết

Hồi quy softmax là mảnh ghép nối liền hồi quy tuyến tính, hồi quy logistic và học sâu. Mạch logic của nó chặt chẽ từ đầu đến cuối:

1. Mỗi lớp một vector trọng số → vector logit $\mathbf{z} = \mathbf{W}^\top\mathbf{x}$.
2. softmax $a_i = e^{z_i}/\sum_j e^{z_j}$ biến logit thành **phân phối xác suất**, với mẹo trừ max để ổn định số học.
3. [MLE](mle-map) trên phân phối Categorical sinh ra **cross-entropy** $-\sum_i y_i \log a_i$.
4. Quy tắc chuỗi cộng đạo hàm softmax $a_i(\delta_{ij}-a_j)$ cho gradient đẹp $\nabla_{\mathbf{z}} = \mathbf{a} - \mathbf{y}$.
5. $C=2$ thu lại [logistic regression](logistic-regression); thay $\mathbf{x}$ bằng đặc trưng học được thì thành **tầng đầu ra** của [mạng nơ-ron](mang-no-ron).

$$\boxed{\;\mathbf{a} = \operatorname{softmax}(\mathbf{W}^\top\mathbf{x}), \quad L = -\sum_i y_i \log a_i, \quad \nabla_{\mathbf{z}} L = \mathbf{a} - \mathbf{y}\;}$$

Ba công thức này — hàm dự đoán, hàm mất mát, và gradient — là tất cả những gì cần để hiểu vì sao softmax + cross-entropy là cặp đôi mặc định trong phân loại.

> Bài tiếp theo đưa softmax regression vào bối cảnh lớn hơn: khi xếp chồng nhiều tầng phi tuyến trước nó, ta được một **mạng nơ-ron** phân loại đầy đủ, huấn luyện trọn vẹn bằng lan truyền ngược từ chính tín hiệu $\mathbf{a} - \mathbf{y}$.

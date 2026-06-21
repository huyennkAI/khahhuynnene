# Hồi quy Logistic (Logistic Regression)

> Hồi quy logistic (logistic regression) là mô hình **phân loại nhị phân (binary classification)** kinh điển: nó không dự đoán trực tiếp nhãn $0/1$ mà ước lượng **xác suất** $P(y=1\mid\mathbf{x})$. Bí quyết là bọc một tổ hợp tuyến tính $\mathbf{w}^\top\mathbf{x}$ qua **hàm sigmoid** để ép kết quả về khoảng $(0,1)$:
>
> $$P(y=1\mid\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^\top\mathbf{x}}}$$
>
> Hàm mất mát của nó — **cross-entropy** — không phải chọn tùy tiện mà **suy ra** từ nguyên lý cực đại hợp lý (MLE) trên phân phối Bernoulli. Và gradient của nó có một dạng đẹp đến kinh ngạc.

---

# 1. Vì sao không dùng hồi quy tuyến tính cho phân loại?

Một ý tưởng ngây thơ: gắn nhãn $y\in\{0,1\}$ rồi khớp một đường hồi quy tuyến tính (linear regression) $\hat{y}=\mathbf{w}^\top\mathbf{x}$, sau đó cắt ngưỡng tại $0.5$. Cách này hỏng vì ba lý do.

**Thứ nhất, đầu ra không bị chặn.** $\mathbf{w}^\top\mathbf{x}$ chạy khắp $(-\infty,+\infty)$, nên ta có thể nhận "xác suất" $-3.2$ hoặc $7.8$ — vô nghĩa khi diễn giải như xác suất.

**Thứ hai, nhạy cảm với điểm ngoại lai (outlier).** Một mẫu dương nằm rất xa biên sẽ kéo lệch đường thẳng, làm dịch chuyển ngưỡng và phân loại sai các mẫu vốn dễ. Hồi quy tuyến tính cố ép đầu ra **bằng đúng** $1$ cho mọi mẫu dương, kể cả mẫu đã rất chắc chắn — một mục tiêu sai.

**Thứ ba, sai giả định mô hình.** MSE (xem [Hàm mất mát & Tối ưu hóa](ham-mat-mat-toi-uu)) tương ứng với giả định nhiễu Gauss trên đầu ra liên tục, trong khi $y$ ở đây là biến **rời rạc nhị phân**. Giả định sinh dữ liệu đúng phải là **Bernoulli**.

Ta cần một hàm: nhận đầu vào tuyến tính $z=\mathbf{w}^\top\mathbf{x}\in\mathbb{R}$, trả ra một số trong $(0,1)$ diễn giải được như xác suất, trơn (để lấy gradient) và đơn điệu tăng. Sigmoid đáp ứng tất cả.

---

# 2. Hàm sigmoid và tính chất đạo hàm

## 2.1. Định nghĩa và trực giác

Hàm **sigmoid** (còn gọi là logistic) là:

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

Vài tính chất hình học cần nắm:

* **Bị chặn:** $\sigma(z)\in(0,1)$ với mọi $z$, tiệm cận $0$ khi $z\to-\infty$ và $1$ khi $z\to+\infty$.
* **Tâm tại $0.5$:** $\sigma(0)=\tfrac12$, là điểm uốn (inflection point).
* **Đối xứng:** $\sigma(-z) = 1-\sigma(z)$. Đây là tính chất rất hay dùng phía dưới.

Trực giác: $z=\mathbf{w}^\top\mathbf{x}$ đo "mức độ tự tin tuyến tính" về lớp dương; sigmoid bóp nó về thang xác suất, mượt mà thay vì nhảy bậc như hàm dấu của [perceptron](perceptron).

## 2.2. Tính chất then chốt: $\sigma'(z)=\sigma(z)\big(1-\sigma(z)\big)$

Đây là tính chất làm cho toàn bộ giải tích của logistic regression trở nên gọn gàng. **Chứng minh** bằng quy tắc đạo hàm hàm hợp. Viết $\sigma(z)=(1+e^{-z})^{-1}$:

$$\sigma'(z) = -\,(1+e^{-z})^{-2}\cdot\frac{d}{dz}\big(1+e^{-z}\big) = -\,(1+e^{-z})^{-2}\cdot(-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}$$

Bây giờ tách khéo léo tử số $e^{-z} = (1+e^{-z})-1$:

$$\sigma'(z) = \frac{(1+e^{-z})-1}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} - \frac{1}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}}\left(1 - \frac{1}{1+e^{-z}}\right)$$

Nhận ra ngay $\dfrac{1}{1+e^{-z}}=\sigma(z)$, nên:

$$\boxed{\;\sigma'(z) = \sigma(z)\big(1-\sigma(z)\big)\;} \qquad\blacksquare$$

Vẻ đẹp ở chỗ đạo hàm biểu diễn **chỉ qua chính giá trị hàm** — không cần tính lại $e^{-z}$. Khi cài đặt, lưu lại $\hat{y}=\sigma(z)$ là tính được gradient bằng $\hat{y}(1-\hat{y})$.

---

# 3. Mô hình xác suất Bernoulli

Đặt $\hat{y}=\sigma(\mathbf{w}^\top\mathbf{x})$. Ta mô hình hóa nhãn như một biến **Bernoulli** với tham số $\hat{y}$:

$$P(y=1\mid\mathbf{x};\mathbf{w}) = \hat{y}, \qquad P(y=0\mid\mathbf{x};\mathbf{w}) = 1-\hat{y}$$

Mẹo gộp hai trường hợp vào **một** biểu thức (vì $y\in\{0,1\}$, một trong hai số mũ luôn bằng $0$):

$$P(y\mid\mathbf{x};\mathbf{w}) = \hat{y}^{\,y}\,(1-\hat{y})^{1-y}$$

Kiểm tra: $y=1$ cho $\hat{y}$, $y=0$ cho $1-\hat{y}$ — đúng. Đây là khối xây dựng để viết hàm hợp lý (likelihood).

> **Lưu ý ký hiệu.** Ta nhét hệ số chệch (bias) vào $\mathbf{w}$ bằng cách thêm một thành phần $1$ vào $\mathbf{x}$, nên $\mathbf{w}^\top\mathbf{x}$ đã bao gồm cả bias. Cách này gọn và phổ biến trong machinelearningcoban.

---

# 4. Hàm mất mát cross-entropy từ MLE

## 4.1. Suy dẫn từ cực đại hợp lý

Cho tập huấn luyện $\{(\mathbf{x}_i,y_i)\}_{i=1}^n$ độc lập. Theo nguyên lý **cực đại hợp lý (MLE)** — xem chi tiết ở [Ước lượng MLE & MAP](mle-map) — ta tìm $\mathbf{w}$ làm **cực đại** xác suất quan sát toàn bộ dữ liệu:

$$\mathcal{L}(\mathbf{w}) = \prod_{i=1}^{n} P(y_i\mid\mathbf{x}_i;\mathbf{w}) = \prod_{i=1}^{n} \hat{y}_i^{\,y_i}\,(1-\hat{y}_i)^{1-y_i}$$

Tích nhiều số trong $(0,1)$ vừa khó tối ưu vừa dễ tràn số (underflow). Lấy **logarit** (đơn điệu tăng nên không đổi nghiệm), biến tích thành tổng:

$$\log\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n} \Big[\, y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \,\Big]$$

Cực đại log-likelihood tương đương cực tiểu **âm log-likelihood**. Chia cho $n$ để được trung bình, ta thu đúng hàm **cross-entropy**:

$$\boxed{\; J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\Big[\, y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \,\Big] \;}$$

Điểm cốt lõi: cross-entropy **không** là một lựa chọn tùy tiện. Nó **chính là** âm log-likelihood của giả định Bernoulli. So với MSE ở [Hàm mất mát & Tối ưu hóa](ham-mat-mat-toi-uu), đây mới là hàm mất mát "đúng nguyên lý" cho phân loại.

## 4.2. Trực giác của cross-entropy

Xét một mẫu dương ($y=1$): mất mát là $-\log\hat{y}$. Nếu mô hình tự tin đúng ($\hat{y}\to1$) thì mất mát $\to0$; nếu mô hình sai nặng ($\hat{y}\to0$) thì mất mát $\to+\infty$. Hàm phạt **rất nặng** với những dự đoán tự tin nhưng sai — đúng điều ta muốn. MSE không có tính chất phạt vô hạn này, nên hội tụ chậm hơn nhiều khi dùng cho phân loại.

---

# 5. Gradient của cross-entropy

## 5.1. Chứng minh dạng đẹp

Đây là kết quả trung tâm. Ta sẽ chứng minh gradient của cross-entropy theo $\mathbf{w}$ có dạng cực gọn:

$$\nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)\,\mathbf{x}_i$$

**Chứng minh.** Xét mất mát một mẫu (bỏ chỉ số $i$ và hệ số $1/n$ cho gọn), với $z=\mathbf{w}^\top\mathbf{x}$ và $\hat{y}=\sigma(z)$:

$$J_0 = -\big[\,y\log\hat{y} + (1-y)\log(1-\hat{y})\,\big]$$

Áp dụng quy tắc dây chuyền theo ba mắt xích $\mathbf{w}\to z\to\hat{y}\to J_0$.

**Mắt xích 1 — đạo hàm $J_0$ theo $\hat{y}$:**

$$\frac{\partial J_0}{\partial\hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

**Mắt xích 2 — đạo hàm $\hat{y}$ theo $z$**, dùng đúng tính chất ở mục 2.2:

$$\frac{\partial\hat{y}}{\partial z} = \sigma'(z) = \hat{y}(1-\hat{y})$$

Nhân hai mắt xích, và đây là chỗ phép màu xảy ra — các thừa số triệt tiêu ngoạn mục:

$$\frac{\partial J_0}{\partial z} = \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right)\hat{y}(1-\hat{y}) = -y(1-\hat{y}) + (1-y)\hat{y}$$

Khai triển: $-y + y\hat{y} + \hat{y} - y\hat{y} = \hat{y} - y$. Vậy:

$$\frac{\partial J_0}{\partial z} = \hat{y} - y$$

**Mắt xích 3 — đạo hàm $z$ theo $\mathbf{w}$:** vì $z=\mathbf{w}^\top\mathbf{x}$ nên $\dfrac{\partial z}{\partial\mathbf{w}}=\mathbf{x}$. Ghép lại:

$$\nabla_{\mathbf{w}} J_0 = \frac{\partial J_0}{\partial z}\cdot\frac{\partial z}{\partial\mathbf{w}} = (\hat{y}-y)\,\mathbf{x}$$

Cộng dồn toàn bộ mẫu và lấy trung bình:

$$\nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)\,\mathbf{x}_i \qquad\blacksquare$$

## 5.2. Ý nghĩa

Gradient là tổng các vector đặc trưng $\mathbf{x}_i$, mỗi cái nhân với **sai số dự đoán** $(\hat{y}_i-y_i)$. Mẫu dự đoán càng sai (lệch giữa xác suất ước lượng và nhãn thật càng lớn) thì càng "kéo" trọng số mạnh. Đáng chú ý: dạng này **trùng khớp** với gradient của hồi quy tuyến tính dùng MSE — không phải trùng hợp, mà do cả hai đều thuộc họ mô hình tuyến tính tổng quát (generalized linear model), nơi sự kết đôi giữa sigmoid và cross-entropy làm triệt tiêu mẫu số.

---

# 6. Tối ưu bằng gradient descent

## 6.1. Vì sao không có nghiệm dạng đóng

Với hồi quy tuyến tính, đặt gradient bằng $0$ cho **phương trình tuyến tính** theo $\mathbf{w}$, giải ra nghiệm dạng đóng (closed-form). Ở đây, điều kiện tối ưu:

$$\sum_{i=1}^{n}\big(\sigma(\mathbf{w}^\top\mathbf{x}_i)-y_i\big)\mathbf{x}_i = 0$$

là **phi tuyến** theo $\mathbf{w}$ (vì $\sigma$ phi tuyến), không tách được $\mathbf{w}$ ra ngoài. Không có công thức tường minh. Ta phải dùng phương pháp **lặp** — gradient descent (xem [Hàm mất mát & Tối ưu hóa](ham-mat-mat-toi-uu)):

$$\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla_{\mathbf{w}} J(\mathbf{w}) = \mathbf{w} - \frac{\eta}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)\,\mathbf{x}_i$$

với $\eta$ là tốc độ học (learning rate).

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train(X, y, eta=0.1, epochs=1000):
    w = np.zeros(X.shape[1])           # X đã thêm cột 1 cho bias
    for _ in range(epochs):
        y_hat = sigmoid(X @ w)
        grad = X.T @ (y_hat - y) / len(y)   # (1/n) Σ (ŷ_i − y_i) x_i
        w -= eta * grad
    return w
```

## 6.2. Tính lồi của hàm mất mát

Tin tốt: cross-entropy của logistic regression là **hàm lồi (convex)** theo $\mathbf{w}$, nên gradient descent hội tụ về **cực tiểu toàn cục**, không kẹt ở cực tiểu địa phương. **Chứng minh phác** dựa vào ma trận Hessian:

$$\nabla^2_{\mathbf{w}} J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}\hat{y}_i(1-\hat{y}_i)\,\mathbf{x}_i\mathbf{x}_i^\top$$

Với mọi vector $\mathbf{v}$, dạng toàn phương là $\mathbf{v}^\top H\mathbf{v} = \tfrac1n\sum_i \hat{y}_i(1-\hat{y}_i)(\mathbf{x}_i^\top\mathbf{v})^2 \ge 0$ vì $\hat{y}_i(1-\hat{y}_i)>0$ và bình phương không âm. Hessian **nửa xác định dương (positive semi-definite)** ⇒ $J$ lồi. $\blacksquare$ Đây là lợi thế lớn so với mạng nơ-ron sâu, nơi hàm mất mát không lồi.

---

# 7. Ranh giới quyết định tuyến tính

Để phân loại, ta cắt ngưỡng tại $0.5$: dự đoán lớp $1$ nếu $\hat{y}\ge0.5$. Vì $\sigma(z)\ge0.5 \iff z\ge0$, ranh giới quyết định (decision boundary) là tập điểm thỏa:

$$\mathbf{w}^\top\mathbf{x} = 0$$

Đây là một **siêu phẳng (hyperplane)** — ranh giới **tuyến tính** trong không gian đặc trưng. Vậy dù sigmoid phi tuyến, logistic regression vẫn là một **bộ phân loại tuyến tính**: nó chỉ phân tách được các lớp khả phân tuyến tính. Muốn ranh giới cong, phải thêm đặc trưng phi tuyến (đa thức, kernel) hoặc chồng nhiều tầng — chính là con đường dẫn tới [mạng nơ-ron](mang-no-ron).

---

# 8. Điều chuẩn chống quá khớp

Khi dữ liệu **khả phân tuyến tính hoàn toàn**, MLE đẩy $\|\mathbf{w}\|\to\infty$ để ép $\hat{y}$ về sát $0$ hoặc $1$ — mô hình quá tự tin và **quá khớp (overfitting)** (xem [Quá khớp & Điều chuẩn](overfitting)). Cách chữa: thêm số hạng **điều chuẩn (regularization)** phạt trọng số lớn. Với $L_2$:

$$J_{\text{reg}}(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\Big] + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$$

Gradient chỉ thêm một số hạng tuyến tính:

$$\nabla_{\mathbf{w}} J_{\text{reg}} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)\mathbf{x}_i + \lambda\mathbf{w}$$

Nhìn từ góc Bayes, $L_2$ tương đương đặt **tiên nghiệm Gauss** lên $\mathbf{w}$ rồi cực đại hậu nghiệm — đúng là ước lượng MAP thay cho MLE (xem [Ước lượng MLE & MAP](mle-map)). Tham số $\lambda$ điều khiển độ mạnh: $\lambda$ lớn ⇒ trọng số nhỏ, mô hình đơn giản hơn.

---

# 9. Mở rộng đa lớp: softmax regression

Logistic regression chỉ xử lý **hai** lớp. Khi có $C>2$ lớp, ta tổng quát sigmoid thành **softmax**: thay một bộ trọng số bằng $C$ bộ $\mathbf{w}_c$, và chuẩn hóa thành phân phối xác suất trên các lớp:

$$P(y=c\mid\mathbf{x}) = \frac{e^{\mathbf{w}_c^\top\mathbf{x}}}{\sum_{k=1}^{C} e^{\mathbf{w}_k^\top\mathbf{x}}}$$

Với $C=2$, softmax **thoái hóa về** đúng sigmoid (chỉ cần đặt $\mathbf{w}=\mathbf{w}_1-\mathbf{w}_0$). Hàm mất mát vẫn là cross-entropy (dạng nhiều lớp), và điều kỳ diệu là gradient vẫn giữ **đúng** dạng đẹp $(\hat{y}-y)\mathbf{x}$ với nhãn one-hot. Chi tiết đầy đủ ở bài [Hồi quy Softmax](softmax-regression).

---

# 10. So sánh với perceptron

[Perceptron](perceptron) cũng là bộ phân loại tuyến tính dùng ranh giới $\mathbf{w}^\top\mathbf{x}=0$, nhưng khác biệt cốt lõi nằm ở **đầu ra**:

| Tiêu chí | Perceptron | Logistic Regression |
| --- | --- | --- |
| Đầu ra | Nhãn cứng $\{0,1\}$ qua hàm dấu | **Xác suất** $\sigma(\mathbf{w}^\top\mathbf{x})\in(0,1)$ |
| Hàm kích hoạt | Bậc thang (không khả vi tại $0$) | Sigmoid (trơn, khả vi mọi nơi) |
| Hàm mất mát | Quy tắc cập nhật heuristic | Cross-entropy (từ MLE, **lồi**) |
| Khi dữ liệu khả phân | Dừng khi phân tách xong | Tiếp tục tinh chỉnh độ tự tin |
| Đầu ra diễn giải được | Không | **Có** — dùng được làm độ tin cậy |

Điểm hơn quan trọng nhất: logistic regression cho **xác suất có hiệu chuẩn (calibrated)**, nên ta biết mô hình "chắc chắn đến đâu" thay vì chỉ một nhãn. Điều này thiết yếu khi cần xếp hạng, đặt ngưỡng theo chi phí, hay kết hợp nhiều mô hình.

---

# 11. Ưu điểm

* **Đầu ra là xác suất diễn giải được** — biết được độ tự tin, đặt ngưỡng linh hoạt theo bài toán.
* **Hàm mất mát lồi** — gradient descent đảm bảo hội tụ về cực tiểu toàn cục (mục 6.2), huấn luyện ổn định.
* **Có nền tảng xác suất chặt chẽ** — suy ra trực tiếp từ MLE trên Bernoulli, không phải heuristic.
* **Gradient cực gọn** — dạng $(\hat{y}-y)\mathbf{x}$ dễ cài đặt, rẻ tính toán, mở rộng tự nhiên sang softmax.
* **Nhẹ và khó quá khớp** — ít tham số, dễ điều chuẩn, là baseline mạnh cho mọi bài phân loại.

---

# 12. Nhược điểm

* **Chỉ vẽ được ranh giới tuyến tính** — không xử lý được dữ liệu khả phân phi tuyến nếu không thêm đặc trưng (mục 7).
* **Không có nghiệm dạng đóng** — bắt buộc tối ưu lặp (mục 6.1), tuy nhờ tính lồi nên không phải vấn đề lớn.
* **Nhạy với đặc trưng tương quan mạnh** — đa cộng tuyến (multicollinearity) làm trọng số bất ổn; cần điều chuẩn.
* **Quá khớp khi dữ liệu khả phân hoàn toàn** — $\|\mathbf{w}\|$ phân kỳ nếu không điều chuẩn (mục 8).

---

# 13. Tổng kết

Hồi quy logistic là cây cầu giữa hồi quy tuyến tính và phân loại xác suất. Toàn bộ mô hình gói gọn trong một dòng — bọc tổ hợp tuyến tính qua sigmoid:

$$P(y=1\mid\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x})$$

Ba mảnh ghép khớp vào nhau một cách tự nhiên: giả định **Bernoulli** sinh ra hàm mất mát **cross-entropy** qua MLE; tính chất $\sigma'=\sigma(1-\sigma)$ làm gradient rút gọn thành dạng đẹp $(\hat{y}-y)\mathbf{x}$; và tính **lồi** đảm bảo tối ưu về cực tiểu toàn cục bằng gradient descent. Ranh giới của nó tuyến tính, sức biểu diễn hạn chế, nhưng đổi lại sự đơn giản, ổn định và khả năng cho xác suất hiệu chuẩn.

> Bài tiếp theo — [Hồi quy Softmax](softmax-regression) — tổng quát logistic regression lên nhiều lớp, và là viên gạch trực tiếp dẫn tới tầng đầu ra của [mạng nơ-ron](mang-no-ron).

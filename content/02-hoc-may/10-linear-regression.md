# Hồi quy tuyến tính (Linear Regression)

> Hồi quy tuyến tính (linear regression) là mô hình học máy **đơn giản nhất nhưng nền tảng nhất**: giả định đầu ra là một **tổ hợp tuyến tính** của các đặc trưng đầu vào.
>
> $$\hat y = \mathbf{w}^\top \mathbf{x} + b$$
>
> Cái đẹp của nó nằm ở chỗ bài toán tối ưu có **nghiệm đóng (closed-form solution)** — phương trình chuẩn (normal equation) — và nghiệm đó có một **ý nghĩa hình học** sáng sủa: phép **chiếu trực giao (orthogonal projection)** vector mục tiêu lên không gian cột của ma trận dữ liệu. Đây là cánh cửa tự nhiên để bước vào học có giám sát (supervised learning).

---

# 1. Mô hình tuyến tính

## 1.1. Trực giác

Cho một tập dữ liệu gồm $N$ mẫu, mỗi mẫu có $d$ đặc trưng (feature) $\mathbf{x} = (x_1, \dots, x_d)$ và một nhãn thực $y \in \mathbb{R}$. Ta muốn dự đoán $y$ từ $\mathbf{x}$. Giả định **đơn giản nhất** có thể là: $y$ phụ thuộc **tuyến tính** vào từng đặc trưng. Ví dụ giá nhà tăng tuyến tính theo diện tích, số phòng ngủ, khoảng cách tới trung tâm.

Mỗi đặc trưng $x_j$ được gán một **trọng số (weight)** $w_j$ đo mức độ ảnh hưởng của nó lên đầu ra, cộng thêm một **độ chệch (bias)** $b$ là giá trị nền khi mọi đặc trưng bằng 0:

$$\hat y = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b = \mathbf{w}^\top \mathbf{x} + b$$

## 1.2. Gộp bias vào trọng số

Để toán học gọn gàng, ta **gộp bias vào vector trọng số** bằng một mẹo kinh điển: thêm một đặc trưng giả luôn bằng 1 vào đầu mỗi vector dữ liệu. Đặt

$$\bar{\mathbf{x}} = (1, x_1, \dots, x_d)^\top, \qquad \mathbf{w} = (b, w_1, \dots, w_d)^\top$$

Khi đó $\hat y = \mathbf{w}^\top \bar{\mathbf{x}}$ và bias trở thành một thành phần bình thường của $\mathbf{w}$. Từ đây ta dùng $\mathbf{x}$ ngầm hiểu là đã thêm thành phần 1.

## 1.3. Dạng ma trận

Xếp $N$ mẫu thành các hàng của một **ma trận dữ liệu (design matrix)** $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$ và các nhãn thành vector $\mathbf{y} \in \mathbb{R}^N$:

$$\mathbf{X} = \begin{pmatrix} 1 & x_{1,1} & \cdots & x_{1,d} \\ 1 & x_{2,1} & \cdots & x_{2,d} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{N,1} & \cdots & x_{N,d} \end{pmatrix}, \qquad \mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{pmatrix}$$

Toàn bộ dự đoán trên cả tập dữ liệu viết gọn trong **một phép nhân ma trận**:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$$

---

# 2. Hàm mất mát bình phương

## 2.1. Định nghĩa

Ta cần một thước đo "độ sai" của dự đoán để tối ưu. Lựa chọn tự nhiên là **sai số bình phương trung bình (Mean Squared Error, MSE)** — trung bình của bình phương phần dư (residual) $y_i - \hat y_i$:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2 = \frac{1}{N} \lVert \mathbf{y} - \mathbf{X}\mathbf{w} \rVert_2^2$$

trong đó $\lVert \cdot \rVert_2$ là chuẩn Euclid. Tham khảo thêm về vai trò của hàm mất mát trong [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu).

## 2.2. Vì sao bình phương?

Có ba lý do khiến bình phương được ưa chuộng hơn trị tuyệt đối:

* **Phạt mạnh sai số lớn** — bình phương khiến một phần dư gấp đôi bị phạt gấp bốn, ép mô hình tránh các sai số lớn.
* **Khả vi mọi nơi** — khác với $|y - \hat y|$ gãy tại 0, hàm bình phương trơn nên dùng được giải tích để tìm cực tiểu.
* **Hàm lồi (convex)** — như chứng minh dưới đây, $\mathcal{L}$ là hàm lồi theo $\mathbf{w}$, nên mọi điểm dừng đều là **cực tiểu toàn cục**, không lo kẹt ở cực tiểu địa phương.

## 2.3. Tính lồi

**Mệnh đề.** $\mathcal{L}(\mathbf{w})$ là hàm lồi theo $\mathbf{w}$.

**Chứng minh.** Bỏ qua hằng số $1/N$, ta khai triển $\lVert \mathbf{y} - \mathbf{X}\mathbf{w} \rVert^2 = \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - 2 \mathbf{y}^\top \mathbf{X} \mathbf{w} + \mathbf{y}^\top \mathbf{y}$. Đây là một **đa thức bậc hai** theo $\mathbf{w}$ với ma trận Hessian:

$$\nabla^2 \mathcal{L} = \frac{2}{N} \mathbf{X}^\top \mathbf{X}$$

Với mọi vector $\mathbf{v}$ ta có $\mathbf{v}^\top (\mathbf{X}^\top \mathbf{X}) \mathbf{v} = \lVert \mathbf{X}\mathbf{v} \rVert^2 \ge 0$, nên Hessian là **nửa xác định dương (positive semi-definite)**. Hessian nửa xác định dương là điều kiện đủ để hàm lồi. $\blacksquare$

---

# 3. Phương trình chuẩn (Normal Equation)

## 3.1. Lấy gradient

Vì $\mathcal{L}$ lồi và khả vi, cực tiểu đạt tại điểm gradient triệt tiêu. Viết $\mathcal{L}(\mathbf{w}) = \tfrac{1}{N}(\mathbf{y} - \mathbf{X}\mathbf{w})^\top (\mathbf{y} - \mathbf{X}\mathbf{w})$ và khai triển:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{N}\left( \mathbf{y}^\top \mathbf{y} - 2\mathbf{w}^\top \mathbf{X}^\top \mathbf{y} + \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} \right)$$

Áp dụng các quy tắc đạo hàm ma trận $\nabla_\mathbf{w}(\mathbf{a}^\top \mathbf{w}) = \mathbf{a}$ và $\nabla_\mathbf{w}(\mathbf{w}^\top \mathbf{A} \mathbf{w}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{w}$ (với $\mathbf{A} = \mathbf{X}^\top\mathbf{X}$ đối xứng nên thành $2\mathbf{A}\mathbf{w}$):

$$\nabla_\mathbf{w} \mathcal{L} = \frac{1}{N}\left( -2\mathbf{X}^\top \mathbf{y} + 2 \mathbf{X}^\top \mathbf{X} \mathbf{w} \right) = \frac{2}{N}\left( \mathbf{X}^\top \mathbf{X} \mathbf{w} - \mathbf{X}^\top \mathbf{y} \right)$$

## 3.2. Suy dẫn nghiệm

Đặt gradient bằng 0 (bỏ hệ số $2/N$):

$$\nabla_\mathbf{w} \mathcal{L} = 0 \;\Longrightarrow\; \mathbf{X}^\top \mathbf{X} \mathbf{w} = \mathbf{X}^\top \mathbf{y}$$

Đẳng thức này được gọi là **phương trình chuẩn (normal equation)**. Khi $\mathbf{X}^\top \mathbf{X}$ **khả nghịch**, nhân trái hai vế với $(\mathbf{X}^\top \mathbf{X})^{-1}$:

$$\boxed{\; \mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} \;}$$

Đây là **nghiệm đóng** của hồi quy tuyến tính: chỉ cần một vài phép nhân ma trận và một phép nghịch đảo, không cần lặp. Ma trận $\mathbf{X}^{\dagger} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top$ chính là **giả nghịch đảo Moore–Penrose (pseudo-inverse)** của $\mathbf{X}$.

## 3.3. Vì sao đây là cực tiểu (chứ không phải cực đại)

Điểm dừng vừa tìm là cực tiểu toàn cục vì Hessian $\frac{2}{N}\mathbf{X}^\top \mathbf{X}$ nửa xác định dương (mục 2.3). Khi $\mathbf{X}^\top\mathbf{X}$ khả nghịch, Hessian **xác định dương ngặt**, nên nghiệm là **duy nhất** và là cực tiểu toàn cục đúng nghĩa. $\blacksquare$

---

# 4. Ý nghĩa hình học: chiếu trực giao

## 4.1. Đặt vấn đề

Vector dự đoán $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ luôn là một **tổ hợp tuyến tính các cột** của $\mathbf{X}$, nên nó bị giới hạn trong **không gian cột (column space)** $\text{Col}(\mathbf{X}) \subseteq \mathbb{R}^N$. Nói chung nhãn thật $\mathbf{y}$ **không** nằm trong không gian con này (dữ liệu có nhiễu). Bài toán trở thành: tìm điểm $\hat{\mathbf{y}}$ trong $\text{Col}(\mathbf{X})$ **gần $\mathbf{y}$ nhất** theo khoảng cách Euclid.

## 4.2. Lời giải hình học

Điểm gần nhất trong một không gian con tới một điểm cho trước chính là **hình chiếu trực giao** của điểm đó lên không gian con. Điều kiện chiếu trực giao là **vector phần dư** $\mathbf{y} - \hat{\mathbf{y}}$ phải **vuông góc với mọi cột** của $\mathbf{X}$:

$$\mathbf{X}^\top (\mathbf{y} - \mathbf{X}\mathbf{w}) = \mathbf{0}$$

Khai triển ra đúng là $\mathbf{X}^\top \mathbf{X} \mathbf{w} = \mathbf{X}^\top \mathbf{y}$ — **chính phương trình chuẩn**. Vậy đại số và hình học cho cùng một kết quả: nghiệm bình phương tối thiểu là phép chiếu trực giao của $\mathbf{y}$ lên $\text{Col}(\mathbf{X})$.

> **Trực giác.** Tên gọi "normal equation" đến từ chữ *normal* nghĩa là **pháp tuyến (vuông góc)**: nghiệm tối ưu là nghiệm khiến sai số vuông góc với không gian biểu diễn của mô hình. Ta không thể "đuổi theo" $\mathbf{y}$ ra ngoài $\text{Col}(\mathbf{X})$, nên điều tốt nhất là bỏ đi đúng phần $\mathbf{y}$ mà mô hình không bao giờ giải thích được.

## 4.3. Ma trận chiếu

Thay nghiệm vào ta được $\hat{\mathbf{y}} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y} = \mathbf{H}\mathbf{y}$, với $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top$ là **ma trận mũ (hat matrix)** — một phép chiếu trực giao. Nó thỏa $\mathbf{H}^2 = \mathbf{H}$ (chiếu hai lần vẫn vậy) và $\mathbf{H}^\top = \mathbf{H}$ (đối xứng) — đúng định nghĩa toán học của một phép chiếu trực giao.

---

# 5. Khi $\mathbf{X}^\top \mathbf{X}$ suy biến: Ridge Regression

## 5.1. Vấn đề suy biến

Phương trình chuẩn giả định $\mathbf{X}^\top \mathbf{X}$ khả nghịch. Điều này **thất bại** khi:

* Số đặc trưng **lớn hơn** số mẫu ($d + 1 > N$), khiến các cột phụ thuộc tuyến tính.
* Hai đặc trưng **tương quan cao (multicollinearity)** — ví dụ "diện tích bằng m²" và "diện tích bằng feet²".

Khi đó $\mathbf{X}^\top \mathbf{X}$ **suy biến (singular)** hoặc gần suy biến, nghịch đảo của nó không tồn tại hoặc cực kỳ không ổn định: một thay đổi nhỏ trong dữ liệu gây dao động khổng lồ trong $\mathbf{w}$. Đây cũng là một dạng [quá khớp (overfitting)](#/overfitting).

## 5.2. Lời giải: thêm điều chuẩn

**Hồi quy ridge (ridge regression)** thêm một số hạng phạt chuẩn $\ell_2$ của trọng số vào hàm mất mát:

$$\mathcal{L}_\text{ridge}(\mathbf{w}) = \lVert \mathbf{y} - \mathbf{X}\mathbf{w} \rVert^2 + \lambda \lVert \mathbf{w} \rVert^2$$

với $\lambda > 0$ là hệ số điều chuẩn (regularization). Lấy gradient và cho bằng 0:

$$-2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) + 2\lambda \mathbf{w} = 0 \;\Longrightarrow\; (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})\,\mathbf{w} = \mathbf{X}^\top \mathbf{y}$$

$$\boxed{\; \mathbf{w}^*_\text{ridge} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y} \;}$$

## 5.3. Vì sao luôn nghịch đảo được

**Mệnh đề.** Với $\lambda > 0$, ma trận $\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}$ luôn khả nghịch.

**Chứng minh.** Với mọi $\mathbf{v} \ne \mathbf{0}$:

$$\mathbf{v}^\top (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}) \mathbf{v} = \lVert \mathbf{X}\mathbf{v} \rVert^2 + \lambda \lVert \mathbf{v} \rVert^2 \ge \lambda \lVert \mathbf{v} \rVert^2 > 0$$

Vậy ma trận **xác định dương ngặt**, mọi trị riêng đều dương, nên định thức khác 0 và ma trận khả nghịch. $\blacksquare$ Về bản chất, ridge "nâng đáy" mọi trị riêng của $\mathbf{X}^\top \mathbf{X}$ thêm $\lambda$ — liên hệ trực tiếp với cách phân rã trị riêng/[phân rã giá trị suy biến (SVD)](#/svd) ổn định hóa bài toán nghịch đảo.

---

# 6. Liên hệ với MLE giả định nhiễu Gauss

## 6.1. Mô hình xác suất

Bình phương tối thiểu không phải lựa chọn tùy tiện — nó là hệ quả của một giả định xác suất rõ ràng. Giả sử mỗi nhãn sinh ra từ mô hình tuyến tính cộng **nhiễu Gauss (Gaussian noise)** độc lập:

$$y_i = \mathbf{w}^\top \mathbf{x}_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$$

Khi đó $y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^\top \mathbf{x}_i, \sigma^2)$.

## 6.2. Cực đại hợp lý dẫn về MSE

Hàm hợp lý (likelihood) của toàn bộ dữ liệu (độc lập) là tích các mật độ Gauss. Lấy logarit (xem chi tiết tại [Ước lượng MLE & MAP](#/mle-map)):

$$\log p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = \sum_{i=1}^N \left[ -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - \mathbf{w}^\top \mathbf{x}_i)^2}{2\sigma^2} \right]$$

Các số hạng không chứa $\mathbf{w}$ là hằng số, nên **cực đại hợp lý** tương đương với **cực tiểu tổng bình phương sai số**:

$$\arg\max_\mathbf{w} \log p(\mathbf{y} \mid \mathbf{X}, \mathbf{w}) = \arg\min_\mathbf{w} \sum_{i=1}^N (y_i - \mathbf{w}^\top \mathbf{x}_i)^2$$

> **Kết luận.** Hồi quy bình phương tối thiểu **chính là** ước lượng hợp lý cực đại (MLE) dưới giả định nhiễu Gauss. Hơn nữa, ridge regression tương ứng với ước lượng **MAP** khi đặt một tiên nghiệm Gauss $\mathcal{N}(0, \tau^2 \mathbf{I})$ lên $\mathbf{w}$ — số hạng phạt $\lambda \lVert \mathbf{w} \rVert^2$ chính là log của tiên nghiệm đó, với $\lambda = \sigma^2/\tau^2$.

---

# 7. Giải bằng Gradient Descent

## 7.1. Vì sao cần lặp khi đã có nghiệm đóng

Nghiệm đóng đòi hỏi nghịch đảo ma trận $(d+1)\times(d+1)$ với chi phí $O(d^3)$, cộng chi phí tính $\mathbf{X}^\top\mathbf{X}$ là $O(Nd^2)$. Khi $d$ lớn (hàng triệu đặc trưng) hoặc $N$ khổng lồ, phép nghịch đảo trở nên **bất khả thi** về thời gian lẫn bộ nhớ. Khi đó ta quay về tối ưu lặp bằng [gradient descent](#/ham-mat-mat-toi-uu).

## 7.2. Quy tắc cập nhật

Dùng gradient đã tính ở mục 3.1, bước cập nhật là:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \frac{2}{N} \mathbf{X}^\top (\mathbf{X}\mathbf{w} - \mathbf{y})$$

với $\eta$ là tốc độ học (learning rate). Vì hàm mất mát **lồi**, gradient descent với $\eta$ đủ nhỏ **đảm bảo hội tụ về cực tiểu toàn cục** — một bảo đảm hiếm có trong học máy. Với dữ liệu rất lớn, dùng **stochastic gradient descent (SGD)** trên từng mini-batch.

```python
import numpy as np

def linreg_gd(X, y, lr=0.01, epochs=1000):
    w = np.zeros(X.shape[1])
    N = len(y)
    for _ in range(epochs):
        grad = (2 / N) * X.T @ (X @ w - y)
        w -= lr * grad
    return w

# Nghiem dong de doi chieu:
# w_closed = np.linalg.solve(X.T @ X, X.T @ y)
```

---

# 8. Giả định và hạn chế

## 8.1. Các giả định cốt lõi

| Giả định | Ý nghĩa | Khi vi phạm |
| --- | --- | --- |
| Quan hệ tuyến tính | $\mathbb{E}[y\mid\mathbf{x}]$ tuyến tính theo $\mathbf{x}$ | Mô hình thiếu khớp (underfit) |
| Nhiễu độc lập, đồng phương sai | $\varepsilon_i$ độc lập, cùng $\sigma^2$ | Ước lượng không hiệu quả |
| Nhiễu Gauss | Hợp thức hóa MSE qua MLE | Mất tối ưu thống kê |
| Không đa cộng tuyến | Các cột $\mathbf{X}$ độc lập | $\mathbf{X}^\top\mathbf{X}$ suy biến |

## 8.2. Hạn chế thực tế

* **Chỉ nắm quan hệ tuyến tính** — không mô tả được tương tác phi tuyến (ví dụ quan hệ hình chữ U). Có thể nới rộng bằng đặc trưng đa thức (polynomial features) nhưng dễ dẫn tới quá khớp.
* **Rất nhạy với điểm ngoại lai (outlier)** — vì bình phương phạt mạnh sai số lớn, một vài điểm bất thường có thể kéo lệch toàn bộ đường hồi quy. Khi cần bền vững, dùng hàm mất mát Huber hoặc hồi quy theo trị tuyệt đối $\ell_1$.

---

# 9. Ưu điểm

* **Nghiệm đóng tường minh** — phương trình chuẩn cho lời giải chính xác trong một bước, không cần lặp hay điều chỉnh siêu tham số.
* **Diễn giải được** — mỗi trọng số $w_j$ nói thẳng mức ảnh hưởng của đặc trưng $j$ lên đầu ra, rất quý trong phân tích.
* **Lồi, bảo đảm tối ưu toàn cục** — không có cực tiểu địa phương; gradient descent luôn hội tụ về nghiệm đúng.
* **Nền tảng lý thuyết vững** — là MLE dưới nhiễu Gauss, gắn kết chặt với thống kê cổ điển và là viên gạch cho các mô hình phức tạp hơn.

---

# 10. Nhược điểm

* **Sức biểu diễn yếu** — chỉ là tuyến tính, thiếu khớp với dữ liệu phi tuyến phức tạp.
* **Nhạy outlier và đa cộng tuyến** — như mục 8, cần ridge hoặc tiền xử lý dữ liệu cẩn thận.
* **Nghịch đảo ma trận tốn kém** — $O(d^3)$ không khả thi khi số chiều rất lớn, buộc quay về gradient descent.

---

# 11. Tổng kết

Hồi quy tuyến tính là điểm hội tụ tuyệt đẹp của ba góc nhìn về cùng một bài toán:

* **Giải tích** — cực tiểu hàm mất mát lồi $\frac{1}{N}\lVert \mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2$ cho phương trình chuẩn $\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$.
* **Hình học** — nghiệm là phép chiếu trực giao $\mathbf{y}$ lên không gian cột của $\mathbf{X}$; phần dư vuông góc với mọi đặc trưng.
* **Xác suất** — chính là MLE dưới giả định nhiễu Gauss, và ridge regression là MAP với tiên nghiệm Gauss.

Khi $\mathbf{X}^\top\mathbf{X}$ suy biến, **ridge regression** $(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\top \mathbf{y}$ vừa khôi phục tính khả nghịch vừa chống quá khớp. Khi dữ liệu quá lớn, **gradient descent** thay thế phép nghịch đảo. Dù đơn giản, hồi quy tuyến tính dạy ta gần như mọi khái niệm trung tâm của học máy: hàm mất mát, tối ưu, điều chuẩn, và mối liên hệ giữa tối ưu với thống kê.

> Bài tiếp theo — **Hồi quy logistic (logistic regression)** — mở rộng ý tưởng tuyến tính này sang bài toán **phân loại** bằng cách bọc đầu ra tuyến tính trong một hàm sigmoid. Xem [Logistic Regression](#/logistic-regression).

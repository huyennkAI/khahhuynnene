# Thuật toán Perceptron (Perceptron Learning Algorithm)

> Perceptron là thuật toán phân loại nhị phân **đầu tiên** trong lịch sử học máy (Rosenblatt, 1958). Ý tưởng cực kỳ đơn giản: tìm một **siêu phẳng (hyperplane)** chia mặt phẳng (hay không gian) thành hai nửa, mỗi nửa ứng với một lớp. Khi gặp một điểm bị phân loại sai, ta **xoay nhẹ** siêu phẳng về phía điểm đó. Điều kỳ diệu là: nếu dữ liệu **khả tách tuyến tính (linearly separable)**, quy trình ngây thơ này được **chứng minh** dừng sau hữu hạn bước.
>
> $$\hat y = \operatorname{sgn}(\mathbf{w}^\top \mathbf{x})$$
>
> Đây vừa là viên gạch nền của mạng nơ-ron, vừa là điểm khởi đầu để hiểu vì sao ta cần đến SVM và logistic regression.

---

# 1. Bài toán: phân loại nhị phân khả tách tuyến tính

Cho tập dữ liệu $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ với mỗi điểm $\mathbf{x}_i \in \mathbb{R}^d$ và nhãn $y_i \in \{-1, +1\}$. Mục tiêu: tìm một hàm dự đoán nhãn cho điểm mới.

Ta nói tập dữ liệu **khả tách tuyến tính** nếu tồn tại một siêu phẳng tách hoàn toàn hai lớp — mọi điểm lớp $+1$ nằm về một phía, mọi điểm lớp $-1$ nằm về phía kia. Hình thức hóa: tồn tại $\mathbf{w}$ sao cho

$$y_i (\mathbf{w}^\top \mathbf{x}_i) > 0 \qquad \forall i.$$

Trực giác: $\mathbf{w}^\top \mathbf{x}$ là khoảng cách có dấu (đến hệ số) từ $\mathbf{x}$ tới siêu phẳng $\mathbf{w}^\top \mathbf{x} = 0$. Nếu điểm và nhãn của nó **cùng dấu**, điểm nằm đúng phía — phân loại đúng. Nếu **trái dấu**, điểm bị phân loại sai.

> **Mẹo bias.** Siêu phẳng tổng quát là $\mathbf{w}^\top \mathbf{x} + b = 0$. Ta gộp $b$ vào $\mathbf{w}$ bằng cách thêm một tọa độ hằng số $1$ vào mỗi điểm: $\bar{\mathbf{x}} = (1, x_1, \dots, x_d)$ và $\bar{\mathbf{w}} = (b, w_1, \dots, w_d)$. Khi đó $\bar{\mathbf{w}}^\top \bar{\mathbf{x}} = \mathbf{w}^\top \mathbf{x} + b$. Từ đây ta giả sử bias đã được gộp và viết gọn $\mathbf{w}, \mathbf{x}$.

---

# 2. Mô hình: hàm dấu

Perceptron dự đoán nhãn bằng **dấu** của tổ hợp tuyến tính:

$$\hat y = \operatorname{sgn}(\mathbf{w}^\top \mathbf{x}) = \begin{cases} +1 & \text{nếu } \mathbf{w}^\top \mathbf{x} \ge 0 \\ -1 & \text{nếu } \mathbf{w}^\top \mathbf{x} < 0 \end{cases}$$

Đây là mô hình tuyến tính giống hệt [hồi quy logistic](logistic-regression), khác ở chỗ thay vì đưa $\mathbf{w}^\top \mathbf{x}$ qua hàm sigmoid để ra **xác suất**, perceptron đưa thẳng qua hàm dấu để ra **nhãn cứng** $\pm 1$. Mặt ra quyết định (decision boundary) là siêu phẳng $\mathbf{w}^\top \mathbf{x} = 0$, vuông góc với vector trọng số $\mathbf{w}$.

Một điểm $\mathbf{x}_i$ bị **phân loại sai** khi và chỉ khi $\hat y_i \ne y_i$, tương đương

$$y_i (\mathbf{w}^\top \mathbf{x}_i) < 0.$$

---

# 3. Hàm mất mát perceptron

Câu hỏi tự nhiên: dùng hàm mất mát nào để tối ưu $\mathbf{w}$? Một lựa chọn ngây thơ là **đếm số điểm sai** (0-1 loss). Nhưng hàm này là **bậc thang**, đạo hàm bằng 0 hầu khắp nơi — không thể dùng gradient descent (xem [hàm mất mát & tối ưu hóa](ham-mat-mat-toi-uu)).

Perceptron thay bằng một hàm khả vi (gần như khắp nơi) chỉ **phạt điểm sai** theo mức độ sai:

$$J(\mathbf{w}) = \sum_{i \in \mathcal{M}(\mathbf{w})} -y_i (\mathbf{w}^\top \mathbf{x}_i)$$

trong đó $\mathcal{M}(\mathbf{w})$ là tập các điểm **đang bị phân loại sai** với trọng số $\mathbf{w}$ hiện tại.

Ý nghĩa từng số hạng: với điểm sai, $y_i(\mathbf{w}^\top \mathbf{x}_i) < 0$, nên $-y_i(\mathbf{w}^\top \mathbf{x}_i) > 0$ — đóng góp **dương** vào mất mát, và càng sai nặng (càng xa siêu phẳng về phía sai) thì phạt càng lớn. Điểm đúng **không** đóng góp gì. Khi mọi điểm đều đúng, $\mathcal{M} = \varnothing$ và $J(\mathbf{w}) = 0$ — giá trị nhỏ nhất có thể (vì $J \ge 0$ luôn).

> **Vì sao không phạt điểm đúng?** Nếu cộng cả $-y_i(\mathbf{w}^\top \mathbf{x}_i)$ của điểm đúng (số hạng **âm**), mô hình sẽ bị "thưởng" vô hạn khi đẩy các điểm đúng ra xa, làm hàm mất mát không bị chặn dưới. Việc chỉ phạt điểm sai khiến $0$ trở thành đáy đúng nghĩa.

---

# 4. Quy tắc cập nhật: suy dẫn từ gradient

Ta tối ưu $J(\mathbf{w})$ bằng **gradient descent ngẫu nhiên (SGD)**: mỗi bước chỉ xét **một** điểm đang bị sai $(\mathbf{x}_i, y_i) \in \mathcal{M}$. Mất mát của riêng điểm đó là

$$J_i(\mathbf{w}) = -y_i (\mathbf{w}^\top \mathbf{x}_i).$$

Lấy gradient theo $\mathbf{w}$ (lưu ý $\nabla_{\mathbf{w}} (\mathbf{w}^\top \mathbf{x}_i) = \mathbf{x}_i$):

$$\nabla_{\mathbf{w}} J_i(\mathbf{w}) = -y_i \mathbf{x}_i.$$

Áp dụng bước cập nhật $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} J_i$ với tốc độ học $\eta = 1$:

$$\boxed{\; \mathbf{w} \leftarrow \mathbf{w} + y_i \mathbf{x}_i \;}$$

Đây chính là **quy tắc học perceptron** kinh điển. Nó chỉ kích hoạt khi gặp điểm sai; gặp điểm đúng thì gradient bằng $\mathbf{0}$, $\mathbf{w}$ giữ nguyên. Vì $\eta$ chỉ thay đổi **độ dài** chứ không đổi **hướng** của siêu phẳng tại mỗi bước (với một điểm), người ta thường cố định $\eta = 1$.

---

# 5. Trực giác hình học: xoay siêu phẳng về phía điểm sai

Vì sao cộng $y_i \mathbf{x}_i$ lại sửa được lỗi? Xét điểm sai có $y_i = +1$ nhưng bị dự đoán $-1$, tức $\mathbf{w}^\top \mathbf{x}_i < 0$. Sau cập nhật $\mathbf{w}' = \mathbf{w} + \mathbf{x}_i$, tích vô hướng mới tại chính điểm đó là:

$$\mathbf{w}'^\top \mathbf{x}_i = (\mathbf{w} + \mathbf{x}_i)^\top \mathbf{x}_i = \mathbf{w}^\top \mathbf{x}_i + \|\mathbf{x}_i\|^2.$$

Ta cộng thêm một lượng **dương** $\|\mathbf{x}_i\|^2 > 0$, kéo $\mathbf{w}'^\top \mathbf{x}_i$ **tăng lên** — tiến về phía $\ge 0$, tức về phía phân loại **đúng**. Với $y_i = -1$, ta trừ đi $\mathbf{x}_i$ và kéo tích vô hướng giảm xuống, cũng về phía đúng.

Hình học: $\mathbf{w}$ là pháp tuyến của siêu phẳng. Cộng $y_i \mathbf{x}_i$ làm pháp tuyến **nghiêng về phía** điểm sai (nếu nhãn $+1$) hoặc nghiêng ra xa (nếu nhãn $-1$), tức **xoay siêu phẳng** sao cho điểm sai có cơ hội rơi về đúng phía. Cập nhật không đảm bảo điểm đó lập tức đúng, nhưng đảm bảo "kéo theo hướng tốt".

---

# 6. Định lý hội tụ của Perceptron

Đây là kết quả lý thuyết đẹp nhất của perceptron, do Novikoff (1962) chứng minh.

> **Định lý (Perceptron Convergence).** Nếu dữ liệu **khả tách tuyến tính**, thuật toán PLA sẽ tìm được một $\mathbf{w}$ tách đúng toàn bộ dữ liệu sau **hữu hạn** lần cập nhật.

## 6.1. Giả thiết

Vì dữ liệu khả tách, tồn tại vector lời giải $\mathbf{w}^*$ với $\|\mathbf{w}^*\| = 1$ và một **biên (margin)** $\gamma > 0$:

$$y_i (\mathbf{w}^{*\top} \mathbf{x}_i) \ge \gamma \qquad \forall i.$$

Gọi $R = \max_i \|\mathbf{x}_i\|$ là **bán kính** bao quanh dữ liệu. Giả sử $\mathbf{w}_0 = \mathbf{0}$, và $\mathbf{w}_k$ là trọng số sau $k$ lần cập nhật.

## 6.2. Ý chứng minh: chặn trên số lần cập nhật

Ý tưởng: theo dõi hai đại lượng — hình chiếu $\mathbf{w}_k^\top \mathbf{w}^*$ (tăng tuyến tính) và độ dài $\|\mathbf{w}_k\|$ (tăng chậm hơn, chỉ như $\sqrt{k}$). Hai tốc độ này không thể cùng tồn tại mãi.

**Bước 1 — hình chiếu lên $\mathbf{w}^*$ tăng ít nhất $\gamma$ mỗi bước.** Mỗi cập nhật dùng một điểm sai $(\mathbf{x}_i, y_i)$:

$$\mathbf{w}_{k}^\top \mathbf{w}^* = (\mathbf{w}_{k-1} + y_i \mathbf{x}_i)^\top \mathbf{w}^* = \mathbf{w}_{k-1}^\top \mathbf{w}^* + y_i (\mathbf{x}_i^\top \mathbf{w}^*) \ge \mathbf{w}_{k-1}^\top \mathbf{w}^* + \gamma.$$

Quy nạp từ $\mathbf{w}_0 = \mathbf{0}$:

$$\mathbf{w}_k^\top \mathbf{w}^* \ge k\gamma.$$

**Bước 2 — độ dài bình phương tăng nhiều nhất $R^2$ mỗi bước.** Vì điểm $\mathbf{x}_i$ đang bị **sai** nên $y_i(\mathbf{w}_{k-1}^\top \mathbf{x}_i) < 0$:

$$\|\mathbf{w}_k\|^2 = \|\mathbf{w}_{k-1} + y_i \mathbf{x}_i\|^2 = \|\mathbf{w}_{k-1}\|^2 + 2 y_i (\mathbf{w}_{k-1}^\top \mathbf{x}_i) + \|\mathbf{x}_i\|^2 \le \|\mathbf{w}_{k-1}\|^2 + \|\mathbf{x}_i\|^2.$$

Số hạng giữa **âm** nên bị bỏ đi an toàn; còn $\|\mathbf{x}_i\|^2 \le R^2$. Quy nạp:

$$\|\mathbf{w}_k\|^2 \le k R^2.$$

**Bước 3 — ép hai chặn.** Theo bất đẳng thức Cauchy–Schwarz và $\|\mathbf{w}^*\| = 1$:

$$k\gamma \le \mathbf{w}_k^\top \mathbf{w}^* \le \|\mathbf{w}_k\|\,\|\mathbf{w}^*\| = \|\mathbf{w}_k\| \le \sqrt{k}\, R.$$

Từ $k\gamma \le \sqrt{k}\,R$ suy ra $\sqrt{k} \le R/\gamma$, tức:

$$\boxed{\; k \le \frac{R^2}{\gamma^2} \;}$$

Số lần cập nhật bị chặn trên bởi $(R/\gamma)^2$ — **hữu hạn**, không phụ thuộc số chiều $d$ hay số điểm $N$. $\blacksquare$

> **Diễn giải.** Biên $\gamma$ càng lớn (dữ liệu càng "tách rời") thì PLA hội tụ càng nhanh; dữ liệu càng "chật" (các lớp gần nhau, $\gamma$ nhỏ) thì cần nhiều bước hơn. Đây cũng là manh mối đầu tiên cho thấy **biên lớn là tốt** — ý tưởng được [SVM](svm) khai thác triệt để.

---

# 7. Cài đặt

```python
import numpy as np

def perceptron(X, y, max_iter=1000):
    # X đã gộp cột bias = 1; y in {-1, +1}
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi) <= 0:          # phân loại sai
                w = w + yi * xi             # w <- w + y_i x_i
                errors += 1
        if errors == 0:                     # không còn điểm sai -> dừng
            break
    return w
```

Vòng lặp duyệt qua dữ liệu, cập nhật ngay khi gặp điểm sai. Nếu một lượt duyệt không còn lỗi nào, thuật toán dừng — theo định lý mục 6, điều này chắc chắn xảy ra với dữ liệu khả tách.

---

# 8. Hạn chế

* **Không hội tụ nếu dữ liệu không khả tách.** Nếu không tồn tại siêu phẳng tách hoàn hảo (luôn còn ít nhất một điểm sai), $\mathcal{M}$ không bao giờ rỗng, $\mathbf{w}$ dao động **mãi mãi** không dừng. PLA không có cơ chế "chấp nhận một ít lỗi". Đây là động lực cho các hàm mất mát mềm hơn (hinge loss, cross-entropy).
* **Nghiệm không duy nhất.** Khi dữ liệu khả tách, có **vô số** siêu phẳng tách đúng, và PLA trả về một cái **bất kỳ** trong số đó tùy thứ tự duyệt và khởi tạo — kể cả những siêu phẳng "sát rạt" một lớp, tổng quát hóa kém. [SVM](svm) khắc phục bằng cách chọn siêu phẳng có **biên lớn nhất**, nghiệm là **duy nhất** và ổn định.
* **Không cho xác suất.** Đầu ra là nhãn cứng $\pm 1$, không cho biết mô hình "chắc chắn" đến đâu. [Hồi quy logistic](logistic-regression) thay hàm dấu bằng sigmoid để xuất **xác suất** $P(y=1\mid\mathbf{x})$, vừa diễn giải được vừa cho gradient mượt ở mọi điểm.

---

# 9. Vai trò lịch sử: tiền thân của nơ-ron

Perceptron chính là **một nơ-ron nhân tạo** đơn lẻ: nhận đầu vào $\mathbf{x}$, tính tổng có trọng số $\mathbf{w}^\top \mathbf{x}$, rồi cho qua một **hàm kích hoạt (activation)** — ở đây là hàm dấu. Đây đúng là cấu trúc một đơn vị trong [mạng nơ-ron](mang-no-ron).

Giới hạn nổi tiếng của perceptron đơn (Minsky & Papert, 1969): nó **không** giải được bài toán **XOR** vì XOR không khả tách tuyến tính. Phát hiện này từng gây ra "mùa đông AI" đầu tiên. Lời giải đến nhiều năm sau: **xếp chồng nhiều lớp perceptron** (multi-layer perceptron) cùng hàm kích hoạt phi tuyến khả vi, huấn luyện bằng lan truyền ngược — chính là mạng nơ-ron sâu hiện đại.

---

# 10. Tổng kết

Perceptron là mô hình tuyến tính tối giản: dự đoán bằng $\hat y = \operatorname{sgn}(\mathbf{w}^\top \mathbf{x})$, học bằng cách lặp lại một quy tắc duy nhất — gặp điểm sai thì $\mathbf{w} \leftarrow \mathbf{w} + y_i \mathbf{x}_i$. Quy tắc này không phải mẹo tùy hứng mà là **SGD trên hàm mất mát chỉ phạt điểm sai** $\sum_{\text{sai}} -y_i \mathbf{w}^\top \mathbf{x}_i$.

Giá trị lý thuyết cốt lõi là **định lý hội tụ**: với dữ liệu khả tách tuyến tính, số lần cập nhật bị chặn bởi $(R/\gamma)^2$ — hữu hạn và không phụ thuộc số chiều. Điểm yếu của perceptron lại mở đường cho các mô hình kế tiếp: không hội tụ khi dữ liệu nhiễu và nghiệm không duy nhất dẫn tới [SVM](svm) (biên lớn nhất); không cho xác suất dẫn tới [hồi quy logistic](logistic-regression); và cấu trúc "tổng trọng số + kích hoạt" của nó chính là tế bào nền của [mạng nơ-ron](mang-no-ron).

> Hiểu perceptron là hiểu hạt nhân của học máy tuyến tính — và là bước đầu tiên trên con đường dẫn đến học sâu.

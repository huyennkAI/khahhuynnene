# SVM lề mềm & Kernel SVM

> [SVM](#/svm) cơ bản (hard-margin) tìm siêu phẳng tách hai lớp với **lề rộng nhất**, nhưng nó đòi hỏi dữ liệu **khả tách tuyến tính tuyệt đối**. Thực tế dữ liệu luôn có **nhiễu** và thường **không tách được bằng đường thẳng**. Bài này nới lỏng SVM theo hai hướng bổ trợ nhau: cho phép phạm lỗi có kiểm soát (**lề mềm / soft margin**), và nâng dữ liệu lên không gian nhiều chiều để tách tuyến tính ở đó (**kernel trick**) mà không phải trả giá tính toán.

---

# 1. Vì sao hard-margin chưa đủ

Hard-margin SVM giải bài toán: tìm $\mathbf{w}, b$ cực đại lề với ràng buộc **mọi điểm** đều nằm đúng phía và ngoài lề,

$$\min_{\mathbf{w}, b} \; \frac12 \lVert \mathbf{w} \rVert^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \;\; \forall i$$

Ràng buộc cứng $\ge 1$ này có hai điểm yếu nghiêm trọng trong thực tế:

* **Không khả tách tuyến tính.** Nếu không tồn tại siêu phẳng nào tách hoàn toàn hai lớp, **không có** nghiệm thỏa mãn mọi ràng buộc — bài toán **vô nghiệm (infeasible)**.
* **Nhạy với nhiễu (noise) và điểm ngoại lai (outlier).** Chỉ **một** điểm bị gán nhãn sai hoặc nằm lệch sâu vào vùng lớp kia cũng buộc lề phải thu hẹp cực mạnh hoặc xoay siêu phẳng lệch hẳn đi, làm mô hình tổng quát hóa kém (liên quan trực tiếp tới [overfitting](#/overfitting)).

Trực giác: hard-margin coi mọi điểm là "thiêng liêng". Ta cần một phiên bản **rộng lượng** hơn — chấp nhận một vài điểm vi phạm, miễn là phần lớn dữ liệu vẫn được tách tốt với lề rộng.

---

# 2. Biến nới lỏng và bài toán lề mềm

## 2.1. Ý tưởng: cho phép vi phạm có trả giá

Với mỗi điểm $\mathbf{x}_i$ ta đưa vào một **biến nới lỏng (slack variable)** $\xi_i \ge 0$ đo **mức độ vi phạm** ràng buộc lề. Ràng buộc cứng được nới thành:

$$y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i, \qquad \xi_i \ge 0$$

Ý nghĩa hình học của $\xi_i$:

* $\xi_i = 0$: điểm nằm đúng phía và **ngoài hoặc trên** lề — hoàn toàn "ngoan".
* $0 < \xi_i \le 1$: điểm lọt **vào trong** lề nhưng vẫn được **phân loại đúng**.
* $\xi_i > 1$: điểm **vượt sang phía sai** — bị phân loại sai.

Ta không muốn các $\xi_i$ lớn tùy tiện (nếu không, đặt $\xi_i$ vô cùng lớn thì ràng buộc nào cũng thỏa). Vì vậy ta **phạt** tổng các vi phạm trong hàm mục tiêu.

## 2.2. Bài toán tối ưu soft-margin

Kết hợp mục tiêu lề rộng (cực tiểu $\tfrac12 \lVert \mathbf{w} \rVert^2$) với mục tiêu ít vi phạm (cực tiểu $\sum_i \xi_i$):

$$\boxed{\;\min_{\mathbf{w}, b, \boldsymbol{\xi}} \; \frac12 \lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^{n} \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i, \;\; \xi_i \ge 0 \;}$$

Đây vẫn là một bài toán **lồi (convex)** — cụ thể là quy hoạch toàn phương (quadratic programming) — nên có nghiệm tối ưu toàn cục duy nhất theo $\mathbf{w}$ (xem [convex optimization](#/convex-optimization)).

## 2.3. Ý nghĩa siêu tham số $C$

Hằng số $C > 0$ điều khiển **sự đánh đổi (trade-off)** giữa hai mục tiêu mâu thuẫn:

* **$C$ lớn** — phạt vi phạm rất nặng ⇒ mô hình cố tách đúng gần như mọi điểm ⇒ lề **hẹp**, sát hard-margin. Rủi ro **overfitting** với nhiễu.
* **$C$ nhỏ** — phạt vi phạm nhẹ ⇒ chấp nhận nhiều điểm lọt lề ⇒ lề **rộng**, mô hình **mượt** hơn. Rủi ro **underfitting** nếu quá nhỏ.

Như vậy $C$ chính là một tham số **điều chuẩn (regularization)**: nó cân bằng giữa **độ phức tạp mô hình** (lề rộng = $\lVert\mathbf{w}\rVert$ nhỏ = đơn giản) và **lỗi trên tập huấn luyện**. Chọn $C$ thường bằng kiểm định chéo (cross-validation), đúng tinh thần kiểm soát [overfitting](#/overfitting).

---

# 3. Dạng tương đương: Hinge Loss

## 3.1. Khử biến nới lỏng

Với mỗi điểm, ràng buộc và mục tiêu cho phép ta xác định $\xi_i$ tối ưu một cách tường minh. Vì ta cực tiểu $\sum \xi_i$ và cần $\xi_i \ge 1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)$ đồng thời $\xi_i \ge 0$, giá trị nhỏ nhất hợp lệ là:

$$\xi_i = \max\!\big(0,\; 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\big)$$

Đặt $f(\mathbf{x}_i) = \mathbf{w}^\top \mathbf{x}_i + b$. Thay $\xi_i$ vào hàm mục tiêu, bài toán có ràng buộc trở thành bài toán **không ràng buộc**:

$$\min_{\mathbf{w}, b} \; \sum_{i=1}^{n} \max\!\big(0,\; 1 - y_i f(\mathbf{x}_i)\big) \;+\; \frac{\lambda}{2} \lVert \mathbf{w} \rVert^2$$

trong đó $\lambda = 1/C$ (chỉ là đổi tỉ lệ giữa hai số hạng). Số hạng đầu là tổng **hàm mất mát bản lề (hinge loss)**, số hạng sau là điều chuẩn $L_2$.

## 3.2. Đọc hiểu hinge loss

Hàm $\ell(z) = \max(0, 1 - z)$ với $z = y_i f(\mathbf{x}_i)$ là **biên (margin)** của điểm:

* $z \ge 1$: điểm phân loại đúng và ngoài lề ⇒ mất mát **bằng 0**. SVM **không thưởng** cho việc đẩy điểm ra xa hơn nữa.
* $z < 1$: mất mát tăng **tuyến tính** theo mức vi phạm, kể cả khi $0 < z < 1$ (đúng nhưng trong lề) lẫn $z < 0$ (sai).

Góc nhìn này cho thấy SVM lề mềm **chính là** một mô hình tuyến tính tối thiểu hóa hinge loss có điều chuẩn — cùng khuôn khổ "[mất mát + điều chuẩn](#/ham-mat-mat-toi-uu)" như hồi quy logistic (chỉ khác hàm mất mát). Đây cũng là lý do SVM huấn luyện được bằng gradient descent (qua subgradient của hinge loss).

---

# 4. Kernel Trick: tách phi tuyến mà không tốn kém

## 4.1. Nâng chiều bằng ánh xạ đặc trưng

Soft-margin xử lý nhiễu, nhưng nếu ranh giới thật sự **cong** (ví dụ một lớp nằm trong vòng tròn bao quanh lớp kia) thì siêu phẳng tuyến tính không bao giờ tách tốt. Ý tưởng: ánh xạ dữ liệu qua một hàm $\phi(\mathbf{x})$ lên **không gian đặc trưng (feature space)** nhiều chiều hơn, nơi dữ liệu **trở nên khả tách tuyến tính**.

Ví dụ kinh điển trong 2D: với $\mathbf{x} = (x_1, x_2)$, đặt

$$\phi(\mathbf{x}) = (x_1^2, \; \sqrt{2}\, x_1 x_2, \; x_2^2)$$

Một đường tròn trong không gian gốc trở thành một mặt phẳng trong không gian 3 chiều mới. Tổng quát, càng nhiều chiều thì càng dễ tách — nhưng tính toán trực tiếp $\phi(\mathbf{x})$ ở không gian rất cao chiều là **cực kỳ tốn kém**, thậm chí **vô hạn chiều**.

## 4.2. Mẹo: chỉ cần tích vô hướng

Chìa khóa nằm ở **bài toán đối ngẫu (dual problem)** của SVM (suy ra từ điều kiện KKT trong [convex optimization](#/convex-optimization)). Bài toán đối ngẫu của soft-margin SVM là:

$$\max_{\boldsymbol{\alpha}} \; \sum_{i=1}^{n} \alpha_i - \frac12 \sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j \, y_i y_j \, \mathbf{x}_i^\top \mathbf{x}_j \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \;\; \sum_i \alpha_i y_i = 0$$

Điều then chốt: dữ liệu **chỉ xuất hiện dưới dạng tích vô hướng** $\mathbf{x}_i^\top \mathbf{x}_j$. Khi nâng chiều, ta thay $\mathbf{x}_i^\top \mathbf{x}_j$ bằng $\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$. Nếu tồn tại một hàm tính trực tiếp tích vô hướng đó **mà không cần** tính $\phi$ rõ ràng, ta được lợi cả hai đầu. Hàm đó gọi là **hàm kernel (kernel function)**:

$$K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top \phi(\mathbf{z})$$

Đây chính là **kernel trick**: thay mọi $\langle \mathbf{x}_i, \mathbf{x}_j \rangle$ bằng $K(\mathbf{x}_i, \mathbf{x}_j)$, ta huấn luyện SVM trong không gian đặc trưng cao chiều **với chi phí của không gian gốc**. Hàm quyết định cũng chỉ phụ thuộc kernel:

$$f(\mathbf{x}) = \sum_{i} \alpha_i y_i \, K(\mathbf{x}_i, \mathbf{x}) + b$$

Tổng chỉ chạy trên các **vector hỗ trợ (support vector)** — những điểm có $\alpha_i > 0$.

---

# 5. Các hàm kernel phổ biến

| Kernel | Công thức $K(\mathbf{x}, \mathbf{z})$ | Đặc điểm |
| --- | --- | --- |
| **Tuyến tính (linear)** | $\mathbf{x}^\top \mathbf{z}$ | $\phi$ = đồng nhất; chính là SVM gốc, hợp dữ liệu nhiều chiều/thưa |
| **Đa thức (polynomial)** | $(\mathbf{x}^\top \mathbf{z} + r)^d$ | Tạo đặc trưng tương tác bậc tới $d$; $r, d$ là siêu tham số |
| **RBF / Gauss** | $\exp(-\gamma \lVert \mathbf{x} - \mathbf{z} \rVert^2)$ | Không gian đặc trưng **vô hạn chiều**; rất linh hoạt, mặc định phổ biến nhất |
| **Sigmoid** | $\tanh(\gamma\, \mathbf{x}^\top \mathbf{z} + r)$ | Gợi nhớ mạng nơ-ron; không phải lúc nào cũng là kernel hợp lệ |

## 5.1. Kernel đa thức

$(\mathbf{x}^\top \mathbf{z} + r)^d$ ngầm tương ứng với một ánh xạ $\phi$ gồm mọi đơn thức (monomial) bậc tới $d$ của các thành phần. Bậc $d$ càng cao, ranh giới càng uốn lượn; $r$ điều khiển trọng số giữa các bậc thấp và cao.

## 5.2. Kernel RBF (Gaussian) — vì sao vô hạn chiều

RBF là kernel được dùng nhiều nhất. Để thấy nó ứng với **không gian đặc trưng vô hạn chiều**, khai triển (với $\gamma = \tfrac12$, một chiều cho gọn) số hạng $\exp(x z)$ thành chuỗi Taylor:

$$\exp(x z) = \sum_{k=0}^{\infty} \frac{(x z)^k}{k!} = \sum_{k=0}^{\infty} \frac{x^k}{\sqrt{k!}} \cdot \frac{z^k}{\sqrt{k!}}$$

Đây là tích vô hướng của hai vector **vô hạn thành phần** $\phi(x) = \big(1, x, \tfrac{x^2}{\sqrt{2!}}, \dots\big)$. Tham số $\gamma$ điều khiển "độ rộng": $\gamma$ lớn ⇒ ảnh hưởng mỗi điểm rất cục bộ ⇒ ranh giới phức tạp (dễ overfitting); $\gamma$ nhỏ ⇒ mượt hơn. Cùng với $C$, cặp $(C, \gamma)$ là hai núm chỉnh chính của SVM-RBF.

---

# 6. Điều kiện Mercer: khi nào một hàm là kernel hợp lệ

Không phải hàm $K$ nào cũng tương ứng với một ánh xạ $\phi$. Một hàm đối xứng $K$ là kernel hợp lệ khi và chỉ khi nó thỏa **điều kiện Mercer (Mercer's condition)**: với **mọi** tập điểm hữu hạn $\{\mathbf{x}_1, \dots, \mathbf{x}_n\}$, **ma trận Gram (Gram matrix)** $\mathbf{K}$ với $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ phải **nửa xác định dương (positive semi-definite)**:

$$\mathbf{c}^\top \mathbf{K} \mathbf{c} \ge 0, \quad \forall\, \mathbf{c} \in \mathbb{R}^n$$

Trực giác: ma trận Gram chứa toàn các tích vô hướng $\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$, nên nó **phải** là ma trận tích vô hướng — và mọi ma trận như vậy đều nửa xác định dương. Khi Mercer thỏa mãn, sự tồn tại của $\phi$ được **bảo đảm** dù ta không bao giờ viết nó ra. Điều kiện này cũng giữ cho bài toán đối ngẫu vẫn **lồi**. (Kernel sigmoid không thỏa Mercer với mọi tham số, nên đôi khi nó không phải kernel "thật".)

---

# 7. Ưu điểm

* **Xử lý nhiễu và dữ liệu không tách được** — biến nới lỏng cho phép vi phạm có kiểm soát, $C$ điều chỉnh đánh đổi lề–lỗi một cách trực tiếp.
* **Tách phi tuyến với chi phí thấp** — kernel trick cho phép làm việc ở không gian cao (thậm chí vô hạn) chiều mà chỉ tốn như không gian gốc.
* **Bài toán lồi** — có nghiệm tối ưu toàn cục, không kẹt ở cực tiểu địa phương như mạng nơ-ron.
* **Nghiệm thưa (sparse)** — hàm quyết định chỉ phụ thuộc các vector hỗ trợ, gọn và dễ diễn giải.

---

# 8. Nhược điểm

* **Tốn kém với dữ liệu lớn** — ma trận Gram có kích thước $n \times n$, huấn luyện kernel SVM thường $O(n^2)$–$O(n^3)$, khó mở rộng cho hàng triệu mẫu.
* **Nhạy với siêu tham số** — chất lượng phụ thuộc mạnh vào $(C, \gamma, d, r)$; cần kiểm định chéo công phu.
* **Chọn kernel là nghệ thuật** — không có quy tắc tổng quát biết trước kernel nào hợp dữ liệu.
* **Không trực tiếp cho xác suất** — đầu ra là khoảng cách dấu, cần hiệu chỉnh (calibration) thêm để có xác suất.

---

# 9. Tổng kết

SVM lề mềm và Kernel SVM nới lỏng [hard-margin SVM](#/svm) theo hai trục độc lập. **Lề mềm** dùng biến nới lỏng $\xi_i$ để chấp nhận vi phạm có trả giá, biến SVM thành bài toán cực tiểu **hinge loss + điều chuẩn**, với $C$ là núm cân bằng giữa lề rộng và lỗi — đúng tinh thần kiểm soát [overfitting](#/overfitting). **Kernel trick** khai thác việc bài toán đối ngẫu chỉ cần **tích vô hướng**, nên ta thay nó bằng hàm kernel $K(\mathbf{x}, \mathbf{z}) = \phi(\mathbf{x})^\top \phi(\mathbf{z})$ để tách phi tuyến mà không phải tính $\phi$ — miễn là $K$ thỏa điều kiện Mercer.

Ghép cả hai, ta được một bộ phân loại vừa **chống nhiễu**, vừa **mềm dẻo phi tuyến**, vẫn giữ tính **lồi** đẹp đẽ của tối ưu — một trong những thuật toán cổ điển mạnh và được dùng rộng rãi nhất trước kỷ nguyên học sâu.

> Mẹo thực hành: bắt đầu với kernel **RBF**, dò lưới $(C, \gamma)$ bằng kiểm định chéo; nếu dữ liệu rất nhiều chiều và thưa (ví dụ văn bản), kernel **tuyến tính** thường đã đủ tốt và nhanh hơn nhiều.

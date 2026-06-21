# Máy vector hỗ trợ (Support Vector Machine)

> Máy vector hỗ trợ (support vector machine, SVM) tìm siêu phẳng phân tách hai lớp dữ liệu sao cho **lề (margin)** — khoảng cách từ điểm gần nhất tới siêu phẳng — là **lớn nhất**. Trong số vô hạn các siêu phẳng tách đúng dữ liệu, SVM chọn ra **một** đường duy nhất, "an toàn" nhất:
>
> $$\min_{\mathbf{w}, b} \; \frac{1}{2}\lVert \mathbf{w} \rVert^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \; \forall i$$
>
> Đây là một bài toán **lồi (convex)** có nghiệm toàn cục duy nhất, và khi chuyển sang dạng đối ngẫu (dual) nó chỉ phụ thuộc vào **tích vô hướng** giữa các điểm dữ liệu — chính chi tiết này mở đường cho **kernel**, biến SVM thành một trong những bộ phân loại mạnh nhất trước kỷ nguyên học sâu.

---

# 1. Động cơ: siêu phẳng nào là tốt nhất?

Xét bài toán phân loại nhị phân với dữ liệu $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, trong đó $\mathbf{x}_i \in \mathbb{R}^d$ và nhãn $y_i \in \{-1, +1\}$. Một **bộ phân loại tuyến tính (linear classifier)** xác định bởi một siêu phẳng:

$$\mathbf{w}^\top \mathbf{x} + b = 0$$

và dự đoán nhãn theo dấu: $\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b)$.

Giả sử dữ liệu **khả tách tuyến tính (linearly separable)** — tồn tại siêu phẳng tách đúng hai lớp. Vấn đề là có **vô hạn** siêu phẳng như vậy. Thuật toán [perceptron](perceptron) dừng lại ở **bất kỳ** siêu phẳng nào tách đúng, phụ thuộc vào thứ tự duyệt dữ liệu và điểm khởi tạo. Nhưng trực giác cho thấy không phải mọi đường tách đều tốt như nhau: một đường đi **sát rạt** các điểm dữ liệu rất dễ phân loại sai khi gặp điểm mới có nhiễu nhỏ.

> Trực giác cốt lõi của SVM: hãy chọn siêu phẳng nằm **xa nhất có thể** khỏi cả hai lớp — tức là tối đa hóa "vùng đệm" trống quanh đường biên. Đường an toàn nhất là đường có khoảng trống hai bên rộng nhất.

"Vùng đệm" đó chính là **lề (margin)**, và SVM còn được gọi là bộ phân loại **lề cực đại (maximum margin classifier)**.

---

# 2. Khoảng cách từ một điểm tới siêu phẳng

Để định lượng lề, ta cần công thức khoảng cách. Cho siêu phẳng $\mathbf{w}^\top \mathbf{x} + b = 0$ và một điểm $\mathbf{x}_0$.

**Mệnh đề.** Khoảng cách từ $\mathbf{x}_0$ tới siêu phẳng là

$$\operatorname{dist}(\mathbf{x}_0) = \frac{\lvert \mathbf{w}^\top \mathbf{x}_0 + b \rvert}{\lVert \mathbf{w} \rVert}$$

**Chứng minh.** Vector $\mathbf{w}$ **vuông góc** với siêu phẳng: với hai điểm bất kỳ $\mathbf{x}_a, \mathbf{x}_b$ trên siêu phẳng, $\mathbf{w}^\top(\mathbf{x}_a - \mathbf{x}_b) = (-b) - (-b) = 0$, nên $\mathbf{w}$ trực giao với mọi hướng nằm trong siêu phẳng. Gọi $\mathbf{p}$ là hình chiếu vuông góc của $\mathbf{x}_0$ lên siêu phẳng. Khi đó $\mathbf{x}_0 - \mathbf{p} = t\,\dfrac{\mathbf{w}}{\lVert \mathbf{w} \rVert}$ với $\lvert t \rvert = \operatorname{dist}(\mathbf{x}_0)$. Vì $\mathbf{p}$ nằm trên siêu phẳng nên $\mathbf{w}^\top \mathbf{p} + b = 0$, suy ra:

$$\mathbf{w}^\top \mathbf{x}_0 + b = \mathbf{w}^\top \mathbf{p} + b + t\,\frac{\mathbf{w}^\top \mathbf{w}}{\lVert \mathbf{w} \rVert} = t\,\lVert \mathbf{w} \rVert$$

Lấy trị tuyệt đối rồi chia cho $\lVert \mathbf{w} \rVert$ cho ngay kết quả. $\blacksquare$

Vì dữ liệu được phân loại đúng khi $y_i(\mathbf{w}^\top \mathbf{x}_i + b) > 0$, ta có thể bỏ trị tuyệt đối: với điểm phân loại đúng, $\lvert \mathbf{w}^\top \mathbf{x}_i + b \rvert = y_i(\mathbf{w}^\top \mathbf{x}_i + b)$. Vậy khoảng cách có dấu (đảm bảo dương) là:

$$\operatorname{dist}(\mathbf{x}_i) = \frac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\lVert \mathbf{w} \rVert}$$

---

# 3. Định nghĩa lề và bài toán tối ưu thô

**Lề (margin)** của một siêu phẳng được định nghĩa là khoảng cách từ **điểm gần nhất** tới nó:

$$\text{margin} = \min_{i} \frac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\lVert \mathbf{w} \rVert}$$

Bài toán lề cực đại viết trực tiếp là:

$$\max_{\mathbf{w}, b} \; \left\{ \frac{1}{\lVert \mathbf{w} \rVert} \min_{i} \; y_i(\mathbf{w}^\top \mathbf{x}_i + b) \right\}$$

Bài toán này **khó giải trực tiếp** vì hàm mục tiêu phức tạp (có $\min$ lồng trong $\max$) và có một sự **bất định về tỉ lệ**: nếu thay $(\mathbf{w}, b)$ bằng $(c\mathbf{w}, cb)$ với $c > 0$, cả tử và mẫu cùng nhân $c$ nên margin **không đổi**. Cùng một siêu phẳng được biểu diễn bởi vô số cặp $(\mathbf{w}, b)$ khác nhau.

> Mẹo chuẩn hóa: vì tỉ lệ tự do, ta **cố định** tỉ lệ bằng cách chuẩn hóa sao cho điểm gần nhất thỏa $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$.

---

# 4. Chuẩn hóa và bài toán chính (primal)

Áp ràng buộc chuẩn hóa: điểm gần siêu phẳng nhất có $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$, còn mọi điểm khác thì $\ge 1$:

$$y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \quad \forall i = 1, \dots, n$$

Với chuẩn hóa này, điểm gần nhất có tử số bằng $1$, nên:

$$\text{margin} = \frac{1}{\lVert \mathbf{w} \rVert}$$

Tối đa hóa $\dfrac{1}{\lVert \mathbf{w} \rVert}$ tương đương với cực tiểu hóa $\lVert \mathbf{w} \rVert$, và để có hàm mục tiêu **trơn, lồi và dễ lấy đạo hàm** ta dùng $\tfrac{1}{2}\lVert \mathbf{w} \rVert^2$. Đây là **bài toán chính (primal problem)** của SVM lề cứng (hard-margin):

$$\boxed{\;\min_{\mathbf{w}, b} \; \frac{1}{2}\lVert \mathbf{w} \rVert^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \; \forall i\;}$$

**Đây là bài toán lồi.** Hàm mục tiêu $\tfrac{1}{2}\lVert \mathbf{w} \rVert^2 = \tfrac12 \mathbf{w}^\top \mathbf{w}$ có Hessian là ma trận đơn vị $I \succ 0$ nên là hàm **lồi ngặt (strictly convex)** — xem [hàm lồi](convex-functions). Các ràng buộc $1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b) \le 0$ là **affine** theo $(\mathbf{w}, b)$, nên miền chấp nhận được là một **đa diện lồi (convex polyhedron)**. Cực tiểu một hàm lồi ngặt trên một tập lồi cho **nghiệm toàn cục duy nhất** — đây là khác biệt căn bản so với [perceptron](perceptron) (nghiệm bất kỳ) và là một thể hiện đẹp của [tối ưu lồi](convex-optimization).

---

# 5. Lagrangian và bài toán đối ngẫu (dual)

Tại sao phải chuyển sang dạng đối ngẫu? Hai lý do: (1) dạng đối ngẫu cho phép áp dụng **kernel**, và (2) nó làm lộ ra cấu trúc **vector hỗ trợ**. Ta dùng phương pháp **nhân tử Lagrange (Lagrange multipliers)** từ [tối ưu lồi](convex-optimization).

## 5.1. Lập hàm Lagrangian

Gán mỗi ràng buộc một nhân tử $\alpha_i \ge 0$. Hàm Lagrangian:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\lVert \mathbf{w} \rVert^2 - \sum_{i=1}^n \alpha_i \big[ y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 \big]$$

Hàm đối ngẫu Lagrange là $g(\boldsymbol{\alpha}) = \min_{\mathbf{w}, b} \mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})$. Vì $\mathcal{L}$ lồi theo $(\mathbf{w}, b)$, cực tiểu đạt tại điểm gradient triệt tiêu.

## 5.2. Khử biến chính

Lấy đạo hàm riêng và cho bằng $0$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = 0 \;\Longrightarrow\; \boxed{\;\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i\;}$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^n \alpha_i y_i = 0 \;\Longrightarrow\; \sum_{i=1}^n \alpha_i y_i = 0$$

Điều kiện đầu cực kỳ quan trọng: **nghiệm $\mathbf{w}$ là tổ hợp tuyến tính của các điểm dữ liệu**, có trọng số $\alpha_i y_i$.

## 5.3. Thay ngược để được hàm đối ngẫu

Thế $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ vào $\mathcal{L}$. Khai triển số hạng bậc hai:

$$\frac{1}{2}\lVert \mathbf{w} \rVert^2 = \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \, \mathbf{x}_i^\top \mathbf{x}_j$$

Số hạng tuyến tính, dùng $\sum_i \alpha_i y_i = 0$ để triệt tiêu $b$:

$$\sum_i \alpha_i y_i \mathbf{w}^\top \mathbf{x}_i = \sum_{i,j} \alpha_i \alpha_j y_i y_j \, \mathbf{x}_i^\top \mathbf{x}_j, \qquad \sum_i \alpha_i y_i b = b \sum_i \alpha_i y_i = 0$$

Gộp lại, các số hạng bậc hai bù trừ một nửa, ta được **bài toán đối ngẫu (dual problem)**:

$$\boxed{\;\max_{\boldsymbol{\alpha}} \; g(\boldsymbol{\alpha}) = \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \, \mathbf{x}_i^\top \mathbf{x}_j\;}$$

$$\text{s.t.} \quad \alpha_i \ge 0 \;\; \forall i, \qquad \sum_{i=1}^n \alpha_i y_i = 0$$

> **Quan sát then chốt.** Bài toán đối ngẫu **chỉ phụ thuộc vào tích vô hướng** $\mathbf{x}_i^\top \mathbf{x}_j$ giữa các cặp điểm dữ liệu, **không** cần truy cập trực tiếp tọa độ $\mathbf{x}_i$. Đây chính là cánh cửa dẫn tới kernel (mục 8).

Bài toán đối ngẫu là một **quy hoạch toàn phương (quadratic programming, QP)** lồi với biến $\boldsymbol{\alpha} \in \mathbb{R}^n$, ràng buộc đơn giản (hộp $\alpha_i \ge 0$ và một ràng buộc đẳng thức tuyến tính) — dễ giải bằng các bộ giải QP tiêu chuẩn hoặc thuật toán SMO.

---

# 6. Đối ngẫu mạnh và điều kiện KKT

Vì bài toán chính là **lồi** và các ràng buộc affine, **điều kiện Slater** thỏa mãn (khi dữ liệu khả tách), nên ta có **đối ngẫu mạnh (strong duality)**: giá trị tối ưu của primal và dual **bằng nhau**, và nghiệm tối ưu thỏa các **điều kiện KKT (Karush–Kuhn–Tucker)** — chi tiết trong [tối ưu lồi](convex-optimization).

Các điều kiện KKT tại nghiệm tối ưu:

1. **Khả thi chính (primal feasibility):** $y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 \ge 0$.
2. **Khả thi đối ngẫu (dual feasibility):** $\alpha_i \ge 0$.
3. **Bù trừ (complementary slackness):** $\alpha_i \big[ y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 \big] = 0$.

Điều kiện thứ ba là cốt lõi. Với mỗi $i$, **tích** của $\alpha_i$ và độ dư ràng buộc phải bằng $0$, nên xảy ra một trong hai khả năng:

- $\alpha_i = 0$: điểm $\mathbf{x}_i$ **không đóng góp** vào $\mathbf{w}$. Đây là các điểm nằm **xa** lề, $y_i(\mathbf{w}^\top \mathbf{x}_i + b) > 1$.
- $\alpha_i > 0$: bắt buộc $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$, tức điểm $\mathbf{x}_i$ nằm **chính xác trên lề**.

---

# 7. Vector hỗ trợ và công thức nghiệm

Những điểm có $\alpha_i > 0$ — nằm đúng trên lề — được gọi là **vector hỗ trợ (support vectors)**. Đây là nguồn gốc tên gọi của thuật toán.

> Nghiệm $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ **chỉ phụ thuộc vào các vector hỗ trợ** (vì các điểm khác có $\alpha_i = 0$). Có thể loại bỏ tất cả những điểm không phải vector hỗ trợ mà nghiệm **không đổi**. Đây là tính chất **thưa (sparse)** rất đẹp: thường chỉ một số ít điểm quyết định toàn bộ siêu phẳng.

**Tính $b$.** Lấy bất kỳ vector hỗ trợ $\mathbf{x}_s$ (có $\alpha_s > 0$), từ $y_s(\mathbf{w}^\top \mathbf{x}_s + b) = 1$ và $y_s \in \{-1, +1\}$ (nên $y_s^2 = 1$):

$$b = y_s - \mathbf{w}^\top \mathbf{x}_s = y_s - \sum_{i \in SV} \alpha_i y_i \, \mathbf{x}_i^\top \mathbf{x}_s$$

Trong thực hành, để ổn định số học, ta lấy **trung bình** $b$ trên toàn bộ tập vector hỗ trợ $SV$.

**Hàm dự đoán** cho điểm mới $\mathbf{x}$:

$$\hat{y} = \operatorname{sign}\left( \sum_{i \in SV} \alpha_i y_i \, \mathbf{x}_i^\top \mathbf{x} + b \right)$$

Một lần nữa, dự đoán **chỉ cần tích vô hướng** giữa điểm mới và các vector hỗ trợ.

---

# 8. Vì sao đối ngẫu mở đường cho kernel

Quan sát rằng cả bài toán đối ngẫu (mục 5.3) lẫn hàm dự đoán (mục 7) **chỉ chứa $\mathbf{x}$ dưới dạng tích vô hướng** $\mathbf{x}_i^\top \mathbf{x}_j$. Ta **không bao giờ** cần biết tọa độ riêng lẻ của các điểm.

Ý tưởng kernel: thay vì làm việc trong không gian gốc, ánh xạ dữ liệu sang một không gian đặc trưng (feature space) nhiều chiều $\phi(\mathbf{x})$ rồi phân tách tuyến tính ở đó. Nhưng tính $\phi(\mathbf{x})$ tường minh có thể rất tốn kém (thậm chí vô hạn chiều). **Mẹo kernel (kernel trick):** thay mọi tích vô hướng bằng một **hàm kernel**:

$$\mathbf{x}_i^\top \mathbf{x}_j \;\longrightarrow\; K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$$

mà tính $K$ trực tiếp **không cần** tính $\phi$. Ví dụ kernel Gauss (RBF) $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2)$ ứng với không gian đặc trưng vô hạn chiều. Nhờ đó SVM phân tách được dữ liệu **phi tuyến** mà chi phí tính toán vẫn nằm trong không gian gốc — chi tiết ở [SVM với kernel](kernel-svm).

> Chính dạng đối ngẫu — chứ không phải dạng chính — mới làm cho kernel khả thi: bài toán chính chứa $\mathbf{w}$ trong không gian đặc trưng (có thể vô hạn chiều, không lưu được), còn bài toán đối ngẫu chỉ chứa $K(\mathbf{x}_i, \mathbf{x}_j)$ là một con số.

---

# 9. So sánh với perceptron và hồi quy logistic

**So với [perceptron](perceptron).** Cả hai đều tìm siêu phẳng tuyến tính cho dữ liệu khả tách. Khác biệt:

- Perceptron dừng ở **bất kỳ** siêu phẳng tách đúng nào — nghiệm **phụ thuộc khởi tạo** và thứ tự dữ liệu, không duy nhất.
- SVM cực tiểu một hàm lồi ngặt nên cho **nghiệm duy nhất**, và đó là siêu phẳng có **lề cực đại** — chọn ra đường "tốt nhất" trong vô số đường mà perceptron có thể trả về. Điều này cho SVM khả năng **tổng quát hóa (generalization)** tốt hơn.

**So với [hồi quy logistic](logistic-regression).** Cả hai là bộ phân loại tuyến tính (có thể kernel hóa). Khác biệt:

- Hồi quy logistic tối ưu **mất mát log (cross-entropy)** trên **mọi** điểm, cho ra **xác suất** $P(y=1\mid\mathbf{x})$; mọi điểm đều ảnh hưởng tới nghiệm dù ít hay nhiều.
- SVM tối ưu **lề** với mất mát hinge; nghiệm **thưa**, chỉ phụ thuộc vào các vector hỗ trợ gần biên, và không cho xác suất trực tiếp. SVM nhạy với hình học biên, logistic nhạy với phân phối tổng thể.

| Tiêu chí | Perceptron | Logistic Regression | SVM (hard-margin) |
| --- | --- | --- | --- |
| Mục tiêu | Tách đúng | Cực đại hợp lý (log-loss) | Lề cực đại |
| Nghiệm | Không duy nhất | Duy nhất | **Duy nhất** |
| Đầu ra | Nhãn | **Xác suất** | Nhãn (+ điểm) |
| Phụ thuộc dữ liệu | Toàn bộ | Toàn bộ | **Chỉ vector hỗ trợ** |
| Kernel hóa | Có (dual) | Có | **Có (dual)** |

---

# 10. Ưu điểm

- **Lề cực đại, tổng quát hóa tốt** — chọn siêu phẳng an toàn nhất trong vô số đường tách đúng, giảm quá khớp so với perceptron.
- **Nghiệm toàn cục duy nhất** — vì là bài toán lồi ngặt, không kẹt ở cực tiểu địa phương.
- **Nghiệm thưa (sparse)** — chỉ một số ít vector hỗ trợ quyết định mô hình; dự đoán nhanh và mô hình gọn.
- **Mở rộng phi tuyến bằng kernel** — nhờ dạng đối ngẫu chỉ dùng tích vô hướng, [kernel](kernel-svm) cho phép phân tách dữ liệu phức tạp mà không cần tính tọa độ trong không gian đặc trưng.

---

# 11. Nhược điểm

- **Đòi hỏi dữ liệu khả tách tuyến tính** — bản hard-margin **vô nghiệm** nếu các lớp chồng lấn hoặc có nhiễu; cần mở rộng sang **soft-margin** (lề mềm) với biến nới lỏng (slack).
- **Nhạy với ngoại lai (outlier)** — một điểm sai nhãn nằm sâu trong lề có thể làm bài toán bất khả thi hoặc bóp méo siêu phẳng.
- **Không cho xác suất trực tiếp** — đầu ra là nhãn/điểm, cần hiệu chỉnh thêm (ví dụ Platt scaling) nếu muốn xác suất, khác với [hồi quy logistic](logistic-regression).
- **Chi phí huấn luyện tăng theo số mẫu** — giải QP đối ngẫu cỡ $O(n^2)$–$O(n^3)$, khó mở rộng cho tập dữ liệu rất lớn.

---

# 12. Tổng kết

SVM lề cứng khởi đi từ một câu hỏi hình học đơn giản — *siêu phẳng nào xa dữ liệu nhất?* — và biến nó thành một bài toán tối ưu **lồi** đẹp đẽ:

$$\min_{\mathbf{w}, b} \frac{1}{2}\lVert \mathbf{w} \rVert^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1$$

Qua mẹo chuẩn hóa, lề trở thành $\dfrac{1}{\lVert \mathbf{w} \rVert}$; qua [tối ưu lồi](convex-optimization) và Lagrangian, ta thu được dạng đối ngẫu chỉ phụ thuộc tích vô hướng và nghiệm $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$; qua KKT, chỉ những **vector hỗ trợ** trên lề ($\alpha_i > 0$) mới định hình siêu phẳng.

Chính dạng đối ngẫu — gọn gàng vì chỉ dùng $\mathbf{x}_i^\top \mathbf{x}_j$ — là di sản lớn nhất của SVM: nó cho phép thay tích vô hướng bằng [kernel](kernel-svm) để bắt phân tách phi tuyến, đồng thời phơi bày sự khác biệt sâu sắc với [perceptron](perceptron) (nghiệm bất kỳ → nghiệm lề cực đại duy nhất) và [hồi quy logistic](logistic-regression) (mọi điểm → chỉ vài vector hỗ trợ).

> Bài tiếp theo — **SVM lề mềm (soft-margin) và kernel** — nới lỏng ràng buộc khả tách bằng biến slack và tham số $C$, rồi thay tích vô hướng bằng hàm kernel để SVM phân loại được cả dữ liệu chồng lấn lẫn dữ liệu phi tuyến.

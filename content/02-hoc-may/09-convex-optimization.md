# Tối ưu lồi & Đối ngẫu (Convex Optimization & Duality)

> Phần lớn các mô hình học máy cuối cùng đều quy về một **bài toán tối ưu**: tìm tham số cực tiểu một hàm mục tiêu, đôi khi kèm **ràng buộc (constraint)**. Khi bài toán có cấu trúc **lồi (convex)**, ta có một sự đảm bảo quý giá: **mọi cực tiểu địa phương đều là cực tiểu toàn cục**. Bài này xây dựng bộ công cụ trung tâm để khai thác cấu trúc đó — **hàm Lagrange**, **bài toán đối ngẫu (dual problem)** và **điều kiện KKT** — chính là nền tảng lý thuyết để giải [Support Vector Machine](#/svm) và phiên bản hạt nhân [Kernel SVM](#/kernel-svm).

---

# 1. Bài toán tối ưu có ràng buộc

## 1.1. Dạng tổng quát

Một bài toán tối ưu (optimization problem) ở dạng chuẩn được viết là:

$$
\begin{aligned}
\min_{x} \quad & f_0(x) \\
\text{sao cho} \quad & f_i(x) \le 0, \quad i = 1, \dots, m \\
& h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
$$

trong đó:

* $x \in \mathbb{R}^n$ là **biến tối ưu (optimization variable)**.
* $f_0(x)$ là **hàm mục tiêu (objective function)** — thứ ta muốn cực tiểu hóa.
* $f_i(x) \le 0$ là các **ràng buộc bất đẳng thức (inequality constraints)**.
* $h_j(x) = 0$ là các **ràng buộc đẳng thức (equality constraints)**.

Mọi bài toán cực đại hóa đều quy về dạng này bằng cách đặt $\max f_0 = -\min(-f_0)$, và mọi ràng buộc dạng $f_i(x) \ge 0$ đều viết lại thành $-f_i(x) \le 0$. Vì vậy dạng trên là đủ tổng quát.

## 1.2. Miền khả thi và nghiệm tối ưu

Tập các điểm $x$ thỏa mãn **tất cả** ràng buộc gọi là **miền khả thi (feasible set)**:

$$
\mathcal{D} = \{ x : f_i(x) \le 0 \;\forall i,\; h_j(x) = 0 \;\forall j \}
$$

Giá trị tối ưu (optimal value) là:

$$
p^\star = \min_{x \in \mathcal{D}} f_0(x)
$$

Một điểm $x^\star \in \mathcal{D}$ đạt $f_0(x^\star) = p^\star$ gọi là **nghiệm tối ưu (optimal solution)**. Nếu miền khả thi rỗng, ta quy ước $p^\star = +\infty$.

> Trực giác: ràng buộc "khoanh vùng" những giá trị $x$ được phép, còn hàm mục tiêu cho biết trong vùng đó điểm nào "tốt nhất". Khó khăn nằm ở chỗ điểm tốt nhất thường nằm **trên biên** của miền khả thi — nơi một vài ràng buộc bất đẳng thức trở thành đẳng thức.

---

# 2. Bài toán lồi

## 2.1. Định nghĩa

Bài toán tối ưu ở mục 1.1 được gọi là **bài toán lồi (convex problem)** nếu thỏa mãn đồng thời:

1. Hàm mục tiêu $f_0$ là **hàm lồi (convex function)**.
2. Mọi hàm ràng buộc bất đẳng thức $f_i$ đều **lồi**.
3. Mọi ràng buộc đẳng thức $h_j$ đều **affine**, tức có dạng $h_j(x) = a_j^\top x - b_j$.

Khi đó miền khả thi $\mathcal{D}$ là một **tập lồi (convex set)** — giao của các tập mức dưới (sub-level set) lồi $\{f_i(x) \le 0\}$ với các siêu phẳng $\{h_j(x) = 0\}$. Nhắc lại định nghĩa hàm lồi (xem chi tiết tại [Hàm lồi](#/convex-functions)): với mọi $x, y$ và mọi $\theta \in [0, 1]$,

$$
f\big(\theta x + (1-\theta) y\big) \le \theta f(x) + (1-\theta) f(y)
$$

## 2.2. Vì sao tính lồi lại quý

Tính chất then chốt của bài toán lồi:

> **Mọi điểm cực tiểu địa phương (local minimum) của một bài toán lồi đều là cực tiểu toàn cục (global minimum).**

Sơ lược lý do: giả sử $x$ là cực tiểu địa phương nhưng tồn tại $y$ khả thi với $f_0(y) < f_0(x)$. Xét điểm trên đoạn nối $z = \theta x + (1-\theta) y$ với $\theta$ gần $1$. Do tập khả thi lồi nên $z$ khả thi, và do $f_0$ lồi:

$$
f_0(z) \le \theta f_0(x) + (1-\theta) f_0(y) < f_0(x)
$$

Nhưng $z$ nằm tùy ý gần $x$ — mâu thuẫn với việc $x$ là cực tiểu địa phương. $\blacksquare$

Nhờ tính chất này, các thuật toán "đi xuống" cục bộ như [Gradient Descent](#/ham-mat-mat-toi-uu) bảo đảm chạm nghiệm toàn cục. Đây là ranh giới phân chia "dễ" và "khó" thực sự trong tối ưu hóa — không phải tuyến tính so với phi tuyến, mà là **lồi so với không lồi**.

---

# 3. Hàm Lagrange

## 3.1. Ý tưởng: gộp ràng buộc vào mục tiêu

Ràng buộc làm bài toán khó vì ta không thể tự do dịch chuyển $x$. Ý tưởng của **Lagrange** là **xóa bỏ ràng buộc** bằng cách "phạt" việc vi phạm chúng: gắn mỗi ràng buộc một **hệ số (nhân tử) Lagrange (Lagrange multiplier)** rồi cộng thẳng vào hàm mục tiêu.

**Hàm Lagrange (Lagrangian)** của bài toán mục 1.1 là:

$$
\mathcal{L}(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^{m} \lambda_i f_i(x) + \sum_{j=1}^{p} \nu_j h_j(x)
$$

trong đó:

* $\lambda = (\lambda_1, \dots, \lambda_m)$ là nhân tử cho ràng buộc bất đẳng thức, **bắt buộc** $\lambda_i \ge 0$.
* $\nu = (\nu_1, \dots, \nu_p)$ là nhân tử cho ràng buộc đẳng thức, **dấu tùy ý**.

$\lambda, \nu$ gọi chung là **biến đối ngẫu (dual variables)**.

## 3.2. Vì sao $\lambda_i \ge 0$

Lý do dấu của $\lambda_i$ không tùy tiện. Nếu $x$ khả thi thì $f_i(x) \le 0$ và $h_j(x) = 0$. Khi đó, với mọi $\lambda \succeq 0$ (mọi thành phần không âm) và mọi $\nu$:

$$
\mathcal{L}(x, \lambda, \nu) = f_0(x) + \underbrace{\sum_i \lambda_i f_i(x)}_{\le 0} + \underbrace{\sum_j \nu_j h_j(x)}_{=0} \le f_0(x)
$$

Tức Lagrangian là một **chặn dưới** của hàm mục tiêu trên miền khả thi. Đây chính là viên gạch để dựng lên đối ngẫu yếu ở mục 5. Nếu cho phép $\lambda_i < 0$, số hạng $\lambda_i f_i(x)$ có thể **dương**, phá vỡ tính chặn dưới này.

---

# 4. Hàm đối ngẫu Lagrange và bài toán đối ngẫu

## 4.1. Hàm đối ngẫu Lagrange

**Hàm đối ngẫu Lagrange (Lagrange dual function)** được định nghĩa là **infimum** của Lagrangian theo $x$ (lấy trên toàn bộ $x$, **không** giới hạn miền khả thi):

$$
g(\lambda, \nu) = \inf_{x} \; \mathcal{L}(x, \lambda, \nu) = \inf_{x} \left( f_0(x) + \sum_i \lambda_i f_i(x) + \sum_j \nu_j h_j(x) \right)
$$

Mỗi cặp $(\lambda, \nu)$ cho ta một con số $g(\lambda, \nu)$ — cận dưới tốt nhất mà Lagrangian tương ứng đạt được.

## 4.2. Tính chất quan trọng: $g$ luôn lõm

> Dù bài toán gốc **có lồi hay không**, hàm đối ngẫu $g(\lambda, \nu)$ **luôn là hàm lõm (concave)**.

Lý do rất đẹp: với mỗi $x$ cố định, $\mathcal{L}(x, \lambda, \nu)$ là hàm **affine** (tuyến tính) theo $(\lambda, \nu)$. Hàm $g$ là **infimum của một họ hàm affine** theo $(\lambda, \nu)$, mà infimum của các hàm affine luôn lõm. Vì vậy bài toán đối ngẫu (cực đại hóa hàm lõm với ràng buộc đơn giản $\lambda \succeq 0$) **luôn là một bài toán lồi** — kể cả khi bài toán gốc không lồi. Đây là một trong những lý do khiến đối ngẫu hữu ích đến vậy.

## 4.3. Bài toán đối ngẫu

Vì $g(\lambda, \nu)$ cho một chặn dưới của $p^\star$ (chứng minh ở mục 5), điều tự nhiên là tìm **chặn dưới tốt nhất**. Ta được **bài toán đối ngẫu Lagrange (Lagrange dual problem)**:

$$
\begin{aligned}
\max_{\lambda, \nu} \quad & g(\lambda, \nu) \\
\text{sao cho} \quad & \lambda \succeq 0
\end{aligned}
$$

Gọi giá trị tối ưu của nó là $d^\star$. Để phân biệt, bài toán ban đầu (mục 1.1) gọi là **bài toán gốc (primal problem)** với giá trị tối ưu $p^\star$.

---

# 5. Đối ngẫu yếu và đối ngẫu mạnh

## 5.1. Đối ngẫu yếu (weak duality)

> **Định lý (đối ngẫu yếu).** Luôn luôn có $d^\star \le p^\star$.

**Chứng minh.** Gọi $\tilde{x}$ là một điểm khả thi bất kỳ của bài toán gốc và $(\lambda, \nu)$ với $\lambda \succeq 0$ là một điểm khả thi của bài toán đối ngẫu. Từ bất đẳng thức ở mục 3.2 và định nghĩa infimum:

$$
g(\lambda, \nu) = \inf_{x} \mathcal{L}(x, \lambda, \nu) \le \mathcal{L}(\tilde{x}, \lambda, \nu) \le f_0(\tilde{x})
$$

Bất đẳng thức $g(\lambda, \nu) \le f_0(\tilde{x})$ đúng với **mọi** cặp khả thi. Lấy max vế trái theo $(\lambda, \nu)$ và min vế phải theo $\tilde{x}$:

$$
d^\star = \max_{\lambda \succeq 0, \nu} g(\lambda, \nu) \le \min_{\tilde{x} \in \mathcal{D}} f_0(\tilde{x}) = p^\star \qquad \blacksquare
$$

Hiệu $p^\star - d^\star \ge 0$ gọi là **khe đối ngẫu (duality gap)**. Đối ngẫu yếu nói rằng khe này không âm — bài toán đối ngẫu luôn cho một **cận dưới** an toàn cho bài toán gốc.

## 5.2. Đối ngẫu mạnh (strong duality)

Khi khe đối ngẫu **bằng 0**, ta nói **đối ngẫu mạnh (strong duality)** thỏa mãn:

$$
d^\star = p^\star
$$

Đây là tình huống lý tưởng: giải bài toán đối ngẫu **tương đương** với giải bài toán gốc. Đối ngẫu mạnh **không phải lúc nào cũng đúng**, nhưng với bài toán **lồi** nó thường đúng dưới một điều kiện chính quy nhẹ.

## 5.3. Điều kiện Slater

> **Điều kiện Slater (Slater's condition).** Nếu bài toán **lồi** và tồn tại một điểm khả thi **chặt (strictly feasible)** — tức một $x$ thỏa $f_i(x) < 0$ với mọi ràng buộc bất đẳng thức (không-affine) và $h_j(x) = 0$ — thì đối ngẫu mạnh thỏa mãn, $d^\star = p^\star$.

Trực giác: nếu miền khả thi có "phần trong" thực sự (không bị bóp dẹt thành một mặt biên), thì không có khe đối ngẫu. Với các ràng buộc affine, ta thậm chí chỉ cần khả thi (không cần chặt). Phần lớn các bài toán học máy phổ biến (SVM, [Hồi quy Logistic](#/logistic-regression) có ràng buộc, v.v.) đều thỏa Slater, nên ta yên tâm chuyển sang giải bài toán đối ngẫu.

---

# 6. Điều kiện KKT

## 6.1. Bốn nhóm điều kiện

Khi đối ngẫu mạnh thỏa mãn và các hàm khả vi, nghiệm tối ưu $(x^\star, \lambda^\star, \nu^\star)$ phải thỏa mãn hệ **điều kiện Karush–Kuhn–Tucker (KKT)** gồm bốn nhóm:

**(1) Khả thi gốc (primal feasibility).** $x^\star$ phải nằm trong miền khả thi:

$$
f_i(x^\star) \le 0 \;\;\forall i, \qquad h_j(x^\star) = 0 \;\;\forall j
$$

**(2) Khả thi đối ngẫu (dual feasibility).** Các nhân tử bất đẳng thức không âm:

$$
\lambda_i^\star \ge 0 \;\;\forall i
$$

**(3) Bù trừ (complementary slackness).** Tích của nhân tử và ràng buộc bằng 0 cho từng ràng buộc:

$$
\lambda_i^\star \, f_i(x^\star) = 0 \;\;\forall i
$$

**(4) Dừng (stationarity).** Gradient của Lagrangian theo $x$ triệt tiêu tại $x^\star$:

$$
\nabla_x \mathcal{L}(x^\star, \lambda^\star, \nu^\star) = \nabla f_0(x^\star) + \sum_i \lambda_i^\star \nabla f_i(x^\star) + \sum_j \nu_j^\star \nabla h_j(x^\star) = 0
$$

## 6.2. Giải thích từng điều kiện

* **Khả thi gốc và đối ngẫu** chỉ là đòi hỏi $x^\star$ và $(\lambda^\star, \nu^\star)$ thực sự là các điểm hợp lệ của hai bài toán.

* **Dừng** là điều kiện đạo hàm bằng 0 quen thuộc, nhưng cho **Lagrangian** thay vì cho mục tiêu trần. Nó nói rằng tại nghiệm, gradient mục tiêu là một tổ hợp tuyến tính của các gradient ràng buộc — lực "muốn giảm $f_0$" bị cân bằng đúng bởi lực "đẩy ra khỏi biên" của các ràng buộc đang hoạt động.

* **Bù trừ** là điều kiện tinh tế và hữu dụng nhất. Vì $\lambda_i^\star \ge 0$ và $f_i(x^\star) \le 0$, tích $\lambda_i^\star f_i(x^\star) \le 0$; điều kiện ép nó bằng 0 nên với mỗi $i$ phải xảy ra **một trong hai**:

$$
\lambda_i^\star > 0 \;\Rightarrow\; f_i(x^\star) = 0 \qquad \text{hoặc} \qquad f_i(x^\star) < 0 \;\Rightarrow\; \lambda_i^\star = 0
$$

Nói cách khác: **ràng buộc nào không "chạm biên" (lỏng, $f_i < 0$) thì nhân tử của nó bằng 0** — nó không ảnh hưởng tới nghiệm. Chỉ những ràng buộc **hoạt động (active)**, tức $f_i(x^\star) = 0$, mới có thể có $\lambda_i^\star > 0$. Đây chính là nguồn gốc của **tính thưa (sparsity)** trong nghiệm SVM.

## 6.3. Vai trò của KKT

Với bài toán **lồi** thỏa Slater, KKT vừa là điều kiện **cần** vừa là điều kiện **đủ**: bất kỳ bộ $(x^\star, \lambda^\star, \nu^\star)$ nào thỏa cả bốn nhóm đều là cặp nghiệm tối ưu gốc–đối ngẫu. Vì vậy "giải bài toán" thường có nghĩa là **giải hệ phương trình/bất phương trình KKT**.

---

# 7. Ứng dụng trong ML

## 7.1. SVM giải qua bài toán đối ngẫu

[Support Vector Machine](#/svm) là ví dụ kinh điển nhất của toàn bộ lý thuyết trên. Bài toán gốc của SVM lề cứng là cực tiểu $\tfrac{1}{2}\|w\|^2$ với các ràng buộc bất đẳng thức $y_i(w^\top x_i + b) \ge 1$. Đây là một **bài toán lồi** (mục tiêu lồi ngặt, ràng buộc affine) và thỏa Slater, nên đối ngẫu mạnh đúng.

Vì sao người ta giải bản **đối ngẫu** thay vì gốc?

* **Số biến hợp lý hơn.** Bài toán đối ngẫu có số biến bằng **số điểm dữ liệu** (mỗi $\lambda_i$ ứng với một điểm), thay vì số chiều đặc trưng. Khi đặc trưng nhiều chiều, đây là lợi thế lớn.
* **Tính thưa nhờ bù trừ.** Theo điều kiện bù trừ (mục 6.2), chỉ những điểm nằm **đúng trên lề** ($f_i(x^\star) = 0$) mới có $\lambda_i^\star > 0$. Những điểm này là các **vector hỗ trợ (support vectors)**; toàn bộ phần còn lại có $\lambda_i^\star = 0$ và không tham gia vào nghiệm. Đây là lý do tên gọi "support vector" và lý do mô hình nhẹ khi dự đoán.
* **Mở đường cho hạt nhân.** Trong bản đối ngẫu, dữ liệu chỉ xuất hiện qua **tích vô hướng** $x_i^\top x_j$. Điều này cho phép thay nó bằng một **hàm hạt nhân (kernel)** $K(x_i, x_j)$ — chính là [Kernel SVM](#/kernel-svm) — để học biên phi tuyến mà không cần tính tường minh ánh xạ đặc trưng nhiều chiều ("mẹo hạt nhân", kernel trick).

## 7.2. Bức tranh tổng quát hơn

Khung Lagrange–đối ngẫu–KKT xuất hiện ở khắp nơi trong học máy: từ ràng buộc chuẩn hóa trong [Hồi quy Logistic](#/logistic-regression) có regularization, tới các bài toán có ràng buộc đẳng thức trong giảm chiều (PCA viết dưới dạng tối ưu có ràng buộc trực giao). Bất cứ khi nào ta thấy một ràng buộc, KKT cho ta một bộ điều kiện tối ưu rõ ràng để bám vào.

---

# 8. Tổng kết

* Bài toán tối ưu chuẩn gồm **mục tiêu** $f_0$, ràng buộc bất đẳng thức $f_i(x) \le 0$ và đẳng thức $h_j(x) = 0$. Khi $f_0, f_i$ **lồi** và $h_j$ **affine**, ta có **bài toán lồi** — nơi cực tiểu địa phương cũng là cực tiểu toàn cục.

* **Hàm Lagrange** $\mathcal{L}(x, \lambda, \nu) = f_0(x) + \sum_i \lambda_i f_i(x) + \sum_j \nu_j h_j(x)$ gộp ràng buộc vào mục tiêu qua các nhân tử, với $\lambda \succeq 0$.

* **Hàm đối ngẫu** $g(\lambda, \nu) = \inf_x \mathcal{L}$ **luôn lõm**, cho ta bài toán đối ngẫu lồi $\max_{\lambda \succeq 0, \nu} g$.

* **Đối ngẫu yếu** $d^\star \le p^\star$ luôn đúng; **đối ngẫu mạnh** $d^\star = p^\star$ đúng cho bài toán lồi thỏa **điều kiện Slater**.

* **KKT** gói gọn tối ưu thành bốn nhóm — dừng, khả thi gốc, khả thi đối ngẫu, bù trừ — và là điều kiện cần và đủ cho bài toán lồi. Riêng **bù trừ** sinh ra tính thưa, là linh hồn của nghiệm [SVM](#/svm) và [Kernel SVM](#/kernel-svm).

> Nắm vững Lagrangian và KKT, bạn sẽ thấy SVM không còn là một thuật toán "trên trời rơi xuống" mà là hệ quả tự nhiên của việc áp lý thuyết đối ngẫu vào một bài toán lồi đơn giản — và đó cũng là chiếc chìa khóa mở ra mẹo hạt nhân.

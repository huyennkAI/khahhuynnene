# Tập lồi & Hàm lồi (Convex Sets & Convex Functions)

> Một bài toán tối ưu **lồi (convex)** có một tính chất kỳ diệu: **mọi cực tiểu cục bộ (local minimum) đều là cực tiểu toàn cục (global minimum)**. Không có "bẫy" cực tiểu địa phương, không có điểm yên ngựa lừa gạt. Đây chính là lý do giới học máy (machine learning) yêu thích các bài toán lồi: nếu xây được một hàm mất mát (loss function) lồi, ta gần như **bảo đảm** tìm được nghiệm tốt nhất.
>
> Bài này xây dựng nền tảng hình học cho điều đó: **tập lồi** (đoạn nối hai điểm nằm trọn trong tập) và **hàm lồi** (đồ thị "võng xuống", luôn nằm trên tiếp tuyến của chính nó).

---

# 1. Trực giác: "võng xuống" và "không có bẫy"

Hãy tưởng tượng thả một viên bi vào một cái bát. Dù thả từ đâu, viên bi cũng lăn về **đúng một** điểm thấp nhất ở đáy. Cái bát đó là hình ảnh trực quan của một **hàm lồi**: bề mặt cong đều một chiều, không có chỗ lồi lõm tạo ra nhiều "đáy" giả.

Ngược lại, một dãy núi gồ ghề có vô số thung lũng — thả bi vào, nó kẹt ở thung lũng gần nhất chứ chưa chắc là chỗ thấp nhất. Đó là hàm **không lồi (non-convex)**, đặc trưng của hầu hết mạng nơ-ron sâu.

Toàn bộ lý thuyết tối ưu lồi (xem [Tối ưu lồi](#/convex-optimization)) là cách **hình thức hóa** trực giác "cái bát" này, để từ một bất đẳng thức hình học đơn giản, ta suy ra được bảo đảm toàn cục. Ta bắt đầu từ khái niệm nền: tập lồi.

---

# 2. Tập lồi (Convex Set)

## 2.1. Định nghĩa

Một tập $\mathcal{C} \subseteq \mathbb{R}^n$ được gọi là **lồi (convex)** nếu với **mọi** hai điểm $x, y \in \mathcal{C}$, toàn bộ **đoạn thẳng** nối $x$ và $y$ cũng nằm trong $\mathcal{C}$. Hình thức:

$$x, y \in \mathcal{C}, \; \lambda \in [0, 1] \;\Rightarrow\; \lambda x + (1 - \lambda) y \in \mathcal{C}$$

Biểu thức $\lambda x + (1-\lambda) y$ khi $\lambda$ chạy từ $0$ đến $1$ quét đúng đoạn thẳng từ $y$ ($\lambda = 0$) tới $x$ ($\lambda = 1$). Trực giác: tập lồi là tập **không có chỗ lõm** — không có "khe", "lỗ" hay "vết khuyết" nào khiến đoạn nối hai điểm phải đi ra ngoài.

## 2.2. Tổ hợp lồi (Convex Combination)

Mở rộng cho nhiều điểm: một **tổ hợp lồi (convex combination)** của các điểm $x_1, \dots, x_k$ là

$$x = \sum_{i=1}^{k} \lambda_i x_i, \qquad \lambda_i \ge 0, \quad \sum_{i=1}^{k} \lambda_i = 1$$

Các trọng số $\lambda_i$ không âm và cộng lại bằng $1$ — đúng dạng một phân phối xác suất. Có thể chứng minh (bằng quy nạp theo $k$) rằng một tập lồi **đóng kín** với mọi tổ hợp lồi: nếu mọi $x_i \in \mathcal{C}$ thì tổ hợp lồi của chúng cũng thuộc $\mathcal{C}$. Tập tất cả tổ hợp lồi của một tập điểm gọi là **bao lồi (convex hull)** — tập lồi nhỏ nhất chứa các điểm đó.

## 2.3. Một vài ví dụ tập lồi

- **Nửa không gian (halfspace)** $\{x : a^\top x \le b\}$. Lấy hai điểm thỏa $a^\top x \le b$ và $a^\top y \le b$, thì $a^\top(\lambda x + (1-\lambda) y) = \lambda a^\top x + (1-\lambda) a^\top y \le \lambda b + (1-\lambda) b = b$. Vậy đoạn nối cũng thỏa $\Rightarrow$ lồi.
- **Hình cầu (ball)** $\{x : \|x - c\| \le r\}$. Theo bất đẳng thức tam giác, đoạn nối hai điểm trong cầu vẫn nằm trong cầu.
- **Siêu phẳng (hyperplane)** $\{x : a^\top x = b\}$, và **không gian con (subspace)** affine nói chung.
- **Khối đa diện (polyhedron)** — giao của hữu hạn nửa không gian, miền nghiệm của bài toán tuyến tính.

## 2.4. Giao các tập lồi vẫn lồi

**Mệnh đề.** Nếu $\mathcal{C}_1, \mathcal{C}_2$ lồi thì $\mathcal{C}_1 \cap \mathcal{C}_2$ lồi. Tổng quát, giao của **một họ bất kỳ** (kể cả vô hạn) các tập lồi vẫn lồi.

**Chứng minh.** Lấy $x, y \in \mathcal{C}_1 \cap \mathcal{C}_2$. Vì $x, y \in \mathcal{C}_1$ (lồi) nên $\lambda x + (1-\lambda) y \in \mathcal{C}_1$; tương tự cho $\mathcal{C}_2$. Vậy đoạn nối nằm trong cả hai tập, tức nằm trong giao. $\blacksquare$

Hệ quả: một khối đa diện — giao của nhiều nửa không gian — luôn lồi. Đây là lý do miền ràng buộc tuyến tính trong các bài toán như [SVM](#/svm) là tập lồi.

---

# 3. Hàm lồi (Convex Function)

## 3.1. Định nghĩa

Cho hàm $f: \mathcal{C} \to \mathbb{R}$ xác định trên một **tập lồi** $\mathcal{C}$. Ta nói $f$ **lồi (convex)** nếu với mọi $x, y \in \mathcal{C}$ và mọi $\lambda \in [0, 1]$:

$$f\big(\lambda x + (1 - \lambda) y\big) \;\le\; \lambda f(x) + (1 - \lambda) f(y)$$

Vế trái là giá trị của hàm tại một điểm **trên đoạn nối**; vế phải là giá trị tương ứng **trên dây cung (chord)** nối $\big(x, f(x)\big)$ với $\big(y, f(y)\big)$. Điều kiện lồi nói rằng: **đồ thị hàm luôn nằm dưới (hoặc trên) dây cung**. Đúng hình ảnh "cái bát" — đáy võng xuống dưới đoạn thẳng nối hai mép.

Nếu bất đẳng thức là **ngặt** ($<$) với mọi $x \ne y$ và $\lambda \in (0,1)$, ta gọi $f$ là **lồi ngặt (strictly convex)** — không có đoạn thẳng nào nằm trên đồ thị, đáy là một điểm duy nhất.

## 3.2. Hàm lõm (Concave Function)

Hàm $f$ là **lõm (concave)** nếu $-f$ lồi, tương đương đảo chiều bất đẳng thức:

$$f\big(\lambda x + (1 - \lambda) y\big) \;\ge\; \lambda f(x) + (1 - \lambda) f(y)$$

Trực giác: hàm lõm "úp xuống" như cái mái vòm, đồ thị nằm **trên** dây cung. Cực đại hóa một hàm lõm tương đương cực tiểu hóa một hàm lồi — nên hai khái niệm là hai mặt của cùng một đồng xu. (Một hàm vừa lồi vừa lõm chính là hàm **affine** $f(x) = a^\top x + b$.)

---

# 4. Điều kiện bậc nhất và bậc hai

Định nghĩa qua dây cung đúng nhưng khó kiểm tra trực tiếp. Khi $f$ khả vi, có hai tiêu chuẩn tiện hơn nhiều.

## 4.1. Điều kiện bậc nhất: đồ thị nằm trên tiếp tuyến

**Định lý (first-order condition).** Giả sử $f$ khả vi trên tập lồi mở $\mathcal{C}$. Khi đó $f$ lồi **khi và chỉ khi** với mọi $x, y \in \mathcal{C}$:

$$f(y) \;\ge\; f(x) + \nabla f(x)^\top (y - x)$$

Vế phải là **xấp xỉ tuyến tính** của $f$ tại $x$ — chính là **tiếp tuyến** (hay siêu phẳng tiếp xúc). Vậy điều kiện này nói: **toàn bộ đồ thị nằm trên mọi tiếp tuyến của nó**. Hình ảnh "cái bát" lần nữa: ở đáy bát, tiếp tuyến nằm ngang và mọi điểm khác đều cao hơn.

**Phác chứng minh (chiều thuận).** Từ định nghĩa lồi, với $\lambda \in (0,1]$:

$$f\big(x + \lambda(y - x)\big) \le (1-\lambda) f(x) + \lambda f(y)$$

Chuyển vế và chia cho $\lambda$:

$$\frac{f\big(x + \lambda(y-x)\big) - f(x)}{\lambda} \le f(y) - f(x)$$

Cho $\lambda \to 0^{+}$, vế trái hội tụ về đạo hàm theo hướng $\nabla f(x)^\top (y - x)$. Suy ra $\nabla f(x)^\top (y-x) \le f(y) - f(x)$, chính là điều phải chứng minh. $\blacksquare$

Hệ quả quan trọng (dùng ở mục 7): nếu $\nabla f(x) = 0$ thì $f(y) \ge f(x)$ với mọi $y$ — tức $x$ là cực tiểu toàn cục.

## 4.2. Điều kiện bậc hai: Hessian nửa xác định dương

**Định lý (second-order condition).** Giả sử $f$ khả vi hai lần trên tập lồi mở $\mathcal{C}$. Khi đó $f$ lồi **khi và chỉ khi** ma trận **Hessian** nửa xác định dương tại mọi điểm:

$$\nabla^2 f(x) \succeq 0 \qquad \text{với mọi } x \in \mathcal{C}$$

Ký hiệu $\nabla^2 f(x) \succeq 0$ nghĩa là $v^\top \nabla^2 f(x)\, v \ge 0$ với mọi vector $v$ — tức **mọi trị riêng (eigenvalue) không âm**, độ cong theo mọi hướng đều không âm. Trong một chiều, điều này quy về tiêu chuẩn quen thuộc $f''(x) \ge 0$: hàm "cong lên". Nếu $\nabla^2 f(x) \succ 0$ (xác định dương ngặt) thì $f$ lồi ngặt.

> **Trực giác.** Hessian đo độ cong. Nửa xác định dương khắp nơi nghĩa là bề mặt **không bao giờ cong xuống** theo bất kỳ hướng nào — đúng định nghĩa của "cái bát" trong không gian nhiều chiều.

---

# 5. Ví dụ hàm lồi và không lồi

| Hàm | Lồi? | Lý do |
| --- | --- | --- |
| $f(x) = x^2$ | Lồi ngặt | $f''(x) = 2 > 0$ |
| $f(x) = e^{x}$ | Lồi ngặt | $f''(x) = e^x > 0$ |
| $f(x) = -\log x$ (trên $x>0$) | Lồi ngặt | $f''(x) = 1/x^2 > 0$ |
| $f(x) = \|x\|$ (norm bất kỳ) | Lồi | Bất đẳng thức tam giác |
| $f(x) = a^\top x + b$ (affine) | Lồi & lõm | $\nabla^2 f = 0 \succeq 0$ |
| $f(x) = \log x$ | Lõm | $f'' = -1/x^2 < 0$ |
| $f(x) = x^3$ | Không lồi | $f'' = 6x$ đổi dấu |
| $f(x) = \sin x$ | Không lồi | Cong lên rồi cong xuống |

Hai ví dụ cuối là **không lồi**: $x^3$ có $f''$ âm khi $x < 0$; $\sin x$ dao động nên không thể "võng" một chiều.

---

# 6. Các phép toán bảo toàn tính lồi

Thay vì kiểm tra Hessian từng lần, ta thường **ghép** các hàm lồi đã biết bằng các phép toán **bảo toàn tính lồi** — đây là cách thực tế để chứng minh một loss phức tạp là lồi.

## 6.1. Tổng có trọng số không âm

Nếu $f_1, f_2$ lồi và $\alpha, \beta \ge 0$, thì $\alpha f_1 + \beta f_2$ lồi. **Chứng minh:** cộng hai bất đẳng thức lồi (đã nhân với hệ số không âm để không đảo chiều) là xong. Hệ quả: tổng các loss trên từng mẫu dữ liệu vẫn lồi, và thêm **hạng phạt chính quy hóa (regularization)** lồi như $\lambda \|w\|^2$ giữ nguyên tính lồi.

## 6.2. Cực đại theo điểm (pointwise maximum)

Nếu $f_1, f_2$ lồi thì $g(x) = \max\{f_1(x), f_2(x)\}$ lồi. **Trực giác:** lấy bao trên của các hàm lồi vẫn cho một mặt "võng xuống". Đây là lý do hàm **hinge loss** $\max(0,\, 1 - y\,\hat{y})$ trong [SVM](#/svm) là lồi — nó là max của hai hàm affine.

## 6.3. Hợp với ánh xạ affine

Nếu $f$ lồi thì $g(x) = f(Ax + b)$ lồi (với $A, b$ là hằng số). **Chứng minh:** ánh xạ affine bảo toàn tổ hợp lồi, nên bất đẳng thức định nghĩa được giữ nguyên. Hệ quả cực kỳ hữu ích: $\|Ax - b\|^2$ lồi theo $x$ vì là hợp của norm-bình phương (lồi) với một ánh xạ affine — chính là dạng của loss trong [Hồi quy tuyến tính](#/linear-regression).

---

# 7. Bất đẳng thức Jensen

Bất đẳng thức định nghĩa lồi là trường hợp hai điểm của một kết quả tổng quát hơn.

**Bất đẳng thức Jensen (Jensen's inequality).** Nếu $f$ lồi và $\lambda_1, \dots, \lambda_k \ge 0$ với $\sum_i \lambda_i = 1$, thì:

$$f\!\left( \sum_{i=1}^{k} \lambda_i x_i \right) \;\le\; \sum_{i=1}^{k} \lambda_i\, f(x_i)$$

Dạng kỳ vọng (dùng nhiều trong xác suất và học máy): với biến ngẫu nhiên $X$,

$$f\big(\mathbb{E}[X]\big) \;\le\; \mathbb{E}\big[f(X)\big]$$

**Chứng minh (quy nạp theo $k$).** Trường hợp $k=2$ chính là định nghĩa. Giả sử đúng tới $k-1$. Với $k$ điểm, đặt $\mu = \sum_{i=1}^{k-1} \lambda_i = 1 - \lambda_k$ và viết tổ hợp lồi thành hai mức:

$$f\!\left( \mu \underbrace{\sum_{i=1}^{k-1} \tfrac{\lambda_i}{\mu} x_i}_{=\,z} + \lambda_k x_k \right) \le \mu\, f(z) + \lambda_k f(x_k)$$

(áp dụng $k=2$). Vì $\sum_{i<k} \frac{\lambda_i}{\mu} = 1$, áp dụng giả thiết quy nạp cho $f(z)$ rồi gộp lại cho kết quả. $\blacksquare$

Jensen là công cụ chứng minh chủ lực: nó cho ELBO trong mô hình sinh, bất đẳng thức trung bình cộng – trung bình nhân, và tính không âm của entropy.

---

# 8. ĐỊNH LÝ then chốt: cực tiểu cục bộ là cực tiểu toàn cục

Đây là lý do sâu xa khiến học máy "thèm" bài toán lồi.

**Định lý.** Cho $f$ là hàm lồi trên tập lồi $\mathcal{C}$. Khi đó **mọi cực tiểu cục bộ của $f$ đều là cực tiểu toàn cục**. Hơn nữa, nếu $f$ lồi ngặt thì cực tiểu (nếu tồn tại) là **duy nhất**.

**Chứng minh (bằng phản chứng).** Giả sử $x^{*}$ là một cực tiểu **cục bộ** nhưng **không** toàn cục. Vậy tồn tại $y \in \mathcal{C}$ với

$$f(y) < f(x^{*})$$

Xét các điểm trên đoạn nối: $z_\lambda = \lambda y + (1-\lambda) x^{*}$ với $\lambda \in (0, 1]$. Vì $\mathcal{C}$ lồi nên $z_\lambda \in \mathcal{C}$. Dùng định nghĩa hàm lồi:

$$f(z_\lambda) \le \lambda f(y) + (1-\lambda) f(x^{*}) < \lambda f(x^{*}) + (1-\lambda) f(x^{*}) = f(x^{*})$$

(bất đẳng thức ngặt thứ hai do $f(y) < f(x^{*})$ và $\lambda > 0$). Vậy với **mọi** $\lambda > 0$, dù nhỏ tùy ý, điểm $z_\lambda$ cho giá trị **nhỏ hơn** $f(x^{*})$. Nhưng khi $\lambda \to 0^{+}$, $z_\lambda$ tiến **sát tùy ý** về $x^{*}$ — tức trong mọi lân cận nhỏ của $x^{*}$ đều có điểm tốt hơn. Điều này **mâu thuẫn** với giả thiết $x^{*}$ là cực tiểu cục bộ. $\blacksquare$

**Tính duy nhất khi lồi ngặt.** Giả sử có hai cực tiểu toàn cục $x_1 \ne x_2$ với $f(x_1) = f(x_2) = m$. Lấy trung điểm $z = \tfrac12 x_1 + \tfrac12 x_2$. Do **lồi ngặt**:

$$f(z) < \tfrac12 f(x_1) + \tfrac12 f(x_2) = m$$

mâu thuẫn vì $m$ là giá trị nhỏ nhất. Vậy cực tiểu là duy nhất. $\blacksquare$

> **Hệ quả vận hành.** Với hàm lồi khả vi, mọi **điểm dừng** ($\nabla f(x) = 0$) đều là cực tiểu toàn cục (kết hợp điều kiện bậc nhất ở 4.1). Nghĩa là chỉ cần gradient descent đưa gradient về $0$ là ta **chắc chắn** đã tìm ra nghiệm tối ưu — không cần lo bị kẹt ở "thung lũng giả".

---

# 9. Vì sao quan trọng cho học máy

- **Bảo đảm tối ưu toàn cục.** Theo Định lý mục 8, nếu loss lồi thì gradient descent (xem [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu)) hội tụ về nghiệm tốt nhất tuyệt đối, không bị bẫy cực tiểu địa phương. Khởi tạo ở đâu cũng về cùng một đáy.
- **Nhiều mô hình kinh điển là lồi.** [Hồi quy tuyến tính](#/linear-regression) (MSE lồi), [Hồi quy logistic](#/logistic-regression) (cross-entropy lồi theo trọng số), và [SVM](#/svm) (hinge loss + phạt $L_2$) đều là **bài toán lồi** — đó là lý do chúng huấn luyện ổn định, nhanh và lặp lại được.
- **Chính quy hóa giữ tính lồi.** Thêm $\lambda \|w\|^2$ (ridge) hay $\lambda \|w\|_1$ (lasso) là cộng một hàm lồi (mục 6.1), nên không phá vỡ bảo đảm toàn cục mà còn làm loss **lồi ngặt** $\Rightarrow$ nghiệm duy nhất.
- **Khung lý thuyết để phân tích.** Ngay cả khi mạng nơ-ron sâu **không lồi**, các công cụ lồi (Jensen, tính lồi cục bộ, hợp affine) vẫn là ngôn ngữ để hiểu hành vi, thiết kế bộ tối ưu và chứng minh hội tụ.

---

# 10. Tổng kết

Hai khái niệm trụ cột của bài:

- **Tập lồi:** đoạn nối hai điểm bất kỳ nằm trọn trong tập. Giao các tập lồi vẫn lồi $\Rightarrow$ miền ràng buộc tuyến tính luôn lồi.
- **Hàm lồi:** đồ thị nằm **dưới mọi dây cung** và **trên mọi tiếp tuyến**; tương đương $\nabla^2 f \succeq 0$.

Từ một bất đẳng thức hình học đơn giản, ta thu được kết quả mạnh nhất: **cực tiểu cục bộ ≡ cực tiểu toàn cục**. Đó là viên ngọc khiến tối ưu lồi trở thành nền móng vững chắc của học máy — bất cứ khi nào ta xây được một loss lồi, ta đã đặt nửa chân vào nghiệm tối ưu.

> Bài tiếp theo — **[Tối ưu lồi](#/convex-optimization)** — dựng trên nền này để xây các thuật toán: điều kiện KKT, đối ngẫu Lagrange, và vì sao gradient descent hội tụ với tốc độ bảo đảm trên hàm lồi.

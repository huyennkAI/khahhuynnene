# Không gian Banach

> Không gian Banach là sân khấu chính của Giải tích hàm: một không gian vector có chuẩn (độ dài, khoảng cách) và **đầy đủ** theo nghĩa mọi chuỗi Cauchy đều hội tụ. Tính đầy đủ nghe có vẻ kỹ thuật, nhưng đây là tính chất căn bản nhất — không có nó, ta không thể "đi tới giới hạn" một cách an toàn, và hầu hết mọi định lý lớn đều sụp đổ.
>
> $$\|x_n - x_m\| \to 0 \implies \exists x \in X : \|x_n - x\| \to 0$$

---

# 1. Ôn tập: không gian metric và không gian vector có chuẩn

## 1.1. Không gian metric

Một **không gian metric** là cặp $(X, d)$ gồm tập $X$ và hàm khoảng cách $d: X \times X \to [0, \infty)$ thỏa:

1. $d(x, y) = 0 \iff x = y$ (đồng nhất)
2. $d(x, y) = d(y, x)$ (đối xứng)
3. $d(x, z) \le d(x, y) + d(y, z)$ (bất đẳng thức tam giác)

Trực giác: metric đo "khoảng cách" giữa hai phần tử. Không gian metric cho phép ta nói về hội tụ, liên tục, và "gần nhau" mà không cần cấu trúc đại số nào.

**Ví dụ nhanh:** Trên $\mathbb{R}^n$, metric Euclid $d(x,y) = \sqrt{\sum_i (x_i - y_i)^2}$; metric taxi $d(x,y) = \sum_i |x_i - y_i|$; metric sup $d(x,y) = \max_i |x_i - y_i|$.

## 1.2. Không gian vector có chuẩn (Normed vector space)

Một **chuẩn (norm)** trên không gian vector thực $X$ là hàm $\|\cdot\| : X \to [0, \infty)$ thỏa:

1. $\|x\| = 0 \iff x = 0$ (xác định dương)
2. $\|\lambda x\| = |\lambda| \|x\|$ với mọi $\lambda \in \mathbb{R}$, $x \in X$ (thuần nhất)
3. $\|x + y\| \le \|x\| + \|y\|$ (bất đẳng thức tam giác)

Chuẩn sinh ra metric tự nhiên: $d(x, y) = \|x - y\|$. Điều này có nghĩa là mọi normed space đều là metric space, nhưng chiều ngược lại không đúng — metric space không nhất thiết có cấu trúc tuyến tính.

Tại sao ta cần chuẩn hơn là chỉ metric? Vì chuẩn **tương thích với cấu trúc tuyến tính**: nó cho phép ta cộng và nhân vô hướng các phần tử và vẫn kiểm soát được "kích thước" của kết quả.

## 1.3. Các chuẩn phổ biến trên $\mathbb{R}^n$

Với $x = (x_1, \dots, x_n) \in \mathbb{R}^n$:

$$\|x\|_1 = \sum_{i=1}^n |x_i|, \qquad \|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}, \qquad \|x\|_\infty = \max_{1 \le i \le n} |x_i|$$

Tổng quát hơn, **chuẩn $\ell^p$** ($1 \le p < \infty$):

$$\|x\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$$

Khi $p \to \infty$ thì $\|x\|_p \to \|x\|_\infty$. Trên $\mathbb{R}^n$ mọi chuẩn đều **tương đương** (equivalent) theo nghĩa tồn tại hằng số $c_1, c_2 > 0$ sao cho $c_1 \|x\|_\alpha \le \|x\|_\beta \le c_2 \|x\|_\alpha$ với mọi $x$. Tính tương đương này **chỉ đúng trong chiều hữu hạn** — đây là điểm khác biệt căn bản giữa chiều hữu hạn và vô hạn.

---

# 2. Chuỗi Cauchy và tính đầy đủ

## 2.1. Định nghĩa

Trong một normed space $(X, \|\cdot\|)$, dãy $(x_n)_{n=1}^\infty$ được gọi là **chuỗi Cauchy** nếu:

$$\forall \varepsilon > 0, \; \exists N \in \mathbb{N} : \; \forall m, n \ge N, \; \|x_n - x_m\| < \varepsilon$$

Nói cách khác, các phần tử của dãy "tụ gần nhau" khi chỉ số đủ lớn.

**Quan sát cơ bản:** Mọi dãy hội tụ đều là chuỗi Cauchy (bởi bất đẳng thức tam giác: $\|x_n - x_m\| \le \|x_n - x\| + \|x - x_m\| \to 0$). Câu hỏi ngược lại — mọi chuỗi Cauchy có hội tụ không? — phụ thuộc vào không gian.

## 2.2. Ví dụ về không đầy đủ

Xét $\mathbb{Q}$ (số hữu tỉ) với giá trị tuyệt đối thông thường. Dãy $x_1 = 1, x_2 = 1.4, x_3 = 1.41, x_4 = 1.414, \dots$ (xấp xỉ thập phân của $\sqrt{2}$) là chuỗi Cauchy trong $\mathbb{Q}$, nhưng giới hạn $\sqrt{2} \notin \mathbb{Q}$. Như vậy $\mathbb{Q}$ không đầy đủ — nó có "lỗ hổng".

$\mathbb{R}$ được xây dựng chính xác để lấp đầy những lỗ hổng này: đây là **hoàn thiện (completion)** của $\mathbb{Q}$.

## 2.3. Không gian Banach

**Định nghĩa:** Một normed vector space $(X, \|\cdot\|)$ được gọi là **không gian Banach** nếu nó **đầy đủ (complete)** — tức là mọi chuỗi Cauchy trong $X$ đều hội tụ đến một phần tử trong $X$.

Đây là định nghĩa trung tâm của chương này. Tên gọi để tưởng nhớ Stefan Banach (1892–1945), nhà toán học người Ba Lan người đặt nền móng cho lý thuyết này.

---

# 3. Các ví dụ quan trọng

## 3.1. $\mathbb{R}^n$ với chuẩn bất kỳ

$\mathbb{R}^n$ với bất kỳ chuẩn nào (kể cả $\|\cdot\|_1$, $\|\cdot\|_2$, $\|\cdot\|_\infty$, hay bất kỳ chuẩn nào) đều là không gian Banach. Điều này xuất phát từ tính đầy đủ của $\mathbb{R}$ và thực tế rằng trên $\mathbb{R}^n$ mọi chuẩn đều tương đương, nên một dãy là Cauchy theo chuẩn này khi và chỉ khi nó Cauchy theo chuẩn kia.

## 3.2. Không gian $\ell^p$ ($1 \le p \le \infty$)

Đây là họ Banach space quan trọng nhất của dãy số:

**Định nghĩa $\ell^p$ ($1 \le p < \infty$):** Tập tất cả các dãy số thực $(x_1, x_2, x_3, \dots)$ sao cho $\sum_{n=1}^\infty |x_n|^p < \infty$, với chuẩn:

$$\|(x_n)\|_p = \left(\sum_{n=1}^\infty |x_n|^p\right)^{1/p}$$

**Định nghĩa $\ell^\infty$:** Tập tất cả các dãy số thực bị chặn, với chuẩn:

$$\|(x_n)\|_\infty = \sup_{n \ge 1} |x_n|$$

**Ví dụ cụ thể:**
- Dãy $x = (1, 1/2, 1/3, 1/4, \dots)$: $\|x\|_2 = \sqrt{\pi^2/6} \approx 1.28$ (hội tụ), nhưng $\|x\|_1 = \sum 1/n = \infty$ (phân kỳ). Vậy $x \in \ell^2$ nhưng $x \notin \ell^1$.
- Dãy $y = (1, 0, 0, 0, \dots)$ (chỉ phần tử đầu khác 0): $y \in \ell^p$ với mọi $p$.
- $c_0$ = tập các dãy hội tụ về 0, với chuẩn $\ell^\infty$, cũng là Banach.

**Hệ thứ tự:** $\ell^1 \subset \ell^2 \subset \ell^p \subset \ell^q \subset \ell^\infty$ với $1 < p < q < \infty$. Nếu $(x_n) \in \ell^1$ thì $(x_n) \in \ell^p$ cho mọi $p \ge 1$.

## 3.3. $C([a,b])$ với chuẩn sup — Banach

Không gian các hàm liên tục $f: [a,b] \to \mathbb{R}$ với chuẩn:

$$\|f\|_\infty = \sup_{x \in [a,b]} |f(x)| = \max_{x \in [a,b]} |f(x)|$$

(dấu sup bằng max vì $f$ liên tục trên compact). Đây là **Banach space**. Tại sao?

Giả sử $(f_n)$ là chuỗi Cauchy trong $(C([a,b]), \|\cdot\|_\infty)$. Với mọi $\varepsilon > 0$, tồn tại $N$ sao cho $\|f_n - f_m\|_\infty < \varepsilon$ với $n, m \ge N$. Điều này có nghĩa là với **mọi** $x \in [a,b]$:

$$|f_n(x) - f_m(x)| \le \|f_n - f_m\|_\infty < \varepsilon$$

Vậy với mỗi $x$ cố định, $(f_n(x))$ là dãy Cauchy trong $\mathbb{R}$, do đó hội tụ đến $f(x) := \lim_{n\to\infty} f_n(x)$. Sự hội tụ là **đều (uniform)** vì $\varepsilon$ không phụ thuộc $x$. Và giới hạn đều của hàm liên tục là liên tục. Vậy $f \in C([a,b])$ và $\|f_n - f\|_\infty \to 0$.

**Ý nghĩa thực tế:** Hội tụ trong $(C([a,b]), \|\cdot\|_\infty)$ chính là **hội tụ đều**. Đây là chuẩn "tự nhiên" nhất để so sánh các hàm liên tục.

## 3.4. $C([a,b])$ với chuẩn $L^1$ — KHÔNG là Banach

Xét cùng không gian hàm liên tục, nhưng với chuẩn khác:

$$\|f\|_1 = \int_a^b |f(x)| \, dx$$

Không gian này **không đầy đủ**. Để thấy tại sao, xây dựng chuỗi Cauchy hội tụ ra ngoài $C([a,b])$.

**Ví dụ cụ thể** (trên $[0,1]$): Cho

$$f_n(x) = \begin{cases} 0 & 0 \le x \le \frac{1}{2} - \frac{1}{n} \\ n\left(x - \frac{1}{2} + \frac{1}{n}\right) & \frac{1}{2} - \frac{1}{n} \le x \le \frac{1}{2} \\ 1 & \frac{1}{2} \le x \le 1 \end{cases}$$

Mỗi $f_n$ liên tục trên $[0,1]$. Ta tính:

$$\|f_n - f_m\|_1 = \int_0^1 |f_n(x) - f_m(x)| \, dx \le \int_{1/2 - 1/\min(m,n)}^{1/2} 1 \, dx = \frac{1}{\min(m,n)} \to 0$$

Vậy $(f_n)$ là chuỗi Cauchy trong $(C([0,1]), \|\cdot\|_1)$. Nhưng giới hạn pointwise là hàm bước Heaviside:

$$f(x) = \begin{cases} 0 & x < 1/2 \\ 1 & x \ge 1/2 \end{cases}$$

Hàm này **không liên tục** tại $x = 1/2$, nên $f \notin C([0,1])$. Chuỗi Cauchy không hội tụ trong không gian — tức là $(C([0,1]), \|\cdot\|_1)$ không đầy đủ, không phải Banach.

**Bài học quan trọng:** Cùng một tập hàm, nhưng chuẩn khác nhau cho kết quả hoàn toàn khác nhau về tính đầy đủ. Không gian thực sự phù hợp với chuẩn $L^1$ là $L^1([0,1])$ — không gian các hàm tích phân Lebesgue, sẽ được đề cập ở File 8.

---

# 4. Chứng minh $\ell^1$ là Banach

Đây là chứng minh kinh điển, thể hiện kỹ thuật chuẩn để kiểm tra tính đầy đủ.

**Định lý:** $(\ell^1, \|\cdot\|_1)$ là không gian Banach.

**Chứng minh:**

Gọi $(x^{(k)})_{k=1}^\infty$ là chuỗi Cauchy trong $\ell^1$, trong đó $x^{(k)} = (x^{(k)}_1, x^{(k)}_2, x^{(k)}_3, \dots)$.

**Bước 1: Tìm giới hạn ứng viên.**

Với mọi $\varepsilon > 0$, tồn tại $K$ sao cho $\|x^{(k)} - x^{(m)}\|_1 < \varepsilon$ với $k, m \ge K$. Vậy với mỗi $j$ cố định:

$$|x^{(k)}_j - x^{(m)}_j| \le \sum_{n=1}^\infty |x^{(k)}_n - x^{(m)}_n| = \|x^{(k)} - x^{(m)}\|_1 < \varepsilon$$

Do đó $(x^{(k)}_j)_k$ là chuỗi Cauchy trong $\mathbb{R}$, hội tụ đến $x_j := \lim_{k\to\infty} x^{(k)}_j$.

Đặt $x = (x_1, x_2, x_3, \dots)$. Ta cần chứng minh (a) $x \in \ell^1$ và (b) $\|x^{(k)} - x\|_1 \to 0$.

**Bước 2: $x \in \ell^1$.**

Với $k \ge K$ và mọi $N$ hữu hạn:

$$\sum_{j=1}^N |x^{(k)}_j - x^{(m)}_j| \le \|x^{(k)} - x^{(m)}\|_1 < \varepsilon$$

Cho $m \to \infty$ (với $N$, $k$ cố định):

$$\sum_{j=1}^N |x^{(k)}_j - x_j| \le \varepsilon$$

Cho $N \to \infty$: $\|x^{(k)} - x\|_1 \le \varepsilon$ với $k \ge K$.

Áp dụng bất đẳng thức tam giác:

$$\|x\|_1 \le \|x - x^{(k)}\|_1 + \|x^{(k)}\|_1 \le \varepsilon + \|x^{(k)}\|_1 < \infty$$

(Vì $x^{(k)} \in \ell^1$.) Vậy $x \in \ell^1$.

**Bước 3: $\|x^{(k)} - x\|_1 \to 0$.**

Từ Bước 2: $\|x^{(k)} - x\|_1 \le \varepsilon$ với $k \ge K$. Vì $\varepsilon > 0$ tùy ý, ta kết luận $\|x^{(k)} - x\|_1 \to 0$. $\square$

**Kỹ thuật tổng quát:** Cùng kiểu lập luận này dùng để chứng minh $\ell^p$ ($1 \le p < \infty$) và $\ell^\infty$ đều là Banach.

---

# 5. Chuỗi tuyệt đối hội tụ trong Banach space

Có một đặc trưng rất hữu ích của Banach space liên quan đến chuỗi (series):

**Định lý:** Một normed space $X$ là Banach khi và chỉ khi mọi **chuỗi tuyệt đối hội tụ (absolutely convergent series)** đều hội tụ, tức là:

$$\sum_{n=1}^\infty \|x_n\| < \infty \implies \sum_{n=1}^\infty x_n \text{ hội tụ trong } X$$

**Chứng minh chiều thuận ($\Rightarrow$):** Đặt $S_N = \sum_{n=1}^N x_n$. Với $m > N$:

$$\|S_m - S_N\| = \left\|\sum_{n=N+1}^m x_n\right\| \le \sum_{n=N+1}^m \|x_n\| \to 0$$

vì $\sum \|x_n\| < \infty$. Vậy $(S_N)$ là chuỗi Cauchy, và do $X$ Banach, nó hội tụ.

**Tại sao hữu ích?** Đặc trưng này cho phép ta chứng minh hội tụ của một chuỗi chỉ bằng cách ước lượng các chuẩn từng số hạng — kỹ thuật quen thuộc từ giải tích thực. Ví dụ: trong $L^2$ (một Banach space), để chứng minh chuỗi Fourier hội tụ, ta chỉ cần kiểm tra $\sum \|f_n\|_2 < \infty$.

---

# 6. Định lý điểm bất động Banach

Đây là ứng dụng trực tiếp và đẹp nhất của tính đầy đủ:

**Định nghĩa:** Ánh xạ $T: X \to X$ là **co (contraction)** nếu tồn tại $0 \le q < 1$ sao cho $\|Tx - Ty\| \le q\|x - y\|$ với mọi $x, y \in X$.

**Định lý Điểm bất động Banach (Banach Fixed Point Theorem):** Nếu $X$ là Banach space và $T: X \to X$ là co, thì $T$ có **đúng một điểm bất động** $x^* \in X$ (tức là $Tx^* = x^*$), và với mọi điểm xuất phát $x_0 \in X$, dãy lặp $x_{n+1} = Tx_n$ hội tụ đến $x^*$ với tốc độ:

$$\|x_n - x^*\| \le \frac{q^n}{1-q} \|x_1 - x_0\|$$

**Chứng minh phác thảo:** Đặt $x_{n+1} = Tx_n$. Ta có:

$$\|x_{n+1} - x_n\| = \|Tx_n - Tx_{n-1}\| \le q\|x_n - x_{n-1}\| \le q^n \|x_1 - x_0\|$$

Dùng bất đẳng thức tam giác với $m > n$:

$$\|x_m - x_n\| \le \sum_{k=n}^{m-1} \|x_{k+1} - x_k\| \le \|x_1 - x_0\| \sum_{k=n}^{m-1} q^k \le \frac{q^n}{1-q}\|x_1 - x_0\| \to 0$$

Vậy $(x_n)$ Cauchy, và do $X$ Banach, hội tụ đến $x^*$. Lấy giới hạn trong $x_{n+1} = Tx_n$: vì $T$ liên tục, $x^* = Tx^*$.

**Các ứng dụng:**
- **Phương trình vi phân:** Định lý Picard-Lindelöf (tồn tại và duy nhất nghiệm ODE) là hệ quả trực tiếp.
- **Phương trình tích phân:** Phương trình Fredholm $f(x) = g(x) + \int K(x,y)f(y)dy$ giải được khi $\|K\|_{L^2}$ đủ nhỏ.
- **Thuật toán lặp:** Nhiều thuật toán trong học máy (ví dụ EM algorithm, một số phương pháp tối ưu) có thể phân tích qua lăng kính điểm bất động.

---

# 7. Hoàn thiện của một normed space

Mọi normed space đều có thể được **hoàn thiện (completed)** thành một Banach space. Nghĩa là:

**Định lý:** Nếu $(X, \|\cdot\|)$ là normed space thì tồn tại Banach space $(\hat{X}, \|\cdot\|_{\hat{X}})$ và đẳng cấu tuyến tính đẳng chuẩn (isometric isomorphism) $\iota: X \to \hat{X}$ sao cho $\iota(X)$ là tập con đặc trong $\hat{X}$. Hơn nữa, $\hat{X}$ là duy nhất (sai biệt đến một đẳng cấu).

**Ý nghĩa:** Mọi không gian "có lỗ hổng" đều có thể lấp đầy bằng cách thêm vào "giới hạn" của tất cả chuỗi Cauchy. Đây chính xác là cách $\mathbb{R}$ được xây dựng từ $\mathbb{Q}$, và cách $L^p$ được xây dựng từ $C([a,b])$.

**Ví dụ:** Hoàn thiện của $(C([0,1]), \|\cdot\|_2)$ là $L^2([0,1])$ — không gian các hàm bình phương tích phân được Lebesgue. Đây là lý do $L^2$ quan trọng hơn $C([0,1])$ trong nhiều ngữ cảnh.

---

# 8. Phân biệt chuẩn tương đương và không tương đương

## 8.1. Chuẩn tương đương

Hai chuẩn $\|\cdot\|_\alpha$ và $\|\cdot\|_\beta$ trên $X$ được gọi là **tương đương** nếu tồn tại $c_1, c_2 > 0$ sao cho:

$$c_1 \|x\|_\alpha \le \|x\|_\beta \le c_2 \|x\|_\alpha \quad \text{với mọi } x \in X$$

Khi hai chuẩn tương đương, mọi khái niệm tôpô (hội tụ, đóng, mở, đầy đủ) đều giống nhau.

**Định lý (chiều hữu hạn):** Trên $\mathbb{R}^n$, mọi hai chuẩn đều tương đương.

## 8.2. Trong chiều vô hạn: chuẩn không tương đương

Trên $C([0,1])$, chuẩn sup $\|\cdot\|_\infty$ và chuẩn $L^1$ $\|\cdot\|_1$ **không tương đương**.

**Bằng chứng:** Xét $f_n(x) = x^n$. Thì $\|f_n\|_\infty = 1$ với mọi $n$, nhưng $\|f_n\|_1 = \int_0^1 x^n dx = \frac{1}{n+1} \to 0$. Không thể có $c_1 \|f_n\|_1 \le \|f_n\|_\infty$ với hằng số $c_1$ độc lập $n$.

Điều này lý giải tại sao $(C([0,1]), \|\cdot\|_\infty)$ là Banach còn $(C([0,1]), \|\cdot\|_1)$ thì không: tính đầy đủ **phụ thuộc vào chuẩn**, không chỉ phụ thuộc vào tập hàm.

---

# 9. Banach space trong học máy

## 9.1. RKHS là Hilbert space (trường hợp đặc biệt)

Trong kernel methods (SVM, Gaussian Processes), **Reproducing Kernel Hilbert Space (RKHS)** là không gian Hilbert — tức là Banach space với tích trong. RKHS $\mathcal{H}_K$ được định nghĩa bởi kernel $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ và có tính chất kỳ diệu: **evaluation functional** $\delta_x(f) = f(x)$ là phiếm hàm tuyến tính **bị chặn** (bounded). Điều này không tầm thường — trong $L^2$ thông thường, $\delta_x$ không bị chặn (vì $f$ chỉ xác định đến tập đo 0).

## 9.2. Không gian hàm mất (loss function space)

Trong lý thuyết học thống kê, ta thường làm việc với các lớp hàm trong $L^p$ hoặc các Sobolev space (là Banach). Tính đầy đủ đảm bảo rằng giới hạn của chuỗi hàm tốt (theo một nghĩa nào đó) vẫn nằm trong lớp hàm đó.

## 9.3. Optimization trên Banach space

Gradient descent trong neural network về cơ bản là lặp trên Banach space (không gian tham số). Sự hội tụ của các phương pháp lặp thường được chứng minh qua điểm bất động Banach hoặc các biến thể.

---

# 10. Tóm tắt và bảng so sánh

| Không gian | Chuẩn | Banach? | Ghi chú |
|---|---|---|---|
| $\mathbb{R}^n$ | $\|\cdot\|_p$ (mọi $p$) | Có | Chiều hữu hạn, mọi chuẩn tương đương |
| $\ell^p$ ($1 \le p \le \infty$) | $\|\cdot\|_p$ | Có | Không gian dãy số |
| $C([a,b])$ | $\|\cdot\|_\infty$ (sup) | Có | Hội tụ $=$ hội tụ đều |
| $C([a,b])$ | $\|\cdot\|_1$ (tích phân) | Không | Hoàn thiện là $L^1([a,b])$ |
| $L^p([a,b])$ | $\|\cdot\|_p$ | Có | Cần tích phân Lebesgue |
| Đa thức $\mathcal{P}([0,1])$ | $\|\cdot\|_\infty$ | Không | Hoàn thiện là $C([0,1])$ |

**Công thức cốt lõi cần nhớ:**

$$X \text{ Banach} \iff \left(\sum_{n=1}^\infty \|x_n\| < \infty \implies \sum_{n=1}^\infty x_n \text{ hội tụ trong } X\right)$$

> Bài tiếp theo — **Toán tử tuyến tính bị chặn** — sẽ khám phá các ánh xạ tuyến tính giữa Banach space, khi nào chúng liên tục, và cách đo "kích thước" của chúng qua chuẩn toán tử. Đây là bước chuyển từ việc nghiên cứu không gian sang nghiên cứu ánh xạ giữa chúng.

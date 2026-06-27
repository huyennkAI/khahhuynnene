# Độ đo ngoài & Sigma-đại số

> Tích phân Riemann có một giới hạn cốt tử: nó chỉ làm việc với các hàm "đủ tốt" trên các tập "đủ tốt". Giải tích Lebesgue phá bỏ giới hạn đó bằng cách xây dựng lại từ đầu khái niệm "đo độ dài" và "tích phân" theo một nguyên tắc khác — không chia nhỏ trục $x$, mà chia nhỏ trục $y$ (miền giá trị). Nền tảng của toàn bộ cấu trúc là **sigma-đại số** và **độ đo**.
>
> $$\lambda^*(E) = \inf\left\{\sum_{k=1}^\infty |I_k| \;:\; E \subset \bigcup_{k=1}^\infty I_k,\; I_k \text{ là khoảng}\right\}$$

---

# 1. Giới hạn của tích phân Riemann

## 1.1. Hàm Dirichlet: ví dụ điển hình

**Hàm Dirichlet** trên $[0,1]$:

$$D(x) = \begin{cases} 1 & x \in \mathbb{Q} \\ 0 & x \notin \mathbb{Q} \end{cases}$$

Hàm này **không tích phân được Riemann**. Tại sao? Với mọi phân hoạch $0 = x_0 < x_1 < \cdots < x_n = 1$, trong mỗi khoảng $[x_{i-1}, x_i]$ đều có cả số hữu tỉ và vô tỉ, nên:

$$\text{Tổng Riemann trên} = \sum_{i=1}^n 1 \cdot (x_i - x_{i-1}) = 1$$
$$\text{Tổng Riemann dưới} = \sum_{i=1}^n 0 \cdot (x_i - x_{i-1}) = 0$$

Tổng trên và dưới không bao giờ hội tụ về nhau, dù lưới phân hoạch mịn đến đâu.

Nhưng về mặt "trực giác", $D(x) = 0$ "hầu hết khắp nơi" (trừ một tập đếm được — tập hữu tỉ $\mathbb{Q}$), và tích phân "đúng" phải là 0. Lebesgue cho phép ta tính được điều này.

## 1.2. Tập Cantor: ví dụ phức tạp hơn

**Tập Cantor** $C$ xây dựng bằng cách bắt đầu từ $[0,1]$ và liên tục loại bỏ đoạn giữa:

- Bước 1: loại $(\frac{1}{3}, \frac{2}{3})$, còn $[0,\frac{1}{3}] \cup [\frac{2}{3},1]$
- Bước 2: loại hai đoạn giữa, còn 4 đoạn độ dài $\frac{1}{9}$
- Bước $n$: còn $2^n$ đoạn, mỗi đoạn độ dài $(\frac{1}{3})^n$

Tổng độ dài bị loại: $\frac{1}{3} + 2 \cdot \frac{1}{9} + 4 \cdot \frac{1}{27} + \cdots = \frac{1/3}{1-2/3} = 1$.

Tập Cantor có **độ đo Lebesgue 0** nhưng có **lực lượng continuum** (tương đương $[0,1]$). Đây là ví dụ về tập "nhỏ" theo đo nhưng "lớn" về tổ hợp.

## 1.3. Vấn đề tổng quát của Riemann

Tích phân Riemann không hoạt động tốt với:
- **Giới hạn pointwise** của dãy hàm khả tích
- Hàm **gián đoạn trên tập phức tạp**
- Các hàm trong **$L^2$** (quan trọng cho chuỗi Fourier)

Lebesgue giải quyết tất cả bằng cách xây dựng nền tảng đo lường chắc chắn hơn.

---

# 2. Độ đo ngoài Lebesgue

## 2.1. Ý tưởng cơ bản

Ý tưởng xuất phát từ câu hỏi: "Chiều dài của tập $E \subset \mathbb{R}$ là bao nhiêu?"

Đối với khoảng $[a,b]$, chiều dài rõ ràng là $b-a$. Với tập phức tạp hơn, ta **xấp xỉ từ bên ngoài** bằng hợp đếm được của khoảng:

**Định nghĩa:** **Độ đo ngoài Lebesgue** của $E \subset \mathbb{R}$:

$$\lambda^*(E) = \inf\left\{\sum_{k=1}^\infty |I_k| \;:\; E \subset \bigcup_{k=1}^\infty I_k, \; I_k = (a_k, b_k)\right\}$$

trong đó infimum lấy trên mọi phủ đếm được bằng khoảng mở.

## 2.2. Tính chất của độ đo ngoài

1. **Không âm:** $\lambda^*(E) \ge 0$ với mọi $E$.
2. **Trống:** $\lambda^*(\emptyset) = 0$.
3. **Đơn điệu:** $A \subset B \Rightarrow \lambda^*(A) \le \lambda^*(B)$.
4. **Bán cộng đếm được (sigma-subadditive):**

$$\lambda^*\left(\bigcup_{k=1}^\infty E_k\right) \le \sum_{k=1}^\infty \lambda^*(E_k)$$

5. **Khoảng:** $\lambda^*([a,b]) = \lambda^*((a,b)) = b - a$.
6. **Bất biến tịnh tiến:** $\lambda^*(E + t) = \lambda^*(E)$ với mọi $t \in \mathbb{R}$.

## 2.3. Tại sao không phải "độ đo" thực sự?

Độ đo ngoài chỉ **bán cộng** ($\le$), không **cộng hoàn toàn**. Với tập đặc biệt (Vitali set, xem Mục 4), tính cộng có thể sai. Để có "độ đo thực sự" với $\sigma$-cộng, ta cần giới hạn vào các tập "đo được" — đây là vai trò của sigma-đại số.

---

# 3. Sigma-đại số

## 3.1. Định nghĩa

**Định nghĩa:** Một **sigma-đại số (sigma-algebra)** trên tập $\Omega$ là họ $\mathcal{F} \subset 2^\Omega$ thỏa:

1. $\Omega \in \mathcal{F}$ (chứa toàn bộ không gian)
2. $A \in \mathcal{F} \Rightarrow A^c = \Omega \setminus A \in \mathcal{F}$ (đóng với phần bù)
3. $A_1, A_2, A_3, \dots \in \mathcal{F} \Rightarrow \bigcup_{n=1}^\infty A_n \in \mathcal{F}$ (đóng với hợp đếm được)

**Hệ quả:** $\emptyset \in \mathcal{F}$, và đóng với giao đếm được ($\bigcap A_n = (\bigcup A_n^c)^c$), hiệu ($A \setminus B = A \cap B^c$).

## 3.2. Ví dụ sigma-đại số

**Ví dụ 1: Hai sigma-đại số tầm thường (trivial)**

$$\mathcal{F}_{\min} = \{\emptyset, \Omega\} \quad \text{(nhỏ nhất)}$$
$$\mathcal{F}_{\max} = 2^\Omega \quad \text{(lớn nhất — tập lũy thừa)}$$

**Ví dụ 2: Sigma-đại số sinh bởi phân hoạch**

Nếu $\Omega = \{1, 2, 3, 4\}$ và xét phân hoạch $\{\{1,2\}, \{3,4\}\}$, sigma-đại số sinh ra là:

$$\{\emptyset, \{1,2\}, \{3,4\}, \{1,2,3,4\}\}$$

**Ví dụ 3: Sigma-đại số Borel — quan trọng nhất**

Trên $\mathbb{R}$, **sigma-đại số Borel** $\mathcal{B}(\mathbb{R})$ là sigma-đại số **nhỏ nhất** chứa mọi khoảng mở:

$$\mathcal{B}(\mathbb{R}) = \sigma(\{(a,b) : a < b\}) = \sigma(\text{mọi tập mở})$$

Phần tử của $\mathcal{B}(\mathbb{R})$ gọi là **tập Borel**. Bao gồm: mọi khoảng mở, đóng, nửa đóng; mọi tập $G_\delta$ (giao đếm được của mở), $F_\sigma$ (hợp đếm được của đóng); và nhiều tập phức tạp hơn.

Hầu hết các tập ta gặp trong thực tế đều là Borel.

## 3.3. Sigma-đại số sinh ra

**Định nghĩa:** Với họ $\mathcal{G} \subset 2^\Omega$ bất kỳ, **sigma-đại số sinh ra bởi $\mathcal{G}$**:

$$\sigma(\mathcal{G}) = \bigcap \{\mathcal{F} : \mathcal{F} \text{ là sigma-đại số}, \mathcal{G} \subset \mathcal{F}\}$$

là sigma-đại số nhỏ nhất chứa $\mathcal{G}$. Giao của các sigma-đại số cũng là sigma-đại số, nên định nghĩa này hợp lệ.

**Ví dụ:** Sigma-đại số Borel $\mathcal{B}(\mathbb{R}) = \sigma(\tau_{\mathbb{R}})$ với $\tau_{\mathbb{R}}$ là tôpô (họ tập mở) trên $\mathbb{R}$. Cũng có thể sinh ra từ:
- Các khoảng $(a, b)$
- Các khoảng $(-\infty, b]$
- Các khoảng $[a, b)$
- Các khoảng $[a, b]$

Mọi họ trên đều sinh ra cùng sigma-đại số Borel.

---

# 4. Tại sao cần sigma-đại số: Tập Vitali không đo được

## 4.1. Các yêu cầu của độ đo

Ta muốn xây dựng "độ đo" $\mu$ trên tập con của $[0,1]$ thỏa:
1. $\mu([a,b]) = b - a$ (độ dài)
2. **Sigma-cộng:** nếu $A = \bigsqcup A_n$ thì $\mu(A) = \sum \mu(A_n)$
3. **Bất biến tịnh tiến:** $\mu(E + t \mod 1) = \mu(E)$

## 4.2. Tập Vitali

**Xây dựng (Vitali, 1905):** Trên $[0,1]$ (hay $\mathbb{R}/\mathbb{Z}$), định nghĩa quan hệ tương đương: $x \sim y \iff x - y \in \mathbb{Q}$.

Mỗi lớp tương đương có dạng $x + \mathbb{Q}$. Bằng **Axiom of Choice**, chọn đúng một đại diện từ mỗi lớp — gọi là tập Vitali $V$.

**Chứng minh $V$ không đo được:**

Xét $V + q = \{v + q : v \in V\}$ với $q \in \mathbb{Q} \cap [-1,1]$. Các tập $V + q$ **rời nhau** (hai phần tử cùng lớp không thể vào $V$), và:

$$[0,1] \subset \bigcup_{q \in \mathbb{Q} \cap [-1,1]} (V + q) \subset [-1,2]$$

Nếu $V$ đo được với $\lambda(V) = \ell$:
- $\ell > 0$: $\sum_q \lambda(V+q) = \sum_q \ell = \infty > \lambda([-1,2]) = 3$. Mâu thuẫn.
- $\ell = 0$: $\lambda([0,1]) \le \sum_q \lambda(V+q) = 0$. Mâu thuẫn.

Vậy $V$ **không đo được Lebesgue**. Sự tồn tại của $V$ phụ thuộc vào Axiom of Choice — trong toán học không có Axiom of Choice (ví dụ mô hình Solovay), mọi tập $\mathbb{R}$ đều đo được!

## 4.3. Bài học

Không thể định nghĩa độ đo trên **toàn bộ $2^\mathbb{R}$** mà thỏa cả ba yêu cầu. Phải giới hạn vào một **sigma-đại số** $\mathcal{L} \subset 2^\mathbb{R}$ — chứa "đủ" các tập mà ta quan tâm, nhưng không phải tất cả.

---

# 5. Tiêu chuẩn Carathéodory và tập đo được

## 5.1. Điều kiện Carathéodory

**Định nghĩa (Carathéodory, 1914):** Tập $E \subset \mathbb{R}$ được gọi là **đo được Lebesgue** nếu với mọi $A \subset \mathbb{R}$:

$$\lambda^*(A) = \lambda^*(A \cap E) + \lambda^*(A \cap E^c)$$

Trực giác: $E$ đo được nếu nó "cắt mọi tập $A$ một cách nhất quán" — cộng hai phần.

**Nhận xét:** Luôn có $\lambda^*(A) \le \lambda^*(A \cap E) + \lambda^*(A \cap E^c)$ (từ tính bán cộng). Điều kiện Carathéodory là chiều ngược: **đẳng thức**.

## 5.2. Sigma-đại số Lebesgue

**Định lý:** Họ $\mathcal{L}$ tất cả các tập đo được Lebesgue là một **sigma-đại số**, và $\lambda = \lambda^*|_{\mathcal{L}}$ là **độ đo** (có tính sigma-cộng đầy đủ trên $\mathcal{L}$).

Hơn nữa:
- $\mathcal{B}(\mathbb{R}) \subset \mathcal{L}$ (mọi tập Borel đo được Lebesgue)
- $\mathcal{L}$ **đầy đủ (complete):** nếu $\lambda(N) = 0$ và $E \subset N$ thì $E \in \mathcal{L}$ (tập con của tập đo không cũng đo được)
- $|\mathcal{L}| > |\mathcal{B}(\mathbb{R})|$: có nhiều tập đo được hơn Borel (tập con của Cantor)

## 5.3. So sánh Borel và Lebesgue

| Tính chất | $\mathcal{B}(\mathbb{R})$ (Borel) | $\mathcal{L}$ (Lebesgue) |
|---|---|---|
| Kích thước | Nhỏ hơn | Lớn hơn ($= 2^{2^{\aleph_0}}$) |
| Sinh ra bởi | Tập mở | Tập Borel + tập con null |
| Đầy đủ | Không | Có |
| Hữu ích cho | Xác suất, tôpô | Tích phân Lebesgue |

Tập Cantor $C$ có $\lambda(C) = 0$, nên mọi tập con của $C$ đo được Lebesgue (và đo 0). Nhưng số tập con của $C$ là $2^{|\mathbb{R}|}$ — lớn hơn $|\mathcal{B}(\mathbb{R})|$. Vậy $\mathcal{L} \supsetneq \mathcal{B}(\mathbb{R})$ thực sự.

---

# 6. Không gian đo (Measure Space)

## 6.1. Định nghĩa

**Định nghĩa:** Bộ ba $(\Omega, \mathcal{F}, \mu)$ là **không gian đo (measure space)** nếu:
- $\Omega$ là tập (không gian mẫu)
- $\mathcal{F}$ là sigma-đại số trên $\Omega$
- $\mu: \mathcal{F} \to [0, +\infty]$ là **độ đo (measure)**: $\mu(\emptyset) = 0$ và $\sigma$-cộng

**$\sigma$-cộng:** Nếu $A_1, A_2, \dots \in \mathcal{F}$ rời nhau đôi một:

$$\mu\left(\bigsqcup_{n=1}^\infty A_n\right) = \sum_{n=1}^\infty \mu(A_n)$$

**Không gian đo xác suất:** $(\Omega, \mathcal{F}, P)$ với $P(\Omega) = 1$.

## 6.2. Ví dụ các độ đo quan trọng

**Độ đo Lebesgue trên $\mathbb{R}^n$:**

$$(\mathbb{R}^n, \mathcal{B}(\mathbb{R}^n), \lambda^n), \quad \lambda^n(I_1 \times \cdots \times I_n) = \prod_i |I_i|$$

**Độ đo đếm (counting measure):**

$$(\mathbb{N}, 2^\mathbb{N}, \#), \quad \#(A) = |A| \text{ (số phần tử của } A)$$

Với độ đo đếm: $\int_\mathbb{N} f \, d\# = \sum_{n=1}^\infty f(n)$ — tích phân là tổng thông thường! Không gian $L^p(\mathbb{N}, \#) = \ell^p$.

**Delta Dirac:**

$$(\mathbb{R}, \mathcal{B}(\mathbb{R}), \delta_a), \quad \delta_a(A) = \mathbf{1}_A(a) = \begin{cases} 1 & a \in A \\ 0 & a \notin A \end{cases}$$

Với $\delta_a$: $\int f \, d\delta_a = f(a)$ — tích phân là đánh giá tại điểm.

**Độ đo Gauss:**

$$(\mathbb{R}, \mathcal{B}(\mathbb{R}), \gamma), \quad d\gamma = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx$$

**Độ đo sản phẩm:** $(\Omega_1 \times \Omega_2, \mathcal{F}_1 \otimes \mathcal{F}_2, \mu_1 \otimes \mu_2)$ — nền tảng của Fubini-Tonelli.

## 6.3. Tính chất cơ bản của độ đo

Từ $\sigma$-cộng suy ra:

**Đơn điệu:** $A \subset B \Rightarrow \mu(A) \le \mu(B)$.

**Liên tục từ dưới:** Nếu $A_n \nearrow A$ (tập tăng dần) thì $\mu(A_n) \nearrow \mu(A)$.

**Liên tục từ trên:** Nếu $A_n \searrow A$ (tập giảm dần) và $\mu(A_1) < \infty$ thì $\mu(A_n) \searrow \mu(A)$.

**Chứng minh tính liên tục từ dưới:**

Đặt $B_1 = A_1$, $B_n = A_n \setminus A_{n-1}$ ($n \ge 2$). Các $B_n$ rời nhau và $\bigsqcup_{k=1}^n B_k = A_n$, $\bigsqcup_{k=1}^\infty B_k = A$.

$$\mu(A) = \sum_{k=1}^\infty \mu(B_k) = \lim_{n\to\infty} \sum_{k=1}^n \mu(B_k) = \lim_{n\to\infty} \mu(A_n)$$

---

# 7. Tập đo không (null set) và hoàn thiện

## 7.1. Tập đo không

**Định nghĩa:** $N \in \mathcal{F}$ là **tập đo không (null set)** nếu $\mu(N) = 0$.

**Ví dụ:**
- Tập hữu hạn hay đếm được trong $\mathbb{R}$: $\lambda(\{x_1, x_2, \dots\}) = 0$.
- Tập Cantor: $\lambda(C) = 0$.
- Mọi tập con đếm được của $\mathbb{R}^n$.

**"Hầu hết khắp nơi" (almost everywhere, a.e.):** Tính chất $P$ đúng "a.e." nếu tập điểm vi phạm có độ đo 0.

## 7.2. Độ đo đầy đủ

**Định nghĩa:** $(\Omega, \mathcal{F}, \mu)$ là **đầy đủ** nếu mọi tập con của null set đều đo được:

$$\mu(N) = 0, E \subset N \Rightarrow E \in \mathcal{F}$$

**Hoàn thiện (Completion):** Với mọi không gian đo $(\Omega, \mathcal{F}, \mu)$, tồn tại mở rộng $(\Omega, \bar{\mathcal{F}}, \bar{\mu})$ đầy đủ, trong đó $\bar{\mathcal{F}}$ là sigma-đại số nhỏ nhất chứa $\mathcal{F}$ và mọi tập con của null set.

Hoàn thiện của $(\mathbb{R}, \mathcal{B}(\mathbb{R}), \lambda)$ là $(\mathbb{R}, \mathcal{L}, \lambda)$ — chính là sigma-đại số Lebesgue.

---

# 8. Xây dựng sigma-đại số trong lý thuyết xác suất

## 8.1. Không gian xác suất

Trong lý thuyết xác suất, $(\Omega, \mathcal{F}, P)$ với $P(\Omega) = 1$ là nền tảng:
- $\Omega$: không gian mẫu (sample space)
- $\mathcal{F}$: các "biến cố đo được"
- $P$: xác suất

## 8.2. Biến ngẫu nhiên

**Biến ngẫu nhiên** $X: \Omega \to \mathbb{R}$ phải là **đo được** theo nghĩa:

$$X^{-1}(B) = \{\omega \in \Omega : X(\omega) \in B\} \in \mathcal{F} \quad \text{với mọi } B \in \mathcal{B}(\mathbb{R})$$

Tức là nghịch ảnh của Borel set phải nằm trong $\mathcal{F}$.

**Sigma-đại số sinh bởi $X$:** $\sigma(X) = \{X^{-1}(B) : B \in \mathcal{B}(\mathbb{R})\}$ là sigma-đại số nhỏ nhất khiến $X$ đo được — nó mã hóa "thông tin" từ $X$.

**Filtration** $(\mathcal{F}_t)_{t \ge 0}$ (trong quá trình ngẫu nhiên): họ sigma-đại số tăng dần $\mathcal{F}_s \subset \mathcal{F}_t$ với $s < t$, biểu diễn "thông tin tích lũy theo thời gian". Nền tảng của lý thuyết martingale và tài chính toán học.

## 8.3. Sigma-đại số tích (Product sigma-algebra)

Cho hai không gian đo $(\Omega_1, \mathcal{F}_1)$ và $(\Omega_2, \mathcal{F}_2)$:

$$\mathcal{F}_1 \otimes \mathcal{F}_2 = \sigma(\{A_1 \times A_2 : A_i \in \mathcal{F}_i\})$$

Đây là sigma-đại số nhỏ nhất trên $\Omega_1 \times \Omega_2$ khiến hai hàm chiếu đo được. Trên $\mathbb{R}^n$:

$$\mathcal{B}(\mathbb{R}^n) = \mathcal{B}(\mathbb{R})^{\otimes n} = \mathcal{B}(\mathbb{R}) \otimes \cdots \otimes \mathcal{B}(\mathbb{R})$$

---

# 9. Hàm đo được (Measurable Function)

## 9.1. Định nghĩa

**Định nghĩa:** Hàm $f: (\Omega, \mathcal{F}) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$ là **đo được (measurable)** nếu:

$$f^{-1}(B) \in \mathcal{F} \quad \text{với mọi } B \in \mathcal{B}(\mathbb{R})$$

Đủ để kiểm tra điều kiện này cho họ sinh ra $\mathcal{B}(\mathbb{R})$:

$$f^{-1}((-\infty, a]) \in \mathcal{F} \quad \text{với mọi } a \in \mathbb{R}$$

## 9.2. Ví dụ hàm đo được

**Hàm liên tục** $f: \mathbb{R} \to \mathbb{R}$ đo được Borel (vì $f^{-1}(\text{mở}) = \text{mở}$).

**Hàm đơn điệu** đo được (vì $f^{-1}((-\infty, a])$ là khoảng hoặc hợp các khoảng).

**Giới hạn pointwise của hàm đo được đo được:** Nếu $f_n$ đo được và $f_n(x) \to f(x)$ với mọi $x$, thì:

$$\{x : f(x) > a\} = \{x : \limsup_n f_n(x) > a\} = \bigcap_k \bigcup_n \{x : f_n(x) > a - 1/k\} \in \mathcal{F}$$

Vậy $f$ đo được. Điều này không đúng cho tích phân Riemann (giới hạn của hàm Riemann khả tích có thể không khả tích Riemann).

## 9.3. Các phép toán bảo toàn tính đo được

Nếu $f, g$ đo được thì $f + g$, $fg$, $|f|$, $\max(f,g)$, $\min(f,g)$ đều đo được. Hợp $g \circ f$ của hàm đo được là đo được (nếu hàm ngoài Borel).

---

# 10. Tóm tắt

Chúng ta đã xây dựng nền tảng cho tích phân Lebesgue:

**Bộ khung cơ bản:**

$$\underbrace{(\Omega, \mathcal{F}, \mu)}_{\text{measure space}} \xrightarrow{f \text{ đo được}} \underbrace{\int_\Omega f \, d\mu}_{\text{tích phân Lebesgue}}$$

**Sigma-đại số Borel vs Lebesgue:**

$$\mathcal{B}(\mathbb{R}) \subsetneq \mathcal{L} \subsetneq 2^\mathbb{R}$$

- $\mathcal{B}$: sinh bởi tập mở, đủ cho hầu hết mục đích thực tế
- $\mathcal{L}$: hoàn thiện của $\mathcal{B}$, phù hợp nhất cho lý thuyết

**Thứ tự quan trọng:**

1. Xây dựng $\lambda^*$ (subadditive) trên $2^\mathbb{R}$
2. Chọn sigma-đại số $\mathcal{L}$ các tập "đo được" (Carathéodory)
3. $\lambda^*|_\mathcal{L}$ là độ đo thực sự (sigma-additive)
4. Định nghĩa hàm đo được và tích phân

> Bài tiếp theo — **Độ đo Lebesgue** — sẽ khám phá sâu hơn các tính chất của độ đo Lebesgue và so sánh sigma-đại số Borel với sigma-đại số Lebesgue, trước khi xây dựng tích phân ở File 7.

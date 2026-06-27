# Toán tử tuyến tính bị chặn

> Trong Giải tích hàm, "hàm" không còn ánh xạ số thực sang số thực — ta nghiên cứu **toán tử (operator)**: ánh xạ từ không gian hàm sang không gian hàm, hoặc từ không gian vô hạn chiều này sang không gian khác. Câu hỏi trung tâm: khi nào toán tử tuyến tính **liên tục**? Câu trả lời đẹp một cách bất ngờ: liên tục **tương đương** với bị chặn (bounded).
>
> $$T \text{ liên tục} \iff T \text{ bị chặn} \iff \|T\| := \sup_{\|x\|=1} \|Tx\| < \infty$$

---

# 1. Toán tử tuyến tính: định nghĩa và ví dụ

## 1.1. Định nghĩa

Cho $X$ và $Y$ là hai normed space (thực hoặc phức). Một **toán tử tuyến tính** (linear operator) là ánh xạ $T: X \to Y$ thỏa:

1. $T(x + y) = Tx + Ty$ với mọi $x, y \in X$
2. $T(\lambda x) = \lambda Tx$ với mọi $\lambda \in \mathbb{R}$ (hoặc $\mathbb{C}$), $x \in X$

Hai điều kiện này gộp thành một: $T(\alpha x + \beta y) = \alpha Tx + \beta Ty$.

**Chú ý về ký hiệu:** Ta thường viết $Tx$ thay cho $T(x)$ khi $T$ là toán tử tuyến tính.

## 1.2. Ví dụ cơ bản

**Ví dụ 1: Ma trận.** Nếu $A \in \mathbb{R}^{m \times n}$ thì $T_A: \mathbb{R}^n \to \mathbb{R}^m$ với $T_A(x) = Ax$ là toán tử tuyến tính. Đây là trường hợp chiều hữu hạn.

**Ví dụ 2: Toán tử dịch chuyển (shift) trên $\ell^2$.**

- *Dịch trái (left shift):* $L(x_1, x_2, x_3, \dots) = (x_2, x_3, x_4, \dots)$
- *Dịch phải (right shift):* $R(x_1, x_2, x_3, \dots) = (0, x_1, x_2, x_3, \dots)$

Cả hai đều là toán tử tuyến tính từ $\ell^2$ sang $\ell^2$. Lưu ý $L \circ R = I$ nhưng $R \circ L \ne I$ (không thể dịch ngược để khôi phục phần tử đầu bị mất).

**Ví dụ 3: Toán tử tích phân (Volterra).** Trên $C([0,1])$:

$$Tf(x) = \int_0^x f(t) \, dt$$

Đây là toán tử tuyến tính ánh xạ $C([0,1])$ vào $C([0,1])$ (thực ra vào $C^1([0,1])$).

**Ví dụ 4: Phiếm hàm (functional).** Trường hợp đặc biệt $Y = \mathbb{R}$ (hoặc $\mathbb{C}$). Ví dụ:

$$\varphi(f) = \int_0^1 f(x) \, dx, \quad \text{trên } C([0,1])$$

$$\delta_a(f) = f(a), \quad \text{evaluation functional tại điểm } a$$

## 1.3. Toán tử vi phân: ví dụ KHÔNG bị chặn

Toán tử đạo hàm $D: C^1([0,1]) \to C([0,1])$ định nghĩa $Df = f'$ là tuyến tính, nhưng **không bị chặn** so với chuẩn $\|\cdot\|_\infty$ (xem Mục 4 bên dưới).

---

# 2. Tương đương giữa liên tục và bị chặn

## 2.1. Định nghĩa bị chặn

Toán tử tuyến tính $T: X \to Y$ được gọi là **bị chặn (bounded)** nếu tồn tại hằng số $C > 0$ sao cho:

$$\|Tx\|_Y \le C \|x\|_X \quad \text{với mọi } x \in X$$

Trực giác: $T$ bị chặn nếu nó không "khuếch đại vô hạn" các vector.

## 2.2. Định lý cơ bản

**Định lý:** Cho $T: X \to Y$ là toán tử tuyến tính giữa hai normed space. Các phát biểu sau tương đương:

(a) $T$ liên tục tại một điểm $x_0 \in X$

(b) $T$ liên tục tại $x = 0$

(c) $T$ liên tục đều (uniformly continuous) trên $X$

(d) $T$ bị chặn

**Chứng minh (a) $\Rightarrow$ (d):**

Giả sử $T$ liên tục tại $x_0$. Khi đó với $\varepsilon = 1$, tồn tại $\delta > 0$ sao cho $\|x - x_0\| < \delta \Rightarrow \|Tx - Tx_0\| < 1$.

Với $x \ne 0$ bất kỳ, đặt $y = x_0 + \frac{\delta}{2} \cdot \frac{x}{\|x\|}$. Thì $\|y - x_0\| = \delta/2 < \delta$, nên:

$$\left\|Ty - Tx_0\right\| = \left\|T\left(\frac{\delta x}{2\|x\|}\right)\right\| = \frac{\delta}{2\|x\|}\|Tx\| < 1$$

Do đó $\|Tx\| < \frac{2\|x\|}{\delta}$, tức là $T$ bị chặn với $C = 2/\delta$.

**Chứng minh (d) $\Rightarrow$ (c):**

Nếu $\|Tx\| \le C\|x\|$ thì $\|Tx - Ty\| = \|T(x-y)\| \le C\|x-y\|$, vậy $T$ liên tục Lipschitz, do đó liên tục đều.

**Bài học:** Với toán tử tuyến tính, mọi khái niệm liên tục (tại một điểm, tại 0, đều, Lipschitz) đều **tương đương**. Đây là điều đặc biệt của tuyến tính — đối với hàm phi tuyến, các khái niệm này khác nhau.

---

# 3. Chuẩn toán tử

## 3.1. Định nghĩa

Với toán tử tuyến tính bị chặn $T: X \to Y$, **chuẩn toán tử** được định nghĩa là:

$$\|T\| = \sup_{\|x\|_X = 1} \|Tx\|_Y$$

Có nhiều cách viết tương đương:

$$\|T\| = \sup_{\|x\|_X \le 1} \|Tx\|_Y = \sup_{x \ne 0} \frac{\|Tx\|_Y}{\|x\|_X} = \inf\{C > 0 : \|Tx\| \le C\|x\| \; \forall x\}$$

**Ý nghĩa:** $\|T\|$ là hệ số khuếch đại lớn nhất của $T$ — "toán tử có thể khuếch đại chuẩn lên bao nhiêu lần trong trường hợp xấu nhất?"

## 3.2. Bất đẳng thức cơ bản

Từ định nghĩa, với mọi $x \in X$:

$$\|Tx\|_Y \le \|T\| \cdot \|x\|_X$$

Và $\|T\|$ là hằng số nhỏ nhất thỏa bất đẳng thức này với mọi $x$.

## 3.3. Tính chất chuẩn toán tử

Với $S, T: X \to Y$ bị chặn và $\lambda \in \mathbb{R}$:

1. $\|T\| \ge 0$ và $\|T\| = 0 \iff T = 0$
2. $\|\lambda T\| = |\lambda| \|T\|$
3. $\|S + T\| \le \|S\| + \|T\|$ (bất đẳng thức tam giác)
4. $\|ST\| \le \|S\| \|T\|$ (khi $T: X \to Y$, $S: Y \to Z$)

## 3.4. Ví dụ tính chuẩn toán tử

**Ví dụ 1: Ma trận.** Với $T_A: (\mathbb{R}^n, \|\cdot\|_2) \to (\mathbb{R}^m, \|\cdot\|_2)$, chuẩn toán tử bằng giá trị kỳ dị lớn nhất (largest singular value) $\sigma_{\max}(A)$.

**Ví dụ 2: Toán tử tích phân.** Trên $(C([0,1]), \|\cdot\|_\infty)$:

$$Tf(x) = \int_0^1 K(x,t) f(t) \, dt$$

Ta có $|Tf(x)| \le \int_0^1 |K(x,t)| |f(t)| \, dt \le \|f\|_\infty \int_0^1 |K(x,t)| \, dt$.

Do đó $\|Tf\|_\infty \le \|f\|_\infty \cdot \sup_x \int_0^1 |K(x,t)| dt$, và thực ra:

$$\|T\| = \sup_{x \in [0,1]} \int_0^1 |K(x,t)| \, dt$$

**Ví dụ 3: Dịch trái trên $\ell^2$.** $\|L(x_1, x_2, \dots)\|_2 = \|(x_2, x_3, \dots)\|_2 \le \|(x_1, x_2, \dots)\|_2$. Và đẳng thức đạt khi $x_1 = 0$. Vậy $\|L\| = 1$. Tương tự $\|R\| = 1$.

---

# 4. Toán tử không bị chặn: toán tử vi phân

## 4.1. Toán tử đạo hàm không bị chặn

Xét $D: C^1([0,1]) \to C([0,1])$ với $Df = f'$, dùng chuẩn sup trên cả hai không gian.

**Xây dựng dãy phản ví dụ:** Cho $f_n(x) = \sin(n\pi x)$. Thì:

$$\|f_n\|_\infty = 1 \quad \text{và} \quad \|Df_n\|_\infty = \|n\pi\cos(n\pi x)\|_\infty = n\pi$$

Vậy $\|Df_n\|_\infty / \|f_n\|_\infty = n\pi \to \infty$, nên $D$ không bị chặn.

## 4.2. Tại sao toán tử vi phân quan trọng trong ứng dụng

Mặc dù không bị chặn, toán tử vi phân cực kỳ quan trọng trong phương trình vi phân, cơ học lượng tử (toán tử động lượng), và xử lý tín hiệu (biến đổi Fourier cho phép "regularize" đạo hàm). Giải tích hàm phát triển lý thuyết riêng cho **toán tử không bị chặn** (unbounded operators) — nhưng đây là chủ đề phức tạp hơn.

**Thực tế:** Trong các bài toán thực tế, người ta thường chuyển qua **yếu (weak) formulation** của phương trình vi phân, trong đó đạo hàm được "chuyển" sang hàm test qua tích phân từng phần, và làm việc trong không gian Sobolev $H^1$ (là Hilbert space). Khi đó toán tử trở nên bị chặn.

---

# 5. Không gian $\mathcal{B}(X,Y)$ các toán tử bị chặn

## 5.1. Không gian $\mathcal{B}(X,Y)$ là Banach

**Định lý:** Nếu $Y$ là Banach space và $X$ là normed space bất kỳ, thì $\mathcal{B}(X,Y)$ — không gian tất cả các toán tử tuyến tính bị chặn từ $X$ đến $Y$ — với chuẩn toán tử, là **Banach space**.

**Chứng minh phác thảo:** Giả sử $(T_n)$ là chuỗi Cauchy trong $\mathcal{B}(X,Y)$. Với $x \in X$ cố định:

$$\|T_n x - T_m x\| \le \|T_n - T_m\| \cdot \|x\| \to 0$$

Vậy $(T_n x)$ là chuỗi Cauchy trong $Y$, hội tụ đến $Tx$. Ánh xạ $T: x \mapsto Tx$ là tuyến tính (giới hạn của toán tử tuyến tính). Ta kiểm tra $T$ bị chặn và $\|T_n - T\| \to 0$. (Chi tiết tương tự chứng minh $\ell^1$ là Banach ở File 1.)

**Trường hợp đặc biệt quan trọng:** $\mathcal{B}(X) := \mathcal{B}(X,X)$ là **Banach algebra** — vừa là Banach space, vừa có phép nhân (hợp toán tử) thỏa $\|ST\| \le \|S\|\|T\|$.

## 5.2. Hệ quả thực tế

Khi $Y = \mathbb{R}$ (hoặc $\mathbb{C}$), ký hiệu $X^* = \mathcal{B}(X, \mathbb{R})$ — đây là **không gian đối ngẫu (dual space)** của $X$. Theo định lý vừa nêu, $X^*$ **luôn là Banach space**, kể cả khi $X$ không đầy đủ.

---

# 6. Không gian đối ngẫu $X^*$

## 6.1. Định nghĩa và cấu trúc

**Không gian đối ngẫu** $X^*$ là tập tất cả các **phiếm hàm tuyến tính liên tục (continuous linear functional)** $f: X \to \mathbb{R}$, với chuẩn:

$$\|f\|_{X^*} = \sup_{\|x\|=1} |f(x)|$$

Mọi phần tử của $X^*$ được gọi là một **phiếm hàm tuyến tính (linear functional)** hoặc **covector** trong ngôn ngữ hình học vi phân.

## 6.2. Ví dụ phiếm hàm tuyến tính trên $C([a,b])$

**Phiếm hàm tích phân:** $f(g) = \int_a^b g(x) \, d\mu(x)$ với $\mu$ là độ đo Borel trên $[a,b]$.

Ta có $|f(g)| \le \|\mu\|_{TV} \cdot \|g\|_\infty$ (với $\|\mu\|_{TV}$ là chuẩn biến thiên toàn phần), nên $f \in (C([a,b]))^*$.

**Định lý Riesz-Markov:** Thực ra *mọi* phiếm hàm tuyến tính liên tục trên $C([a,b])$ đều có dạng này với $\mu$ là độ đo Radon có dấu (signed Radon measure). Đây là mô tả đầy đủ của $(C([a,b]))^*$.

**Trường hợp đặc biệt:** $f(g) = g(a)$ (evaluation tại $a$) tương ứng với $\mu = \delta_a$ (delta Dirac). Chuẩn: $\|f\| = 1$.

## 6.3. Không gian đối ngẫu của $\ell^p$

Đây là kết quả cơ bản:

$$(\ell^p)^* \cong \ell^q, \quad \frac{1}{p} + \frac{1}{q} = 1, \quad 1 < p < \infty$$

**Ý nghĩa cụ thể:** Mọi phiếm hàm tuyến tính liên tục $f: \ell^p \to \mathbb{R}$ có dạng:

$$f(x) = \sum_{n=1}^\infty y_n x_n$$

với $(y_n) \in \ell^q$ duy nhất, và $\|f\|_{(\ell^p)^*} = \|(y_n)\|_q$.

**Chứng minh phác thảo:** Đặt $e_n = (0, \dots, 0, 1, 0, \dots)$ (1 ở vị trí thứ $n$). Mọi $x = \sum x_n e_n$ (hội tụ trong $\ell^p$). Bằng tính tuyến tính và liên tục của $f$:

$$f(x) = \sum_{n=1}^\infty x_n f(e_n) = \sum_{n=1}^\infty x_n y_n$$

trong đó $y_n = f(e_n)$. Cần chứng minh $(y_n) \in \ell^q$ — điều này xuất phát từ bất đẳng thức Hölder và ước lượng chuẩn.

**Trường hợp biên:**
- $(\ell^1)^* \cong \ell^\infty$: mọi phiếm hàm liên tục trên $\ell^1$ tương ứng với dãy bị chặn.
- $(\ell^\infty)^* \supsetneq \ell^1$: không gian đối ngẫu của $\ell^\infty$ **lớn hơn** $\ell^1$, chứa cả các phiếm hàm "lạ" liên quan đến Hahn-Banach và Axiom of Choice.

---

# 7. Phiếm hàm tuyến tính và dạng tuyến tính

## 7.1. Dạng của phiếm hàm tuyến tính bị chặn

Trong hầu hết các không gian "cụ thể", phiếm hàm tuyến tính liên tục có dạng tường minh:

| Không gian $X$ | $X^*$ | Dạng của $f \in X^*$ |
|---|---|---|
| $\mathbb{R}^n$ ($\|\cdot\|_2$) | $\mathbb{R}^n$ | $f(x) = \langle a, x \rangle = \sum a_i x_i$ |
| $\ell^p$ ($1 < p < \infty$) | $\ell^q$ | $f(x) = \sum y_n x_n$, $(y_n) \in \ell^q$ |
| $\ell^1$ | $\ell^\infty$ | $f(x) = \sum y_n x_n$, $(y_n) \in \ell^\infty$ |
| $L^p(\mu)$ ($1 < p < \infty$) | $L^q(\mu)$ | $f(g) = \int fg \, d\mu$, $g \in L^q$ |
| $H$ Hilbert | $H$ (tự đối ngẫu) | $f(x) = \langle x, y \rangle$, $y \in H$ duy nhất |

## 7.2. Liên hệ với lý thuyết xác suất và học máy

Trong thống kê và học máy, các **phiếm hàm tuyến tính** xuất hiện tự nhiên:
- **Giá trị kỳ vọng:** $E_\mu[f] = \int f \, d\mu$ là phiếm hàm tuyến tính liên tục trên $C(\mathcal{X})$.
- **Maximum Mean Discrepancy (MMD):** $\text{MMD}(P,Q) = \sup_{\|f\|_{\mathcal{H}} \le 1} |E_P[f] - E_Q[f]|$ là chuẩn toán tử.
- **SVM:** Siêu phẳng phân loại $f(x) = \langle w, x \rangle + b$ là phiếm hàm tuyến tính trong $\ell^2$.

---

# 8. Toán tử bị chặn trong không gian Hilbert: xem trước

Khi $X = Y = H$ là Hilbert space (không gian Banach đặc biệt có tích trong), toán tử bị chặn có thêm cấu trúc: **toán tử adjoint** $T^*$ định nghĩa bởi:

$$\langle Tx, y \rangle = \langle x, T^*y \rangle \quad \text{với mọi } x, y \in H$$

Và $\|T^*\| = \|T\|$, $\|T^*T\| = \|T\|^2$. Đây sẽ là chủ đề của File 10 và 11.

---

# 9. Kernel trick và RKHS: ứng dụng trong ML

## 9.1. Evaluation functional trong RKHS

Trong học máy, **Reproducing Kernel Hilbert Space (RKHS)** $\mathcal{H}_K$ được định nghĩa bởi kernel $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ với tính chất đặc trưng:

$$f(x) = \langle f, K(\cdot, x) \rangle_{\mathcal{H}_K} \quad \text{với mọi } f \in \mathcal{H}_K, x \in \mathcal{X}$$

Tính chất này gọi là **reproducing property**. Hệ quả: evaluation functional $\delta_x: f \mapsto f(x)$ là phiếm hàm tuyến tính **bị chặn** trong $\mathcal{H}_K$:

$$|f(x)| = |\langle f, K(\cdot,x)\rangle| \le \|f\|_{\mathcal{H}_K} \|K(\cdot,x)\|_{\mathcal{H}_K} = \|f\|_{\mathcal{H}_K} \sqrt{K(x,x)}$$

Điều này rất đặc biệt — trong $L^2([0,1])$, evaluation tại một điểm cụ thể **không xác định** (vì hàm trong $L^2$ chỉ xác định đến tập đo 0) và chắc chắn không bị chặn.

## 9.2. Kernel trick

**Kernel trick** về bản chất là: thay vì làm việc trong không gian đặc trưng $\mathcal{F}$ (có thể vô hạn chiều), ta thay mọi tích trong $\langle \phi(x), \phi(y)\rangle_\mathcal{F}$ bằng $K(x,y)$. Điều này chỉ đúng vì:

$$K(x,y) = \langle K(\cdot,x), K(\cdot,y)\rangle_{\mathcal{H}_K}$$

Toán tử bị chặn trong RKHS — cụ thể là toán tử covariance trong Gaussian Process — có phổ với nghĩa Giải tích hàm, và chứa mọi thông tin về độ mượt của hàm.

---

# 10. Định lý mở rộng Hahn-Banach (xem trước)

Một hệ quả quan trọng sẽ được chứng minh ở File 4:

**Định lý Hahn-Banach (dạng chuẩn):** Nếu $M \subset X$ là không gian con và $f: M \to \mathbb{R}$ là phiếm hàm tuyến tính bị chặn với $\|f\|_{M^*} \le 1$, thì tồn tại $\tilde{f}: X \to \mathbb{R}$ tuyến tính bị chặn với $\tilde{f}|_M = f$ và $\|\tilde{f}\|_{X^*} = \|f\|_{M^*}$.

**Hệ quả quan trọng:** Với mọi $x_0 \ne 0$ trong $X$, tồn tại $f \in X^*$ sao cho $f(x_0) = \|x_0\|$ và $\|f\| = 1$. Điều này nói rằng **chuẩn của $x$ được xác định bởi các phiếm hàm tuyến tính liên tục**:

$$\|x\| = \sup_{f \in X^*, \|f\| \le 1} |f(x)|$$

---

# 11. Tóm tắt

Toán tử tuyến tính là đối tượng nghiên cứu trung tâm của Giải tích hàm. Điểm mấu chốt:

**Định lý tương đương:** $T: X \to Y$ tuyến tính thì

$$T \text{ liên tục} \iff T \text{ liên tục tại 0} \iff T \text{ bị chặn}$$

**Chuẩn toán tử:** $\|T\| = \sup_{\|x\|=1} \|Tx\|$ đo "hệ số khuếch đại tối đa".

**Không gian $\mathcal{B}(X,Y)$:** Đầy đủ nếu $Y$ đầy đủ — không gian toán tử có cấu trúc phong phú.

**Không gian đối ngẫu $X^*$:** Luôn là Banach, mô tả các "quan sát tuyến tính" trên $X$.

| Tính chất | Toán tử TẾ (chiều hữu hạn) | Toán tử vô hạn chiều |
|---|---|---|
| Tuyến tính $\Rightarrow$ liên tục? | Có | Không (cần thêm điều kiện) |
| Bị chặn $\Leftrightarrow$ liên tục? | Có | Có |
| Mọi chuẩn tương đương? | Có | Không |
| Phổ tập hợp hữu hạn? | Có | Không |

> Bài tiếp theo — **Bộ ba định lý lớn** — sẽ tiết lộ những hệ quả sâu sắc nhất khi ta làm việc trên không gian Banach đầy đủ: tính hữu hạn của chuẩn trong nhiều tình huống xuất hiện "miễn phí" từ lý luận tôpô.

# Không gian $L^p$

> Không gian $L^p$ là nơi mọi lý thuyết tích phân hội tụ: chúng là Banach space, mang chuẩn đo "kích thước" hàm theo nghĩa tích phân, và bất đẳng thức Hölder liên kết chúng thành một họ đẹp đẽ. Đặc biệt, $L^2$ là Hilbert space — không gian "tốt nhất" cho chuỗi Fourier và cơ học lượng tử.
>
> $$\|f\|_p = \left(\int |f|^p \, d\mu\right)^{1/p}, \quad \frac{1}{p} + \frac{1}{q} = 1 \implies \int |fg| \le \|f\|_p \|g\|_q$$

---

# 1. Định nghĩa không gian $L^p$

## 1.1. $L^p$ cho $1 \le p < \infty$

Cho $(\Omega, \mathcal{F}, \mu)$ là không gian đo. **Không gian $L^p(\mu)$** (hay $L^p(\Omega, \mathcal{F}, \mu)$) là tập:

$$L^p(\mu) = \left\{f: \Omega \to \mathbb{R} \text{ đo được} \;:\; \int_\Omega |f|^p \, d\mu < \infty\right\}$$

với chuẩn:

$$\|f\|_p = \left(\int_\Omega |f|^p \, d\mu\right)^{1/p}$$

**Chú ý kỹ thuật:** Hai hàm $f$ và $g$ được đồng nhất (xem là cùng phần tử của $L^p$) nếu $f = g$ a.e. Kỹ thuật hơn: $L^p$ là không gian thương $\tilde{L}^p / \sim$ với $f \sim g \iff f = g$ a.e.

**Lý do:** $\|f\|_p = 0 \iff f = 0$ a.e. (không phải $f = 0$ mọi nơi). Nếu không đồng nhất, $\|\cdot\|_p$ không phải chuẩn mà chỉ là *semi-norm*.

## 1.2. $L^\infty$ và essential supremum

**Định nghĩa:** $L^\infty(\mu)$ là tập các hàm đo được **bị chặn về bản chất (essentially bounded)**:

$$\|f\|_\infty = \text{ess}\sup |f| = \inf\{M \ge 0 : \mu(\{|f| > M\}) = 0\}$$

Đây là "chuẩn sup" nhưng bỏ qua tập đo 0. Ví dụ: $f(x) = \begin{cases} 1000 & x = 0 \\ 1 & x \ne 0 \end{cases}$ thì $\|f\|_\infty = 1$ (dù sup thông thường là 1000, vì $\{0\}$ có độ đo 0).

## 1.3. Liên hệ với $\ell^p$

Với $(\mathbb{N}, 2^\mathbb{N}, \#)$ (độ đo đếm):

$$L^p(\mathbb{N}, \#) = \ell^p, \quad \|f\|_{L^p(\#)} = \left(\sum_{n=1}^\infty |f(n)|^p\right)^{1/p} = \|(f(n))\|_{\ell^p}$$

Vậy $\ell^p$ là trường hợp riêng của $L^p$ — toàn bộ lý thuyết $L^p$ bao quát cả hai.

---

# 2. Bất đẳng thức Young

## 2.1. Bất đẳng thức Young

**Bổ đề Young:** Với $a, b \ge 0$ và $p, q > 1$, $\frac{1}{p} + \frac{1}{q} = 1$:

$$ab \le \frac{a^p}{p} + \frac{b^q}{q}$$

**Chứng minh hình học:** Xét đồ thị $y = x^{p-1}$ (hay $x = y^{q-1}$ do $q = p/(p-1)$). Diện tích hình chữ nhật $[0,a] \times [0,b]$ ($=ab$) không vượt diện tích tổng hai vùng dưới:

$$ab \le \int_0^a x^{p-1} \, dx + \int_0^b y^{q-1} \, dy = \frac{a^p}{p} + \frac{b^q}{q} \quad \square$$

Đẳng thức khi $a^p = b^q$, tức $a = b^{q/p}$.

---

# 3. Bất đẳng thức Hölder

## 3.1. Phát biểu

**Định lý (Hölder):** Cho $1 \le p \le \infty$ và $\frac{1}{p} + \frac{1}{q} = 1$ (Hölder conjugate). Nếu $f \in L^p(\mu)$ và $g \in L^q(\mu)$, thì $fg \in L^1(\mu)$ và:

$$\int |fg| \, d\mu \le \|f\|_p \|g\|_q$$

**Trường hợp đặc biệt:**
- $p = q = 2$: bất đẳng thức Cauchy-Schwarz $\int |fg| \le \|f\|_2 \|g\|_2$.
- $p = 1, q = \infty$: $\int |fg| \le \|f\|_1 \|g\|_\infty$.

## 3.2. Chứng minh từ Young

Nếu $\|f\|_p = 0$ hoặc $\|g\|_q = 0$, kết quả hiển nhiên. Giả sử cả hai $> 0$.

Đặt $\tilde{f} = f/\|f\|_p$ và $\tilde{g} = g/\|g\|_q$ (chuẩn hóa về 1). Áp dụng Young với $a = |\tilde{f}(x)|$ và $b = |\tilde{g}(x)|$:

$$|\tilde{f}(x)||\tilde{g}(x)| \le \frac{|\tilde{f}(x)|^p}{p} + \frac{|\tilde{g}(x)|^q}{q}$$

Tích phân hai vế:

$$\int |\tilde{f}||\tilde{g}| \, d\mu \le \frac{1}{p}\int |\tilde{f}|^p + \frac{1}{q}\int |\tilde{g}|^q = \frac{1}{p} \cdot 1 + \frac{1}{q} \cdot 1 = 1$$

Vậy $\int |fg| = \|f\|_p \|g\|_q \int |\tilde{f}||\tilde{g}| \le \|f\|_p \|g\|_q$. $\square$

## 3.3. Điều kiện đẳng thức

Đẳng thức trong Hölder xảy ra khi $|\tilde{f}|^p = |\tilde{g}|^q$ a.e., tức $|f|^p/\|f\|_p^p = |g|^q/\|g\|_q^q$ a.e.

## 3.4. Hölder tổng quát

**Bất đẳng thức Hölder tổng quát:** Nếu $\frac{1}{p_1} + \cdots + \frac{1}{p_k} = 1$ thì:

$$\int |f_1 \cdots f_k| \le \|f_1\|_{p_1} \cdots \|f_k\|_{p_k}$$

---

# 4. Bất đẳng thức Minkowski (Bất đẳng thức Tam giác trong $L^p$)

## 4.1. Phát biểu

**Định lý (Minkowski):** Với $1 \le p \le \infty$ và $f, g \in L^p(\mu)$:

$$\|f + g\|_p \le \|f\|_p + \|g\|_p$$

Đây chính là bất đẳng thức tam giác cho chuẩn $\|\cdot\|_p$.

## 4.2. Chứng minh (cho $1 < p < \infty$)

$$\|f + g\|_p^p = \int |f + g|^p \le \int (|f| + |g|)|f+g|^{p-1}$$

$$= \int |f| \cdot |f+g|^{p-1} + \int |g| \cdot |f+g|^{p-1}$$

Áp dụng Hölder cho mỗi số hạng với exp $p$ và $q = p/(p-1)$:

$$\int |f| \cdot |f+g|^{p-1} \le \|f\|_p \cdot \||f+g|^{p-1}\|_q = \|f\|_p \cdot \|f+g\|_p^{p-1}$$

Tương tự cho số hạng kia. Kết hợp:

$$\|f+g\|_p^p \le (\|f\|_p + \|g\|_p) \|f+g\|_p^{p-1}$$

Chia hai vế cho $\|f+g\|_p^{p-1}$: $\|f+g\|_p \le \|f\|_p + \|g\|_p$. $\square$

---

# 5. Định lý Riesz-Fischer: $L^p$ là Banach

## 5.1. Phát biểu

**Định lý Riesz-Fischer:** Với $1 \le p \le \infty$, không gian $L^p(\mu)$ là **Banach space** (đầy đủ với chuẩn $\|\cdot\|_p$).

## 5.2. Chứng minh (cho $1 \le p < \infty$)

Dùng đặc trưng chuỗi tuyệt đối hội tụ (File 1): đủ chứng minh mọi chuỗi tuyệt đối hội tụ trong $L^p$ hội tụ.

Giả sử $\sum_{n=1}^\infty \|f_n\|_p < \infty$. Đặt $g = \sum_{n=1}^\infty |f_n|$. Bằng Minkowski:

$$\left\|\sum_{n=1}^N |f_n|\right\|_p \le \sum_{n=1}^N \|f_n\|_p \le \sum_{n=1}^\infty \|f_n\|_p =: M < \infty$$

Áp dụng MCT (hàm không âm tăng đơn điệu $\sum_{n=1}^N |f_n| \nearrow g$):

$$\int g^p = \lim_N \int \left(\sum_{n=1}^N |f_n|\right)^p \le M^p < \infty$$

Vậy $g \in L^p$, và đặc biệt $g < \infty$ a.e. Do đó chuỗi $\sum f_n$ hội tụ tuyệt đối a.e. đến $f = \sum_{n=1}^\infty f_n$.

Kiểm tra $f \in L^p$: $|f| \le g$, nên $\|f\|_p \le \|g\|_p < \infty$.

Kiểm tra $\|f - \sum_{n=1}^N f_n\|_p \to 0$: $\left|f - \sum_{n=1}^N f_n\right| \le g$ và $\to 0$ a.e. DCT cho $\|f - \sum_{n=1}^N f_n\|_p \to 0$. $\square$

---

# 6. Không gian đối ngẫu: $(L^p)^* \cong L^q$

## 6.1. Phát biểu

**Định lý (Duality of $L^p$):** Cho $1 \le p < \infty$ và $\frac{1}{p} + \frac{1}{q} = 1$. Ánh xạ $\Phi: L^q(\mu) \to (L^p(\mu))^*$ định bởi:

$$\Phi(g)(f) = \int fg \, d\mu$$

là **đẳng cấu đẳng chuẩn (isometric isomorphism)**:

$$\|Phi(g)\|_{(L^p)^*} = \|g\|_{L^q}$$

và $\Phi$ surjective.

## 6.2. Ý nghĩa

Mọi phiếm hàm tuyến tính liên tục $T: L^p \to \mathbb{R}$ đều có dạng $T(f) = \int fg$ với $g \in L^q$ duy nhất. Điều này:

1. **Biểu diễn tường minh $(L^p)^*$:** không cần trừu tượng.
2. **Kết nối Hölder:** bất đẳng thức Hölder là "bị chặn" của phiếm hàm $\Phi(g)$.

## 6.3. Chứng minh tồn tại $g$ (phác thảo)

Với $T \in (L^p)^*$, định nghĩa độ đo có dấu $\nu(A) = T(\mathbf{1}_A)$. Kiểm tra $\nu \ll \mu$ (nếu $\mu(A) = 0$ thì $\mathbf{1}_A = 0$ trong $L^p$, nên $T(\mathbf{1}_A) = 0$). Radon-Nikodym: $\nu(A) = \int_A g \, d\mu$ với $g$ đo được. Kiểm tra $g \in L^q$ bằng cách chọn $f$ có dấu phù hợp và dùng bị chặn của $T$.

## 6.4. Trường hợp đặc biệt và ngoại lệ

| $p$ | $q$ | $(L^p)^*$ |
|---|---|---|
| $1$ | $\infty$ | $L^\infty$ (khi $\mu$ $\sigma$-hữu hạn) |
| $(1, \infty)$ | $(1, \infty)$ | $L^q$ |
| $2$ | $2$ | $L^2$ (tự đối ngẫu) |
| $\infty$ | $1$ | $(L^\infty)^* \supsetneq L^1$ (lớn hơn) |

Trường hợp $p = \infty$ đặc biệt: $(L^\infty)^*$ không phải $L^1$ mà là không gian đo finitely additive — chứa các "phiếm hàm kỳ lạ" từ Hahn-Banach.

---

# 7. $L^2$: Hilbert space đặc biệt

## 7.1. Tích trong trên $L^2$

$L^2(\mu)$ có **tích trong**:

$$\langle f, g \rangle = \int_\Omega f \cdot g \, d\mu$$

Kiểm tra:
- $\langle f, f \rangle = \int |f|^2 = \|f\|_2^2 \ge 0$
- $\langle f, f \rangle = 0 \iff f = 0$ a.e.
- Song tuyến tính: $\langle \alpha f + \beta g, h \rangle = \alpha\langle f,h\rangle + \beta\langle g,h\rangle$
- Đối xứng: $\langle f,g\rangle = \langle g,f\rangle$

Vậy $L^2$ là **không gian Hilbert** (Banach space với tích trong tương thích).

## 7.2. Cauchy-Schwarz trong $L^2$

$$|\langle f, g \rangle| = \left|\int fg\right| \le \|f\|_2 \|g\|_2$$

Đây chính là Hölder với $p = q = 2$.

## 7.3. Trực giao trong $L^2$

Hai hàm $f, g \in L^2$ là **trực giao** ($f \perp g$) nếu $\langle f, g\rangle = 0$.

Ví dụ: Hàm $\sin(nx)$ và $\cos(mx)$ trực giao trên $[-\pi, \pi]$ cho mọi $m, n$ nguyên (nền tảng của chuỗi Fourier).

---

# 8. Các bao hàm thức giữa $L^p$

## 8.1. Trên tập có độ đo hữu hạn

Nếu $\mu(\Omega) < \infty$ và $1 \le p \le q \le \infty$, thì:

$$L^q(\mu) \subset L^p(\mu) \quad \text{và} \quad \|f\|_p \le \mu(\Omega)^{1/p - 1/q} \|f\|_q$$

**Chứng minh:** Áp dụng Hölder với $f' = |f|^p$ và $g' = 1$, exp $q/p$ và $(q/p)' = q/(q-p)$:

$$\int |f|^p \le \left(\int |f|^q\right)^{p/q} \cdot \mu(\Omega)^{1-p/q}$$

Nâng lũy thừa $1/p$: $\|f\|_p \le \|f\|_q \cdot \mu(\Omega)^{1/p - 1/q}$.

## 8.2. Trên $\mathbb{N}$ (dãy số)

Với $1 \le p \le q \le \infty$:

$$\ell^p \subset \ell^q \quad \text{và} \quad \|x\|_q \le \|x\|_p$$

(Ngược chiều so với trên tập hữu hạn!) Ví dụ: $\ell^1 \subset \ell^2 \subset \ell^\infty$.

## 8.3. Trên $\mathbb{R}$ (tập vô hạn với $\lambda$)

Không gian $L^p(\mathbb{R})$ và $L^q(\mathbb{R})$ **không có bao hàm thức** nào giữa chúng (với $p \ne q$). Ví dụ:

- $f(x) = \mathbf{1}_{[0,1]}$: $f \in L^p$ với mọi $p$.
- $g(x) = x^{-1/3}\mathbf{1}_{[0,1]}$: $g \in L^2$ nhưng $g \notin L^4$ (và $g \in L^q$ với $q < 3$).
- $h(x) = \mathbf{1}_{[1,\infty)}/(1+x)$: $h \in L^2$ nhưng $h \notin L^1$.

---

# 9. Mật độ và xấp xỉ trong $L^p$

## 9.1. Tập đặc trong $L^p$

**Định lý:** Các họ sau **đặc trong $L^p(\mathbb{R}^n)$** ($1 \le p < \infty$):

1. **Hàm đơn giản:** $\left\{\sum a_i \mathbf{1}_{E_i} : E_i \text{ đo được hữu hạn}, a_i \in \mathbb{R}\right\}$
2. **Hàm đặc trưng của hộp:** $\mathbf{1}_Q$ với $Q$ là hình chữ nhật
3. **Hàm liên tục compact:** $C_c(\mathbb{R}^n)$
4. **Đa thức:** Weierstrass trên $[a,b]$ (mật độ trong $L^p([a,b])$)

## 9.2. Xấp xỉ bởi hàm trơn: Mollification

**Định nghĩa:** Hàm **mollifier** $\rho_\varepsilon(x) = \varepsilon^{-n}\rho(x/\varepsilon)$ với $\rho \in C_c^\infty$, $\rho \ge 0$, $\int \rho = 1$.

**Định lý:** Nếu $f \in L^p(\mathbb{R}^n)$ ($1 \le p < \infty$) thì $f * \rho_\varepsilon \in C^\infty$ và $\|f * \rho_\varepsilon - f\|_p \to 0$.

Điều này cho thấy $C_c^\infty(\mathbb{R}^n)$ đặc trong $L^p(\mathbb{R}^n)$ — cực kỳ quan trọng cho các lập luận xấp xỉ.

---

# 10. Ứng dụng trong học máy

## 10.1. Hàm mất mát trong $L^p$

Các hàm mất mát phổ biến trong ML tương ứng với chuẩn $L^p$:

- **MSE (Mean Squared Error):** $\|y - \hat{y}\|_2^2 = \int (y-\hat{y})^2$ — chuẩn $L^2$ bình phương.
- **MAE (Mean Absolute Error):** $\|y - \hat{y}\|_1$ — chuẩn $L^1$.
- **Huber loss:** Kết hợp $L^1$ và $L^2$, ổn định hơn với outlier.

Lý thuyết $L^p$ giải thích khi nào bài toán tối thiểu hóa có nghiệm và nghiệm có tính chất gì.

## 10.2. Regularization qua chuẩn $L^p$

- **$L^2$ regularization (Ridge):** $\min \|Xw - y\|_2^2 + \lambda\|w\|_2^2$ — nghiệm tường minh, mượt mà.
- **$L^1$ regularization (Lasso):** $\min \|Xw - y\|_2^2 + \lambda\|w\|_1$ — khuyến khích sparsity (nhiều tham số bằng 0).

Tính chất geometric của quả cầu đơn vị $L^1$ (có góc nhọn) giải thích tại sao Lasso cho nghiệm thưa.

## 10.3. Không gian $L^2$ và biến đổi Fourier trong tín hiệu

Biến đổi Fourier $\mathcal{F}: L^2(\mathbb{R}) \to L^2(\mathbb{R})$ là toán tử **unitary** (bảo toàn tích trong). Parseval:

$$\int_\mathbb{R} |f(x)|^2 \, dx = \int_\mathbb{R} |\hat{f}(\xi)|^2 \, d\xi$$

"Năng lượng" (chuẩn $L^2$) bảo toàn khi biến đổi Fourier — nền tảng của lý thuyết tín hiệu và nén dữ liệu.

## 10.4. $L^\infty$ và robustness

Các bài toán minimax trong lý thuyết trò chơi và ML đối thủ (adversarial ML) liên quan đến chuẩn $L^\infty$:

$$\max_{\|\delta\|_\infty \le \varepsilon} \mathcal{L}(f(x + \delta), y)$$

(Tấn công adversarial: tìm nhiễu $\delta$ nhỏ theo $L^\infty$ làm model mắc lỗi.) Chuẩn $L^\infty$ đo "nhiễu tối đa tại một điểm", phù hợp với tấn công pixel-wise trong ảnh.

---

# 11. Bảng tổng kết không gian $L^p$

| $p$ | Không gian | Đặc trưng | Ứng dụng |
|---|---|---|---|
| 1 | $L^1$ | Tích phân được | Xác suất, MAE |
| 2 | $L^2$ | Hilbert, chuỗi Fourier | Tín hiệu, MSE, QM |
| $\infty$ | $L^\infty$ | Bị chặn b.c. | Minimax, adversarial |
| $(1,\infty)$ | $L^p$ | Phản xạ | Lý thuyết tổng quát |

**Công thức chính:**

$$\text{Hölder: } \|fg\|_1 \le \|f\|_p\|g\|_q, \quad \frac{1}{p}+\frac{1}{q}=1$$

$$\text{Riesz-Fischer: } L^p(\mu) \text{ là Banach space}$$

$$\text{Duality: } (L^p)^* \cong L^q \quad (1 \le p < \infty)$$

> Bài tiếp theo — **Không gian Hilbert** — sẽ tập trung vào $L^2$ và cấu trúc phong phú của nó: tích trong, trực giao, cơ sở trực chuẩn, và chuỗi Fourier như một mở rộng tự nhiên của khái niệm tọa độ.

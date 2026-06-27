# Bổ đề Zorn & Hahn–Banach

> Định lý Hahn–Banach là công cụ "xây dựng" cơ bản nhất của Giải tích hàm: nó đảm bảo rằng không gian đối ngẫu $X^*$ luôn đủ phong phú để phân biệt mọi điểm trong $X$. Không giống ba định lý lớn (cần Banach space và completeness), Hahn–Banach chỉ cần normed space — nhưng cần một công cụ đại số mạnh hơn: **Bổ đề Zorn** (tương đương Axiom of Choice).
>
> $$\text{Hahn-Banach:} \quad \exists \tilde{f}: X \to \mathbb{R} \text{ tuyến tính}, \; \tilde{f}|_M = f, \; \|\tilde{f}\|_{X^*} = \|f\|_{M^*}$$

---

# 1. Công cụ chuẩn bị: Bổ đề Zorn

## 1.1. Thứ tự bộ phận và tập có thứ tự

**Định nghĩa:** Tập $P$ với quan hệ $\le$ là **tập có thứ tự bộ phận (partially ordered set, poset)** nếu:
1. $a \le a$ (phản xạ)
2. $a \le b$ và $b \le a$ $\Rightarrow$ $a = b$ (phản đối xứng)
3. $a \le b$ và $b \le c$ $\Rightarrow$ $a \le c$ (bắc cầu)

"Bộ phận" vì có thể tồn tại các phần tử không so sánh được với nhau (không có $a \le b$ hay $b \le a$).

**Ví dụ:** Tập $2^S$ (mọi tập con của $S$) với quan hệ $\subset$. Tập $\mathbb{R}$ với $\le$ thông thường là **tập có thứ tự toàn phần (totally ordered / chain)**.

**Định nghĩa:** Trong poset $(P, \le)$:
- **Chuỗi (chain):** tập con $C \subset P$ mà mọi hai phần tử đều so sánh được.
- **Chặn trên (upper bound):** $u \in P$ sao cho $a \le u$ với mọi $a \in C$.
- **Phần tử cực đại (maximal element):** $m \in P$ sao cho không có $a \in P$ với $a > m$ (tức $m \le a \Rightarrow a = m$).

## 1.2. Bổ đề Zorn

**Bổ đề Zorn:** Nếu $(P, \le)$ là poset khác rỗng và mọi chuỗi trong $P$ đều có chặn trên trong $P$, thì $P$ có **phần tử cực đại**.

**Quan hệ với Axiom of Choice:** Bổ đề Zorn, Axiom of Choice, và Định lý thứ tự tốt (Well-Ordering Theorem) là **ba phát biểu tương đương** trong lý thuyết tập hợp ZF. Không thể chứng minh hay bác bỏ chúng từ các tiên đề ZF thuần túy.

**Cách dùng thực tế:** Để chứng minh "tồn tại phần tử cực đại", ta:
1. Định nghĩa poset $P$ phù hợp
2. Kiểm tra mọi chuỗi trong $P$ có chặn trên
3. Kết luận $P$ có phần tử cực đại

## 1.3. Cơ sở Hamel (Hamel Basis)

**Định nghĩa:** **Cơ sở Hamel** (hay cơ sở đại số) của không gian vector $X$ là tập $B \subset X$ sao cho mọi phần tử của $X$ biểu diễn được như **tổ hợp hữu hạn** (finite linear combination) của các phần tử $B$, và $B$ độc lập tuyến tính.

**Định lý (từ Bổ đề Zorn):** Mọi không gian vector (thực hoặc phức) đều có cơ sở Hamel.

**Chứng minh:** Đặt $P$ = tập tất cả các tập con độc lập tuyến tính của $X$, có thứ tự bởi $\subset$. Mọi chuỗi $C_1 \subset C_2 \subset \cdots$ trong $P$ có chặn trên $\bigcup C_i \in P$ (vì mọi tổ hợp hữu hạn từ $\bigcup C_i$ chỉ dùng hữu hạn phần tử, nên nằm trong một $C_k$). Bổ đề Zorn suy ra $P$ có phần tử cực đại $B$. Ta cần kiểm tra $B$ sinh ra toàn $X$: nếu $x \notin \text{span}(B)$ thì $B \cup \{x\}$ độc lập tuyến tính, mâu thuẫn cực đại của $B$. $\square$

**Cơ sở Hamel vs. cơ sở Schauder:** Cơ sở Schauder (trong không gian Banach) cho phép tổ hợp **vô hạn** (chuỗi hội tụ). Cơ sở Hamel chỉ dùng tổ hợp hữu hạn. Cơ sở Hamel của không gian vô hạn chiều **không đếm được** (Baire), nhưng cơ sở Schauder có thể đếm được (ví dụ $\ell^2$ có cơ sở Schauder $\{e_n\}$).

---

# 2. Định lý Hahn–Banach: dạng phân tích (Analytic form)

## 2.1. Phát biểu

**Định lý Hahn–Banach (dạng thực):** Cho $X$ là **không gian vector thực** (không cần chuẩn, không cần đầy đủ), $M \subset X$ là không gian con (subspace), và $f: M \to \mathbb{R}$ là phiếm hàm tuyến tính. Giả sử có **hàm sublinear** $p: X \to \mathbb{R}$ sao cho:

$$f(m) \le p(m) \quad \text{với mọi } m \in M$$

Thì tồn tại phiếm hàm tuyến tính $\tilde{f}: X \to \mathbb{R}$ mở rộng $f$ (tức $\tilde{f}|_M = f$) và thỏa:

$$\tilde{f}(x) \le p(x) \quad \text{với mọi } x \in X$$

**Nhắc lại:** Hàm $p: X \to \mathbb{R}$ là **sublinear** nếu $p(\lambda x) = \lambda p(x)$ với $\lambda \ge 0$, và $p(x+y) \le p(x) + p(y)$.

## 2.2. Dạng chuẩn cho normed space

**Hệ quả (dạng chuẩn nhất):** Nếu $X$ là normed space, $M \subset X$ là không gian con, và $f: M \to \mathbb{R}$ tuyến tính với $\|f\|_{M^*} \le 1$ (tức $|f(m)| \le \|m\|$ với mọi $m \in M$), thì tồn tại $\tilde{f}: X \to \mathbb{R}$ tuyến tính với:
- $\tilde{f}|_M = f$ (mở rộng)
- $\|\tilde{f}\|_{X^*} = \|f\|_{M^*}$ (không tăng chuẩn)

**Chứng minh:** Áp dụng Hahn-Banach tổng quát với $p(x) = \|f\|_{M^*} \cdot \|x\|$. Khi đó $f(m) \le |f(m)| \le \|f\|_{M^*}\|m\| = p(m)$. Và $\tilde{f}(x) \le p(x)$, $\tilde{f}(-x) \le p(-x)$, nên $|\tilde{f}(x)| \le p(x) = \|f\|_{M^*}\|x\|$. $\square$

## 2.3. Chứng minh Hahn–Banach (phác thảo)

**Bước 1: Mở rộng một chiều.**

Cho $x_0 \notin M$, đặt $M_1 = M + \mathbb{R} x_0 = \{m + t x_0 : m \in M, t \in \mathbb{R}\}$. Ta cần tìm $c = \tilde{f}(x_0) \in \mathbb{R}$ sao cho:

$$f(m) + tc \le p(m + t x_0) \quad \forall m \in M, t \in \mathbb{R}$$

Chia thành $t > 0$ và $t < 0$:
- $t > 0$: $c \le p\left(\frac{m}{t} + x_0\right) - f\left(\frac{m}{t}\right)$ với mọi $m$, tức $c \le \inf_{m \in M} [p(m + x_0) - f(m)]$.
- $t < 0$ (đặt $t = -s, s > 0$): $-c \le p\left(\frac{m}{s} - x_0\right) - f\left(\frac{m}{s}\right)$, tức $c \ge \sup_{m \in M} [f(m) - p(m - x_0)]$.

Ta cần $\sup[\cdots] \le c \le \inf[\cdots]$. Điều này có thể nếu $\sup \le \inf$, tức:

$$f(m_1) - p(m_1 - x_0) \le p(m_2 + x_0) - f(m_2) \quad \forall m_1, m_2 \in M$$

Hay $f(m_1 + m_2) \le p(m_1 - x_0) + p(m_2 + x_0) \le p(m_1 + m_2)$ ✓

Vậy luôn chọn được $c$ phù hợp.

**Bước 2: Dùng Bổ đề Zorn để mở rộng đến toàn $X$.**

Đặt $P$ = tập tất cả các mở rộng hợp lệ $(N, g)$ trong đó $M \subset N$ là không gian con và $g: N \to \mathbb{R}$ tuyến tính với $g|_M = f$ và $g \le p$ trên $N$. Định nghĩa thứ tự $(N_1, g_1) \le (N_2, g_2)$ nếu $N_1 \subset N_2$ và $g_2|_{N_1} = g_1$.

Mọi chuỗi $(N_\alpha, g_\alpha)$ có chặn trên $(\bigcup N_\alpha, g)$ trong đó $g(x) = g_\alpha(x)$ với $x \in N_\alpha$ (nhất quán vì chuỗi). Bổ đề Zorn suy ra $P$ có phần tử cực đại $(N^*, \tilde{f})$. Nếu $N^* \ne X$, Bước 1 cho mở rộng lớn hơn — mâu thuẫn cực đại. Vậy $N^* = X$. $\square$

---

# 3. Hệ quả của Hahn–Banach

## 3.1. Phiếm hàm chuẩn hóa tại một điểm

**Hệ quả 1:** Với mọi $x_0 \ne 0$ trong normed space $X$, tồn tại $f \in X^*$ sao cho:

$$f(x_0) = \|x_0\| \quad \text{và} \quad \|f\|_{X^*} = 1$$

**Chứng minh:** Xét $M = \mathbb{R} \cdot x_0$ và $f(\lambda x_0) = \lambda \|x_0\|$. Thì $|f(\lambda x_0)| = |\lambda| \|x_0\| = \|\lambda x_0\|$, nên $\|f\|_{M^*} = 1$. Hahn-Banach mở rộng $f$ lên toàn $X$ mà không tăng chuẩn. $\square$

## 3.2. $X^*$ phân tách điểm (separates points)

**Hệ quả 2:** Nếu $x \ne y$ trong $X$, tồn tại $f \in X^*$ với $f(x) \ne f(y)$.

**Chứng minh:** Áp dụng Hệ quả 1 cho $x_0 = x - y \ne 0$: có $f$ với $f(x - y) = \|x - y\| \ne 0$. $\square$

**Ý nghĩa:** Không gian đối ngẫu $X^*$ "nhìn thấy" mọi sự khác biệt trong $X$. Đây là nền tảng để phân tích $X$ qua các phép đo tuyến tính.

## 3.3. Chuẩn biểu diễn bằng phiếm hàm

**Hệ quả 3:**

$$\|x\| = \sup_{\substack{f \in X^* \\ \|f\| \le 1}} |f(x)| = \max_{\substack{f \in X^* \\ \|f\| = 1}} f(x)$$

**Chứng minh:** Chiều $\le$: $|f(x)| \le \|f\|\|x\| \le \|x\|$. Chiều $\ge$: tồn tại $f$ với $f(x) = \|x\|$, $\|f\| = 1$. $\square$

Đây là "mô tả đối ngẫu" của chuẩn — cực kỳ hữu ích trong tối ưu hóa và hình học lồi.

## 3.4. Tính bất khả xích phân của không gian con đóng

**Hệ quả 4:** Nếu $M \subset X$ là không gian con đóng và $x_0 \notin M$, thì tồn tại $f \in X^*$ sao cho $f|_M = 0$ và $f(x_0) \ne 0$.

**Chứng minh:** Xét khoảng cách $d = d(x_0, M) > 0$ (vì $M$ đóng). Định nghĩa $g: M + \mathbb{R} x_0 \to \mathbb{R}$ bởi $g(m + tx_0) = t \cdot d$. Thì $\|g\| = 1$ và mở rộng bằng Hahn-Banach. $\square$

**Ý nghĩa:** Các phiếm hàm tuyến tính liên tục phân biệt được điểm với không gian con đóng — là nền tảng của hình học lồi và phân tích tối ưu.

---

# 4. Định lý Hahn–Banach: dạng hình học (Separation Theorem)

## 4.1. Phát biểu

**Định lý tách tập lồi (Geometric Hahn-Banach):** Trong normed space $X$, nếu $A$ và $B$ là hai tập lồi **không giao nhau** và $A$ có nội điểm, thì tồn tại phiếm hàm tuyến tính liên tục $f: X \to \mathbb{R}$ và hằng số $c$ sao cho:

$$f(a) \le c \le f(b) \quad \text{với mọi } a \in A, b \in B$$

Tức là, siêu phẳng $\{x : f(x) = c\}$ **phân tách** $A$ và $B$.

**Dạng mạnh hơn:** Nếu $A$ và $B$ đóng, $A$ compact và $A \cap B = \emptyset$, thì có thể phân tách **nghiêm ngặt**: $f(a) < c_1 < c_2 < f(b)$.

## 4.2. Hàm Minkowski (Gauge functional)

Công cụ chính: với tập lồi $C$ chứa 0 trong nội điểm, **hàm Minkowski** $\mu_C: X \to [0,\infty)$:

$$\mu_C(x) = \inf\{t > 0 : x \in tC\}$$

Hàm Minkowski là sublinear và $\{x : \mu_C(x) < 1\} = \text{int}(C)$. Dùng hàm Minkowski để biến định lý hình học thành dạng phân tích.

## 4.3. Ứng dụng trong học máy: SVM

Support Vector Machine phân loại nhị phân tìm **siêu phẳng phân tách tối ưu** giữa hai lớp dữ liệu. Về lý thuyết, bài toán này tồn tại nghiệm chính xác vì Separation Theorem đảm bảo tồn tại siêu phẳng khi hai tập lồi (convex hulls của hai lớp) không giao nhau.

Bài toán SVM:

$$\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(\langle w, x_i\rangle + b) \ge 1$$

là bài toán tối ưu lồi, và điều kiện KKT (liên quan đến Hahn-Banach qua dạng hình học) cho nghiệm tối ưu dưới dạng **support vectors**.

---

# 5. Không gian phản xạ (Reflexive Banach Space)

## 5.1. Nhúng chuẩn vào $X^{**}$

Với normed space $X$, xét **nhúng tự nhiên (canonical embedding)** $J: X \to X^{**}$:

$$J(x)(f) = f(x), \quad x \in X, f \in X^*$$

Đây là nhúng **đẳng chuẩn (isometric)**: $\|J(x)\|_{X^{**}} = \|x\|_X$ (từ Hệ quả 3 của Hahn-Banach).

Vậy **mọi normed space** nhúng đẳng chuẩn vào $X^{**}$. Câu hỏi: khi nào $J$ là song ánh?

## 5.2. Định nghĩa không gian phản xạ

**Định nghĩa:** Banach space $X$ được gọi là **phản xạ (reflexive)** nếu nhúng tự nhiên $J: X \to X^{**}$ là song ánh, tức $J(X) = X^{**}$.

Hay viết $X \cong X^{**}$ (đẳng cấu đẳng chuẩn theo nhúng tự nhiên).

**Chú ý quan trọng:** Phải dùng đúng nhúng tự nhiên $J$. Có thể tồn tại đẳng cấu $X \cong X^{**}$ không qua $J$ dù $X$ không phản xạ (trường hợp "James space").

## 5.3. Ví dụ

**$\ell^p$ ($1 < p < \infty$) là phản xạ:**

$$(\ell^p)^* \cong \ell^q \quad (1/p + 1/q = 1), \quad (\ell^q)^* \cong \ell^p$$

Vậy $(\ell^p)^{**} \cong \ell^p$. Nhúng tự nhiên đúng là song ánh.

**$\ell^1$ không phản xạ:**

$$(\ell^1)^* = \ell^\infty, \quad (\ell^\infty)^* \supsetneq \ell^1$$

Không gian $(\ell^\infty)^*$ chứa $\ell^1$ (theo nhúng tự nhiên) nhưng còn lớn hơn nhiều — nó chứa các phần tử "kỳ lạ" như giới hạn Banach (Banach limit).

**$L^p(\mu)$ ($1 < p < \infty$) là phản xạ:** Tương tự $\ell^p$, với $(L^p)^* \cong L^q$.

**$L^1(\mu)$ không phản xạ:** $(L^1)^* \cong L^\infty$ nhưng $(L^\infty)^*$ lớn hơn $L^1$.

**$C([0,1])$ không phản xạ:** $(C([0,1]))^*$ là độ đo Radon có dấu, và không gian đó không phản xạ.

## 5.4. Tầm quan trọng của tính phản xạ

**Định lý:** Nếu $X$ phản xạ thì mọi dãy bị chặn trong $X$ có **dãy con hội tụ yếu (weakly convergent subsequence)**.

Đây là "compact tương đối" trong hội tụ yếu — cực kỳ quan trọng trong tối ưu hóa và phương trình vi phân:

**Ứng dụng trong bài toán biến phân:** Để giải $\min_{u \in V} F(u)$ trong Sobolev space $V = H^1(\Omega)$ (là phản xạ):
1. Lấy dãy cực tiểu $(u_n)$ bị chặn trong $V$.
2. Vì $V$ phản xạ, có dãy con $u_{n_k} \rightharpoonup u^*$ yếu.
3. Nếu $F$ bán liên tục dưới theo hội tụ yếu: $F(u^*) \le \liminf F(u_{n_k})$.
4. Kết luận $u^*$ là cực tiểu.

Đây là lược đồ "phương pháp trực tiếp trong giải tích biến phân" — nền tảng của nhiều phương pháp số.

---

# 6. Phân tích sâu: $(\ell^\infty)^*$ và giới hạn Banach

## 6.1. Cấu trúc $(\ell^\infty)^*$

Mọi phiếm hàm tuyến tính liên tục trên $\ell^\infty$ có dạng tích phân đối với **độ đo finitely additive** trên $\mathbb{N}$ (hay tổng quát hơn, theo Stone-Čech compactification). Không gian này rất lớn và "phi xây dựng được" (non-constructive).

## 6.2. Giới hạn Banach

**Định nghĩa (từ Hahn-Banach):** **Giới hạn Banach** là phiếm hàm tuyến tính $L: \ell^\infty \to \mathbb{R}$ thỏa:
1. $L \ge 0$: nếu $x_n \ge 0$ với mọi $n$ thì $L(x) \ge 0$.
2. $L$ bất biến dịch: $L(x_1, x_2, \dots) = L(x_2, x_3, \dots)$.
3. $L$ mở rộng $\lim$: nếu $\lim_{n\to\infty} x_n = \ell$ tồn tại thì $L(x) = \ell$.

**Tồn tại từ Hahn-Banach:** Xây dựng bằng cách mở rộng phiếm hàm $\lim: c \to \mathbb{R}$ (trên không gian dãy hội tụ $c \subset \ell^\infty$) lên $\ell^\infty$, kết hợp điều kiện bất biến dịch (thông qua phương pháp Cesàro).

**Ứng dụng:** Trong lý thuyết ergodic, giới hạn Banach liên quan đến "giá trị trung bình" của dãy không hội tụ. Trong lý thuyết xác suất phi chu trình (non-standard analysis).

---

# 7. Hahn–Banach phức (complex version)

## 7.1. Phát biểu cho không gian phức

Hahn-Banach cũng đúng cho không gian vector phức $X$ với phiếm hàm $f: X \to \mathbb{C}$:

**Định lý:** Cho $M \subset X$ không gian con phức và $f: M \to \mathbb{C}$ tuyến tính với $|f(m)| \le p(m)$ (với $p$ seminorm). Thì tồn tại $\tilde{f}: X \to \mathbb{C}$ tuyến tính mở rộng $f$ với $|\tilde{f}(x)| \le p(x)$.

**Kỹ thuật chứng minh:** Tách phần thực và phần ảo: $f = u + iv$ với $u = \text{Re}(f)$ và $v = \text{Im}(f)$. Do tính tuyến tính phức, $v(m) = -u(im)$, nên chỉ cần mở rộng phần thực $u$ (là phiếm hàm thực tuyến tính), rồi đặt $\tilde{f}(x) = \tilde{u}(x) - i\tilde{u}(ix)$.

## 7.2. Ứng dụng trong cơ học lượng tử

Trong cơ học lượng tử, **không gian Hilbert phức** $\mathcal{H}$ là không gian trạng thái. Phiếm hàm tuyến tính liên tục $\langle \psi | \in \mathcal{H}^*$ (ký hiệu bra của Dirac) tương ứng với một vector trạng thái $|\psi\rangle$ theo định lý Riesz (xem File 10). Hahn-Banach đảm bảo rằng bra và ket "hoán đổi được" trong mọi tình huống hợp lý.

---

# 8. Tổng kết và ứng dụng thực tiễn

## 8.1. Bảng tóm tắt

| Kết quả | Ý nghĩa |
|---|---|
| Hahn-Banach (phân tích) | Mở rộng phiếm hàm tuyến tính mà không tăng chuẩn |
| Hahn-Banach (hình học) | Tách hai tập lồi bằng siêu phẳng |
| $X^*$ phân tách điểm | $\|x\| = \sup_{\|f\|\le 1} |f(x)|$ |
| Nhúng $J: X \to X^{**}$ | Mọi normed space nhúng vào đối ngẫu kép |
| $X$ phản xạ | $X \cong X^{**}$, dãy bị chặn có dãy con hội tụ yếu |

## 8.2. Đối ngẫu trong tối ưu hóa lồi

Nhiều bài toán tối ưu trong học máy có dạng primal-dual:

$$\min_{x \in X} f(x) + g(Ax) \longleftrightarrow \max_{y \in Y} -f^*(-A^*y) - g^*(y)$$

trong đó $f^*, g^*$ là **biến đổi Fenchel-Legendre** (conjugate function), và $A^*: Y^* \to X^*$ là **adjoint operator** (từ Hahn-Banach và cấu trúc đối ngẫu). Điều kiện tối ưu mạnh (strong duality) tương đương với điều kiện Slater — một dạng của Separation Theorem.

## 8.3. Chuỗi lồng đối ngẫu

$$\ell^1 \subset \ell^2 \subset \ell^p \subset \ell^q \subset \ell^\infty \quad (1 < p < q < \infty)$$

$$(\ell^1)^* = \ell^\infty \supset \ell^q \supset \ell^p \supset \ell^2 \supset \ell^1 = (\ell^\infty)^* \text{ (phần con)}$$

Đây cho thấy sự "đảo ngược" của bao hàm thức qua đối ngẫu — một biểu hiện của duality.

> Bài tiếp theo — **Độ đo ngoài & Sigma-đại số** — sẽ xây dựng nền tảng cho tích phân Lebesgue, cho phép ta làm việc với không gian $L^p$ — những không gian Banach "đúng đắn" nhất để chứa hàm khả tích.

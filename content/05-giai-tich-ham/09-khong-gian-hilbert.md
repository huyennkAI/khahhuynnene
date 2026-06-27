# Không gian Hilbert & Chuỗi Fourier

> Không gian Hilbert mang cấu trúc phong phú nhất trong các không gian Banach: khái niệm **góc** và **chiếu vuông góc** xuất hiện, và mọi phần tử có thể "phân tích" thành các thành phần theo mọi hướng trực giao. Chuỗi Fourier là hiện thân cụ thể của ý tưởng này: biểu diễn hàm như "tọa độ vô hạn chiều" theo hệ sóng điều hòa.
>
> $$\|f\|^2 = \sum_{n=-\infty}^{\infty} |\langle f, e_n\rangle|^2, \quad e_n(x) = \frac{e^{inx}}{\sqrt{2\pi}} \quad \text{(Parseval)}$$

---

# 1. Không gian tiền-Hilbert và Không gian Hilbert

## 1.1. Tích trong (Inner Product)

**Định nghĩa:** Trên không gian vector thực $H$, **tích trong** là ánh xạ $\langle \cdot, \cdot \rangle: H \times H \to \mathbb{R}$ thỏa:

1. $\langle x, x \rangle \ge 0$ với mọi $x$, và $\langle x, x \rangle = 0 \iff x = 0$
2. $\langle x, y \rangle = \langle y, x \rangle$ (đối xứng)
3. $\langle \alpha x + \beta y, z \rangle = \alpha\langle x,z\rangle + \beta\langle y,z\rangle$ (tuyến tính theo biến đầu)

**Chuẩn từ tích trong:** $\|x\| = \sqrt{\langle x, x\rangle}$.

## 1.2. Bất đẳng thức Cauchy-Schwarz

**Định lý:** $|\langle x, y\rangle| \le \|x\|\|y\|$, đẳng thức khi $x$ và $y$ tỉ lệ.

**Chứng minh:** Với $t \in \mathbb{R}$:
$$0 \le \|x - ty\|^2 = \|x\|^2 - 2t\langle x,y\rangle + t^2\|y\|^2$$

Đây là tam thức bậc hai không âm theo $t$, nên discriminant $\le 0$:

$$4\langle x,y\rangle^2 - 4\|x\|^2\|y\|^2 \le 0 \implies |\langle x,y\rangle| \le \|x\|\|y\| \quad \square$$

Từ Cauchy-Schwarz suy ra bất đẳng thức tam giác (chuẩn từ tích trong là chuẩn hợp lệ).

## 1.3. Không gian Hilbert

**Định nghĩa:** Không gian tiền-Hilbert (inner product space) $H$ được gọi là **không gian Hilbert** nếu đầy đủ với chuẩn $\|x\| = \sqrt{\langle x,x\rangle}$.

Vậy: Không gian Hilbert = Banach space đặc biệt có tích trong.

**Ví dụ:**
- $\mathbb{R}^n$ với $\langle x,y\rangle = \sum x_i y_i$
- $\ell^2$ với $\langle x,y\rangle = \sum x_n y_n$
- $L^2(\mu)$ với $\langle f,g\rangle = \int fg \, d\mu$

## 1.4. Định luật hình bình hành (Parallelogram Law)

**Định lý:** Trong mọi không gian Hilbert:

$$\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2$$

**Chứng minh:** Khai triển $\|x\pm y\|^2 = \|x\|^2 \pm 2\langle x,y\rangle + \|y\|^2$ rồi cộng.

**Ý nghĩa:** Định luật hình bình hành là **đặc trưng** của không gian Hilbert: một Banach space là Hilbert khi và chỉ khi chuẩn thỏa định luật hình bình hành. Ví dụ: $L^1$ không phải Hilbert vì không thỏa (lấy $f = \mathbf{1}_{[0,1]}$, $g = \mathbf{1}_{[1,2]}$: $\|f+g\|_1 = \|f-g\|_1 = 2$, $2\|f\|_1^2 + 2\|g\|_1^2 = 4 \ne 4 + 4$... thực ra cần ví dụ cụ thể hơn).

---

# 2. Trực giao và bổ sung trực giao

## 2.1. Trực giao

**Định nghĩa:** $x \perp y$ (trực giao) nếu $\langle x, y\rangle = 0$.

**Định lý Pythagoras:** Nếu $x \perp y$ thì $\|x + y\|^2 = \|x\|^2 + \|y\|^2$.

**Chứng minh:** $\|x+y\|^2 = \langle x+y, x+y\rangle = \|x\|^2 + 2\langle x,y\rangle + \|y\|^2 = \|x\|^2 + \|y\|^2$. $\square$

## 2.2. Bổ sung trực giao

Với $M \subset H$ là tập con bất kỳ:

$$M^\perp = \{y \in H : \langle x, y\rangle = 0 \; \forall x \in M\}$$

**Tính chất:** $M^\perp$ luôn là **không gian con đóng** (vì $y \mapsto \langle x, y\rangle$ liên tục, và $M^\perp = \bigcap_{x \in M} \ker(\langle x, \cdot\rangle)$).

**Bao trực giao kép:** $(M^\perp)^\perp = \overline{\text{span}(M)}$ (bao đóng của span).

---

# 3. Tính lồi và chiếu trong Hilbert space

## 3.1. Bổ đề: khoảng cách đến tập lồi đóng

**Định lý:** Trong Hilbert space $H$, nếu $C$ là tập lồi đóng khác rỗng và $x \in H$, thì tồn tại **duy nhất** $m^* \in C$ gần $x$ nhất:

$$\|x - m^*\| = \inf_{m \in C} \|x - m\| =: d(x, C)$$

**Chứng minh (tồn tại):** Lấy dãy $(m_n)$ trong $C$ với $\|x - m_n\| \to d = d(x,C)$. Dùng định luật hình bình hành cho $x - m_n$ và $x - m_k$:

$$\|m_n - m_k\|^2 = 2\|x-m_n\|^2 + 2\|x-m_k\|^2 - 4\left\|x - \frac{m_n+m_k}{2}\right\|^2$$

Vì $C$ lồi, $(m_n+m_k)/2 \in C$, nên $\|x - (m_n+m_k)/2\| \ge d$. Do đó:

$$\|m_n - m_k\|^2 \le 2\|x-m_n\|^2 + 2\|x-m_k\|^2 - 4d^2 \to 0$$

Vậy $(m_n)$ Cauchy, hội tụ đến $m^* \in C$ (đóng), và $\|x - m^*\| = d$.

**Đặc trưng $m^*$:** $m^* \in C$ gần $x$ nhất $\iff \langle x - m^*, m - m^*\rangle \le 0$ với mọi $m \in C$.

## 3.2. Phân tách trực tiếp Hilbert = M ⊕ M⊥

**Định lý:** Nếu $M \subset H$ là không gian con **đóng**, thì:

$$H = M \oplus M^\perp$$

Nghĩa là mọi $x \in H$ phân tách duy nhất thành $x = m + m^\perp$ với $m \in M$ và $m^\perp \in M^\perp$.

**Chứng minh:** Lấy $m = P_M x$ (chiếu lên $M$), $m^\perp = x - m$. Kiểm tra $m^\perp \perp M$: với $u \in M$ và $t \in \mathbb{R}$:

$$\|x - (m + tu)\|^2 \ge \|x - m\|^2 \implies \text{(khai triển)} \implies \langle x - m, u\rangle = 0$$

Vậy $m^\perp = x - m \in M^\perp$. Tính duy nhất: nếu $x = m_1 + m_1^\perp = m_2 + m_2^\perp$ thì $m_1 - m_2 = m_2^\perp - m_1^\perp \in M \cap M^\perp = \{0\}$. $\square$

---

# 4. Hệ trực chuẩn và Bất đẳng thức Bessel

## 4.1. Hệ trực chuẩn

**Định nghĩa:** Họ $(e_\alpha)_{\alpha \in I}$ trong $H$ là **hệ trực chuẩn (orthonormal system, ONS)** nếu:

$$\langle e_\alpha, e_\beta \rangle = \delta_{\alpha\beta} = \begin{cases} 1 & \alpha = \beta \\ 0 & \alpha \ne \beta \end{cases}$$

**Hệ trực chuẩn đầy đủ (ONB — Orthonormal Basis):** ONS $(e_n)_{n \in I}$ là đầy đủ nếu $\text{span}\{e_n\}$ đặc trong $H$ (hay $(e_n)^\perp = \{0\}$).

## 4.2. Hệ số Fourier

Với $(e_n)$ là ONS trong $H$ và $x \in H$, **hệ số Fourier** của $x$ theo $(e_n)$ là:

$$\hat{x}_n = \langle x, e_n \rangle$$

**Trực giác:** Đây là "tọa độ" của $x$ theo hướng $e_n$ — generalization của tọa độ trong $\mathbb{R}^n$.

## 4.3. Bất đẳng thức Bessel

**Định lý:** Với ONS $(e_n)_{n=1}^\infty$ và $x \in H$:

$$\sum_{n=1}^\infty |\langle x, e_n\rangle|^2 \le \|x\|^2$$

**Chứng minh:** Với $N$ hữu hạn, đặt $S_N = \sum_{n=1}^N \langle x, e_n\rangle e_n$ (chiếu lên span hữu hạn chiều). Khi đó $x - S_N \perp e_n$ với $n \le N$, nên Pythagoras:

$$\|x\|^2 = \|S_N\|^2 + \|x - S_N\|^2 \ge \|S_N\|^2 = \sum_{n=1}^N |\langle x, e_n\rangle|^2$$

Cho $N \to \infty$: $\sum_{n=1}^\infty |\langle x, e_n\rangle|^2 \le \|x\|^2$. $\square$

**Hệ quả:** Chuỗi $\sum \hat{x}_n e_n$ hội tụ trong $H$ (vì $\sum \|\hat{x}_n e_n\|^2 = \sum |\hat{x}_n|^2 < \infty$ và $H$ Hilbert — dùng đặc trưng Banach).

---

# 5. Đẳng thức Parseval và ONB

## 5.1. Định lý Parseval

**Định lý:** ONS $(e_n)$ là ONB $\iff$ **đẳng thức Parseval**: với mọi $x \in H$:

$$\sum_{n=1}^\infty |\langle x, e_n\rangle|^2 = \|x\|^2$$

Và khi đó: $x = \sum_{n=1}^\infty \langle x, e_n\rangle e_n$ (hội tụ trong $H$).

**Chứng minh:** Đặt $r = x - \sum \hat{x}_n e_n$. Thì $\langle r, e_n\rangle = \hat{x}_n - \hat{x}_n = 0$ với mọi $n$. Nếu $(e_n)$ là ONB thì $r \perp \text{span}\{e_n\}$, và vì $\text{span}\{e_n\}$ đặc, $r = 0$. Khi đó $\|x\|^2 = \|\sum \hat{x}_n e_n\|^2 = \sum |\hat{x}_n|^2$. $\square$

**Mở rộng cho tích trong:**

$$\langle x, y\rangle = \sum_{n=1}^\infty \langle x, e_n\rangle \overline{\langle y, e_n\rangle}$$

## 5.2. Phân tích $L^2$ qua ONB

Khi $H = L^2([a,b])$ có ONB $(e_n)$, mọi hàm $f \in L^2$ phân tích thành:

$$f = \sum_{n=1}^\infty \langle f, e_n\rangle e_n \quad \text{(hội tụ trong } L^2\text{)}$$

$$\|f\|_2^2 = \sum_{n=1}^\infty |\langle f, e_n\rangle|^2$$

Đây là "khai triển theo ONB" — generalization vô hạn chiều của biểu diễn vector theo cơ sở trực chuẩn.

---

# 6. Chuỗi Fourier trong $L^2([-\pi, \pi])$

## 6.1. Hệ hàm mũ

Trên $L^2([-\pi,\pi], \lambda/2\pi)$ (với độ đo $d\mu = dx/(2\pi)$), xét:

$$e_n(x) = e^{inx}, \quad n \in \mathbb{Z}$$

Ta kiểm tra $\{e_n\}$ là ONS:

$$\langle e_n, e_m\rangle = \frac{1}{2\pi}\int_{-\pi}^\pi e^{inx} e^{-imx} \, dx = \frac{1}{2\pi}\int_{-\pi}^\pi e^{i(n-m)x} \, dx = \delta_{nm}$$

**Hệ số Fourier:** $\hat{f}(n) = \langle f, e_n\rangle = \frac{1}{2\pi}\int_{-\pi}^\pi f(x) e^{-inx} \, dx$

**Chuỗi Fourier:** $Sf(x) = \sum_{n=-\infty}^\infty \hat{f}(n) e^{inx}$

## 6.2. Tính đầy đủ: Định lý

**Định lý:** Hệ $\{e_n\}_{n \in \mathbb{Z}}$ là **ONB** của $L^2([-\pi, \pi])$.

**Hệ quả (Parseval):** Với mọi $f \in L^2([-\pi, \pi])$:

$$\sum_{n=-\infty}^\infty |\hat{f}(n)|^2 = \frac{1}{2\pi}\int_{-\pi}^\pi |f(x)|^2 \, dx = \|f\|_2^2$$

$$\text{và } \|f - S_N f\|_2 \to 0 \quad \text{(hội tụ trong } L^2\text{)}$$

## 6.3. Chứng minh tính đầy đủ: Định lý Fejér

**Phương pháp:** Đủ chứng minh: nếu $f \in L^2$ và $\hat{f}(n) = 0$ với mọi $n$, thì $f = 0$ a.e.

**Định lý Fejér:** Tổng Cesàro (Fejér):

$$\sigma_N f(x) = \frac{1}{N} \sum_{k=0}^{N-1} S_k f(x) = \int_{-\pi}^\pi f(t) F_N(x-t) \, \frac{dt}{2\pi}$$

với **nhân Fejér** $F_N(t) = \frac{1}{N}\left(\frac{\sin(Nt/2)}{\sin(t/2)}\right)^2 \ge 0$.

**Tính chất nhân Fejér:**
- $F_N \ge 0$ (không âm)
- $\frac{1}{2\pi}\int_{-\pi}^\pi F_N(t) \, dt = 1$ (chuẩn hóa)
- $F_N(t) \to 0$ đều trên $[\delta, 2\pi - \delta]$ với $\delta > 0$ (approximate identity)

Nếu $\hat{f}(n) = 0$ với mọi $n$, thì $S_k f = 0$ với mọi $k$, nên $\sigma_N f = 0$. Nhưng từ tính chất nhân Fejér (approximate identity): $\sigma_N f \to f$ trong $L^2$. Vậy $f = 0$. $\square$

## 6.4. Bảng hệ số Fourier thường gặp

| Hàm $f(x)$ | $\hat{f}(n)$ |
|---|---|
| $1$ | $\delta_{n,0}$ |
| $\cos(kx)$ | $\frac{1}{2}(\delta_{n,k} + \delta_{n,-k})$ |
| $\sin(kx)$ | $\frac{1}{2i}(\delta_{n,k} - \delta_{n,-k})$ |
| $x$ (trên $(-\pi,\pi)$) | $\frac{(-1)^{n+1}}{in}$ ($n \ne 0$), $0$ ($n=0$) |
| $x^2$ | $\frac{2(-1)^n}{n^2}$ ($n \ne 0$), $\frac{\pi^2}{3}$ ($n=0$) |
| $|x|$ | $-\frac{2}{\pi n^2}$ ($n$ lẻ), $0$ ($n$ chẵn $\ne 0$), $\frac{\pi}{2}$ ($n=0$) |

---

# 7. So sánh các loại hội tụ

## 7.1. Ba khái niệm hội tụ

Với chuỗi Fourier $S_N f = \sum_{|n| \le N} \hat{f}(n) e^{inx}$:

**Hội tụ trong $L^2$:** $\|S_N f - f\|_2 \to 0$ — đúng với **mọi** $f \in L^2$. Đây là kết quả tổng quát nhất.

**Hội tụ pointwise:** $S_N f(x) \to f(x)$ với $x$ cố định. Phức tạp hơn nhiều:
- Đúng với a.e. $x$ nếu $f \in L^2$ (Định lý Carleson, 1966 — cực khó)
- Có thể sai tại điểm cụ thể (từ Banach-Steinhaus — xem File 3)

**Hội tụ đều:** $\|S_N f - f\|_\infty \to 0$ — chỉ đúng nếu $f$ có tính chất đặc biệt (ví dụ Hölder liên tục):

**Định lý (Dirichlet-Jordan):** Nếu $f$ biến thiên hữu hạn (bounded variation) trên $[-\pi, \pi]$, thì $S_N f(x) \to \frac{f(x^+) + f(x^-)}{2}$.

## 7.2. Hiện tượng Gibbs

Với hàm bước $f(x) = \text{sgn}(x)$ trên $(-\pi, \pi)$, chuỗi Fourier hội tụ pointwise nhưng có "vọt" (overshoot) khoảng 9% ở lân cận điểm gián đoạn, dù $N \to \infty$:

$$\max_x |S_N f(x)| \approx \frac{2}{\pi} \int_0^\pi \frac{\sin t}{t} \, dt \approx 1.179 > 1 = \|f\|_\infty$$

Hiện tượng Gibbs không mất đi dù chuỗi Fourier hội tụ trong $L^2$.

## 7.3. Bảng so sánh

| Loại hội tụ | Điều kiện đủ | Mạnh yếu |
|---|---|---|
| $L^2$ | $f \in L^2$ | Yếu nhất, nhưng rộng nhất |
| Pointwise a.e. | $f \in L^2$ (Carleson) | Mạnh hơn, khó hơn |
| Pointwise (mọi điểm) | BV, Hölder | Điều kiện ngặt hơn |
| Đều | Hölder liên tục | Mạnh nhất |

---

# 8. Không gian Hilbert vô hạn chiều và ONB

## 8.1. Mọi Hilbert space tách biệt đều đồng cấu với $\ell^2$

**Định nghĩa:** $H$ là **separable** (tách biệt) nếu có tập con đếm được đặc trong $H$.

**Định lý:** Mọi Hilbert space vô hạn chiều tách biệt đều đẳng cấu đẳng chuẩn (isometrically isomorphic) với $\ell^2$.

**Chứng minh phác thảo:** Nếu $H$ tách biệt, áp dụng Gram-Schmidt cho tập đếm được đặc để được ONB đếm được $(e_n)$. Ánh xạ $U: H \to \ell^2$, $Ux = (\langle x, e_n\rangle)_n$ là isometric isomorphism. $\square$

**Hệ quả:** Về mặt trừu tượng, $L^2([0,1])$ và $L^2(\mathbb{R})$ và $\ell^2$ đều "giống nhau" (đồng cấu với nhau). Sự khác nhau nằm ở cấu trúc thêm (các toán tử cụ thể, như multiplication operator, shift, v.v.).

## 8.2. Ví dụ ONB trên các Hilbert space khác nhau

**Trên $L^2([0,1])$:** $(e_n)_{n \ge 0}$ với $e_0 = 1$, $e_n(x) = \sqrt{2}\cos(n\pi x)$, $e_n(x) = \sqrt{2}\sin(n\pi x)$.

**Hàm Legendre:** $\{P_n\}$ là ONB của $L^2([-1,1], dx)$ sau khi chuẩn hóa.

**Hàm Hermite:** $\{H_n e^{-x^2/2}\}$ chuẩn hóa là ONB của $L^2(\mathbb{R}, dx)$ — xuất hiện trong cơ học lượng tử (dao động tử điều hòa).

**Wavelet:** Hệ $\{\psi_{j,k}(x) = 2^{j/2}\psi(2^j x - k) : j, k \in \mathbb{Z}\}$ với $\psi$ là mother wavelet phù hợp (Daubechies, Haar) là ONB của $L^2(\mathbb{R})$ — nền tảng của nén ảnh JPEG2000.

---

# 9. Ứng dụng: chuỗi Fourier trong học máy và xử lý tín hiệu

## 9.1. Biến đổi Fourier rời rạc (DFT)

Trên $\mathbb{Z}_N = \{0, 1, \dots, N-1\}$ (độ đo đếm), ONB là:

$$e_k(n) = \frac{1}{\sqrt{N}} e^{2\pi i kn/N}, \quad k = 0, 1, \dots, N-1$$

**DFT:** $\hat{x}_k = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} x_n e^{-2\pi ikn/N}$

Đây là phép biến đổi unitary trong $\mathbb{C}^N$, và FFT (Fast Fourier Transform) tính DFT trong $O(N \log N)$ thay vì $O(N^2)$.

## 9.2. Spectral representation trong học máy

**Graph Fourier Transform:** Với đồ thị $G = (V, E)$, Laplacian $L = D - A$. Các vector riêng của $L$ ($Lu_k = \lambda_k u_k$) là "hệ Fourier" trên đồ thị. **Graph Neural Network (GNN)** trong miền phổ: $\hat{f}(\lambda_k) = \langle f, u_k\rangle$ là hệ số Fourier trên đồ thị.

## 9.3. Attention mechanism: tích trong

**Scaled Dot-Product Attention** trong Transformer:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Hàng thứ $i$ của $QK^T/\sqrt{d_k}$ chính là $(Q_i^T K_j/\sqrt{d_k})_j$ — tương đương tích trong giữa query $Q_i$ và các key $K_j$ trong không gian $\mathbb{R}^{d_k}$. Attention có thể nhìn qua lăng kính kernel method trong $\ell^2$.

---

# 10. Định lý Stone-Weierstrass (xem trước)

Định lý Weierstrass nói rằng đa thức đặc trong $C([a,b])$ theo chuẩn sup. Dạng tổng quát (Stone-Weierstrass): nếu đại số $A \subset C(K)$ tách điểm và chứa hàm hằng, thì $A$ đặc trong $C(K)$.

**Hệ quả cho Fourier:** Đa thức lượng giác $\{\sum a_n \cos nx + b_n \sin nx\}$ đặc trong $C([-\pi,\pi])$ (theo sup), và do đó trong $L^2([-\pi,\pi])$ (vì $C$ đặc trong $L^2$). Đây là một cách chứng minh ONB của Fourier không dùng nhân Fejér.

---

# 11. Tóm tắt

**Chuỗi lý thuyết:**

$$\text{Tích trong} \xrightarrow{\text{Hilbert}} \text{Phân tích trực giao} \xrightarrow{\text{ONB}} \text{Chuỗi Fourier} \xrightarrow{\text{Parseval}} \text{Bảo toàn chuẩn}$$

**Công thức cốt lõi:**

$$H = \bigoplus_n \mathbb{R} \cdot e_n \quad (\text{trực hòa tổng trực tiếp})$$

$$x = \sum_n \langle x, e_n\rangle e_n, \quad \|x\|^2 = \sum_n |\langle x, e_n\rangle|^2 \quad (\text{Parseval})$$

**Hội tụ chuỗi Fourier trong $L^2$:**

$$\left\|f - \sum_{|n| \le N} \hat{f}(n) e_n\right\|_2 \to 0 \quad \text{với mọi } f \in L^2$$

| Mức độ | Hilbert | Banach | Normed |
|---|---|---|---|
| Chuẩn | Có | Có | Có |
| Tích trong | Có | Không nhất thiết | Không |
| Hoàn toàn | Có | Có | Không nhất thiết |
| Phân tích trực giao | Có | Không | Không |

> Bài tiếp theo — **Chiếu giao & Định lý Riesz** — sẽ khai thác đầy đủ cấu trúc tích trong để giải các bài toán cực tiểu hóa và biểu diễn phiếm hàm tuyến tính liên tục — nền tảng của kernel methods trong ML.

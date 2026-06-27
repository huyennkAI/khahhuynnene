# Độ đo Lebesgue

> Độ đo Lebesgue là hệ thống đo lường "tối ưu" trên $\mathbb{R}^n$: nó mở rộng chiều dài/diện tích/thể tích thông thường đến mọi tập "đo được", bảo toàn tính bất biến tịnh tiến và là độ đo đầy đủ. Trên $\mathbb{R}$, nó là **duy nhất** (đến thừa số nhân) với những tính chất này.
>
> $$\lambda([a,b]) = b - a, \quad \lambda(A + t) = \lambda(A), \quad \lambda\left(\bigsqcup_n A_n\right) = \sum_n \lambda(A_n)$$

---

# 1. Sigma-đại số Lebesgue và tính đầy đủ

## 1.1. Từ Borel đến Lebesgue

Ta đã xây dựng (File 5):
- Sigma-đại số Borel: $\mathcal{B}(\mathbb{R}) = \sigma(\text{tập mở})$
- Độ đo ngoài Lebesgue: $\lambda^*$ trên $2^\mathbb{R}$
- Sigma-đại số Lebesgue: $\mathcal{L}$ = các tập thỏa Carathéodory

**Quan hệ:** $\mathcal{B}(\mathbb{R}) \subsetneq \mathcal{L} \subsetneq 2^\mathbb{R}$

**Định lý:** $\mathcal{L}$ là hoàn thiện (completion) của $\mathcal{B}(\mathbb{R})$ theo $\lambda$, nghĩa là:

$$\mathcal{L} = \{A \cup N : A \in \mathcal{B}(\mathbb{R}), N \subset M \text{ với } M \in \mathcal{B}(\mathbb{R}), \lambda(M) = 0\}$$

Nói đơn giản: tập Lebesgue đo được là tập Borel "cộng thêm" một tập con của null set Borel.

## 1.2. Lực lượng của $\mathcal{L}$

- $|\mathcal{B}(\mathbb{R})| = 2^{\aleph_0} = |\mathbb{R}|$ (lực lượng continuum)
- $|\mathcal{L}| = 2^{2^{\aleph_0}} = 2^{|\mathbb{R}|}$ (lớn hơn đáng kể)

Vì sao $|\mathcal{L}|$ lớn hơn? Tập Cantor $C$ có $\lambda(C) = 0$, nên $|2^C| = 2^{|\mathbb{R}|}$ tập con của $C$ đều đo được Lebesgue. Nhưng $|\mathcal{B}(\mathbb{R})| = |\mathbb{R}|$ — không đủ để chứa hết.

## 1.3. Tính đầy đủ và tầm quan trọng

**Tính đầy đủ** của $\mathcal{L}$: nếu $\lambda(N) = 0$ và $E \subset N$ thì $E \in \mathcal{L}$ và $\lambda(E) = 0$.

**Tại sao quan trọng?** Trong giải tích: nếu $f = g$ "a.e." (almost everywhere, trừ tập đo 0) và $f$ đo được thì $g$ cũng đo được (trong $\mathcal{L}$). Điều này cho phép ta đồng nhất hai hàm bằng nhau a.e. — nền tảng của không gian $L^p$.

Trong $\mathcal{B}(\mathbb{R})$, điều này có thể sai! Một tập con của tập Cantor (đo 0) không nhất thiết là Borel.

---

# 2. Tính chất của độ đo Lebesgue

## 2.1. Các tính chất cơ bản

Độ đo Lebesgue $\lambda$ trên $(\mathbb{R}, \mathcal{L})$ thỏa:

**1. Độ dài khoảng:**
$$\lambda([a,b]) = \lambda((a,b)) = \lambda([a,b)) = b - a$$

(điểm đầu cuối không ảnh hưởng vì $\lambda(\{a\}) = 0$)

**2. Sigma-cộng (sigma-additivity):** Với $A_n \in \mathcal{L}$ rời nhau đôi một:
$$\lambda\left(\bigsqcup_{n=1}^\infty A_n\right) = \sum_{n=1}^\infty \lambda(A_n)$$

**3. Bất biến tịnh tiến (translation invariance):**
$$\lambda(A + t) = \lambda(A) \quad \forall t \in \mathbb{R}$$

**4. Bất biến co giãn:**
$$\lambda(cA) = |c| \lambda(A) \quad \forall c \in \mathbb{R}$$

**5. Đơn điệu:** $A \subset B \Rightarrow \lambda(A) \le \lambda(B)$.

**6. Liên tục:**
- Từ dưới: $A_n \nearrow A \Rightarrow \lambda(A_n) \nearrow \lambda(A)$
- Từ trên: $A_n \searrow A$, $\lambda(A_1) < \infty \Rightarrow \lambda(A_n) \searrow \lambda(A)$

## 2.2. Ví dụ tính liên tục

**Từ dưới:** $A_n = [0, 1-1/n]$, $A = [0,1)$. $\lambda(A_n) = 1-1/n \to 1 = \lambda(A)$.

**Từ trên (cần $\lambda(A_1) < \infty$):** $A_n = [n, \infty)$. $A = \bigcap A_n = \emptyset$. $\lambda(A_n) = \infty$ với mọi $n$, nhưng $\lambda(\emptyset) = 0 \ne \lim \infty$. Vậy điều kiện $\lambda(A_1) < \infty$ là **cần thiết**.

## 2.3. Độ đo Lebesgue trên $\mathbb{R}^n$

Trên $\mathbb{R}^n$, độ đo Lebesgue $\lambda^n$ thỏa:

$$\lambda^n([a_1,b_1] \times \cdots \times [a_n,b_n]) = \prod_{i=1}^n (b_i - a_i)$$

và là duy nhất (đến thừa số nhân) với các tính chất: sigma-cộng, bất biến tịnh tiến, và chuẩn hóa.

**Bất biến xoay:** $\lambda^n(RA) = \lambda^n(A)$ với $R$ là phép quay (rotation). Kết hợp với tịnh tiến: $\lambda^n$ bất biến với **mọi đẳng cự (isometry)** của $\mathbb{R}^n$.

---

# 3. Borel vs Lebesgue: phân tích chi tiết

## 3.1. Ví dụ tập Borel không là Lebesgue

Không có! Mọi tập Borel đều là Lebesgue ($\mathcal{B} \subset \mathcal{L}$).

## 3.2. Ví dụ tập Lebesgue không là Borel

Tập con của Cantor không là Borel. Cụ thể:

**Hàm Cantor (Devil's Staircase)** $\varphi: [0,1] \to [0,1]$: liên tục, đơn điệu tăng, $\varphi(0) = 0$, $\varphi(1) = 1$, hằng số trên mọi khoảng mở trong $[0,1] \setminus C$. Đặc điểm: $\varphi'(x) = 0$ a.e. (trên $[0,1] \setminus C$ có độ đo 1), nhưng $\varphi(0) \ne \varphi(1)$!

**Xây dựng tập Lebesgue không Borel:** Đặt $\psi(x) = \varphi(x) + x$, ánh xạ $[0,1] \to [0,2]$ homeomorphic. Nếu $V \subset [0,1]$ là tập Vitali (không đo được), thì $V' = \psi(V \cap C)$ là tập **đo được Lebesgue** (vì $\psi(C)$ có độ đo 1 và chứa $V'$, nhưng cần lập luận kỹ). Thực ra các ví dụ này phức tạp; điểm chính là $\mathcal{L} \supsetneq \mathcal{B}$ thật sự.

## 3.3. Hàm đo được Lebesgue vs Borel

**Hàm Borel đo được:** $f^{-1}(B) \in \mathcal{B}(\mathbb{R})$ với mọi $B \in \mathcal{B}(\mathbb{R})$.

**Hàm Lebesgue đo được:** $f^{-1}(B) \in \mathcal{L}$ với mọi $B \in \mathcal{B}(\mathbb{R})$.

Mọi hàm Borel đo được là Lebesgue đo được. Chiều ngược có thể sai, nhưng thực ra "mọi hàm Lebesgue đo được bằng a.e. với một hàm Borel đo được" — trong thực hành không có sự khác biệt.

---

# 4. Hàm đo được: phân tích sâu

## 4.1. Đặc trưng hàm đo được

Hàm $f: \mathbb{R} \to \mathbb{R}$ đo được (Lebesgue) khi và chỉ khi một trong các điều sau đây đúng:

- $f^{-1}((-\infty, a)) \in \mathcal{L}$ với mọi $a \in \mathbb{R}$
- $f^{-1}((-\infty, a]) \in \mathcal{L}$ với mọi $a \in \mathbb{R}$
- $f^{-1}((a, \infty)) \in \mathcal{L}$ với mọi $a \in \mathbb{R}$
- $f^{-1}([a, \infty)) \in \mathcal{L}$ với mọi $a \in \mathbb{R}$
- $f^{-1}((a, b)) \in \mathcal{L}$ với mọi $a < b$

(Các điều kiện này tương đương vì các họ sinh ra này đều sinh ra $\mathcal{B}(\mathbb{R})$.)

## 4.2. Hàm không đo được

**Định lý:** Với Axiom of Choice, tồn tại hàm $f: \mathbb{R} \to \{0,1\}$ không đo được — chính là $\mathbf{1}_V$ với $V$ là tập Vitali.

Trong thực hành, mọi hàm được xây dựng một cách "hiển nhiên" (công thức, giới hạn, tích phân) đều đo được.

## 4.3. Xấp xỉ bởi hàm đơn giản

**Định lý:** Mọi hàm đo được không âm $f: \Omega \to [0, \infty]$ là giới hạn tăng dần của hàm đơn giản (simple function):

$$0 \le \phi_1 \le \phi_2 \le \cdots \le f, \quad \phi_n \nearrow f \text{ pointwise}$$

**Xây dựng:** Đặt

$$\phi_n(x) = \sum_{k=1}^{n 2^n} \frac{k-1}{2^n} \mathbf{1}_{\left\{\frac{k-1}{2^n} \le f(x) < \frac{k}{2^n}\right\}}(x) + n \cdot \mathbf{1}_{\{f(x) \ge n\}}(x)$$

Hàm $\phi_n$ đơn giản, đo được, và $\phi_n \nearrow f$.

---

# 5. Liên hệ giữa Riemann và Lebesgue

## 5.1. Định lý Lebesgue về hàm Riemann khả tích

**Định lý Lebesgue-Vitali:** Hàm bị chặn $f: [a,b] \to \mathbb{R}$ là Riemann khả tích khi và chỉ khi tập điểm gián đoạn của $f$ có **độ đo Lebesgue 0**.

**Ví dụ:**
- $f(x) = 1/x$ trên $(0,1]$: Riemann khả tích (improper), Lebesgue khả tích với $\int_0^1 1/x \, dx = \infty$.
- Hàm Dirichlet $D$: tập điểm gián đoạn là $[0,1]$ (mọi điểm đều gián đoạn), độ đo $> 0$, không Riemann khả tích. Nhưng $\int D \, d\lambda = 0$ (Lebesgue).
- Hàm bậc thang (finitely many jumps): Riemann khả tích (tập gián đoạn hữu hạn, đo 0).

## 5.2. Khi Riemann và Lebesgue khác nhau

Nếu $f$ Riemann khả tích trên $[a,b]$ thì $f$ cũng Lebesgue khả tích và hai tích phân bằng nhau:

$$\int_a^b f(x) dx \text{ (Riemann)} = \int_{[a,b]} f \, d\lambda$$

Nhưng ngược lại không đúng: $\int D \, d\lambda = 0$ nhưng $\int D$ (Riemann) không xác định.

---

# 6. Tập đo không quan trọng

## 6.1. Tập đo không và "hầu hết khắp nơi"

**Mệnh đề "almost everywhere" (a.e.):** Tính chất $P(x)$ đúng "a.e." nếu $\lambda(\{x : P(x) \text{ sai}\}) = 0$.

**Ví dụ:**
- $f_n(x) = x^n$ trên $[0,1]$: $f_n(x) \to 0$ với mọi $x \in [0,1)$ và $f_n(1) = 1$. Vậy $f_n \to 0$ a.e. (trừ điểm $x = 1$, đo 0).
- Hàm Cantor $\varphi$: $\varphi'(x) = 0$ a.e. (trên $[0,1] \setminus C$, độ đo 1).

## 6.2. Tập Cantor: điểm kỳ diệu của lý thuyết đo

Tập Cantor tiêu chuẩn $C \subset [0,1]$:

| Tính chất | Giá trị |
|---|---|
| $\lambda(C)$ | 0 (đo Lebesgue) |
| $|C|$ | $2^{\aleph_0}$ (lực lượng continuum) |
| Tập đóng? | Có |
| Tập compact? | Có |
| Có nội điểm? | Không (nowhere dense) |
| Tổng hợp đếm được? | Không (không thể viết $C = \bigcup_{n} F_n$ với $F_n$ nowhere dense, trừ chính nó) |

Nói cách khác: $C$ là tập "lớn" về kích thước (nhiều điểm) nhưng "nhỏ" về độ đo.

**Tập Cantor béo (Fat Cantor set):** Biến thể trong đó độ đo dương! Thay vì loại đoạn giữa độ dài $1/3^n$, ta loại đoạn giữa độ dài $\epsilon/3^n$ với $\sum \epsilon/3^n = \epsilon < 1$. Kết quả: tập đóng, nowhere dense, nhưng $\lambda(C_\epsilon) = 1 - \epsilon > 0$.

---

# 7. Độ đo trên $\mathbb{R}^n$: định lý Fubini sơ bộ

## 7.1. Tích độ đo và sigma-đại số

Trên $\mathbb{R}^n = \mathbb{R}^m \times \mathbb{R}^k$ ($m + k = n$), độ đo Lebesgue là **tích độ đo**:

$$\lambda^n = \lambda^m \otimes \lambda^k$$

trên sigma-đại số tích $\mathcal{L}^m \otimes \mathcal{L}^k$.

**Chú ý tinh tế:** $\mathcal{L}^m \otimes \mathcal{L}^k \subsetneq \mathcal{L}^{m+k}$. Sigma-đại số Lebesgue trên $\mathbb{R}^{m+k}$ lớn hơn tích sigma-đại số Lebesgue. Nhưng trên $\mathcal{L}^m \otimes \mathcal{L}^k$, hai độ đo trùng nhau.

## 7.2. Lát cắt (Slices) và tích phân lặp

**Định lý (Fubini-Tonelli, dạng đo):** Với $E \in \mathcal{L}^{m+k}$:
- Với $\lambda^m$-a.e. $x \in \mathbb{R}^m$, lát cắt $E_x = \{y \in \mathbb{R}^k : (x,y) \in E\}$ đo được Lebesgue.
- Hàm $x \mapsto \lambda^k(E_x)$ đo được.
- $\lambda^{m+k}(E) = \int_{\mathbb{R}^m} \lambda^k(E_x) \, d\lambda^m(x)$.

Đây là nền tảng tích phân lặp cho hàm nhiều biến.

---

# 8. Độ đo tuyệt đối liên tục và độ đo kỳ dị

## 8.1. Độ đo tuyệt đối liên tục

**Định nghĩa:** Độ đo $\nu$ (có dấu hoặc không) là **tuyệt đối liên tục** so với $\mu$ (ký hiệu $\nu \ll \mu$) nếu:

$$\mu(N) = 0 \Rightarrow \nu(N) = 0$$

**Ví dụ:** Nếu $f \ge 0$ đo được, $\nu(A) = \int_A f \, d\mu$ thì $\nu \ll \mu$.

## 8.2. Định lý Radon-Nikodym

**Định lý:** Nếu $(\Omega, \mathcal{F}, \mu)$ là không gian đo $\sigma$-hữu hạn và $\nu \ll \mu$, thì tồn tại hàm đo được duy nhất $f: \Omega \to [0,\infty]$ (gọi là **đạo hàm Radon-Nikodym**) sao cho:

$$\nu(A) = \int_A f \, d\mu \quad \text{với mọi } A \in \mathcal{F}$$

Ký hiệu: $f = \frac{d\nu}{d\mu}$ (đạo hàm Radon-Nikodym).

**Ứng dụng:** Mật độ xác suất (probability density function, PDF) chính là đạo hàm Radon-Nikodym của phân phối xác suất so với độ đo Lebesgue: $P(A) = \int_A p(x) \, dx$, $p = dP/d\lambda$.

## 8.3. Phân tích Lebesgue-Radon-Nikodym

**Định lý:** Mọi độ đo $\sigma$-hữu hạn $\nu$ phân tích duy nhất thành:

$$\nu = \nu_{ac} + \nu_s$$

trong đó $\nu_{ac} \ll \lambda$ (tuyệt đối liên tục) và $\nu_s \perp \lambda$ (kỳ dị — có giá đỡ tập đo 0).

**Ví dụ:** Độ đo $\nu = \frac{1}{2}\lambda|_{[0,1]} + \frac{1}{2}\delta_0$:
- $\nu_{ac} = \frac{1}{2}\lambda|_{[0,1]}$ (tuyệt đối liên tục, mật độ $\frac{1}{2}\mathbf{1}_{[0,1]}$)
- $\nu_s = \frac{1}{2}\delta_0$ (kỳ dị với $\lambda$, tập trung tại điểm 0)

---

# 9. Ứng dụng trong xác suất và học máy

## 9.1. Mật độ xác suất (PDF)

Biến ngẫu nhiên liên tục $X$ có **hàm mật độ xác suất (PDF)** $f_X$ nghĩa là:

$$P(X \in B) = \int_B f_X(x) \, d\lambda(x), \quad f_X \ge 0, \quad \int_\mathbb{R} f_X = 1$$

Đây chính là: phân phối của $X$ (độ đo xác suất trên $\mathbb{R}$) là tuyệt đối liên tục so với Lebesgue, và $f_X = dP_X/d\lambda$.

## 9.2. Kỳ vọng là tích phân Lebesgue

$$E[g(X)] = \int_\Omega g(X(\omega)) \, dP(\omega) = \int_\mathbb{R} g(x) f_X(x) \, d\lambda(x)$$

Đây là **công thức đổi biến** cho độ đo đẩy đi (pushforward measure).

## 9.3. KL Divergence

**KL Divergence** giữa hai phân phối $P \ll Q$:

$$D_{KL}(P \| Q) = \int \log\frac{dP}{dQ} \, dP = \int p(x) \log\frac{p(x)}{q(x)} \, d\lambda(x)$$

Điều kiện $P \ll Q$ (tuyệt đối liên tục) là cần để KL divergence xác định. Nếu $P$ không tuyệt đối liên tục so với $Q$, $D_{KL} = +\infty$.

## 9.4. Không gian $L^2$ và xử lý tín hiệu

Không gian $L^2(\mathbb{R}, \lambda)$ — tập các hàm tích phân bình phương được — là không gian "đúng đắn" cho xử lý tín hiệu. Biến đổi Fourier $\mathcal{F}: L^2 \to L^2$ là toán tử unitary (đẳng cấu đẳng chuẩn), và Parseval's theorem:

$$\int_\mathbb{R} |f(x)|^2 \, dx = \int_\mathbb{R} |\hat{f}(\xi)|^2 \, d\xi$$

chính là bảo toàn "năng lượng" của tín hiệu.

---

# 10. Tóm tắt

**Bức tranh tổng thể của độ đo Lebesgue:**

$$\underbrace{\mathcal{B}(\mathbb{R})}_{\text{Borel}} \subsetneq \underbrace{\mathcal{L}}_{\text{Lebesgue}} \subsetneq \underbrace{2^\mathbb{R}}_{\text{tất cả}}$$

$$\underbrace{\lambda^*}_{\text{độ đo ngoài}} \xrightarrow{\text{Carathéodory}} \underbrace{\lambda|_\mathcal{L}}_{\text{độ đo thực sự}}$$

**Các tính chất quyết định:**
1. Sigma-cộng (additivity)
2. Bất biến tịnh tiến (translation invariance)  
3. Đầy đủ (completeness)
4. Duy nhất (uniqueness up to scaling)

**Từ Riemann đến Lebesgue:** Cùng cho kết quả với hàm Riemann khả tích, nhưng Lebesgue làm việc được với lớp hàm lớn hơn nhiều — và "ổn định" hơn dưới giới hạn.

| Tích phân | Khả tích khi | Giới hạn |
|---|---|---|
| Riemann | $f$ gián đoạn trên tập đo 0 | Khó trao đổi $\lim$ và $\int$ |
| Lebesgue | $f$ đo được và $\int |f| < \infty$ | DCT, MCT đảm bảo đổi thứ tự |

> Bài tiếp theo — **Tích phân Lebesgue** — sẽ xây dựng tích phân từng bước (simple function → không âm → tổng quát) và chứng minh các định lý hội tụ cơ bản: MCT, Fatou, DCT.

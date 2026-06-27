# Tích phân Lebesgue

> Tích phân Lebesgue xây dựng từ dưới lên: trước tiên cho hàm đơn giản (giá trị hữu hạn trên tập đo được), rồi mở rộng bằng xấp xỉ. Điểm mạnh cốt lõi là ba định lý hội tụ — MCT, Fatou, DCT — cho phép **trao đổi thứ tự giữa giới hạn và tích phân** dưới các điều kiện rõ ràng. Đây là điều tích phân Riemann không thể làm.
>
> $$\int f \, d\mu = \sup\left\{\int \phi \, d\mu : 0 \le \phi \le f, \; \phi \text{ simple}\right\}$$

---

# 1. Hàm đơn giản (Simple Function)

## 1.1. Định nghĩa

**Định nghĩa:** Hàm $\phi: \Omega \to \mathbb{R}$ là **hàm đơn giản (simple function)** nếu $\phi$ đo được và nhận hữu hạn giá trị phân biệt:

$$\phi = \sum_{i=1}^n a_i \mathbf{1}_{E_i}$$

trong đó $a_1, \dots, a_n \in \mathbb{R}$ là các giá trị phân biệt, $E_i = \phi^{-1}(\{a_i\}) \in \mathcal{F}$ rời nhau, và $\Omega = \bigsqcup E_i$.

**Biểu diễn chuẩn:** Phân rã $\phi$ thành indicator function của các tập đo được.

## 1.2. Tích phân của hàm đơn giản

**Định nghĩa:** Với $\phi = \sum_{i=1}^n a_i \mathbf{1}_{E_i} \ge 0$ (không âm):

$$\int_\Omega \phi \, d\mu = \sum_{i=1}^n a_i \mu(E_i)$$

(Dùng quy ước $0 \cdot \infty = 0$.)

**Ví dụ:**
- $\phi = 3 \cdot \mathbf{1}_{[0,2]} + 5 \cdot \mathbf{1}_{(2,4]}$: $\int \phi \, d\lambda = 3 \cdot 2 + 5 \cdot 2 = 16$.
- $\phi = \mathbf{1}_{[0,1] \cap \mathbb{Q}}$: $\int \phi \, d\lambda = 1 \cdot \lambda([0,1] \cap \mathbb{Q}) = 1 \cdot 0 = 0$.

## 1.3. Tính chất tuyến tính và đơn điệu

Với $\phi, \psi$ hàm đơn giản không âm:

$$\int (\alpha\phi + \beta\psi) \, d\mu = \alpha\int\phi \, d\mu + \beta\int\psi \, d\mu \quad (\alpha, \beta \ge 0)$$

$$\phi \le \psi \implies \int\phi \, d\mu \le \int\psi \, d\mu$$

---

# 2. Tích phân của hàm không âm

## 2.1. Định nghĩa bằng xấp xỉ từ dưới

**Định nghĩa:** Với $f: \Omega \to [0, +\infty]$ đo được:

$$\int_\Omega f \, d\mu = \sup\left\{\int_\Omega \phi \, d\mu : \phi \text{ simple}, \; 0 \le \phi \le f\right\}$$

Đây là supremum của tích phân tất cả các hàm đơn giản không âm "nằm dưới $f$".

**Tính nhất quán:** Nếu $f$ chính là hàm đơn giản thì định nghĩa mới trùng với định nghĩa trực tiếp ở Mục 1.

## 2.2. Tích phân trên tập con

Với $A \in \mathcal{F}$:

$$\int_A f \, d\mu = \int_\Omega f \cdot \mathbf{1}_A \, d\mu$$

**Tính chất:** $A \subset B \Rightarrow \int_A f \, d\mu \le \int_B f \, d\mu$ (đơn điệu theo miền).

## 2.3. Tích phân bằng 0

**Mệnh đề:** $\int_\Omega f \, d\mu = 0 \iff f = 0$ a.e.

**Chứng minh:** Nếu $\{f > 0\} = \bigcup_n \{f > 1/n\}$ và $\mu(\{f > 0\}) > 0$, thì $\mu(\{f > 1/n\}) > 0$ với một $n$ nào đó, và $\int f \ge \frac{1}{n}\mu(\{f > 1/n\}) > 0$. Chiều ngược: nếu $f = 0$ a.e., mọi $\phi \le f$ có $\phi = 0$ a.e., nên $\int \phi = 0$, vậy $\int f = 0$. $\square$

---

# 3. Định lý hội tụ đơn điệu (MCT - Monotone Convergence Theorem)

## 3.1. Phát biểu

**Định lý (MCT — Lebesgue, 1904):** Nếu $(f_n)$ là dãy hàm đo được không âm và **tăng đơn điệu** ($f_n \le f_{n+1}$ a.e.) với $f_n \nearrow f$ pointwise a.e., thì:

$$\lim_{n\to\infty} \int f_n \, d\mu = \int f \, d\mu$$

(Cả hai vế có thể bằng $+\infty$.)

## 3.2. Ví dụ minh họa

**Ví dụ 1:** $f_n(x) = x \cdot \mathbf{1}_{[0,n]}(x)$ trên $[0,\infty)$ với $\lambda$.

$f_n \nearrow f(x) = x$ pointwise. $\int f_n = \int_0^n x \, dx = n^2/2 \to \infty = \int_0^\infty x \, dx$. MCT đúng.

**Ví dụ 2:** $f_n(x) = \frac{1}{n} \mathbf{1}_{[0,n]}(x)$.

$f_n \to 0$ pointwise, nhưng $\int f_n = 1$ với mọi $n$, trong khi $\int 0 = 0$. **MCT KHÔNG áp dụng được** vì $(f_n)$ không tăng đơn điệu (mà giảm).

## 3.3. Chứng minh MCT

Vì $f_n \nearrow f$, có $\int f_n \le \int f_{n+1} \le \int f$, nên giới hạn $L = \lim \int f_n \le \int f$.

Cần chứng minh $L \ge \int f$, tức là $L \ge \int \phi$ với mọi $\phi$ đơn giản $0 \le \phi \le f$.

Fix $\phi$ và $0 < c < 1$. Đặt $A_n = \{f_n \ge c\phi\}$. Vì $f_n \nearrow f \ge \phi$, ta có $A_n \nearrow \Omega$ (a.e.).

$$\int f_n \ge \int_{A_n} f_n \ge c\int_{A_n} \phi$$

Vì $\phi$ đơn giản và $A_n \nearrow \Omega$, theo tính liên tục của độ đo từ dưới:

$$\int_{A_n} \phi \nearrow \int_\Omega \phi$$

Vậy $L \ge c \int \phi$. Cho $c \to 1^-$: $L \ge \int \phi$. Vì $\phi$ tùy ý: $L \ge \int f$. $\square$

## 3.4. Hệ quả: cộng chuỗi và tích phân

**Hệ quả:** Nếu $f_n \ge 0$ đo được thì:

$$\int \sum_{n=1}^\infty f_n \, d\mu = \sum_{n=1}^\infty \int f_n \, d\mu$$

**Chứng minh:** Đặt $g_N = \sum_{n=1}^N f_n \nearrow g = \sum f_n$. MCT: $\int g = \lim_N \int g_N = \lim_N \sum_{n=1}^N \int f_n = \sum \int f_n$. $\square$

---

# 4. Bổ đề Fatou

## 4.1. Phát biểu

**Bổ đề Fatou:** Nếu $(f_n)$ là dãy hàm đo được **không âm**, thì:

$$\int \liminf_{n\to\infty} f_n \, d\mu \le \liminf_{n\to\infty} \int f_n \, d\mu$$

**Nhắc lại:** $\liminf_{n\to\infty} f_n(x) = \lim_{n\to\infty} \inf_{k \ge n} f_k(x) = \sup_n \inf_{k \ge n} f_k(x)$.

## 4.2. Chứng minh

Đặt $g_n = \inf_{k \ge n} f_k$. Thì $g_n \le f_n$ và $g_n \nearrow \liminf f_n$.

$$\int g_n \le \int f_n \implies \liminf \int g_n \le \liminf \int f_n$$

Nhưng $g_n \nearrow g = \liminf f_n$, nên MCT:

$$\int g = \lim \int g_n = \liminf \int g_n \le \liminf \int f_n$$

Do đó $\int \liminf f_n \le \liminf \int f_n$. $\square$

## 4.3. Khi đẳng thức không đạt

Bổ đề Fatou là **bất đẳng thức chặt** nói chung. Ví dụ:

$f_n = \mathbf{1}_{[n, n+1]}$ trên $\mathbb{R}$: $f_n(x) \to 0$ với mọi $x$ (nên $\liminf f_n = 0$), $\int f_n = 1$ với mọi $n$.

$$0 = \int \liminf f_n < \liminf \int f_n = 1$$

"Khối lượng" $\int f_n = 1$ "chạy trốn ra vô cực". Fatou chỉ nói $\le$, không $=$.

---

# 5. Tích phân hàm có dấu tổng quát

## 5.1. Phân tách dương và âm

Với hàm đo được $f: \Omega \to [-\infty, +\infty]$ tổng quát, viết:

$$f = f^+ - f^-, \quad f^+ = \max(f, 0), \quad f^- = \max(-f, 0)$$

Cả $f^+$ và $f^-$ không âm và đo được.

**Định nghĩa:** $f$ là **khả tích (integrable)** nếu $\int f^+ \, d\mu < \infty$ và $\int f^- \, d\mu < \infty$. Khi đó:

$$\int f \, d\mu = \int f^+ \, d\mu - \int f^- \, d\mu$$

Ký hiệu $f \in L^1(\mu)$ khi $\int |f| \, d\mu < \infty$ (tương đương cả hai phần hữu hạn).

## 5.2. Tính chất tuyến tính đầy đủ

Với $f, g \in L^1(\mu)$ và $\alpha, \beta \in \mathbb{R}$:

$$\int (\alpha f + \beta g) \, d\mu = \alpha \int f \, d\mu + \beta \int g \, d\mu$$

## 5.3. Bất đẳng thức tam giác

$$\left|\int f \, d\mu\right| \le \int |f| \, d\mu$$

---

# 6. Định lý hội tụ chi phối (DCT - Dominated Convergence Theorem)

## 6.1. Phát biểu

**Định lý (DCT — Lebesgue):** Giả sử:
- $f_n: \Omega \to \mathbb{R}$ đo được với mọi $n$
- $f_n \to f$ pointwise a.e.
- Tồn tại $g \in L^1(\mu)$ sao cho $|f_n| \le g$ a.e. với mọi $n$ (**điều kiện chi phối**)

Thì $f \in L^1(\mu)$ và:

$$\lim_{n\to\infty} \int f_n \, d\mu = \int f \, d\mu$$

Và hơn nữa, $\int |f_n - f| \, d\mu \to 0$ (hội tụ trong $L^1$).

## 6.2. Chứng minh từ Fatou

Áp dụng Fatou cho $g + f_n \ge 0$:

$$\int (g + f) = \int \liminf(g + f_n) \le \liminf \int (g + f_n) = \int g + \liminf \int f_n$$

Nên $\int f \le \liminf \int f_n$.

Áp dụng Fatou cho $g - f_n \ge 0$:

$$\int (g - f) \le \liminf \int (g - f_n) = \int g - \limsup \int f_n$$

Nên $\limsup \int f_n \le \int f$.

Kết hợp: $\limsup \int f_n \le \int f \le \liminf \int f_n$, vậy $\lim \int f_n = \int f$. $\square$

## 6.3. Ví dụ áp dụng DCT

**Ví dụ 1: Tính giới hạn dưới dấu tích phân**

$$\lim_{n\to\infty} \int_0^\infty \frac{n \sin(x/n)}{x(1+x^2)} \, dx$$

Hàm $f_n(x) = \frac{n\sin(x/n)}{x(1+x^2)} \to \frac{1}{1+x^2}$ (vì $\lim_{t\to 0} \frac{\sin t}{t} = 1$, đặt $t=x/n$).

Chi phối: $|f_n(x)| = \frac{n|\sin(x/n)|}{x(1+x^2)} \le \frac{1}{1+x^2} =: g(x)$ (dùng $|\sin t| \le |t|$).

$g \in L^1([0,\infty))$ (tích phân $= \pi/2$). Vậy DCT: giới hạn $= \int_0^\infty \frac{1}{1+x^2} dx = \pi/2$.

**Ví dụ 2: Phân biệt với trường hợp không có chi phối**

$f_n(x) = n \mathbf{1}_{[0,1/n]}$: $f_n \to 0$ pointwise (trên $(0,\infty)$), nhưng $\int f_n = 1 \not\to 0$.

Không có hàm chi phối $g \in L^1[0,1]$ với $|f_n| \le g$. DCT **không áp dụng được**.

**Ví dụ 3: Đổi thứ tự tích phân và chuỗi**

Xét $\int_0^\infty \sum_{n=1}^\infty a_n(x) \, dx$ với $a_n \ge 0$. Từ MCT (cho $g_N = \sum_{n=1}^N a_n \nearrow \sum a_n$):

$$\int \sum a_n = \sum \int a_n$$

Hay từ DCT nếu có chi phối $|\sum a_n| \le g \in L^1$.

---

# 7. So sánh Riemann và Lebesgue

## 7.1. Ưu điểm của Lebesgue

| Tính chất | Riemann | Lebesgue |
|---|---|---|
| Đổi thứ tự $\lim$ và $\int$ | Cần hội tụ đều | MCT, DCT (yếu hơn) |
| Hàm tích phân được | Gián đoạn đo 0 | Đo được và $L^1$ |
| Tập miền | Khoảng $[a,b]$ | Tập đo được bất kỳ |
| Không gian $L^p$ | Không đầy đủ | Đầy đủ (Banach) |
| Fubini-Tonelli | Tích phân lặp phức tạp | Định lý sạch đẹp |

## 7.2. Hàm Dirichlet một lần nữa

$$\int_{[0,1]} D \, d\lambda = \int \mathbf{1}_\mathbb{Q} \, d\lambda = \lambda(\mathbb{Q} \cap [0,1]) = 0$$

Vì $\mathbb{Q}$ đếm được, $\lambda(\mathbb{Q}) = 0$. Lebesgue tính được, Riemann không.

## 7.3. Riemann khả tích $\Rightarrow$ Lebesgue khả tích

Nếu $f$ bị chặn trên $[a,b]$ và Riemann khả tích, thì:

$$\int_a^b f(x) \, dx \text{ (Riemann)} = \int_{[a,b]} f \, d\lambda \text{ (Lebesgue)}$$

Chiều ngược không đúng (như hàm Dirichlet cho thấy).

---

# 8. Định lý Fubini-Tonelli

## 8.1. Định lý Tonelli (hàm không âm)

**Định lý Tonelli:** Cho $(\Omega_1, \mathcal{F}_1, \mu_1)$ và $(\Omega_2, \mathcal{F}_2, \mu_2)$ là không gian đo $\sigma$-hữu hạn. Nếu $f: \Omega_1 \times \Omega_2 \to [0,+\infty]$ đo được so với $\mathcal{F}_1 \otimes \mathcal{F}_2$, thì:

$$\int_{\Omega_1 \times \Omega_2} f \, d(\mu_1 \otimes \mu_2) = \int_{\Omega_1}\left(\int_{\Omega_2} f(x,y) \, d\mu_2(y)\right) d\mu_1(x) = \int_{\Omega_2}\left(\int_{\Omega_1} f(x,y) \, d\mu_1(x)\right) d\mu_2(y)$$

**Ý nghĩa:** Với hàm không âm, luôn có thể đổi thứ tự tích phân.

## 8.2. Định lý Fubini (hàm khả tích)

**Định lý Fubini:** Nếu $f \in L^1(\mu_1 \otimes \mu_2)$ (tức $\int|f| \, d(\mu_1 \otimes \mu_2) < \infty$), thì:
- Với $\mu_1$-a.e. $x$: $y \mapsto f(x,y) \in L^1(\mu_2)$
- Hàm $x \mapsto \int_{\Omega_2} f(x,y) \, d\mu_2(y)$ thuộc $L^1(\mu_1)$
- Và tích phân có thể đổi thứ tự.

## 8.3. Ví dụ: Tính tích phân bằng Fubini

**Ví dụ:** Tính $\int_0^1 \int_x^1 \frac{e^{-y}}{y} \, dy \, dx$.

Trực tiếp khó. Dùng Fubini đổi thứ tự tích phân lặp. Miền $0 \le x \le y \le 1$, nên:

$$\int_0^1 \int_0^y \frac{e^{-y}}{y} \, dx \, dy = \int_0^1 e^{-y} \, dy = 1 - e^{-1}$$

## 8.4. Ứng dụng: tích phân Gauss

$$\left(\int_{-\infty}^\infty e^{-x^2} dx\right)^2 = \int_{\mathbb{R}^2} e^{-(x^2+y^2)} \, dx \, dy$$

(Fubini cho hàm không âm). Chuyển sang tọa độ cực:

$$= \int_0^{2\pi}\int_0^\infty e^{-r^2} r \, dr \, d\theta = 2\pi \cdot \frac{1}{2} = \pi$$

Vậy $\int_{-\infty}^\infty e^{-x^2} dx = \sqrt{\pi}$.

---

# 9. Tích phân trong $L^p$ và quan hệ với chuỗi Fourier

## 9.1. Khai triển Fourier và DCT

Cho $f \in L^1([-\pi, \pi])$, hệ số Fourier:

$$\hat{f}(n) = \frac{1}{2\pi}\int_{-\pi}^\pi f(x) e^{-inx} \, dx$$

Giả sử $|f(x)| \le g(x)$ với $g \in L^1$. Khi đó với $|f(x) e^{-inx}| \le g(x)$, DCT cho phép tính:

$$\lim_{n\to\infty} \int f(x) e^{inx} \, dx = \int \lim_{n\to\infty} f(x) e^{inx} \, dx$$

Nếu giới hạn pointwise là 0 (từ Riemann-Lebesgue: $\hat{f}(n) \to 0$ khi $|n| \to \infty$).

## 9.2. Định lý Riemann-Lebesgue

**Định lý:** Nếu $f \in L^1([-\pi, \pi])$ thì $\hat{f}(n) \to 0$ khi $|n| \to \infty$.

**Chứng minh:** Đầu tiên với $f = \mathbf{1}_{[a,b]}$: $\hat{f}(n) = \frac{e^{-ina} - e^{-inb}}{2\pi i n} \to 0$. Mở rộng bằng tính tuyến tính cho hàm đơn giản, rồi dùng DCT (hoặc xấp xỉ $L^1$) cho $L^1$ tổng quát.

---

# 10. Đo độ tích phân và tính liên tục

## 10.1. Tính liên tục tuyệt đối của tích phân

**Mệnh đề:** Nếu $f \in L^1(\mu)$, thì với mọi $\varepsilon > 0$, tồn tại $\delta > 0$ sao cho:

$$\mu(A) < \delta \implies \left|\int_A f \, d\mu\right| < \varepsilon$$

**Chứng minh:** Xét $f_n = f \cdot \mathbf{1}_{|f| \le n}$ (truncation). $|f_n| \le n$, nên $\left|\int_A f_n\right| \le n\mu(A)$. Chọn $n$ lớn: $\int |f - f_n| < \varepsilon/2$. Chọn $\delta = \varepsilon/(2n)$. $\square$

## 10.2. Ứng dụng: tích phân dọc theo đường

Trong lý thuyết tích phân Stokes và Green, tích phân dọc theo đường cong $\int_C f \, ds$ viết lại qua tham số hoá và tích phân Lebesgue — đảm bảo tính độc lập với cách tham số hóa.

---

# 11. Tóm tắt ba định lý hội tụ

| Định lý | Điều kiện | Kết luận |
|---|---|---|
| **MCT** | $0 \le f_n \nearrow f$ a.e. | $\int f_n \to \int f$ (cả hai vế $\in [0, \infty]$) |
| **Fatou** | $f_n \ge 0$ | $\int \liminf f_n \le \liminf \int f_n$ |
| **DCT** | $f_n \to f$ a.e., $|f_n| \le g \in L^1$ | $\int f_n \to \int f$, $\|f_n - f\|_1 \to 0$ |

**Quan hệ:** MCT $\Rightarrow$ Fatou $\Rightarrow$ DCT (mỗi định lý sau mạnh hơn theo một nghĩa).

**Ký ức trực giác:**
- MCT: khối lượng không bị "mất" khi hàm tăng.
- Fatou: khối lượng có thể bị "mất" (chạy ra vô cực), nhưng giới hạn dưới của tổng $\ge$ tổng của giới hạn dưới.
- DCT: nếu có "giám hộ" $g$ kiểm soát khối lượng không bị mất, thì hội tụ pointwise kéo theo hội tụ tích phân.

$$\text{MCT (tăng)} \quad \subset \quad \text{Fatou} \quad \subset \quad \text{DCT (có chi phối)}$$

> Bài tiếp theo — **Không gian $L^p$** — sẽ xây dựng không gian hàm dựa trên tích phân Lebesgue, chứng minh bất đẳng thức Hölder và Minkowski, và định lý Riesz-Fischer (đầy đủ của $L^p$).

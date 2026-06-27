# Bộ ba định lý lớn (Ba định lý nền tảng của Giải tích hàm)

> Ba định lý nền tảng — Baire, Banach-Steinhaus, và Open Mapping — đều xuất phát từ một nguồn: **tính đầy đủ** của không gian Banach. Chúng biến tính chất tôpô trừu tượng (completeness) thành những phát biểu cụ thể, mạnh mẽ về toán tử tuyến tính. Ngạc nhiên lớn nhất là chúng "cho không" — sự bị chặn, tính mở, tính liên tục xuất hiện mà không cần kiểm tra trực tiếp.
>
> $$\text{Completeness} \xrightarrow{\text{Baire}} \text{Category} \xrightarrow{\text{Banach-Steinhaus}} \text{Uniform Boundedness} \xrightarrow{\text{Open Mapping}} \text{Bounded Inverse}$$

---

# 1. Định lý Baire (Baire Category Theorem)

## 1.1. Tập nowhere dense và category

**Định nghĩa:** Tập $A$ trong không gian metric $X$ được gọi là **nowhere dense** (không có nội điểm ở đâu, hay "nơi chốn thưa") nếu phần trong của bao đóng của $A$ là rỗng:

$$\text{int}(\overline{A}) = \emptyset$$

Trực giác: $A$ không chứa bất kỳ quả cầu mở nào. Tập $\mathbb{Z}$ trong $\mathbb{R}$ là nowhere dense (nó đóng và không có nội điểm). Cantor set cũng là nowhere dense.

**Category của Baire:**
- Tập **category 1 (meager)**: hợp đếm được của các tập nowhere dense.
- Tập **category 2**: không phải category 1.

Trực giác: tập category 1 là "nhỏ bé" về mặt tôpô — mặc dù có thể có kích thước dương theo nghĩa độ đo.

## 1.2. Định lý Baire

**Định lý Baire:** Không gian metric **đầy đủ** (complete metric space) là tập **category 2** trong chính nó — tức là không thể biểu diễn như hợp đếm được của các tập nowhere dense.

Phát biểu tương đương: Nếu $X$ là complete metric space và $X = \bigcup_{n=1}^\infty F_n$ với $F_n$ đóng, thì ít nhất một $F_n$ có **nội điểm** (interior point) — tức $\text{int}(F_n) \ne \emptyset$.

## 1.3. Chứng minh Định lý Baire

Giả sử $X = \bigcup_{n=1}^\infty F_n$ với $F_n$ đóng, $\text{int}(F_n) = \emptyset$ với mọi $n$. Ta sẽ dẫn đến mâu thuẫn bằng cách xây dựng điểm không thuộc $\bigcup F_n$.

Vì $\text{int}(F_1) = \emptyset$, phần bù $U_1 = X \setminus F_1$ là tập mở đặc khắp nơi (open dense). Chọn quả cầu mở $B(x_1, r_1)$ với $r_1 < 1/2$ nằm trong $U_1$.

Vì $\text{int}(F_2) = \emptyset$, tập $U_2 = X \setminus F_2$ đặc, nên cắt quả cầu $B(x_1, r_1/2)$ không rỗng. Chọn $B(x_2, r_2) \subset B(x_1, r_1/2) \setminus F_2$ với $r_2 < r_1/2$.

Tiếp tục, ta được dãy quả cầu lồng nhau $B(x_1, r_1) \supset B(x_2, r_2) \supset \cdots$ với $r_n \to 0$ và $B(x_n, r_n) \cap F_n = \emptyset$.

Vì $r_n \to 0$, dãy $(x_n)$ là Cauchy. Vì $X$ đầy đủ, $x_n \to x^*$ cho $x^* \in X$. Nhưng $x^* \in \overline{B(x_n, r_n)} \subset X \setminus F_n$ với mọi $n$, vậy $x^* \notin \bigcup F_n$. Mâu thuẫn. $\square$

## 1.4. Hệ quả: $\mathbb{R}$ không đếm được

**Corollary:** $\mathbb{R}$ không đếm được.

**Chứng minh:** $\mathbb{R}$ đầy đủ. Nếu $\mathbb{R}$ đếm được, $\mathbb{R} = \{r_1, r_2, r_3, \dots\}$. Mỗi $F_n = \{r_n\}$ đóng và nowhere dense. Thì $\mathbb{R} = \bigcup F_n$ là hợp đếm được nowhere dense — vi phạm Baire. $\square$

**Hệ quả 2:** Không gian Banach vô hạn chiều **không có cơ sở Hamel đếm được** (Hamel basis là cơ sở tuyến tính, không phải Schauder basis). Nói cách khác, không thể biểu diễn mọi phần tử như tổ hợp hữu hạn của đếm được nhiều vector cố định.

## 1.5. Ứng dụng: chuỗi Fourier

Định lý Baire chứng minh rằng tập hàm liên tục mà chuỗi Fourier phân kỳ tại $x = 0$ là tập **category 2** (rất "lớn"). Cụ thể hơn, tập $\{f \in C([-\pi,\pi]) : \sup_N |S_N f(0)| = \infty\}$ là $G_\delta$ đặc (dense $G_\delta$). Đây là hệ quả của Banach-Steinhaus, xem Mục 2.

---

# 2. Định lý Banach-Steinhaus (Uniform Boundedness Principle)

## 2.1. Phát biểu

**Định lý Banach-Steinhaus:** Cho $X$ là **Banach space**, $Y$ là normed space, và $(T_\alpha)_{\alpha \in I}$ là họ toán tử tuyến tính bị chặn $T_\alpha \in \mathcal{B}(X, Y)$. Nếu với mọi $x \in X$:

$$\sup_{\alpha \in I} \|T_\alpha x\| < \infty \quad \text{(pointwise bounded)}$$

Thì:

$$\sup_{\alpha \in I} \|T_\alpha\| < \infty \quad \text{(uniformly bounded)}$$

**Phát biểu thô:** "Bị chặn điểm" (với mọi điểm cố định, họ toán tử bị chặn) $\Rightarrow$ "Bị chặn đều" (hằng số chặn đồng đều với mọi điểm).

## 2.2. Ý nghĩa

Điều này **không** hiển nhiên! Từ giả thiết, với mỗi $x$, ta có $\|T_\alpha x\| \le C_x$ nhưng hằng số $C_x$ **phụ thuộc** vào $x$. Kết luận: có thể chọn **một** hằng số $C$ không phụ thuộc $x$ và $\alpha$.

**Ví dụ về tại sao cần đầy đủ:** Nếu $X$ không đầy đủ, định lý sai. Lấy $X$ = đa thức trên $[0,1]$ với chuẩn sup, và $T_n p = n \cdot p(1/n) \cdot (1 - 1/n)^n$. Có thể xây dựng phản ví dụ cho định lý trên không gian không đầy đủ.

## 2.3. Chứng minh từ Baire

Đặt $F_n = \{x \in X : \sup_\alpha \|T_\alpha x\| \le n\} = \bigcap_\alpha \{x : \|T_\alpha x\| \le n\}$.

Mỗi $F_n$ **đóng** (giao của các tập đóng, vì $x \mapsto \|T_\alpha x\|$ liên tục).

Từ giả thiết pointwise bounded: $X = \bigcup_{n=1}^\infty F_n$.

Theo **Baire**, có $F_{n_0}$ với $\text{int}(F_{n_0}) \ne \emptyset$, tức tồn tại $x_0 \in F_{n_0}$ và $r > 0$ sao cho $B(x_0, r) \subset F_{n_0}$.

Với $\|x\| \le r$, ta có $x_0 + x \in B(x_0, r) \subset F_{n_0}$, nên:

$$\|T_\alpha x\| = \|T_\alpha(x_0 + x) - T_\alpha x_0\| \le \|T_\alpha(x_0 + x)\| + \|T_\alpha x_0\| \le n_0 + n_0 = 2n_0$$

Vậy với $\|x\| = 1$: $\|T_\alpha x\| = \frac{1}{r}\|T_\alpha(rx)\| \le \frac{2n_0}{r}$.

Do đó $\sup_\alpha \|T_\alpha\| \le \frac{2n_0}{r} < \infty$. $\square$

## 2.4. Corollary: Phân kỳ chuỗi Fourier

**Ứng dụng quan trọng:** Cho $f \in C([-\pi, \pi])$ và $S_N f(x) = \sum_{|k| \le N} \hat{f}(k) e^{ikx}$ là chuỗi Fourier riêng phần. Xét các toán tử $T_N: C([-\pi,\pi]) \to \mathbb{R}$:

$$T_N f = S_N f(0) = \int_{-\pi}^\pi f(t) D_N(t) dt$$

trong đó $D_N(t) = \frac{\sin((N+1/2)t)}{2\pi \sin(t/2)}$ là nhân Dirichlet. Ta tính:

$$\|T_N\| = \int_{-\pi}^\pi |D_N(t)| dt = \frac{4}{\pi^2} \log N + O(1) \to \infty$$

Nếu $\sup_N |T_N f| < \infty$ với **mọi** $f \in C$, Banach-Steinhaus suy ra $\sup_N \|T_N\| < \infty$ — mâu thuẫn. Vậy **tồn tại** $f \in C$ sao cho $S_N f(0)$ phân kỳ.

Hơn nữa, bằng Baire, tập $f$ như vậy là category 2 — "hầu hết" hàm liên tục có chuỗi Fourier phân kỳ tại 0!

## 2.5. Corollary: Hội tụ yếu và chuẩn

**Hệ quả:** Nếu $x_n \rightharpoonup x$ (hội tụ yếu trong Banach space $X$), thì $\sup_n \|x_n\| < \infty$.

**Chứng minh:** Đồng nhất $x_n$ với toán tử $\hat{x}_n: X^* \to \mathbb{R}$ qua $\hat{x}_n(f) = f(x_n)$. Hội tụ yếu $\Leftrightarrow$ pointwise bounded: $\sup_n |\hat{x}_n(f)| = \sup_n |f(x_n)| < \infty$ với mọi $f \in X^*$. Banach-Steinhaus (trên $X^*$, là Banach) suy ra $\sup_n \|\hat{x}_n\| < \infty$. Và $\|\hat{x}_n\| = \|x_n\|$ (từ Hahn-Banach). $\square$

---

# 3. Định lý ánh xạ mở (Open Mapping Theorem)

## 3.1. Phát biểu

**Định lý:** Cho $X$ và $Y$ là **Banach space** và $T \in \mathcal{B}(X, Y)$ là toán tử tuyến tính bị chặn **surjective** (toàn ánh, $T(X) = Y$). Thì $T$ là **ánh xạ mở** — nghĩa là ảnh của mọi tập mở là tập mở.

**Phát biểu tương đương:** Tồn tại $c > 0$ sao cho $B_Y(0, c) \subset T(B_X(0,1))$ (quả cầu đơn vị trong $Y$ nằm trong ảnh của quả cầu đơn vị $X$).

## 3.2. Ý nghĩa và trực giác

Vì sao đây không hiển nhiên? Nếu $T$ là song ánh (bijective), ta biết $T^{-1}$ tồn tại trên tập, nhưng không rõ $T^{-1}$ có liên tục không. Nói cách khác: tập đóng trong $X$ có ảnh đóng trong $Y$ không? Không nhất thiết (kể cả với song ánh). Nhưng **tập mở** thì có — đây là nội dung định lý.

**Phản ví dụ khi không surjective:** Injection $T: \ell^2 \hookrightarrow \ell^2$ định bởi $T(x_n) = (x_n/n)$ là đơn ánh, bị chặn, nhưng không mở (ảnh của quả cầu đơn vị không chứa bất kỳ quả cầu nào trong $\ell^2$).

## 3.3. Chứng minh phác thảo

**Bước 1:** Vì $T$ surjective, $Y = \bigcup_{n=1}^\infty T(B_X(0,n)) = \bigcup_{n=1}^\infty \overline{T(B_X(0,n))}$.

Theo **Baire** (trên $Y$ đầy đủ), có $n_0$ sao cho $\overline{T(B_X(0,n_0))}$ có nội điểm. Bằng tính đồng đều và tuyến tính, thực ra $\overline{T(B_X(0,1))}$ chứa một quả cầu $B_Y(y_0, \varepsilon)$ với $\varepsilon > 0$.

**Bước 2:** Bằng tính đối xứng (vì $T(-x) = -Tx$), suy ra $B_Y(0, \varepsilon) \subset \overline{T(B_X(0,1))}$.

**Bước 3 (phần kỹ thuật):** Chuyển từ bao đóng sang bao chính xác, dùng tính đầy đủ của $X$: bằng lập luận lặp, $B_Y(0, \varepsilon/2) \subset T(B_X(0,1))$.

Cụ thể: với $y \in Y$, $\|y\| < \varepsilon/2$, ta tìm $x_1 \in X$ với $\|x_1\| < 1$ và $\|y - Tx_1\| < \varepsilon/4$, rồi $x_2$ với $\|x_2\| < 1/2$ và $\|y - Tx_1 - Tx_2\| < \varepsilon/8$, v.v. Đặt $x = \sum x_k$; dãy hội tụ tuyệt đối trong $X$ Banach, và $Tx = y$. $\square$

## 3.4. Định lý nghịch đảo bị chặn (Bounded Inverse Theorem)

**Hệ quả quan trọng nhất:**

**Định lý:** Nếu $T \in \mathcal{B}(X, Y)$ là song ánh tuyến tính (bijective) giữa hai Banach space thì $T^{-1} \in \mathcal{B}(Y, X)$, tức là **nghịch đảo cũng bị chặn**.

**Chứng minh:** $T$ surjective nên mở (theo Open Mapping). Vì $T$ đơn ánh, $T^{-1}: Y \to X$ xác định trên toàn $Y$. Tập mở trong $X$ có ảnh thuận là tập mở trong $Y$, tức là nghịch ảnh của $T^{-1}$ đối với tập mở là mở — nghĩa là $T^{-1}$ liên tục. $\square$

**Ứng dụng:** Xét hai chuẩn $\|\cdot\|_1$ và $\|\cdot\|_2$ trên cùng không gian $X$ sao cho $X$ đầy đủ với cả hai, và $\|x\|_1 \le C\|x\|_2$. Áp dụng Bounded Inverse cho $I: (X, \|\cdot\|_2) \to (X, \|\cdot\|_1)$ suy ra $I^{-1}$ bị chặn, tức hai chuẩn **tương đương**.

---

# 4. Định lý đồ thị đóng (Closed Graph Theorem)

## 4.1. Phát biểu

**Định nghĩa:** Toán tử $T: X \to Y$ có **đồ thị đóng** nếu tập $\Gamma(T) = \{(x, Tx) : x \in X\} \subset X \times Y$ là tập đóng, tức là:

$$x_n \to x \text{ và } Tx_n \to y \implies y = Tx$$

**Định lý Đồ thị Đóng:** Nếu $X$ và $Y$ là Banach space và $T: X \to Y$ là toán tử tuyến tính có đồ thị đóng, thì $T$ bị chặn.

## 4.2. Ý nghĩa và trực giác

Để chứng minh $T$ liên tục, thông thường ta cần: "$x_n \to x \Rightarrow Tx_n \to Tx$". Điều kiện đồ thị đóng yếu hơn: ta không cần biết $Tx_n$ hội tụ, mà chỉ cần biết nếu $Tx_n$ hội tụ đến đâu đó thì giới hạn đó phải là $Tx$.

Đây là một kỹ thuật chứng minh rất hữu ích: thay vì ước lượng trực tiếp $\|Tx\|$, ta chứng minh đồ thị đóng — thường dễ hơn.

## 4.3. Chứng minh từ Open Mapping

Xét $X \times Y$ với chuẩn $\|(x,y)\| = \|x\|_X + \|y\|_Y$ (hay tương đương $\max(\|x\|, \|y\|)$). Vì $X, Y$ Banach, $X \times Y$ cũng là Banach.

$\Gamma(T)$ là không gian con đóng của $X \times Y$, do đó cũng là Banach. Xét ánh xạ:

$$\pi_X: \Gamma(T) \to X, \quad \pi_X(x, Tx) = x$$

Đây là toán tử tuyến tính bị chặn (vì $\|\pi_X(x, Tx)\| = \|x\| \le \|(x,Tx)\|$) và song ánh. Theo **Bounded Inverse**, $\pi_X^{-1}: X \to \Gamma(T)$ bị chặn. Khi đó:

$$Tx = \pi_Y(\pi_X^{-1}(x))$$

là hợp của hai toán tử bị chặn, nên $T$ bị chặn. $\square$

## 4.4. Ứng dụng: chứng minh bị chặn không cần ước lượng trực tiếp

**Ví dụ:** $T: H \to H$ là toán tử tuyến tính trên Hilbert space $H$, tự liên hợp (self-adjoint). Giả sử ta biết $\langle Tx, y\rangle = \langle x, Ty\rangle$. Không giả định gì về sự bị chặn.

Kiểm tra đồ thị đóng: nếu $x_n \to x$ và $Tx_n \to z$, thì với mọi $y$:

$$\langle z, y\rangle = \lim_n \langle Tx_n, y\rangle = \lim_n \langle x_n, Ty\rangle = \langle x, Ty\rangle = \langle Tx, y\rangle$$

Vậy $z = Tx$ (vì tích trong tách điểm). Đồ thị đóng, do đó $T$ bị chặn — không cần ước lượng trực tiếp!

---

# 5. Sơ đồ liên hệ giữa ba định lý

```
Tính đầy đủ (Completeness)
          │
          ▼
   Định lý Baire
   (X không phải category 1)
          │
    ┌─────┴─────┐
    ▼           ▼
Banach-        Open Mapping
Steinhaus      Theorem
(pointwise     (surjective ⟹ open)
⟹ uniform)         │
    │               ▼
    │         Bounded Inverse
    │         (bijection ⟹ T⁻¹ bounded)
    │               │
    └───────────────┘
                    ▼
            Closed Graph Theorem
            (closed graph ⟹ bounded)
```

---

# 6. Ví dụ tổng hợp: chuỗi Fourier và Banach-Steinhaus

## 6.1. Setup

Trên $X = C([-\pi, \pi])$ (hàm liên tục, chuẩn sup), xét các toán tử:

$$S_N: f \mapsto S_N f, \quad S_N f(x) = \sum_{k=-N}^N \hat{f}(k) e^{ikx}$$

trong đó $\hat{f}(k) = \frac{1}{2\pi}\int_{-\pi}^\pi f(t) e^{-ikt} dt$.

## 6.2. Tính chuẩn $\|S_N\|$

$$S_N f(x) = \int_{-\pi}^\pi f(t) D_N(x-t) \frac{dt}{2\pi}$$

với $D_N(t) = \sum_{k=-N}^N e^{ikt} = \frac{\sin((N+\frac{1}{2})t)}{\sin(t/2)}$ là nhân Dirichlet.

Do đó:

$$\|S_N\| = \|D_N\|_{L^1} = \frac{1}{2\pi}\int_{-\pi}^\pi |D_N(t)| dt \sim \frac{4}{\pi^2} \log N \to \infty$$

## 6.3. Kết luận từ Banach-Steinhaus

Từ $\sup_N \|S_N\| = \infty$, Banach-Steinhaus cho ta: **không thể có** $\sup_N \|S_N f\|_\infty < \infty$ với mọi $f \in C([-\pi,\pi])$.

Nghĩa là tồn tại $f$ liên tục sao cho $S_N f$ không bị chặn đều. Hơn nữa (qua Baire):

$$\left\{f \in C([-\pi,\pi]) : \sup_N \|S_N f\|_\infty = \infty\right\} \text{ là tập category 2 trong } C([-\pi,\pi])$$

Hay nói cách khác: hàm liên tục có chuỗi Fourier hội tụ đều là **ngoại lệ** (category 1), không phải quy tắc!

---

# 7. Ứng dụng Bounded Inverse trong giải hệ phương trình

## 7.1. Phương trình toán tử

Nhiều bài toán khoa học và kỹ thuật có dạng: **Giải $Tu = f$** trong đó $T: X \to Y$ là toán tử tuyến tính bị chặn, $f \in Y$ là cho trước, cần tìm $u \in X$.

**Điều kiện đủ để bài toán đặt chỉnh (well-posed):** $T$ là song ánh Banach-Banach. Khi đó Bounded Inverse đảm bảo $u = T^{-1}f$ phụ thuộc liên tục vào dữ liệu $f$:

$$\|u_1 - u_2\|_X = \|T^{-1}(f_1 - f_2)\|_X \le \|T^{-1}\| \|f_1 - f_2\|_Y$$

## 7.2. Ví dụ: phương trình tích phân Fredholm

Xét phương trình $u - Tu = f$ trên $L^2(\Omega)$ với $T$ compact. Nếu $1$ không là trị riêng của $T$, thì $(I - T)^{-1}$ tồn tại và bị chặn (theo lý thuyết Fredholm và Bounded Inverse). Đây là nền tảng lý thuyết giải phương trình tích phân.

---

# 8. Tóm tắt ba định lý

| Định lý | Giả thiết | Kết luận |
|---|---|---|
| **Baire** | $X$ complete metric, $X = \bigcup F_n$ ($F_n$ đóng) | $\exists n: \text{int}(F_n) \ne \emptyset$ |
| **Banach-Steinhaus** | $X$ Banach, $(T_\alpha) \subset \mathcal{B}(X,Y)$, pointwise bounded | Uniformly bounded: $\sup \|T_\alpha\| < \infty$ |
| **Open Mapping** | $X, Y$ Banach, $T \in \mathcal{B}(X,Y)$ surjective | $T$ ánh xạ mở |
| **Bounded Inverse** | $X, Y$ Banach, $T$ bijection trong $\mathcal{B}(X,Y)$ | $T^{-1} \in \mathcal{B}(Y,X)$ |
| **Closed Graph** | $X, Y$ Banach, $\Gamma(T)$ đóng trong $X \times Y$ | $T \in \mathcal{B}(X,Y)$ |

**Nguồn gốc chung:** Mọi định lý đều cần **tính đầy đủ** và đều dựa vào Định lý Baire (trực tiếp hoặc gián tiếp qua Open Mapping).

**Tại sao quan trọng trong thực hành?** Các định lý này cho phép ta kết luận về sự bị chặn, tính liên tục, hay tính khả nghịch của toán tử **mà không cần tính toán chuẩn trực tiếp** — chỉ cần kiểm tra các tính chất định tính (surjective, đồ thị đóng, pointwise bounded).

> Bài tiếp theo — **Hahn–Banach** — sẽ cung cấp công cụ xây dựng phiếm hàm tuyến tính liên tục, trả lời câu hỏi: $X^*$ đủ lớn để "nhìn thấy" mọi sự khác nhau trong $X$ không?

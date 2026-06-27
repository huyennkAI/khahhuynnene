# Chiếu giao & Định lý Riesz

> Trong Hilbert space, mọi phiếm hàm tuyến tính liên tục đều **"nhìn thấy"** thông qua tích trong với một vector cụ thể. Đây là Định lý biểu diễn Riesz — một trong những định lý đẹp và ứng dụng rộng nhất của Giải tích hàm. Kết hợp với định lý chiếu, nó cho phép giải các bài toán cực tiểu hóa bậc hai một cách tường minh.
>
> $$\forall f \in H^* \; \exists! \; y_f \in H : f(x) = \langle x, y_f\rangle \; \forall x \in H, \quad \|f\|_{H^*} = \|y_f\|_H$$

---

# 1. Định lý chiếu lên không gian con đóng

## 1.1. Phát biểu

**Định lý chiếu:** Nếu $M$ là không gian con **đóng** của Hilbert space $H$, thì với mọi $x \in H$, tồn tại **duy nhất** phần tử $P_M x \in M$ (chiếu của $x$ lên $M$) gần $x$ nhất:

$$\|x - P_M x\| = \min_{m \in M} \|x - m\|$$

Hơn nữa, $P_M x$ đặc trưng bởi:

$$P_M x \in M \quad \text{và} \quad x - P_M x \in M^\perp$$

tức là $\langle x - P_M x, m\rangle = 0$ với mọi $m \in M$.

## 1.2. Phân tách trực tiếp

Từ định lý chiếu, mọi $x \in H$ phân tách **duy nhất** thành:

$$x = \underbrace{P_M x}_{\in M} + \underbrace{(x - P_M x)}_{\in M^\perp}$$

Ký hiệu: $P_{M^\perp} x = x - P_M x$ là chiếu lên $M^\perp$. Vậy:

$$P_M + P_{M^\perp} = I \quad \text{(toán tử đơn vị)}$$

## 1.3. Toán tử chiếu $P_M$: tính chất

Toán tử chiếu $P_M: H \to H$ (với ảnh là $M$) thỏa:

1. **Tuyến tính:** $P_M(\alpha x + \beta y) = \alpha P_M x + \beta P_M y$
2. **Bị chặn:** $\|P_M\| = 1$ (khi $M \ne \{0\}$)
3. **Idempotent:** $P_M^2 = P_M$ (chiếu hai lần = chiếu một lần)
4. **Tự liên hợp:** $\langle P_M x, y\rangle = \langle x, P_M y\rangle$ (vì $\langle P_M x, y\rangle = \langle P_M x, P_M y\rangle + \langle P_M x, P_{M^\perp} y\rangle = \langle P_M x, P_M y\rangle$)

**Chứng minh $\|P_M\| = 1$:** Từ Pythagoras: $\|x\|^2 = \|P_M x\|^2 + \|P_{M^\perp} x\|^2 \ge \|P_M x\|^2$, nên $\|P_M x\| \le \|x\|$ (bị chặn bởi 1). Chuẩn đạt 1 khi $x \in M$.

## 1.4. Đặc trưng toán tử chiếu

**Định lý:** $T: H \to H$ là toán tử chiếu lên một không gian con đóng khi và chỉ khi $T^2 = T$ và $T^* = T$ (tự liên hợp và idempotent).

---

# 2. Bài toán cực tiểu hóa trên tập lồi đóng

## 2.1. Định lý tổng quát

**Định lý:** Trong Hilbert space $H$, nếu $C$ là tập **lồi đóng** khác rỗng và $x \in H$, thì:

(a) Tồn tại **duy nhất** $m^* \in C$ với $\|x - m^*\| = d(x, C) := \inf_{m \in C}\|x - m\|$.

(b) **Đặc trưng tối ưu:** $m^* \in C$ gần $x$ nhất khi và chỉ khi:

$$\langle x - m^*, m - m^*\rangle \le 0 \quad \forall m \in C$$

(c) Khi $C = M$ (không gian con đóng), điều kiện tối ưu trở thành $x - m^* \perp M$.

## 2.2. Chứng minh đặc trưng

**($\Leftarrow$):** Giả sử $\langle x - m^*, m - m^*\rangle \le 0$ với mọi $m \in C$. Thì:

$$\|x - m\|^2 = \|x - m^* + m^* - m\|^2 = \|x-m^*\|^2 + 2\langle x-m^*, m^*-m\rangle + \|m-m^*\|^2$$

$$\ge \|x-m^*\|^2 + \|m-m^*\|^2 \ge \|x-m^*\|^2$$

(Dùng $\langle x - m^*, m^* - m\rangle = -\langle x - m^*, m - m^*\rangle \ge 0$.)

**($\Rightarrow$):** Nếu $m^*$ là cực tiểu, với $m \in C$ và $t \in [0,1]$, điểm $(1-t)m^* + tm = m^* + t(m - m^*) \in C$ (lồi). Khai triển:

$$\|x - m^* - t(m-m^*)\|^2 \ge \|x - m^*\|^2$$

$$\Rightarrow -2t\langle x - m^*, m - m^*\rangle + t^2\|m-m^*\|^2 \ge 0$$

Chia $t > 0$ và cho $t \to 0^+$: $\langle x - m^*, m - m^*\rangle \le 0$. $\square$

## 2.3. Ứng dụng: chiếu lên hyperplane và halfspace

**Chiếu lên siêu phẳng:** $C = \{m : \langle a, m\rangle = b\}$ ($a \ne 0$). Đặc trưng tối ưu:

$$m^* = x - \frac{\langle a,x\rangle - b}{\|a\|^2} a$$

**Chiếu lên nửa không gian:** $C = \{m : \langle a, m\rangle \le b\}$:

$$m^* = \begin{cases} x & \text{nếu } \langle a, x\rangle \le b \\ x - \frac{\langle a,x\rangle - b}{\|a\|^2} a & \text{nếu } \langle a, x\rangle > b \end{cases}$$

---

# 3. Áp dụng chiếu: bình phương tối thiểu (Least Squares)

## 3.1. Bài toán bình phương tối thiểu

Cho $A: H \to K$ là toán tử tuyến tính bị chặn và $b \in K$. Bài toán **bình phương tối thiểu**: tìm $x^* \in H$ cực tiểu $\|Ax - b\|_K^2$.

Nghiệm tồn tại nếu $b \in \overline{A(H)}$ và duy nhất trên $(\ker A)^\perp$.

**Đặc trưng:** Nghiệm bình phương tối thiểu thỏa **phương trình normal**:

$$A^* A x^* = A^* b$$

trong đó $A^*: K \to H$ là adjoint của $A$ ($\langle Ax, y\rangle_K = \langle x, A^*y\rangle_H$).

**Giải thích hình học:** $Ax^*$ là chiếu của $b$ lên $\overline{A(H)}$, và $x^* = A^+ b$ (nghịch đảo Moore-Penrose).

## 3.2. Ví dụ: hồi quy tuyến tính

Trong ML, bài toán hồi quy: tìm $w^* \in \mathbb{R}^d$ cực tiểu $\|Xw - y\|_2^2$ với $X \in \mathbb{R}^{n \times d}$.

Nghiệm: $w^* = (X^TX)^{-1}X^Ty$ (nếu $X^TX$ khả nghịch).

Đây chính là phương trình normal với $A = X$, $A^* = X^T$: $X^TXw^* = X^Ty$.

---

# 4. Định lý biểu diễn Riesz

## 4.1. Phát biểu

**Định lý Riesz (Representation Theorem):** Cho $H$ là Hilbert space (thực). Với mọi phiếm hàm tuyến tính liên tục $f: H \to \mathbb{R}$ (tức $f \in H^*$), tồn tại **duy nhất** $y_f \in H$ sao cho:

$$f(x) = \langle x, y_f\rangle \quad \text{với mọi } x \in H$$

Hơn nữa: $\|f\|_{H^*} = \|y_f\|_H$.

**Hệ quả:** $H^* \cong H$ (đẳng cấu đẳng chuẩn qua ánh xạ $f \mapsto y_f$).

## 4.2. Chứng minh

**Tồn tại:**

Nếu $f = 0$, lấy $y_f = 0$.

Nếu $f \ne 0$, đặt $M = \ker f = \{x : f(x) = 0\}$. Vì $f$ liên tục, $M$ là không gian con đóng. Vì $f \ne 0$, $M \ne H$, nên $M^\perp \ne \{0\}$.

Chọn $z \in M^\perp$, $z \ne 0$. Với $x \in H$, đặt $u = x - \frac{f(x)}{f(z)} z$. Kiểm tra $f(u) = f(x) - \frac{f(x)}{f(z)} f(z) = 0$, nên $u \in M = (\ker f)$. Do $z \in M^\perp$:

$$\langle u, z\rangle = 0 \implies \langle x, z\rangle = \frac{f(x)}{f(z)}\|z\|^2$$

$$\implies f(x) = \left\langle x, \frac{f(z)}{\|z\|^2} z\right\rangle = \langle x, y_f\rangle$$

với $y_f = \frac{f(z)}{\|z\|^2} z \in M^\perp$.

**Duy nhất:** Nếu $\langle x, y_f\rangle = \langle x, y_f'\rangle$ với mọi $x$, thì $\langle x, y_f - y_f'\rangle = 0$ với mọi $x$. Lấy $x = y_f - y_f'$: $\|y_f - y_f'\|^2 = 0$, vậy $y_f = y_f'$.

**Chuẩn:** $\|f\|_{H^*} = \sup_{\|x\|=1}|f(x)| = \sup_{\|x\|=1} |\langle x, y_f\rangle| = \|y_f\|_H$ (từ Cauchy-Schwarz và đẳng thức khi $x = y_f/\|y_f\|$). $\square$

## 4.3. Ý nghĩa của định lý Riesz

**$H$ "tự đối ngẫu":** Khác với $L^p$ ($p \ne 2$) nơi $(L^p)^* = L^q \ne L^p$, trong Hilbert space $H^* \cong H$.

**Mọi phiếm hàm là "tích trong với một vector":** Không có phiếm hàm "kỳ lạ" trong $H^*$.

**Dạng bra-ket trong cơ học lượng tử:** Định lý Riesz là nền tảng toán học cho ký hiệu Dirac: $\langle \psi|$ (bra) $\in H^*$ tương ứng với $|\psi\rangle$ (ket) $\in H$ qua $\langle \phi|\psi\rangle = \langle |\phi\rangle, |\psi\rangle\rangle$.

---

# 5. Hệ quả của Định lý Riesz

## 5.1. $H$ phản xạ

Từ Riesz: $H^* \cong H$, nên $H^{**} \cong (H^*)^* \cong H^* \cong H$. Vậy mọi Hilbert space phản xạ.

**Hệ quả:** Trong Hilbert space, mọi dãy bị chặn có dãy con **hội tụ yếu**.

## 5.2. Biểu diễn tích trong

Từ định lý Riesz, ánh xạ $\Phi: H \to H^*$ định bởi $\Phi(y)(x) = \langle x, y\rangle$ là đẳng cấu đẳng chuẩn. Với $H$ phức, $\Phi$ là **anti-linear isometry** (không linear mà conjugate-linear vì $\langle x, \alpha y\rangle = \bar\alpha \langle x, y\rangle$).

## 5.3. Adjoint tự nhiên từ Riesz

Với $T: H \to K$ bị chặn (giữa hai Hilbert space), **adjoint** $T^*: K \to H$ định nghĩa qua:

$$\langle Tx, y\rangle_K = \langle x, T^*y\rangle_H \quad \forall x \in H, y \in K$$

**Tồn tại từ Riesz:** Với $y \in K$ cố định, $x \mapsto \langle Tx, y\rangle_K$ là phiếm hàm tuyến tính liên tục trên $H$ (bị chặn bởi $\|T\|\|y\|$). Riesz: tồn tại $z_y \in H$ duy nhất với $\langle Tx, y\rangle_K = \langle x, z_y\rangle_H$. Đặt $T^*y = z_y$.

---

# 6. Toán tử tự liên hợp (Self-adjoint) trên Hilbert space

## 6.1. Định nghĩa

**Toán tử tự liên hợp (Hermitian):** $T: H \to H$ bị chặn thỏa $T^* = T$, tức là:

$$\langle Tx, y\rangle = \langle x, Ty\rangle \quad \forall x, y \in H$$

**Ví dụ:**
- Multiplication operator $M_\phi: f \mapsto \phi f$ trên $L^2$ với $\phi$ thực.
- Toán tử tích phân $Tf(x) = \int K(x,y)f(y)dy$ với $K(x,y) = K(y,x)$.
- Laplacian $-\Delta$ trên miền thích hợp.
- Mọi ma trận đối xứng $A = A^T$ trong $\mathbb{R}^n$.

## 6.2. Tính chất

Với $T$ tự liên hợp:

1. **Trị riêng thực:** $Tu = \lambda u \Rightarrow \lambda \in \mathbb{R}$.
2. **Trị riêng khác nhau, vector riêng trực giao:** $Tu = \lambda u$, $Tv = \mu v$, $\lambda \ne \mu$ $\Rightarrow u \perp v$.
3. **Biểu diễn phổ (trường hợp chiều hữu hạn):** $T = \sum_i \lambda_i P_i$ với $P_i$ là chiếu lên không gian riêng.

**Chứng minh trị riêng thực:** $\lambda \|u\|^2 = \langle Tu, u\rangle = \langle u, Tu\rangle = \bar\lambda\|u\|^2 \Rightarrow \lambda = \bar\lambda \in \mathbb{R}$.

**Chứng minh trực giao:** $\lambda \langle u, v\rangle = \langle Tu, v\rangle = \langle u, Tv\rangle = \mu\langle u, v\rangle$, nên $(\lambda - \mu)\langle u,v\rangle = 0$, vậy $\langle u,v\rangle = 0$.

---

# 7. Bài toán cực tiểu hóa bậc hai và ứng dụng ML

## 7.1. Dạng tổng quát

Nhiều bài toán ML có dạng:

$$\min_{f \in \mathcal{H}} J(f) = \min_{f \in \mathcal{H}} \left[\sum_{i=1}^n L(y_i, f(x_i)) + \lambda \|f\|_{\mathcal{H}}^2\right]$$

trong đó $\mathcal{H}$ là RKHS. Đây là bài toán cực tiểu hóa trên Hilbert space.

**Định lý representer (Kimeldorf-Wahba):** Nghiệm tối ưu $f^*$ có dạng:

$$f^*(x) = \sum_{i=1}^n \alpha_i K(x, x_i)$$

Nghĩa là bài toán vô hạn chiều rút về bài toán hữu hạn chiều trong $\alpha \in \mathbb{R}^n$.

**Chứng minh:** Phân tách $f = f_{||\text{data}} + f_\perp$ trong đó $f_{||\text{data}} \in \text{span}\{K(\cdot, x_i)\}$ và $f_\perp \perp \text{span}\{K(\cdot, x_i)\}$. Do tính reproducing: $f(x_i) = \langle f, K(\cdot, x_i)\rangle = \langle f_{||\text{data}}, K(\cdot, x_i)\rangle$ — phần vuông góc không ảnh hưởng đến dữ liệu nhưng tăng chuẩn. Bỏ $f_\perp$ giảm $J$. $\square$

## 7.2. Ridge Regression trong RKHS

$$\min_{f \in \mathcal{H}_K} \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda\|f\|_{\mathcal{H}_K}^2$$

Từ representer theorem: $f^*(x) = \sum_i \alpha_i K(x, x_i)$ với $\alpha = (K + \lambda I)^{-1} y$ (ma trận kernel $K_{ij} = K(x_i, x_j)$).

## 7.3. Gaussian Process Regression

Trong Gaussian Process (GP), hàm hậu nghiệm (posterior):

$$f^* = \mathbb{E}[f|y] = K(X, \cdot)^T (K(X,X) + \sigma^2 I)^{-1} y$$

Đây chính là nghiệm ridge regression trong RKHS — cho thấy GP và kernel ridge regression là hai mặt của cùng một bài toán (Bayesian vs frequentist).

---

# 8. Liên hệ với lý thuyết chiều và tập tuyến tính

## 8.1. Khoảng cách từ điểm đến không gian con

Với không gian con đóng $M = \overline{\text{span}}\{e_1, e_2, \dots, e_k\}$ ($e_i$ trực chuẩn), chiếu:

$$P_M x = \sum_{i=1}^k \langle x, e_i\rangle e_i$$

$$d(x, M)^2 = \|x - P_M x\|^2 = \|x\|^2 - \sum_{i=1}^k |\langle x, e_i\rangle|^2$$

Đây là công thức khoảng cách từ vector đến không gian con — dùng trong phân tích thành phần chính (PCA).

## 8.2. PCA qua lăng kính chiếu Hilbert

PCA tìm không gian con $M$ $k$-chiều cực đại hóa phương sai chiếu:

$$\max_{M: \dim M = k} \sum_i \|P_M x_i\|^2 = \max_{M: \dim M = k} \sum_i \left(\|x_i\|^2 - d(x_i, M)^2\right)$$

Nghiệm: $M$ = không gian riêng của ma trận hiệp phương sai $\Sigma = \frac{1}{n}\sum x_i x_i^T$ ứng với $k$ trị riêng lớn nhất.

---

# 9. Định lý biểu diễn cho $L^2(\mu)$

## 9.1. Riesz trên $L^2$

Áp dụng cho $H = L^2(\Omega, \mu)$: mọi $f \in (L^2)^*$ có dạng $f(g) = \int gh \, d\mu$ với $h \in L^2$ duy nhất, và $\|f\|_{(L^2)^*} = \|h\|_{L^2}$.

Điều này nhất quán với biết đã biết: $(L^p)^* \cong L^q$, và với $p = q = 2$ thì $L^2$ tự đối ngẫu.

## 9.2. Riesz và Radon-Nikodym

Định lý Riesz trên Hilbert và định lý Radon-Nikodym (File 6) liên quan sâu sắc:

- Radon-Nikodym: $\nu \ll \mu \Rightarrow \nu = f \cdot \mu$ (biểu diễn độ đo).
- Riesz: $T \in (L^2)^* \Rightarrow T(\cdot) = \langle \cdot, h\rangle$ (biểu diễn phiếm hàm).

Cả hai đều là "biểu diễn qua mật độ/hàm" — thể hiện nguyên lý chung: mọi vật thể đối ngẫu trong thế giới đo đếm đều là tích phân/tích trong với một hàm.

---

# 10. Tóm tắt

**Chuỗi suy luận:**

$$\text{Hilbert} \xrightarrow{\text{Chiếu}} \text{Phân tách} H = M \oplus M^\perp \xrightarrow{\text{Riesz}} H^* \cong H$$

**Các định lý cốt lõi:**

| Định lý | Phát biểu | Ứng dụng |
|---|---|---|
| Chiếu lên tập lồi đóng | Tồn tại và duy nhất $m^*$, đặc trưng bởi bất đẳng thức | Least squares, convex opt |
| Phân tách Hilbert | $H = M \oplus M^\perp$ với $M$ đóng | ONB, PCA |
| Định lý Riesz | $H^* \cong H$ qua $f \leftrightarrow y_f$ | Kernel methods, GP |
| Representer Theorem | Min trong RKHS = bài toán hữu hạn chiều | SVM, kernel regression |

**Điều kiện tối ưu của chiếu:**

$$\boxed{\langle x - P_C x, m - P_C x\rangle \le 0 \quad \forall m \in C}$$

**Định lý Riesz:**

$$\boxed{f \in H^* \implies \exists! y_f \in H : f(x) = \langle x, y_f\rangle, \; \|f\| = \|y_f\|}$$

> Bài tiếp theo — **Toán tử compact** — sẽ nghiên cứu một lớp toán tử đặc biệt quan trọng: chúng xử lý tập bị chặn thành tập compact, có phổ đẹp, và là cầu nối giữa chiều hữu hạn và chiều vô hạn.

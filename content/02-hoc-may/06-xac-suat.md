# Ôn tập Xác suất cho Machine Learning

> Học máy là nghệ thuật **suy luận trong điều kiện bất định (uncertainty)**. Dữ liệu nhiễu, nhãn thiếu, mô hình không bao giờ hoàn hảo — và ngôn ngữ duy nhất diễn đạt được sự bất định ấy một cách nhất quán chính là **xác suất (probability)**.
>
> Bài này gom lại những viên gạch xác suất mà mọi thuật toán ML đứng trên: biến ngẫu nhiên, phân phối, kỳ vọng, và viên ngọc trung tâm — **định lý Bayes**:
>
> $$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$
>
> Nắm chắc những thứ này, các mô hình như [Naive Bayes](#/naive-bayes) hay [MLE & MAP](#/mle-map) sẽ hiện ra không phải như công thức cần học thuộc, mà như **hệ quả tất yếu** của vài quy tắc đơn giản.

---

# 1. Biến ngẫu nhiên và phân phối

## 1.1. Biến ngẫu nhiên là gì

Một **biến ngẫu nhiên (random variable)** $X$ là một quy tắc gán mỗi kết quả của một phép thử ngẫu nhiên với một con số. Tung đồng xu mà gán "sấp" $\mapsto 0$, "ngửa" $\mapsto 1$ thì $X$ là biến ngẫu nhiên. Trong ML, $X$ có thể là điểm ảnh, là từ trong câu, là đặc trưng (feature) của một mẫu dữ liệu.

Ta chia biến ngẫu nhiên thành hai loại:

* **Rời rạc (discrete)** — nhận giá trị trong một tập đếm được (số chấm xúc xắc, nhãn lớp). Mô tả bằng **hàm khối xác suất (probability mass function, pmf)**.
* **Liên tục (continuous)** — nhận giá trị trên một khoảng số thực (chiều cao, nhiệt độ). Mô tả bằng **hàm mật độ xác suất (probability density function, pdf)**.

## 1.2. pmf, pdf và cdf

Với biến **rời rạc**, pmf $p(x) = P(X = x)$ trả về xác suất tại đúng một giá trị. Nó thỏa:

$$p(x) \ge 0, \qquad \sum_{x} p(x) = 1$$

Với biến **liên tục**, ta **không** nói về $P(X = x)$ (luôn bằng 0), mà về mật độ $p(x)$ — xác suất tích lũy trên một khoảng:

$$P(a \le X \le b) = \int_a^b p(x)\, dx, \qquad p(x) \ge 0, \qquad \int_{-\infty}^{\infty} p(x)\, dx = 1$$

Lưu ý quan trọng: $p(x)$ là **mật độ**, không phải xác suất, nên nó **có thể lớn hơn 1**. Chỉ có tích phân của nó mới bị chặn trong $[0, 1]$.

Cả hai loại đều có chung một mô tả: **hàm phân phối tích lũy (cumulative distribution function, cdf)**:

$$F_X(x) = P(X \le x)$$

cdf không giảm, đi từ $0$ lên $1$. Với biến liên tục, $F_X(x) = \int_{-\infty}^x p(t)\, dt$, nên đạo hàm của cdf chính là pdf: $p(x) = F_X'(x)$. Mối liên hệ này là nền cho **mẹo CDF nghịch** dùng để lấy mẫu và cho công thức đổi biến trong [Normalizing Flows](#/normalizing-flows).

---

# 2. Xác suất đồng thời, biên và có điều kiện

Khi có nhiều biến ngẫu nhiên cùng lúc, mọi thông tin nằm trong **phân phối đồng thời (joint distribution)** $P(X, Y)$ — xác suất để đồng thời $X = x$ và $Y = y$.

## 2.1. Phân phối biên (marginal)

Từ phân phối đồng thời, ta "tổng hết" một biến để thu về phân phối của biến còn lại. Đây là **quy tắc tổng (sum rule)** hay **lấy biên (marginalization)**:

$$P(X) = \sum_{y} P(X, Y = y) \qquad \text{(rời rạc)}, \qquad p(x) = \int p(x, y)\, dy \qquad \text{(liên tục)}$$

Trực giác: muốn biết xác suất của $X$ mà không quan tâm $Y$, ta cộng dồn mọi cách $Y$ có thể xảy ra.

## 2.2. Xác suất có điều kiện

**Xác suất có điều kiện (conditional probability)** trả lời: biết $B$ đã xảy ra, $A$ có khả năng đến đâu?

$$P(A \mid B) = \frac{P(A, B)}{P(B)}, \qquad P(B) > 0$$

Trực giác hình học: ta **thu hẹp vũ trụ** lại còn đúng vùng $B$, rồi đo phần $A$ chiếm trong vùng đó. Mẫu số $P(B)$ chính là phép **chuẩn hóa lại** để xác suất trong vũ trụ mới vẫn cộng thành 1.

## 2.3. Quy tắc nhân (chain rule)

Nhân chéo định nghĩa trên cho ta **quy tắc nhân (product rule)**:

$$P(A, B) = P(A \mid B)\, P(B) = P(B \mid A)\, P(A)$$

Mở rộng cho nhiều biến là **quy tắc chuỗi (chain rule)** — xương sống của mọi mô hình tự hồi quy (autoregressive):

$$P(X_1, X_2, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid X_1, \dots, X_{i-1})$$

Đẳng thức này **luôn đúng**, không cần giả thiết gì. Mọi sự đơn giản hóa (như giả định độc lập của [Naive Bayes](#/naive-bayes)) chỉ là cách **cắt bớt** các điều kiện ở vế phải để bài toán khả thi.

---

# 3. Độc lập và độc lập có điều kiện

## 3.1. Độc lập

Hai biến cố $A, B$ **độc lập (independent)** nếu biết cái này không cho ta thông tin gì về cái kia:

$$P(A \mid B) = P(A) \quad \Longleftrightarrow \quad P(A, B) = P(A)\, P(B)$$

Dạng tích bên phải đối xứng hơn và thường được lấy làm định nghĩa. Khi đó quy tắc chuỗi sụp đổ thành tích các xác suất đơn lẻ — đây chính là điều khiến tính toán trở nên rẻ.

## 3.2. Độc lập có điều kiện

Tinh tế và quan trọng hơn cho ML là **độc lập có điều kiện (conditional independence)**: $A$ và $B$ độc lập **khi đã biết** $C$:

$$P(A, B \mid C) = P(A \mid C)\, P(B \mid C)$$

Ký hiệu $A \perp B \mid C$. Điều này **không** kéo theo $A, B$ độc lập vô điều kiện, và ngược lại. Ví dụ: chiều cao và vốn từ vựng của một đứa trẻ tương quan với nhau — nhưng **khi cố định tuổi**, chúng trở nên độc lập; tuổi là nguyên nhân chung làm cả hai cùng tăng.

Giả định cốt lõi của [Naive Bayes](#/naive-bayes) đúng là dạng này: các đặc trưng độc lập có điều kiện **khi đã biết nhãn lớp** $C$:

$$P(x_1, \dots, x_d \mid C) = \prod_{i=1}^{d} P(x_i \mid C)$$

Nhờ đó, thay vì ước lượng một phân phối đồng thời $d$ chiều khổng lồ, ta chỉ cần $d$ phân phối một chiều — "naive" mà hiệu quả đến bất ngờ.

---

# 4. Định lý Bayes

## 4.1. Suy dẫn

Định lý Bayes không phải một tiên đề mới — nó rơi ra trực tiếp từ quy tắc nhân. Viết $P(A, B)$ theo hai chiều:

$$P(A, B) = P(A \mid B)\, P(B) = P(B \mid A)\, P(A)$$

Cho hai vế phải bằng nhau rồi chia cho $P(B)$:

$$\boxed{\; P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)} \;}$$

Mẫu số thường được khai triển bằng quy tắc tổng (định lý xác suất toàn phần) trên các giả thuyết $A_k$ loại trừ nhau:

$$P(B) = \sum_{k} P(B \mid A_k)\, P(A_k)$$

## 4.2. Ý nghĩa: prior, likelihood, posterior

Đây là chỗ Bayes biến từ một mẹo đại số thành **triết lý suy luận**. Đặt $A = \theta$ (giả thuyết/tham số) và $B = \mathcal{D}$ (dữ liệu quan sát):

$$\underbrace{P(\theta \mid \mathcal{D})}_{\text{hậu nghiệm}} = \frac{\overbrace{P(\mathcal{D} \mid \theta)}^{\text{hợp lý}} \; \overbrace{P(\theta)}^{\text{tiên nghiệm}}}{\underbrace{P(\mathcal{D})}_{\text{bằng chứng}}}$$

| Thành phần | Tên | Ý nghĩa |
| --- | --- | --- |
| $P(\theta)$ | **Tiên nghiệm (prior)** | Niềm tin về $\theta$ *trước khi* thấy dữ liệu |
| $P(\mathcal{D} \mid \theta)$ | **Hợp lý (likelihood)** | Dữ liệu phù hợp đến đâu với một $\theta$ cho trước |
| $P(\theta \mid \mathcal{D})$ | **Hậu nghiệm (posterior)** | Niềm tin về $\theta$ *sau khi* cập nhật bằng dữ liệu |
| $P(\mathcal{D})$ | **Bằng chứng (evidence)** | Hằng số chuẩn hóa, độc lập với $\theta$ |

Vì $P(\mathcal{D})$ không phụ thuộc $\theta$, ta thường viết gọn:

$$P(\theta \mid \mathcal{D}) \propto P(\mathcal{D} \mid \theta)\, P(\theta)$$

Đây chính là cây cầu sang [MLE & MAP](#/mle-map): cực đại likelihood $P(\mathcal{D} \mid \theta)$ cho ra **MLE**; cực đại posterior $P(\mathcal{D} \mid \theta)\, P(\theta)$ cho ra **MAP** — MAP chỉ là MLE có thêm prior đóng vai trò điều chuẩn (regularization).

---

# 5. Kỳ vọng, phương sai, hiệp phương sai

## 5.1. Kỳ vọng

**Kỳ vọng (expectation)** là giá trị trung bình của $X$ nếu lặp phép thử vô số lần — trọng tâm của phân phối:

$$\mathbb{E}[X] = \sum_{x} x\, p(x) \quad \text{(rời rạc)}, \qquad \mathbb{E}[X] = \int x\, p(x)\, dx \quad \text{(liên tục)}$$

Tổng quát hơn, kỳ vọng của một hàm: $\mathbb{E}[g(X)] = \sum_x g(x)\, p(x)$ (luật của thống kê viên vô thức, LOTUS).

## 5.2. Tính tuyến tính của kỳ vọng (chứng minh)

Một trong những tính chất hữu dụng nhất của ML, vì nó đúng **kể cả khi các biến phụ thuộc nhau**:

$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$$

**Chứng minh** (trường hợp rời rạc). Dùng phân phối đồng thời $p(x, y)$ và LOTUS:

$$\mathbb{E}[aX + bY] = \sum_{x}\sum_{y} (ax + by)\, p(x, y)$$

Tách tổng thành hai phần và đưa hằng số ra ngoài:

$$= a \sum_{x}\sum_{y} x\, p(x, y) + b \sum_{x}\sum_{y} y\, p(x, y)$$

Lấy biên ($\sum_y p(x,y) = p(x)$ ở số hạng đầu, $\sum_x p(x,y) = p(y)$ ở số hạng sau):

$$= a \sum_{x} x\, p(x) + b \sum_{y} y\, p(y) = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] \qquad \blacksquare$$

Bước lấy biên là chìa khóa: ta **không** dùng độc lập ở bất kỳ đâu — đó là lý do tính tuyến tính luôn đúng.

## 5.3. Phương sai

**Phương sai (variance)** đo độ phân tán quanh kỳ vọng:

$$\operatorname{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big] = \mathbb{E}[X^2] - \big(\mathbb{E}[X]\big)^2$$

Đẳng thức rút gọn thứ hai suy ra bằng cách khai triển bình phương rồi dùng tính tuyến tính. Khác với kỳ vọng, phương sai **không** tuyến tính: $\operatorname{Var}(aX) = a^2 \operatorname{Var}(X)$.

## 5.4. Hiệp phương sai

**Hiệp phương sai (covariance)** đo mức hai biến cùng dao động:

$$\operatorname{Cov}(X, Y) = \mathbb{E}\big[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])\big] = \mathbb{E}[XY] - \mathbb{E}[X]\,\mathbb{E}[Y]$$

Nếu $X, Y$ độc lập thì $\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y]$, nên $\operatorname{Cov}(X, Y) = 0$. **Cảnh báo:** chiều ngược lại **không** đúng — hiệp phương sai bằng 0 (không tương quan) không kéo theo độc lập. Tổng quát:

$$\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y)$$

Với dữ liệu nhiều chiều, mọi hiệp phương sai từng cặp gom thành **ma trận hiệp phương sai (covariance matrix)** $\Sigma$ — đối xứng, nửa xác định dương, và là tham số trung tâm của phân phối Gauss nhiều chiều.

---

# 6. Các phân phối thường gặp

## 6.1. Bảng tổng quan

| Phân phối | Loại | Tham số | Kỳ vọng | Phương sai |
| --- | --- | --- | --- | --- |
| Bernoulli | rời rạc | $p$ | $p$ | $p(1-p)$ |
| Categorical | rời rạc | $\boldsymbol{\pi}$ | — | — |
| Nhị thức (Binomial) | rời rạc | $n, p$ | $np$ | $np(1-p)$ |
| Gauss 1 chiều | liên tục | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ |
| Laplace | liên tục | $\mu, b$ | $\mu$ | $2b^2$ |
| Beta | liên tục | $\alpha, \beta$ | $\frac{\alpha}{\alpha+\beta}$ | — |

## 6.2. Bernoulli và Categorical

**Bernoulli** mô hình hóa một phép thử nhị phân (đúng/sai, có/không):

$$p(x) = p^x (1-p)^{1-x}, \qquad x \in \{0, 1\}$$

Đây là phân phối ngầm sau **hồi quy logistic (logistic regression)**: mô hình dự đoán tham số $p = \sigma(w^\top x)$ cho nhãn nhị phân.

**Categorical** là tổng quát lên $K$ lớp với vector xác suất $\boldsymbol{\pi} = (\pi_1, \dots, \pi_K)$, $\sum_k \pi_k = 1$:

$$p(x = k) = \pi_k$$

Đây là phân phối ngầm sau lớp softmax trong phân loại đa lớp.

## 6.3. Nhị thức (Binomial)

Tổng của $n$ phép thử Bernoulli độc lập cùng tham số $p$ cho **phân phối nhị thức** — đếm số lần "thành công":

$$p(k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad k = 0, 1, \dots, n$$

Kỳ vọng $np$ và phương sai $np(1-p)$ suy ra ngay từ tính tuyến tính của kỳ vọng áp dụng cho tổng các Bernoulli.

## 6.4. Gauss một chiều

**Phân phối Gauss (Gaussian / normal)** là phân phối quan trọng nhất trong ML — vừa vì định lý giới hạn trung tâm, vừa vì toán học của nó cực kỳ tiện:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$

Lấy logarit, mọi thứ thành đa thức bậc hai: $\log p(x) = -\frac{(x-\mu)^2}{2\sigma^2} + \text{const}$. Đây chính là lý do **cực đại likelihood Gauss tương đương cực tiểu sai số bình phương (MSE)** — cây cầu nối xác suất với [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu).

## 6.5. Gauss nhiều chiều

Tổng quát lên $D$ chiều với vector trung bình $\boldsymbol{\mu}$ và ma trận hiệp phương sai $\Sigma$:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\!\left( -\tfrac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)$$

Số hạng $(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})$ là **khoảng cách Mahalanobis** bình phương — khoảng cách Euclid được "chuẩn hóa" theo hình dạng của đám mây dữ liệu. Khi $\Sigma$ là ma trận đường chéo, các chiều độc lập và mật độ tách thành tích các Gauss một chiều.

## 6.6. Beta, Dirichlet và Laplace (nhắc)

* **Beta** $\text{Beta}(\alpha, \beta)$ trên $[0, 1]$ là phân phối của một *xác suất*. Nó là **liên hợp (conjugate prior)** của Bernoulli/Nhị thức: prior Beta gặp likelihood nhị thức cho ra posterior vẫn là Beta — khiến cập nhật Bayes thành phép cộng đếm đơn giản.
* **Dirichlet** là tổng quát của Beta lên vector xác suất, liên hợp của Categorical/Đa thức — nền của các mô hình chủ đề (topic model).
* **Laplace** $p(x) \propto \exp(-|x - \mu|/b)$ có "đuôi nhọn" với trị tuyệt đối thay vì bình phương. Cực đại likelihood Laplace tương đương cực tiểu **sai số tuyệt đối (MAE)**, và Laplace prior trên trọng số tạo ra điều chuẩn **L1** (sparse) — đối ứng với Gauss prior tạo ra L2.

---

# 7. Vài khái niệm cần cho ML

* **Entropy** $H(X) = -\sum_x p(x) \log p(x)$ — đo độ bất định trung bình của một phân phối. Xuất hiện trực tiếp trong hàm mất mát cross-entropy.
* **Phân kỳ KL (Kullback–Leibler divergence)** $D_{\mathrm{KL}}(p \,\|\, q) = \mathbb{E}_{x \sim p}\big[\log \frac{p(x)}{q(x)}\big] \ge 0$ — đo khoảng cách (không đối xứng) giữa hai phân phối; cực tiểu hóa nó là mục tiêu ngầm của hầu hết huấn luyện sinh.
* **Bất đẳng thức Jensen.** Với hàm lồi $\varphi$: $\varphi(\mathbb{E}[X]) \le \mathbb{E}[\varphi(X)]$. Đây là công cụ suy ra $D_{\mathrm{KL}} \ge 0$ và cận dưới ELBO của VAE.
* **Định lý giới hạn trung tâm (CLT).** Trung bình của nhiều biến độc lập cùng phân phối tiệm cận Gauss — lý do Gauss xuất hiện ở khắp nơi và lý do ta hay giả định nhiễu là Gauss.
* **Bất đẳng thức Chebyshev.** $P(|X - \mu| \ge k\sigma) \le 1/k^2$ — cho cận xác suất một biến lệch xa trung bình, nền của nhiều cận tổng quát hóa (generalization bound).

---

# 8. Tổng kết

Mọi mô hình học máy, dù phức tạp đến đâu, đều được lắp từ vài viên gạch trong bài này:

* **pmf/pdf/cdf** mô tả một biến ngẫu nhiên; **đồng thời, biên, có điều kiện** mô tả nhiều biến cùng lúc, nối với nhau qua **quy tắc tổng** và **quy tắc nhân**.
* **Định lý Bayes** — rơi thẳng ra từ quy tắc nhân — cho ta cách **cập nhật niềm tin** từ prior qua likelihood thành posterior. Đây là khung suy luận thống nhất đứng sau [MLE & MAP](#/mle-map) và [Naive Bayes](#/naive-bayes).
* **Kỳ vọng** (tuyến tính, luôn đúng), **phương sai** và **hiệp phương sai** tóm tắt một phân phối thành vài con số có thể tính và tối ưu.
* Họ **phân phối** chuẩn (Bernoulli, Categorical, Nhị thức, Gauss, Beta/Dirichlet, Laplace) là từ vựng để giả định về dữ liệu — và mỗi giả định lại sinh ra một hàm mất mát cụ thể.

> Điểm mấu chốt cần mang theo: **mỗi hàm mất mát là một giả định xác suất trá hình**. MSE là Gauss, cross-entropy là Bernoulli/Categorical, MAE là Laplace, điều chuẩn L2/L1 là Gauss/Laplace prior. Khi đã nhìn ra điều đó, ML không còn là một túi mẹo rời rạc, mà là **suy luận Bayes áp dụng nhất quán**.
>
> Bài tiếp theo — **[Naive Bayes](#/naive-bayes)** — là ứng dụng trực tiếp đầu tiên: ghép định lý Bayes với giả định độc lập có điều kiện để dựng một bộ phân loại đơn giản mà mạnh mẽ đến không ngờ.

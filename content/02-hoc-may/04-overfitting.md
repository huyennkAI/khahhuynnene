# Overfitting & Điều chuẩn (Regularization)

> Một mô hình tốt không phải là mô hình **khớp hoàn hảo** dữ liệu huấn luyện, mà là mô hình **tổng quát hóa (generalize)** tốt lên dữ liệu chưa từng thấy.
>
> Khi mô hình quá phức tạp, nó học cả **nhiễu (noise)** trong dữ liệu huấn luyện — hiện tượng **quá khớp (overfitting)**. Khi mô hình quá đơn giản, nó bỏ sót quy luật thật — **dưới khớp (underfitting)**. Bài này phân tích nguồn gốc của sự đánh đổi đó qua **phân rã thiên lệch–phương sai (bias–variance decomposition)**, rồi giới thiệu các công cụ kiểm soát độ phức tạp: **kiểm định chéo (cross-validation)** và **điều chuẩn (regularization)** L2/L1.

---

# 1. Overfitting và Underfitting

## 1.1. Trực giác

Giả sử dữ liệu thật sinh ra từ một hàm $f(x)$ cộng nhiễu ngẫu nhiên:

$$y = f(x) + \varepsilon, \qquad \mathbb{E}[\varepsilon] = 0,\; \operatorname{Var}(\varepsilon) = \sigma^2$$

Mục tiêu là tìm $\hat f$ xấp xỉ $f$. Nhưng ta chỉ quan sát được tập huấn luyện hữu hạn $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ đã **lẫn nhiễu** $\varepsilon$. Nếu cố ép $\hat f$ đi qua **đúng** mọi điểm $(x_i, y_i)$, mô hình sẽ học cả phần nhiễu — vốn không lặp lại ở dữ liệu mới.

Ví dụ kinh điển: khớp đa thức bậc cao vào vài điểm dữ liệu. Đa thức bậc $1$ (đường thẳng) quá cứng — **underfitting**; đa thức bậc $N-1$ đi qua mọi điểm nhưng dao động dữ dội giữa các điểm — **overfitting**; một bậc trung gian nắm được quy luật mà không chạy theo nhiễu.

## 1.2. Định nghĩa

- **Quá khớp (overfitting):** mô hình đạt sai số **rất thấp** trên tập huấn luyện nhưng sai số **cao** trên tập kiểm tra. Khoảng cách lớn giữa hai sai số gọi là **khoảng hở tổng quát hóa (generalization gap)**.
- **Dưới khớp (underfitting):** mô hình có sai số **cao trên cả hai** tập, do không đủ sức biểu diễn quy luật.

Ta phân biệt hai loại sai số:

$$\underbrace{E_{\text{train}}}_{\text{lỗi huấn luyện}} \quad\text{và}\quad \underbrace{E_{\text{test}} \approx \mathbb{E}_{(x,y)}\big[L(y, \hat f(x))\big]}_{\text{lỗi tổng quát hóa}}$$

| Tình huống | $E_{\text{train}}$ | $E_{\text{test}}$ | Chẩn đoán |
| --- | --- | --- | --- |
| Underfitting | Cao | Cao | Thiên lệch lớn (high bias) |
| Vừa khớp | Thấp | Thấp | Cân bằng tốt |
| Overfitting | Rất thấp | Cao | Phương sai lớn (high variance) |

Vấn đề trung tâm của học máy: **giảm $E_{\text{test}}$**, chứ không phải $E_{\text{train}}$. Hai loại lỗi này liên hệ với nhau qua sự đánh đổi ở mục 2.

---

# 2. Đánh đổi thiên lệch–phương sai

## 2.1. Phát biểu

Độ phức tạp mô hình không thể tối ưu cả hai mặt cùng lúc:

- **Thiên lệch (bias)** — sai số do **giả định mô hình quá đơn giản**, không nắm được quy luật thật. Mô hình cứng nhắc ⇒ bias cao.
- **Phương sai (variance)** — sai số do mô hình **quá nhạy với tập huấn luyện cụ thể**. Đổi tập dữ liệu một chút thì $\hat f$ thay đổi nhiều ⇒ variance cao.

Tăng độ phức tạp thì bias giảm nhưng variance tăng, và ngược lại. Tổng của chúng có một điểm cực tiểu — đó là độ phức tạp tối ưu.

## 2.2. Phân rã sai số kỳ vọng (suy dẫn)

Xét một điểm $x$ cố định, nhãn thật $y = f(x) + \varepsilon$ với $\mathbb{E}[\varepsilon]=0$, $\operatorname{Var}(\varepsilon)=\sigma^2$. Dự đoán $\hat f(x)$ là biến ngẫu nhiên (phụ thuộc tập huấn luyện ngẫu nhiên). Ta phân rã sai số bình phương kỳ vọng:

$$\mathbb{E}\big[(y - \hat f(x))^2\big]$$

trong đó kỳ vọng lấy theo **cả** nhiễu $\varepsilon$ và tập huấn luyện. Đặt $\bar f(x) = \mathbb{E}[\hat f(x)]$ là dự đoán trung bình. Vì $\varepsilon$ độc lập với $\hat f$ và $f(x)$ là hằng số, ta tách:

$$\mathbb{E}\big[(y - \hat f)^2\big] = \mathbb{E}\big[(f + \varepsilon - \hat f)^2\big] = \mathbb{E}\big[(f - \hat f)^2\big] + \underbrace{\mathbb{E}[\varepsilon^2]}_{\sigma^2} + 2\,\mathbb{E}[\varepsilon]\,\mathbb{E}[f - \hat f]$$

Số hạng chéo triệt tiêu vì $\mathbb{E}[\varepsilon]=0$. Tiếp tục khai triển $\mathbb{E}[(f - \hat f)^2]$ bằng cách thêm bớt $\bar f$:

$$\mathbb{E}\big[(f - \hat f)^2\big] = \mathbb{E}\big[(f - \bar f + \bar f - \hat f)^2\big]$$

$$= \underbrace{(f - \bar f)^2}_{\text{hằng số}} + \mathbb{E}\big[(\bar f - \hat f)^2\big] + 2(f - \bar f)\,\mathbb{E}[\bar f - \hat f]$$

Số hạng chéo lại triệt tiêu vì $\mathbb{E}[\bar f - \hat f] = \bar f - \mathbb{E}[\hat f] = 0$. Vậy ta thu được công thức **phân rã thiên lệch–phương sai**:

$$\boxed{\;\mathbb{E}\big[(y - \hat f(x))^2\big] = \underbrace{(f(x) - \bar f(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\big[(\hat f(x) - \bar f(x))^2\big]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}\;}$$

Ba thành phần có ý nghĩa rõ ràng:

- $\text{Bias}^2$ — bình phương khoảng cách giữa dự đoán **trung bình** và sự thật. Mô hình quá đơn giản ⇒ $\bar f$ lệch xa $f$ ⇒ bias lớn.
- $\text{Variance}$ — mức dao động của dự đoán quanh trung bình của nó khi đổi tập huấn luyện. Mô hình quá linh hoạt ⇒ variance lớn.
- $\sigma^2$ — **nhiễu không thể giảm (irreducible error)**, là chặn dưới của sai số bất kỳ mô hình nào.

> **Hệ quả then chốt.** Vì $\sigma^2$ cố định, mọi nỗ lực cải thiện chỉ tác động vào bias và variance. Điều chuẩn (mục 5) là cách **đánh đổi có chủ đích**: chấp nhận tăng nhẹ bias để giảm mạnh variance, nhằm hạ tổng sai số tổng quát hóa.

---

# 3. Đường cong học và chẩn đoán

## 3.1. Lỗi huấn luyện so với lỗi kiểm định

Vẽ $E_{\text{train}}$ và $E_{\text{val}}$ theo **độ phức tạp mô hình** (bậc đa thức, số tham số, số vòng lặp...) cho ta một bức tranh chẩn đoán:

- $E_{\text{train}}$ **giảm đơn điệu** khi tăng độ phức tạp (mô hình càng mạnh càng khớp tốt dữ liệu đã thấy).
- $E_{\text{val}}$ có hình **chữ U**: giảm trong vùng underfitting, chạm đáy tại độ phức tạp tối ưu, rồi **tăng** trở lại khi bắt đầu overfitting.

Điểm overfitting nhận ra qua việc $E_{\text{val}} \gg E_{\text{train}}$ và $E_{\text{val}}$ đang đi lên.

## 3.2. Đường cong học theo cỡ dữ liệu (learning curve)

Một biến thể khác: vẽ sai số theo **số lượng mẫu huấn luyện** $N$ (giữ nguyên mô hình).

- Mô hình **bias cao**: cả $E_{\text{train}}$ và $E_{\text{val}}$ hội tụ tới một mức **cao** và sát nhau. Thêm dữ liệu **không** giúp ích — cần mô hình mạnh hơn.
- Mô hình **variance cao**: $E_{\text{train}}$ thấp, $E_{\text{val}}$ cao, khoảng hở **lớn** nhưng đang **thu hẹp** khi $N$ tăng. Thêm dữ liệu hoặc điều chuẩn sẽ giúp ích.

> Đường cong học cho biết nên đầu tư vào **dữ liệu** (variance cao) hay vào **mô hình/đặc trưng** (bias cao) — tránh lãng phí công sức sai hướng.

---

# 4. Tập kiểm định và kiểm định chéo

## 4.1. Vì sao cần tập riêng

Không được dùng tập huấn luyện để ước lượng $E_{\text{test}}$ (sẽ lạc quan giả tạo), cũng không được dùng tập kiểm tra (test set) để **chọn** siêu tham số (sẽ rò rỉ thông tin). Vì vậy ta chia ba phần:

- **Train** — học tham số $w$.
- **Validation** — chọn **siêu tham số (hyperparameter)** như độ phức tạp, $\lambda$.
- **Test** — chỉ dùng **một lần cuối** để báo cáo hiệu năng trung thực.

## 4.2. Kiểm định chéo k-phần (k-fold cross-validation)

Khi dữ liệu ít, tách riêng một tập validation cố định gây lãng phí. Thay vào đó, chia dữ liệu huấn luyện thành $k$ phần (fold) bằng nhau. Lặp $k$ lần: mỗi lần dùng $k-1$ phần để huấn luyện và **1 phần còn lại** để kiểm định, rồi lấy trung bình:

$$\text{CV}(k) = \frac{1}{k} \sum_{j=1}^{k} E_{\text{val}}^{(j)}$$

Mọi mẫu đều được dùng để kiểm định đúng một lần và để huấn luyện $k-1$ lần. Trường hợp $k = N$ gọi là **leave-one-out (LOOCV)** — gần như không thiên lệch nhưng tốn $N$ lần huấn luyện. Thực tế thường chọn $k = 5$ hoặc $k = 10$ để cân bằng độ tin cậy và chi phí.

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

scores = []
for lam in [0.01, 0.1, 1.0, 10.0]:
    model = Ridge(alpha=lam)
    cv = cross_val_score(model, X, y, cv=5,
                         scoring="neg_mean_squared_error")
    scores.append((lam, -cv.mean()))
best_lambda = min(scores, key=lambda t: t[1])[0]
```

> Quy trình chuẩn: dùng cross-validation trên tập train+val để **chọn $\lambda$**, huấn luyện lại trên toàn bộ dữ liệu đó với $\lambda$ tốt nhất, rồi đánh giá **một lần** trên test. Xem thêm [Đánh giá phân loại](#/danh-gia-phan-loai) cho các độ đo cụ thể.

---

# 5. Điều chuẩn L2 và L1

## 5.1. Ý tưởng: phạt độ lớn tham số

Overfitting thường đi kèm các trọng số $w$ có **độ lớn rất lớn** (để đường cong dao động mạnh khớp nhiễu). Ý tưởng điều chuẩn (regularization): thêm vào hàm mất mát một **số hạng phạt** tỉ lệ với độ lớn của $w$, ép mô hình ưu tiên nghiệm "đơn giản":

$$\mathcal{L}_{\text{reg}}(w) = \underbrace{\mathcal{L}_{\text{data}}(w)}_{\text{khớp dữ liệu}} + \lambda\, R(w)$$

Tham số $\lambda \ge 0$ điều khiển cường độ phạt: $\lambda = 0$ là không điều chuẩn (dễ overfit); $\lambda$ lớn ép $w \to 0$ (dễ underfit). Đây chính là một **núm vặn** trực tiếp lên sự đánh đổi bias–variance ở mục 2.

## 5.2. Điều chuẩn L2 (Ridge) — co hệ số

L2 phạt **bình phương chuẩn Euclid** của trọng số. Với hồi quy tuyến tính (xem [Linear Regression](#/linear-regression)):

$$\mathcal{L}_{\text{ridge}}(w) = \frac{1}{N}\sum_{i=1}^{N} \big(y_i - w^\top x_i\big)^2 + \lambda \lVert w \rVert_2^2, \qquad \lVert w \rVert_2^2 = \sum_{j} w_j^2$$

**Nghiệm dạng đóng.** Viết dưới dạng ma trận với $X \in \mathbb{R}^{N\times d}$. Lấy đạo hàm theo $w$ và cho bằng $0$:

$$\nabla_w \big[(y - Xw)^\top(y - Xw) + N\lambda\, w^\top w\big] = -2X^\top(y - Xw) + 2N\lambda\, w = 0$$

$$\Rightarrow \quad \boxed{\;\hat w_{\text{ridge}} = \big(X^\top X + N\lambda\, I\big)^{-1} X^\top y\;}$$

So với hồi quy thường $(X^\top X)^{-1}X^\top y$, ta cộng thêm $N\lambda I$ vào đường chéo. Điều này luôn làm ma trận **khả nghịch** (kể cả khi $X^\top X$ suy biến) và **co** mọi hệ số về phía $0$ một cách trơn tru. L2 thường **không** đưa hệ số về đúng $0$ — nó chỉ làm chúng nhỏ lại.

## 5.3. Điều chuẩn L1 (Lasso) — nghiệm thưa

L1 phạt **tổng trị tuyệt đối** của trọng số:

$$\mathcal{L}_{\text{lasso}}(w) = \frac{1}{N}\sum_{i=1}^{N} \big(y_i - w^\top x_i\big)^2 + \lambda \lVert w \rVert_1, \qquad \lVert w \rVert_1 = \sum_j |w_j|$$

Khác biệt cốt lõi: L1 đẩy nhiều hệ số về **đúng bằng $0$**, tạo ra **nghiệm thưa (sparse)** — tức là **chọn đặc trưng (feature selection)** tự động.

**Vì sao L1 tạo nghiệm thưa.** Có hai cách hiểu.

*Góc nhìn hình học.* Xem điều chuẩn như ràng buộc $R(w) \le t$. Vùng ràng buộc của L1 là hình **kim cương** (có các đỉnh nhọn nằm trên trục), còn của L2 là hình **tròn** (trơn). Đường đồng mức của hàm mất mát dữ liệu (ellipse) chạm vùng ràng buộc tại điểm tối ưu. Với hình kim cương, điểm chạm rất dễ rơi vào **đỉnh nhọn** — nơi một số tọa độ bằng $0$. Hình tròn không có đỉnh nên hiếm khi cho tọa độ $0$.

*Góc nhìn gradient (proximal/soft-thresholding).* Xét bài toán một chiều, gọi $z$ là nghiệm bình phương tối thiểu không phạt. Nghiệm L1 là phép **co mềm (soft-thresholding)**:

$$w^\star = \operatorname{sign}(z)\,\max\big(|z| - \lambda,\, 0\big)$$

Mọi $z$ có $|z| \le \lambda$ đều bị **đặt về $0$** — đây chính là cơ chế tạo sparsity. Trong khi đó nghiệm L2 chỉ là phép co theo tỉ lệ $w^\star = z/(1+\lambda)$, **không bao giờ** chạm $0$ với $z \ne 0$.

| Tiêu chí | L2 (Ridge) | L1 (Lasso) |
| --- | --- | --- |
| Số hạng phạt | $\lambda\lVert w\rVert_2^2$ | $\lambda\lVert w\rVert_1$ |
| Tác dụng lên hệ số | Co nhỏ (shrink) | Đưa về **đúng $0$** |
| Nghiệm thưa | Không | **Có** (chọn đặc trưng) |
| Khả vi tại $0$ | Có | Không (cần subgradient) |
| Nghiệm dạng đóng | Có | Không (giải lặp) |

> **Elastic Net** kết hợp cả hai: $\lambda_1\lVert w\rVert_1 + \lambda_2\lVert w\rVert_2^2$, vừa chọn đặc trưng vừa ổn định khi các đặc trưng tương quan cao.

---

# 6. Liên hệ điều chuẩn với MAP

Điều chuẩn không phải là một mẹo tùy tiện — nó tương đương với **ước lượng cực đại hậu nghiệm (MAP)** dưới một **tiên nghiệm (prior)** trên $w$. Hãy nhắc lại từ [MLE & MAP](#/mle-map): MAP cực đại hậu nghiệm $p(w \mid \mathcal{D}) \propto p(\mathcal{D}\mid w)\,p(w)$, tức cực tiểu

$$-\log p(\mathcal{D}\mid w) - \log p(w)$$

Số hạng đầu là hàm mất mát likelihood (ví dụ MSE với nhiễu Gauss); số hạng sau, $-\log p(w)$, **chính là** số hạng điều chuẩn.

## 6.1. L2 tương ứng tiên nghiệm Gauss

Đặt prior Gauss tâm $0$: $w_j \sim \mathcal{N}(0, \tau^2)$, độc lập. Khi đó:

$$-\log p(w) = -\sum_j \log\!\left(\frac{1}{\sqrt{2\pi}\,\tau}\exp\!\Big(-\frac{w_j^2}{2\tau^2}\Big)\right) = \frac{1}{2\tau^2}\sum_j w_j^2 + \text{const}$$

Đây đúng là số hạng L2 với $\lambda = \dfrac{1}{2\tau^2}$. **Tin tưởng $w$ nhỏ** (prior hẹp, $\tau$ nhỏ) tương ứng $\lambda$ **lớn**.

## 6.2. L1 tương ứng tiên nghiệm Laplace

Đặt prior Laplace tâm $0$: $p(w_j) = \dfrac{1}{2b}\exp\!\big(-|w_j|/b\big)$. Khi đó:

$$-\log p(w) = \frac{1}{b}\sum_j |w_j| + \text{const}$$

Đây là số hạng L1 với $\lambda = 1/b$. Phân phối Laplace có **đỉnh nhọn** và đuôi dày tại $0$ — nó đặt nhiều khối xác suất ngay tại $w_j = 0$, giải thích bằng ngôn ngữ Bayes vì sao L1 cho nghiệm thưa.

> **Tổng quát.** Mọi điều chuẩn $\lambda R(w)$ đều đọc được như một prior $p(w) \propto e^{-\lambda R(w)}$. Đây là cầu nối sâu sắc giữa quan điểm **tối ưu** (thêm số hạng phạt) và quan điểm **xác suất** (đặt niềm tin tiên nghiệm). Khi $\lambda \to 0$, prior trở nên "phẳng" và MAP suy biến về MLE.

---

# 7. Các kỹ thuật điều chuẩn khác

## 7.1. Dừng sớm (early stopping)

Trong huấn luyện lặp (gradient descent — xem [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu)), theo dõi $E_{\text{val}}$ sau mỗi epoch. $E_{\text{train}}$ giảm liên tục, nhưng $E_{\text{val}}$ chạm đáy rồi tăng lên (bắt đầu overfit). **Dừng sớm** nghĩa là ngừng tại điểm $E_{\text{val}}$ thấp nhất và giữ lại trọng số ở đó.

Số vòng lặp đóng vai trò như một **núm điều chuẩn ngầm**: càng huấn luyện lâu, $w$ càng đi xa khỏi điểm khởi tạo nhỏ, độ phức tạp hiệu dụng càng tăng. Với mô hình tuyến tính, có thể chứng minh early stopping **xấp xỉ** điều chuẩn L2.

## 7.2. Dropout (cho mạng nơ-ron)

Trong mỗi bước huấn luyện, **dropout** ngẫu nhiên "tắt" mỗi nơ-ron với xác suất $p$ (đặt đầu ra về $0$). Điều này ngăn các nơ-ron **đồng thích nghi (co-adaptation)** quá mức, buộc mạng học các đặc trưng dư thừa, bền vững hơn. Có thể hiểu dropout như huấn luyện ngầm một **tập hợp (ensemble)** khổng lồ các mạng con chia sẻ trọng số. Khi suy luận, bật toàn bộ nơ-ron và chia tỉ lệ phù hợp.

## 7.3. Tăng cường dữ liệu (data augmentation)

Vì variance cao thường do **thiếu dữ liệu**, ta sinh thêm mẫu huấn luyện bằng các phép biến đổi **bảo toàn nhãn**: với ảnh là lật, xoay, cắt, đổi sáng; với văn bản là thay từ đồng nghĩa. Mô hình học được tính **bất biến (invariance)** với các biến đổi đó, giảm overfitting mà không cần thu thập dữ liệu mới.

---

# 8. Ưu điểm và khi nào dùng

- **Cải thiện tổng quát hóa** — điều chuẩn giảm variance, hạ $E_{\text{test}}$, đặc biệt khi **ít dữ liệu** so với số tham số.
- **Ổn định nghiệm** — L2 làm $X^\top X + N\lambda I$ luôn khả nghịch, xử lý tốt **đa cộng tuyến (multicollinearity)**.
- **Chọn đặc trưng tự động** — L1 cho nghiệm thưa, hữu ích khi nghi ngờ nhiều đặc trưng vô ích; tăng tính diễn giải.
- **Rẻ và phổ quát** — chỉ thêm một số hạng vào hàm mất mát; áp dụng cho hầu hết mô hình (hồi quy, [Logistic Regression](#/logistic-regression), mạng nơ-ron).

Dùng L2 làm mặc định; dùng L1 khi cần chọn đặc trưng; dùng dropout/early stopping/augmentation cho mạng nơ-ron sâu; luôn chọn $\lambda$ bằng cross-validation.

---

# 9. Hạn chế

- **Phải dò siêu tham số** — $\lambda$, tỉ lệ dropout, lịch early stopping đều cần tìm kiếm tốn kém qua cross-validation.
- **Thêm bias** — điều chuẩn quá mạnh ($\lambda$ quá lớn) gây **underfitting**; phải cân bằng.
- **Nhạy với thang đo** — số hạng phạt phụ thuộc độ lớn đặc trưng, nên **bắt buộc chuẩn hóa (standardize)** dữ liệu trước; thường **không** phạt hệ số chệch (bias term).
- **L1 kém ổn định** — khi các đặc trưng tương quan cao, L1 chọn tùy ý một trong số chúng; lúc đó Elastic Net tốt hơn.
- **Không thay thế dữ liệu tốt** — với mô hình bias cao, điều chuẩn không giúp; cần đặc trưng/kiến trúc tốt hơn (xem mục 3.2).

---

# 10. Tổng kết

Sai số tổng quát hóa được phân rã chính xác thành ba phần:

$$\mathbb{E}\big[(y - \hat f)^2\big] = \text{Bias}^2 + \text{Variance} + \sigma^2$$

Trong đó $\sigma^2$ là nhiễu không thể giảm. Toàn bộ nghệ thuật chống overfitting là **kiểm soát độ phức tạp mô hình** để đặt nó vào đúng điểm cân bằng bias–variance:

- **Chẩn đoán** bằng đường cong học và lỗi train so với validation.
- **Ước lượng trung thực** bằng tập kiểm định và k-fold cross-validation.
- **Đánh đổi có chủ đích** bằng điều chuẩn L2 (co hệ số, prior Gauss) và L1 (nghiệm thưa, prior Laplace) — hai mặt của cùng một câu chuyện MAP.
- **Bổ sung** bằng early stopping, dropout và data augmentation cho mô hình lớn.

> Một mô hình giỏi không phải là mô hình nhớ giỏi nhất, mà là mô hình **quên đúng những thứ nên quên** — tức là quên đi nhiễu, giữ lại quy luật. Điều chuẩn chính là cách ta dạy mô hình **biết quên**.

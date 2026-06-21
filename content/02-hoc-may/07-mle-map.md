# Maximum Likelihood và Maximum A Posteriori (MLE & MAP)

> Khi có một tập dữ liệu $\mathcal{D}$ và một mô hình tham số $\theta$, câu hỏi cốt lõi là: **chọn $\theta$ nào?** Hai nguyên lý kinh điển trả lời câu hỏi này.
>
> $$\theta_{MLE} = \arg\max_\theta \; p(\mathcal{D} \mid \theta), \qquad \theta_{MAP} = \arg\max_\theta \; p(\theta \mid \mathcal{D})$$
>
> MLE chọn tham số làm dữ liệu **hợp lý nhất**; MAP thêm vào **niềm tin trước (prior)** về tham số. Bài này dẫn cả hai từ trực giác đến chứng minh, và cho thấy MLE chính là mặt sau của [cực tiểu cross-entropy/MSE](#/ham-mat-mat-toi-uu), còn MAP chính là mặt sau của **điều chuẩn (regularization)**.

---

# 1. Likelihood và ước lượng hợp lý cực đại

## 1.1. Likelihood là gì?

Giả sử dữ liệu $\mathcal{D} = \{x_1, \dots, x_n\}$ được sinh ra bởi một phân phối có tham số $\theta$. **Hàm hợp lý (likelihood function)** đo *mức độ một bộ tham số $\theta$ giải thích được dữ liệu quan sát*:

$$\mathcal{L}(\theta) = p(\mathcal{D} \mid \theta)$$

Điểm tinh tế: về mặt biểu thức, $p(\mathcal{D} \mid \theta)$ giống một phân phối xác suất, nhưng cách nhìn **ngược lại**. Khi xem là phân phối, ta cố định $\theta$ và để $\mathcal{D}$ thay đổi (tổng theo $\mathcal{D}$ bằng 1). Khi xem là **likelihood**, ta cố định $\mathcal{D}$ (đã quan sát rồi) và để $\theta$ thay đổi — likelihood **không** phải là phân phối theo $\theta$ và **không** tích phân về 1. Đây là lý do thuật ngữ riêng "hợp lý" tồn tại bên cạnh "xác suất". (Xem nền tảng ở bài [Xác suất](#/xac-suat).)

## 1.2. Nguyên lý cực đại hợp lý

**Ý tưởng.** Trong vô số $\theta$ có thể, hãy chọn $\theta$ làm cho dữ liệu ta thực sự quan sát trở nên **hợp lý nhất**:

$$\theta_{MLE} = \arg\max_\theta \; p(\mathcal{D} \mid \theta)$$

Đây là **ước lượng hợp lý cực đại (Maximum Likelihood Estimation, MLE)**. Trực giác rất tự nhiên: nếu một bộ tham số khiến dữ liệu quan sát gần như không bao giờ xảy ra, ta nên nghi ngờ bộ tham số đó.

## 1.3. Giả định i.i.d. và log-likelihood

Tích trực tiếp $p(\mathcal{D} \mid \theta)$ thường rất khó tối ưu. Hai mẹo chuẩn làm nó dễ hơn nhiều.

**Giả định i.i.d. (independent and identically distributed).** Nếu các điểm dữ liệu **độc lập và cùng phân phối**, xác suất đồng thời tách thành tích:

$$p(\mathcal{D} \mid \theta) = \prod_{i=1}^{n} p(x_i \mid \theta)$$

**Lấy logarit.** Tích của $n$ số trong $[0,1]$ tiến rất nhanh về 0 (tràn số dưới — underflow) và đạo hàm của tích thì cồng kềnh. Vì $\log$ là hàm **đồng biến ngặt**, $\arg\max$ không đổi khi ta lấy log. Ta định nghĩa **log-likelihood**:

$$\ell(\theta) = \log p(\mathcal{D} \mid \theta) = \sum_{i=1}^{n} \log p(x_i \mid \theta)$$

$$\boxed{\;\theta_{MLE} = \arg\max_\theta \; \sum_{i=1}^{n} \log p(x_i \mid \theta)\;}$$

Tích đã biến thành **tổng** — đạo hàm trở nên đơn giản và ổn định về số học. Đây là dạng ta sẽ tối ưu trong mọi ví dụ dưới đây.

---

# 2. Suy dẫn MLE cho phân phối Bernoulli

## 2.1. Đặt bài toán: tung đồng xu

Cho một đồng xu có xác suất ra mặt ngửa là $\theta \in [0,1]$. Tung $n$ lần, thu được kết quả $x_i \in \{0, 1\}$ (1 = ngửa). Phân phối **Bernoulli** của một lần tung:

$$p(x_i \mid \theta) = \theta^{x_i} (1 - \theta)^{1 - x_i}$$

(Kiểm tra: $x_i = 1 \Rightarrow \theta$; $x_i = 0 \Rightarrow 1 - \theta$.) Gọi $m = \sum_i x_i$ là số lần ngửa.

## 2.2. Lập log-likelihood

Theo giả định i.i.d. và lấy log:

$$\ell(\theta) = \sum_{i=1}^{n} \Big[ x_i \log \theta + (1 - x_i) \log(1 - \theta) \Big] = m \log \theta + (n - m) \log(1 - \theta)$$

## 2.3. Cực đại bằng đạo hàm = 0

Lấy đạo hàm theo $\theta$ và cho bằng 0:

$$\frac{d\ell}{d\theta} = \frac{m}{\theta} - \frac{n - m}{1 - \theta} = 0$$

Quy đồng:

$$m(1 - \theta) = (n - m)\theta \;\Longrightarrow\; m - m\theta = n\theta - m\theta \;\Longrightarrow\; m = n\theta$$

$$\boxed{\;\theta_{MLE} = \frac{m}{n} = \frac{\text{số lần ngửa}}{\text{tổng số lần tung}}\;}$$

Đây đúng là **tần suất quan sát (empirical frequency)** — kết quả khớp hoàn hảo với trực giác. Đạo hàm bậc hai $-\frac{m}{\theta^2} - \frac{n-m}{(1-\theta)^2} < 0$ xác nhận đây là cực **đại**. Nguyên lý này là nền tảng của [phân loại logistic](#/logistic-regression) và [Naive Bayes](#/naive-bayes), nơi các xác suất cũng được ước lượng bằng tần suất.

---

# 3. Suy dẫn MLE cho phân phối Gauss

## 3.1. Đặt bài toán

Cho dữ liệu thực $x_1, \dots, x_n$ giả định sinh từ phân phối **Gauss** $\mathcal{N}(\mu, \sigma^2)$:

$$p(x_i \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(x_i - \mu)^2}{2\sigma^2} \right)$$

Ta cần ước lượng đồng thời $\mu$ và $\sigma^2$.

## 3.2. Log-likelihood

$$\ell(\mu, \sigma^2) = \sum_{i=1}^{n} \left[ -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2} \right] = -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2$$

## 3.3. Ước lượng trung bình $\mu$

Đạo hàm riêng theo $\mu$ (chỉ số hạng cuối phụ thuộc $\mu$):

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^{n} (x_i - \mu) = 0 \;\Longrightarrow\; \sum_{i=1}^{n} x_i = n\mu$$

$$\boxed{\;\mu_{MLE} = \frac{1}{n} \sum_{i=1}^{n} x_i\;}$$

Đúng là **trung bình mẫu (sample mean)**.

## 3.4. Ước lượng phương sai $\sigma^2$

Đặt $v = \sigma^2$ rồi đạo hàm theo $v$:

$$\frac{\partial \ell}{\partial v} = -\frac{n}{2v} + \frac{1}{2v^2} \sum_{i=1}^{n} (x_i - \mu)^2 = 0$$

Nhân hai vế với $2v^2$:

$$-nv + \sum_{i=1}^{n} (x_i - \mu)^2 = 0$$

$$\boxed{\;\sigma^2_{MLE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_{MLE})^2\;}$$

Đúng là **phương sai mẫu (sample variance)**. Lưu ý ước lượng này chia cho $n$ (không phải $n-1$) nên hơi **chệch (biased)** — đây là một dấu hiệu sớm rằng MLE không phải lúc nào cũng hoàn hảo.

---

# 4. MLE chính là cực tiểu cross-entropy / MSE

Một quan sát đẹp: **cực đại log-likelihood tương đương cực tiểu một hàm mất mát quen thuộc** — cầu nối trực tiếp tới bài [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu).

## 4.1. Hồi quy Gauss ↔ MSE

Giả sử $y_i = f_\theta(x_i) + \varepsilon_i$ với nhiễu $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, tức $p(y_i \mid x_i, \theta) = \mathcal{N}(f_\theta(x_i), \sigma^2)$. Bỏ các hằng số không phụ thuộc $\theta$ trong log-likelihood ở mục 3.2:

$$\ell(\theta) = -\frac{1}{2\sigma^2} \sum_{i=1}^{n} \big(y_i - f_\theta(x_i)\big)^2 + \text{const}$$

Cực đại $\ell(\theta)$ chính là **cực tiểu tổng bình phương sai số**:

$$\theta_{MLE} = \arg\min_\theta \sum_{i=1}^{n} \big(y_i - f_\theta(x_i)\big)^2$$

Vậy **MSE không phải lựa chọn tùy tiện** — nó là MLE dưới giả định nhiễu Gauss. Đây chính là nền lý thuyết của [hồi quy tuyến tính](#/linear-regression).

## 4.2. Phân loại ↔ cross-entropy

Với nhãn nhị phân và mô hình cho $p(y_i = 1 \mid x_i) = \hat{y}_i$, likelihood của một điểm là Bernoulli $\hat{y}_i^{y_i}(1-\hat{y}_i)^{1-y_i}$. Âm log-likelihood:

$$-\ell(\theta) = -\sum_{i=1}^{n} \Big[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \Big]$$

— đây **đúng là cross-entropy**. Cực đại likelihood = cực tiểu cross-entropy. Đây là hàm mất mát của [phân loại logistic](#/logistic-regression).

| Giả định mô hình | MLE tương đương cực tiểu |
| --- | --- |
| Nhiễu Gauss (hồi quy) | Mean Squared Error (MSE) |
| Đầu ra Bernoulli (phân loại nhị phân) | Cross-Entropy (nhị phân) |
| Đầu ra Categorical (đa lớp) | Cross-Entropy (đa lớp) |

---

# 5. Hạn chế của MLE: overfit khi ít dữ liệu

MLE chỉ nhìn vào dữ liệu, **không có niềm tin nào trước đó**. Khi dữ liệu ít, điều này gây họa.

**Ví dụ kinh điển.** Tung đồng xu $n = 3$ lần, ra **cả 3 lần ngửa**. MLE cho:

$$\theta_{MLE} = \frac{3}{3} = 1$$

tức kết luận đồng xu **không bao giờ ra sấp** — một suy diễn cực đoan và gần như chắc chắn sai. Một xác suất 0 cứng nhắc còn gây thảm họa trong [Naive Bayes](#/naive-bayes): một từ chưa từng xuất hiện sẽ làm toàn bộ tích xác suất bằng 0.

Đây chính là biểu hiện của [quá khớp (overfitting)](#/overfitting): mô hình bám sát mẫu nhỏ ngẫu nhiên thay vì quy luật chung. Trực giác con người lại nói: ta *biết trước* đồng xu thường khá cân, nên kể cả 3 lần ngửa cũng không tin $\theta = 1$. Cách đưa **niềm tin trước** đó vào toán học chính là MAP.

---

# 6. Maximum A Posteriori (MAP)

## 6.1. Lật bài toán bằng định lý Bayes

Thay vì hỏi "dữ liệu hợp lý đến đâu với $\theta$?", MAP hỏi "**$\theta$ nào khả dĩ nhất sau khi thấy dữ liệu?**" — tức cực đại **hậu nghiệm (posterior)** $p(\theta \mid \mathcal{D})$. Dùng [định lý Bayes](#/xac-suat):

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{p(\mathcal{D})}$$

Mẫu số $p(\mathcal{D})$ **không phụ thuộc $\theta$**, nên không ảnh hưởng $\arg\max$. Do đó:

$$\theta_{MAP} = \arg\max_\theta \; p(\theta \mid \mathcal{D}) = \arg\max_\theta \; p(\mathcal{D} \mid \theta)\, p(\theta)$$

$$\boxed{\;\theta_{MAP} = \arg\max_\theta \; \Big[ \log p(\mathcal{D} \mid \theta) + \log p(\theta) \Big]\;}$$

So với MLE, MAP chỉ **thêm một số hạng** $\log p(\theta)$ — phần thưởng/phạt dựa trên niềm tin trước về tham số.

## 6.2. Vai trò của prior

$p(\theta)$ là **phân phối tiên nghiệm (prior)**, mã hóa hiểu biết của ta *trước khi* thấy dữ liệu. Trong ví dụ đồng xu, ta có thể đặt prior tập trung quanh $\theta = 0.5$ (tin đồng xu khá cân). Khi đó MAP sẽ "kéo" ước lượng về phía $0.5$, tránh kết luận cực đoan $\theta = 1$ — đúng như trực giác.

Prior hoạt động như một **lực điều chuẩn (regularization)**: nó hạn chế tham số khỏi chiều theo mọi ngẫu nhiên của mẫu nhỏ, giảm [quá khớp](#/overfitting).

---

# 7. MAP suy ra điều chuẩn L2 và L1

Đây là kết quả quan trọng nhất của bài: **chọn prior nào thì ra dạng điều chuẩn ấy**. Xét hồi quy với nhiễu Gauss (mục 4.1), âm log-likelihood là $\frac{1}{2\sigma^2}\sum_i (y_i - x_i^\top\theta)^2$.

## 7.1. Prior Gauss ↔ điều chuẩn L2 (Ridge)

Đặt prior Gauss tâm 0 cho từng tham số: $\theta_j \sim \mathcal{N}(0, \tau^2)$, tức

$$p(\theta) \propto \exp\!\left( -\frac{1}{2\tau^2} \sum_j \theta_j^2 \right) \;\Longrightarrow\; \log p(\theta) = -\frac{1}{2\tau^2} \sum_j \theta_j^2 + \text{const} = -\frac{1}{2\tau^2}\|\theta\|_2^2 + \text{const}$$

Mục tiêu MAP (lấy âm để chuyển thành cực tiểu):

$$\theta_{MAP} = \arg\min_\theta \; \underbrace{\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - x_i^\top\theta)^2}_{\text{từ likelihood}} + \underbrace{\frac{1}{2\tau^2}\|\theta\|_2^2}_{\text{từ prior}}$$

Nhân với $2\sigma^2$ và đặt $\lambda = \sigma^2/\tau^2$:

$$\boxed{\;\theta_{MAP} = \arg\min_\theta \; \sum_{i=1}^{n}(y_i - x_i^\top\theta)^2 + \lambda \|\theta\|_2^2\;}$$

Đây **chính xác là điều chuẩn L2 (Ridge regression)**. Hệ số $\lambda$ phản ánh độ chặt của prior: prior càng hẹp ($\tau$ nhỏ) → $\lambda$ lớn → phạt mạnh các trọng số lớn.

## 7.2. Prior Laplace ↔ điều chuẩn L1 (Lasso)

Đổi prior thành **Laplace** tâm 0: $p(\theta) \propto \exp\!\left( -\frac{1}{b}\sum_j |\theta_j| \right)$, nên

$$\log p(\theta) = -\frac{1}{b}\sum_j |\theta_j| + \text{const} = -\frac{1}{b}\|\theta\|_1 + \text{const}$$

Lặp lại lập luận trên, đặt $\lambda = 2\sigma^2/b$:

$$\boxed{\;\theta_{MAP} = \arg\min_\theta \; \sum_{i=1}^{n}(y_i - x_i^\top\theta)^2 + \lambda \|\theta\|_1\;}$$

Đây là **điều chuẩn L1 (Lasso)**, nổi tiếng vì đẩy nhiều trọng số về **đúng 0** (chọn lọc đặc trưng). Lý do hình học: chóp nhọn của phân phối Laplace tại 0 ưu tiên mạnh các nghiệm thưa. Cả hai dạng điều chuẩn là vũ khí chống [quá khớp](#/overfitting).

```python
import numpy as np

# MLE Bernoulli: tan suat quan sat
heads, n = 3, 3
theta_mle = heads / n            # = 1.0 -> overfit

# MAP voi prior Beta(a, b) (cong nghiem cho Bernoulli)
a, b = 2, 2                       # prior tin dong xu kha can, tam ~0.5
theta_map = (heads + a - 1) / (n + a + b - 2)   # = 0.6, hop ly hon
print(theta_mle, theta_map)
```

---

# 8. Khi nhiều dữ liệu, MAP hội tụ về MLE

Hãy nhìn lại mục tiêu MAP, viết theo tổng:

$$\theta_{MAP} = \arg\max_\theta \; \underbrace{\sum_{i=1}^{n} \log p(x_i \mid \theta)}_{\text{lớn dần theo } n} + \underbrace{\log p(\theta)}_{\text{cố định}}$$

Số hạng likelihood là **tổng của $n$ số hạng** nên độ lớn của nó tăng tuyến tính theo $n$. Trong khi đó $\log p(\theta)$ **không đổi** dù $n$ lớn đến đâu. Vì vậy:

$$\frac{\log p(\theta)}{\sum_{i=1}^{n} \log p(x_i \mid \theta)} \xrightarrow{\;n \to \infty\;} 0$$

Ảnh hưởng của prior **loãng dần** khi dữ liệu nhiều. Trong giới hạn, prior bị "nhấn chìm" (data swamps the prior) và $\theta_{MAP} \to \theta_{MLE}$.

Đây là một thông điệp thực tế quan trọng:

> **Ít dữ liệu** → prior cứu ta khỏi kết luận cực đoan (MAP hơn hẳn MLE).
> **Nhiều dữ liệu** → dữ liệu tự nói lên tất cả, prior trở nên không quan trọng (MAP $\approx$ MLE).

Nói cách khác, điều chuẩn (chính là prior trá hình) hữu ích nhất khi dữ liệu khan hiếm — đúng lúc nguy cơ [quá khớp](#/overfitting) cao nhất.

---

# 9. Ưu điểm và khi nào dùng

| Tiêu chí | MLE | MAP |
| --- | --- | --- |
| Mục tiêu | $\arg\max p(\mathcal{D}\mid\theta)$ | $\arg\max p(\mathcal{D}\mid\theta)\,p(\theta)$ |
| Dùng prior? | Không | Có |
| Tương đương | Cực tiểu cross-entropy/MSE | Cực tiểu loss **có điều chuẩn** |
| Khi ít dữ liệu | Dễ overfit, cực đoan | Ổn định nhờ prior |
| Khi nhiều dữ liệu | Tốt | Hội tụ về MLE |

* **Dùng MLE** khi dữ liệu dồi dào, hoặc khi không có niềm tin trước đáng tin cậy về tham số. Nó đơn giản, không cần chọn prior, và là chuẩn mặc định của hầu hết mô hình học sâu.
* **Dùng MAP** khi dữ liệu ít/nhiễu, hoặc khi ta có hiểu biết trước hợp lý. Vì MAP $\equiv$ điều chuẩn, mỗi lần thêm L2/L1 vào hàm mất mát là ta đã ngầm làm MAP.
* **Lưu ý chung:** cả hai chỉ trả về **một điểm ước lượng** (point estimate). Nếu cần định lượng độ bất định đầy đủ, phải dùng **suy diễn Bayes đầy đủ** (giữ toàn bộ phân phối hậu nghiệm thay vì lấy đỉnh của nó).

---

# 10. Tổng kết

MLE và MAP là hai trụ cột để **đi từ dữ liệu đến tham số**. Toàn bộ bài quy về một cặp công thức đối xứng đẹp:

$$\theta_{MLE} = \arg\max_\theta \sum_i \log p(x_i \mid \theta), \qquad \theta_{MAP} = \arg\max_\theta \Big[ \sum_i \log p(x_i \mid \theta) + \log p(\theta) \Big]$$

Ba ý cần nhớ:

1. **MLE = mặt sau của hàm mất mát.** Nhiễu Gauss cho ra MSE, đầu ra Bernoulli/Categorical cho ra cross-entropy. Khi tối ưu các loss đó (bài [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu)), ta đang làm MLE.
2. **MAP = MLE + prior = mặt sau của điều chuẩn.** Prior Gauss sinh ra L2 (Ridge), prior Laplace sinh ra L1 (Lasso) — công cụ trực tiếp chống [quá khớp](#/overfitting).
3. **Prior loãng dần theo dữ liệu.** Khi $n \to \infty$, $\theta_{MAP} \to \theta_{MLE}$; prior chỉ thực sự quan trọng khi dữ liệu khan hiếm.

> Bài tiếp theo sẽ áp dụng hai nguyên lý này một cách cụ thể trong [hồi quy tuyến tính](#/linear-regression) và [phân loại logistic](#/logistic-regression), nơi MLE/MAP biến thành các thuật toán huấn luyện thực sự.

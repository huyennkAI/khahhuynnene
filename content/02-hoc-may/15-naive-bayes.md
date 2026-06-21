# Bộ phân loại Naive Bayes (Naive Bayes Classifier)

> Naive Bayes là một bộ phân loại (classifier) **xác suất** dựa trên định lý Bayes, cộng thêm một giả định **"ngây thơ"** (naive): khi đã biết lớp, các đặc trưng (feature) là **độc lập có điều kiện** với nhau. Giả định này gần như luôn **sai** trong thực tế, nhưng nó biến một bài toán bất khả thi thành một phép nhân vài xác suất — và kỳ lạ thay, mô hình vẫn phân loại **rất tốt**, đặc biệt với văn bản (lọc spam, phân loại chủ đề).
>
> $$P(c \mid \mathbf{x}) \;\propto\; P(c)\prod_{i=1}^{d} P(x_i \mid c)$$

---

# 1. Động cơ: phân loại bằng định lý Bayes

Cho một điểm dữ liệu mô tả bởi vector đặc trưng $\mathbf{x} = (x_1, x_2, \dots, x_d)$, ta muốn gán nó vào một trong các lớp $c \in \{1, 2, \dots, C\}$. Cách tiếp cận xác suất tự nhiên là hỏi: **lớp nào có xác suất cao nhất khi đã quan sát được $\mathbf{x}$?** Tức là ta cần **xác suất hậu nghiệm (posterior)** $P(c \mid \mathbf{x})$.

Đại lượng này khó ước lượng trực tiếp, nhưng [định lý Bayes](xac-suat) cho ta một lối vòng:

$$P(c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c)\, P(c)}{P(\mathbf{x})}$$

Các thành phần có tên gọi và ý nghĩa rõ ràng:

* $P(c)$ — **xác suất tiên nghiệm (prior)** của lớp, tức tỉ lệ mỗi lớp trong dữ liệu, biết được **trước khi** nhìn $\mathbf{x}$.
* $P(\mathbf{x} \mid c)$ — **likelihood**, xác suất sinh ra đúng quan sát $\mathbf{x}$ nếu nó thuộc lớp $c$.
* $P(\mathbf{x})$ — **bằng chứng (evidence)**, xác suất quan sát $\mathbf{x}$ nói chung.

Vì ta chỉ cần **so sánh** các lớp với cùng một $\mathbf{x}$, mẫu số $P(\mathbf{x})$ là hằng số chung và **bỏ được**:

$$P(c \mid \mathbf{x}) \;\propto\; P(\mathbf{x} \mid c)\, P(c)$$

Dấu $\propto$ nghĩa là "tỉ lệ thuận với". Vậy bài toán quy về ước lượng prior $P(c)$ (dễ — đếm tần suất lớp) và likelihood $P(\mathbf{x} \mid c)$ (khó).

---

# 2. Vì sao likelihood là một bài toán bất khả thi

Hãy thử ước lượng $P(\mathbf{x} \mid c) = P(x_1, x_2, \dots, x_d \mid c)$ một cách trung thực, không giả định gì. Đây là một **phân phối đồng thời (joint distribution)** trên $d$ biến.

Giả sử mỗi đặc trưng $x_i$ chỉ nhận **2 giá trị** (có/không). Khi đó số tổ hợp $\mathbf{x}$ khả dĩ là $2^d$. Để ước lượng đầy đủ phân phối đồng thời, ta cần đếm tần suất của **từng tổ hợp** trong mỗi lớp — tức cần cỡ $2^d$ tham số cho mỗi lớp. Với $d = 50$ đặc trưng, đó là hơn $10^{15}$ con số: không có bộ dữ liệu nào đủ lớn, và hầu hết tổ hợp **chưa bao giờ xuất hiện** nên xác suất ước lượng bằng 0.

Đây chính là **lời nguyền số chiều (curse of dimensionality)** thể hiện trong xác suất. Ta cần một cách **phân rã** $P(\mathbf{x} \mid c)$ thành các mảnh nhỏ ước lượng được.

---

# 3. Giả định "ngây thơ": độc lập có điều kiện

Theo quy tắc dây chuyền (chain rule) của xác suất, ta luôn có thể viết chính xác:

$$P(x_1, \dots, x_d \mid c) = P(x_1 \mid c)\, P(x_2 \mid x_1, c)\, P(x_3 \mid x_1, x_2, c) \cdots$$

Mỗi số hạng càng về sau càng phụ thuộc nhiều biến — đó là nguồn gốc của $2^d$ tham số. **Giả định ngây thơ** của Naive Bayes cắt phăng mọi phụ thuộc chéo: khi đã biết lớp $c$, các đặc trưng là **độc lập có điều kiện (conditionally independent)** với nhau:

$$P(x_i \mid x_1, \dots, x_{i-1}, c) = P(x_i \mid c)$$

Áp vào quy tắc dây chuyền, tích "co lại" thành dạng cực kỳ gọn:

$$P(\mathbf{x} \mid c) = \prod_{i=1}^{d} P(x_i \mid c)$$

Thay vào công thức Bayes (mục 1), ta được **trái tim** của Naive Bayes:

$$\boxed{\;P(c \mid \mathbf{x}) \;\propto\; P(c)\prod_{i=1}^{d} P(x_i \mid c)\;}$$

**Vì sao giả định này làm bài toán khả thi?** Thay vì $2^d$ tham số cho mỗi lớp, giờ ta chỉ cần ước lượng $d$ phân phối **một chiều** $P(x_i \mid c)$ — mỗi cái chỉ cần đếm tần suất một đặc trưng trong một lớp. Số tham số giảm từ **lũy thừa** xuống **tuyến tính** theo $d$. Mỗi $P(x_i \mid c)$ lại có rất nhiều mẫu để ước lượng, nên ước lượng ổn định ngay cả khi dữ liệu ít.

> **Lưu ý về "độc lập có điều kiện" ≠ "độc lập".** Naive Bayes **không** nói các đặc trưng độc lập tuyệt đối. Nó chỉ nói chúng độc lập **một khi đã cố định lớp**. Ví dụ trong email, từ "trúng" và từ "thưởng" tương quan mạnh nói chung; nhưng nếu ta đã biết email là spam, việc xuất hiện "trúng" được coi là không cho thêm thông tin về "thưởng" nữa.

---

## 3.1. Vì sao thường vẫn hiệu quả dù giả định sai?

Giả định độc lập gần như luôn sai (các từ trong câu rõ ràng phụ thuộc nhau). Nhưng Naive Bayes vẫn phân loại tốt vì lý do tinh tế: **ta không cần ước lượng xác suất đúng, ta chỉ cần xếp đúng thứ tự các lớp.**

Quyết định phân loại chỉ phụ thuộc lớp nào có posterior **lớn nhất**, không phụ thuộc giá trị posterior chính xác. Khi các đặc trưng phụ thuộc nhau, Naive Bayes thường **đếm trùng** bằng chứng và đẩy posterior của lớp thắng ra rất gần $0$ hoặc $1$ (ước lượng xác suất kém — xem mục 9). Nhưng miễn là **đúng lớp vẫn được đẩy lên cao nhất**, nhãn dự đoán vẫn đúng. Nói cách khác, ranh giới quyết định (decision boundary) khá bền vững với sai lệch của giả định độc lập.

---

# 4. Quy tắc quyết định: MAP

Để ra nhãn cuối cùng, ta chọn lớp có posterior cao nhất — chính là quy tắc **cực đại hậu nghiệm (Maximum A Posteriori, MAP)**, đã gặp ở bài [Ước lượng MLE & MAP](mle-map):

$$\hat{c} = \arg\max_{c \in \{1,\dots,C\}} \; P(c \mid \mathbf{x}) = \arg\max_{c} \; P(c)\prod_{i=1}^{d} P(x_i \mid c)$$

Liên hệ với MLE: nếu ta **bỏ** thừa số prior $P(c)$ (coi mọi lớp ngang nhau tiên nghiệm), quy tắc rút về cực đại likelihood $\arg\max_c \prod_i P(x_i \mid c)$. Prior chính là phần "MAP" nhiều hơn "MLE": nó kéo dự đoán nghiêng về lớp phổ biến khi bằng chứng yếu.

---

# 5. Các biến thể: chọn mô hình cho $P(x_i \mid c)$

Công thức Naive Bayes chưa nói $P(x_i \mid c)$ có **dạng** gì — đó là nơi các biến thể khác nhau. Việc chọn dạng phụ thuộc bản chất đặc trưng (xem thêm [Feature Engineering](feature-engineering)).

## 5.1. Gaussian Naive Bayes (đặc trưng liên tục)

Khi đặc trưng là **số thực liên tục** (chiều cao, cân nặng, cường độ điểm ảnh), ta mô hình hóa mỗi $P(x_i \mid c)$ bằng một phân phối **Gauss** một chiều:

$$P(x_i \mid c) = \frac{1}{\sqrt{2\pi \sigma_{c,i}^2}} \exp\!\left( -\frac{(x_i - \mu_{c,i})^2}{2\sigma_{c,i}^2} \right)$$

Huấn luyện chỉ là ước lượng MLE: với mỗi lớp $c$ và mỗi đặc trưng $i$, lấy **trung bình** $\mu_{c,i}$ và **phương sai** $\sigma_{c,i}^2$ của các mẫu thuộc lớp đó.

## 5.2. Multinomial Naive Bayes (đếm — văn bản)

Đây là biến thể kinh điển cho **phân loại văn bản** với biểu diễn **túi từ (bag of words)**: mỗi văn bản là một vector đếm $x_i$ = số lần từ thứ $i$ trong từ điển xuất hiện. Mô hình coi văn bản như rút lặp $n$ lần từ một túi từ của lớp:

$$P(x_i \mid c) = \theta_{c,i}, \qquad \sum_{i=1}^{V} \theta_{c,i} = 1$$

trong đó $\theta_{c,i}$ là xác suất một từ bất kỳ trong văn bản lớp $c$ rơi đúng vào từ $i$, ước lượng bằng tần suất:

$$\theta_{c,i} = \frac{N_{c,i}}{\sum_{j=1}^{V} N_{c,j}}$$

với $N_{c,i}$ là tổng số lần từ $i$ xuất hiện trong toàn bộ văn bản lớp $c$, và $V$ là kích thước từ điển. Multinomial NB quan tâm **số lần** một từ xuất hiện.

## 5.3. Bernoulli Naive Bayes (có/không — văn bản)

Bernoulli NB cũng dành cho văn bản nhưng dùng đặc trưng **nhị phân**: $x_i \in \{0, 1\}$ chỉ cho biết từ $i$ **có mặt hay không**, bất kể bao nhiêu lần. Mỗi đặc trưng là một phép thử Bernoulli:

$$P(x_i \mid c) = p_{c,i}^{\,x_i}\,(1 - p_{c,i})^{\,1 - x_i}$$

với $p_{c,i}$ là xác suất từ $i$ xuất hiện trong một văn bản lớp $c$. Khác biệt then chốt với Multinomial: Bernoulli **phạt rõ ràng sự vắng mặt** của từ (số hạng $1 - p_{c,i}$ khi $x_i = 0$), nên thường tốt cho văn bản **ngắn**; Multinomial tốt hơn cho văn bản dài, giàu tần suất.

---

# 6. Làm trơn Laplace: tránh xác suất 0

Có một cái bẫy chết người trong ước lượng tần suất. Giả sử trong lúc huấn luyện, từ "khuyến mãi" **chưa bao giờ** xuất hiện trong email **ham** (không spam). Khi đó $P(\text{"khuyến mãi"} \mid \text{ham}) = 0$. Vì posterior là **tích**, một thừa số $0$ sẽ kéo **toàn bộ** $P(\text{ham} \mid \mathbf{x})$ về $0$ — dù mọi từ khác đều hét lên rằng email này là ham. Một từ chưa từng thấy có quyền phủ quyết tất cả: điều này vô lý.

**Làm trơn Laplace (Laplace / additive smoothing)** sửa lỗi bằng cách giả vờ ta đã thấy mỗi khả năng thêm $\alpha$ lần (thường $\alpha = 1$). Với Multinomial NB:

$$\theta_{c,i} = \frac{N_{c,i} + \alpha}{\sum_{j=1}^{V} N_{c,j} + \alpha V}$$

Mẫu số cộng $\alpha V$ để tổng xác suất vẫn bằng $1$. Hệ quả: **không xác suất nào còn bằng 0** nữa, mọi từ chỉ "đẩy" chứ không "phủ quyết". Nhìn từ [MAP](mle-map), đây đúng là ước lượng MAP với **prior Dirichlet** đối xứng — $\alpha$ đóng vai trò "số đếm ảo" thể hiện niềm tin tiên nghiệm rằng mọi từ đều có thể xảy ra.

---

# 7. Dùng log để tránh tràn số (underflow)

Posterior là tích của $d$ xác suất, mỗi cái nhỏ hơn $1$. Với văn bản dài ($d$ hàng nghìn từ), tích này nhỏ đến mức số thực dấu phẩy động làm tròn thành $0$ — gọi là **tràn số dưới (underflow)**. Lúc đó mọi lớp đều cho $0$ và không so sánh được.

Mẹo chuẩn: lấy **logarit**. Vì $\log$ là hàm **đồng biến ngặt**, nó **không đổi** thứ tự các lớp, nên $\arg\max$ giữ nguyên. Tích biến thành tổng — vừa ổn định số học, vừa nhanh hơn:

$$\hat{c} = \arg\max_{c}\; \left[ \log P(c) + \sum_{i=1}^{d} \log P(x_i \mid c) \right]$$

Cộng các số âm vừa phải thì an toàn, thay vì nhân các số tí hon. Đây là lý do mọi cài đặt Naive Bayes thực tế đều làm việc trong **không gian log**.

---

# 8. Ví dụ: lọc spam

Ghép tất cả lại trong bài toán kinh điển — phân loại email thành **spam** hay **ham**, dùng Multinomial NB trên bag of words.

1. **Tiền xử lý.** Tách email thành các từ (tokenize), bỏ từ dừng (stop words), dựng từ điển kích thước $V$ — đây là phần [feature engineering](feature-engineering).
2. **Học prior.** $P(\text{spam}) = $ tỉ lệ email spam trong tập huấn luyện, tương tự $P(\text{ham})$.
3. **Học likelihood.** Với mỗi từ $i$, tính $\theta_{\text{spam},i}$ và $\theta_{\text{ham},i}$ bằng tần suất **có làm trơn Laplace** (mục 6).
4. **Dự đoán.** Với email mới, tính điểm log cho cả hai lớp rồi chọn lớp lớn hơn:

$$\text{score}(c) = \log P(c) + \sum_{i \in \text{email}} x_i \log \theta_{c,i}$$

Email có nhiều từ như "miễn phí", "trúng thưởng", "nhấp vào đây" sẽ đẩy $\text{score}(\text{spam})$ lên cao và bị gắn cờ.

```python
import numpy as np

def train_nb(X, y, alpha=1.0):
    classes = np.unique(y)
    log_prior, log_likelihood = {}, {}
    for c in classes:
        Xc = X[y == c]                      # các văn bản thuộc lớp c
        log_prior[c] = np.log(len(Xc) / len(X))
        counts = Xc.sum(axis=0) + alpha     # làm trơn Laplace
        log_likelihood[c] = np.log(counts / counts.sum())
    return log_prior, log_likelihood

def predict(x, log_prior, log_likelihood, classes):
    scores = {c: log_prior[c] + (x * log_likelihood[c]).sum() for c in classes}
    return max(scores, key=scores.get)      # quy tắc MAP trong không gian log
```

---

# 9. Ưu điểm

* **Nhanh và rẻ** — huấn luyện chỉ là **đếm tần suất** trong một lượt duyệt dữ liệu (không lặp tối ưu), dự đoán là vài phép cộng log. Quy mô lên hàng triệu văn bản dễ dàng.
* **Cần ít dữ liệu** — vì chỉ ước lượng $d$ phân phối một chiều thay vì phân phối đồng thời $2^d$, mô hình ổn định ngay cả với tập huấn luyện nhỏ.
* **Rất tốt cho văn bản** — với số chiều cực lớn (từ điển hàng chục nghìn từ), Naive Bayes vẫn vững nhờ giả định độc lập làm gọn bài toán; nó là **baseline mạnh** cho phân loại spam, chủ đề, cảm xúc.
* **Dễ diễn giải** — có thể đọc trực tiếp từ nào "thiên về" lớp nào qua tỉ số $\log \theta_{\text{spam},i} - \log \theta_{\text{ham},i}$.

---

# 10. Nhược điểm

* **Giả định độc lập gần như luôn sai** — khi các đặc trưng tương quan mạnh (ví dụ cụm từ, từ đồng nghĩa), Naive Bayes **đếm trùng** bằng chứng, làm giảm độ chính xác so với mô hình nắm được tương tác (logistic regression, cây tăng cường).
* **Ước lượng xác suất kém hiệu chỉnh (poorly calibrated)** — như đã nói ở mục 3.1, posterior thường bị đẩy về sát $0$ hoặc $1$. Nhãn dự đoán có thể đúng, nhưng **con số xác suất** không đáng tin để dùng làm ngưỡng hay xếp hạng độ tin cậy.
* **Nhạy với đặc trưng dư thừa** — đặc trưng tương quan bị tính nhiều lần làm lệch kết quả; cần [feature engineering](feature-engineering)/chọn lọc đặc trưng để giảm trùng lặp.
* **Cần làm trơn và chọn biến thể đúng** — quên làm trơn Laplace gây lỗi xác suất 0; chọn nhầm Multinomial/Bernoulli/Gaussian cho sai loại đặc trưng làm hỏng giả định phân phối.

---

# 11. Tổng kết

Naive Bayes là minh chứng đẹp rằng một **giả định sai nhưng hữu ích** có thể đánh bại sự cầu toàn. Bằng cách giả định độc lập có điều kiện, nó biến phân phối đồng thời $2^d$ chiều bất khả thi thành một **tích** các xác suất một chiều ước lượng dễ dàng:

$$P(c \mid \mathbf{x}) \;\propto\; P(c)\prod_{i=1}^{d} P(x_i \mid c)$$

Quy tắc quyết định là [MAP](mle-map) — chọn lớp posterior cao nhất; trong thực tế ta làm việc trong **không gian log** để tránh tràn số và dùng **làm trơn Laplace** để tránh xác suất 0. Tùy loại đặc trưng mà chọn Gaussian (liên tục), Multinomial (đếm) hay Bernoulli (có/không) cho $P(x_i \mid c)$.

Dù giả định độc lập sai, Naive Bayes vẫn **xếp đúng thứ tự lớp** trong rất nhiều bài toán — đặc biệt văn bản nhiều chiều — nên đến nay vẫn là một **baseline nhanh, nhẹ, đáng kính nể** mà mọi bài toán phân loại nên thử trước tiên.

> Bài tiếp theo sẽ chuyển từ mô hình **sinh (generative)** — học $P(\mathbf{x} \mid c)$ rồi đảo qua Bayes — sang mô hình **phân biệt (discriminative)** học thẳng $P(c \mid \mathbf{x})$ như **hồi quy logistic (logistic regression)**, vốn không cần giả định độc lập và cho xác suất hiệu chỉnh tốt hơn.

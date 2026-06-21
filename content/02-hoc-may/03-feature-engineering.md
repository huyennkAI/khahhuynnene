# Kỹ thuật đặc trưng (Feature Engineering)

> Mô hình học máy không "nhìn" thế giới như con người — nó chỉ nhìn thấy những **con số** mà ta đưa vào. Kỹ thuật đặc trưng (feature engineering) là nghệ thuật **biến dữ liệu thô thành biểu diễn số** mà mô hình học được tốt nhất. Một nguyên lý xuyên suốt: **rác vào, rác ra (garbage in, garbage out)** — biểu diễn dữ liệu tồi thì không thuật toán nào cứu nổi, còn đặc trưng tốt có thể khiến cả mô hình tuyến tính đơn giản vượt mặt mạng nơ-ron sâu.

---

# 1. Vì sao biểu diễn dữ liệu quan trọng

Một thuật toán học máy, ở mức trừu tượng nhất, là một ánh xạ từ **vector đặc trưng (feature vector)** $\mathbf{x} \in \mathbb{R}^d$ sang dự đoán $\hat{y}$. Mọi tri thức mà mô hình có được đều chảy qua $\mathbf{x}$. Nếu thông tin cần thiết **không** hiện diện trong $\mathbf{x}$, hoặc bị mã hóa theo cách mô hình khó khai thác, thì việc học là vô vọng.

Hãy lấy một ví dụ kinh điển: dự đoán giá nhà. Cho trước **ngày xây dựng** (định dạng `2010-03-15`), một mô hình tuyến tính không thể trực tiếp dùng chuỗi này. Nhưng nếu ta trích ra đặc trưng **"tuổi căn nhà"** $= 2026 - 2010 = 16$, quan hệ giữa tuổi và giá trở nên gần như tuyến tính và mô hình học ngay được. Cùng một dữ liệu thô, hai biểu diễn khác nhau, kết quả một trời một vực.

## 1.1. Garbage in, garbage out

Nguyên lý này nói rằng chất lượng đầu ra bị chặn trên bởi chất lượng đầu vào. Cụ thể với học máy:

* Nếu đặc trưng **nhiễu** hoặc **không liên quan**, mô hình sẽ học theo nhiễu (dẫn tới [quá khớp - overfitting](#/overfitting)).
* Nếu đặc trưng **thiếu** thông tin then chốt, không có lượng dữ liệu nào bù lại được.
* Nếu đặc trưng bị **rò rỉ** thông tin tương lai (xem mục 8), mô hình "ăn gian" lúc huấn luyện và sụp đổ khi triển khai.

> **Trực giác.** Mô hình giống một học sinh thông minh nhưng chỉ học đúng những gì sách giáo khoa viết. Feature engineering chính là việc viết lại sách giáo khoa sao cho dễ hiểu nhất.

## 1.2. Quy trình tổng quát

Một chu trình feature engineering điển hình gồm bốn nhóm thao tác, mỗi nhóm sẽ được khai triển ở các mục sau:

| Nhóm | Mục tiêu | Ví dụ kỹ thuật |
| --- | --- | --- |
| Mã hóa hạng mục | Biến nhãn chữ thành số | one-hot, label, hashing |
| Biến đổi số | Đưa về thang đo/dạng phân phối tốt | z-score, min-max, log, binning |
| Xử lý thiếu | Lấp giá trị khuyết | mean/median, indicator |
| Trích từ phi cấu trúc | Số hóa văn bản, ảnh | bag-of-words, TF-IDF, bag-of-features |

---

# 2. Đặc trưng cho dữ liệu hạng mục (categorical)

Dữ liệu hạng mục là dữ liệu nhận giá trị trong một tập hữu hạn các **hạng mục (category)** không có thứ tự số học tự nhiên, ví dụ màu sắc $\{\text{đỏ}, \text{xanh}, \text{vàng}\}$ hay thành phố. Mô hình cần số, nên ta phải mã hóa.

## 2.1. One-hot encoding

One-hot encoding biến một biến hạng mục có $K$ giá trị thành một vector nhị phân $K$ chiều, trong đó đúng **một** thành phần bằng $1$, còn lại bằng $0$. Với màu $\{\text{đỏ}, \text{xanh}, \text{vàng}\}$:

$$\text{đỏ} \mapsto (1, 0, 0), \quad \text{xanh} \mapsto (0, 1, 0), \quad \text{vàng} \mapsto (0, 0, 1)$$

Ưu điểm: **không áp đặt thứ tự giả** giữa các hạng mục — khoảng cách Euclid giữa hai hạng mục bất kỳ đều bằng $\sqrt{2}$, phản ánh đúng việc chúng "khác nhau như nhau". Đây là lựa chọn mặc định cho mô hình tuyến tính như [hồi quy logistic](#/logistic-regression).

Nhược điểm: khi $K$ rất lớn (ví dụ mã bưu chính có hàng nghìn giá trị), số chiều bùng nổ, gây tốn bộ nhớ và làm ma trận đặc trưng **thưa (sparse)**.

## 2.2. Label encoding

Label encoding gán cho mỗi hạng mục một số nguyên: $\text{đỏ} \mapsto 0,\ \text{xanh} \mapsto 1,\ \text{vàng} \mapsto 2$. Gọn (chỉ một chiều), nhưng **nguy hiểm với mô hình tuyến tính** vì nó áp đặt một thứ tự và khoảng cách giả: mô hình sẽ "tin" rằng $\text{vàng} - \text{đỏ} = 2$ và $\text{xanh}$ nằm chính giữa, điều hoàn toàn vô nghĩa.

Label encoding chỉ thực sự phù hợp khi:

* Biến **có thứ tự thật** (ordinal), ví dụ $\{\text{nhỏ}, \text{vừa}, \text{lớn}\} \mapsto \{0, 1, 2\}$.
* Hoặc mô hình là **cây quyết định (decision tree)** / rừng ngẫu nhiên, vốn chia theo ngưỡng nên không bị giả định tuyến tính làm hại.

## 2.3. Feature hashing (hashing trick)

Khi số hạng mục cực lớn hoặc chưa biết trước (ví dụ ID người dùng trong hệ thống streaming), one-hot bất khả thi. **Hashing trick** dùng một hàm băm $h$ để ánh xạ mỗi hạng mục $c$ vào một trong $m$ ô cố định ($m \ll K$):

$$\phi(c) = \mathbf{e}_{\,h(c) \bmod m}$$

trong đó $\mathbf{e}_j$ là vector đơn vị thứ $j$. Kích thước đặc trưng được **khống chế cứng** ở $m$ bất kể bao nhiêu hạng mục. Cái giá là **đụng độ (collision)**: hai hạng mục khác nhau có thể rơi vào cùng ô, gây mất mát thông tin nhỏ — thường chấp nhận được nếu $m$ đủ lớn.

---

# 3. Đặc trưng số: chuẩn hóa thang đo

Các đặc trưng số thường có **đơn vị và độ lớn rất khác nhau**: tuổi (0–100) và thu nhập (0–$10^9$). Nếu để nguyên, đặc trưng có thang lớn sẽ **lấn át** trong các thuật toán dựa trên khoảng cách hoặc gradient. Chuẩn hóa (feature scaling) đưa mọi đặc trưng về thang đo so sánh được.

## 3.1. Chuẩn hóa z-score (standardization)

Standardization trừ trung bình rồi chia độ lệch chuẩn, biến mỗi đặc trưng thành phân phối có trung bình $0$ và phương sai $1$:

$$x' = \frac{x - \mu}{\sigma}, \qquad \mu = \frac{1}{n}\sum_{i=1}^n x_i, \quad \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2}$$

Đây là lựa chọn phổ biến nhất, đặc biệt khi dữ liệu gần Gauss. Nó **không bị giới hạn miền** (giá trị có thể âm hoặc lớn) nhưng giữ được hình dạng phân phối.

## 3.2. Chuẩn hóa min-max

Min-max co dữ liệu vào khoảng $[0, 1]$ (hoặc $[a, b]$ tổng quát):

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Phù hợp khi cần miền giá trị cố định (ví dụ đầu vào ảnh về $[0,1]$, hoặc tầng kích hoạt cần khoảng bị chặn). Nhược điểm: **rất nhạy với điểm ngoại lai (outlier)** — một giá trị cực lớn sẽ nén toàn bộ phần còn lại về gần $0$.

| Phương pháp | Công thức | Miền sau biến đổi | Nhạy outlier |
| --- | --- | --- | --- |
| Z-score | $(x-\mu)/\sigma$ | $\mathbb{R}$, mean $0$ var $1$ | Trung bình |
| Min-max | $(x-x_{\min})/(x_{\max}-x_{\min})$ | $[0, 1]$ | **Cao** |
| Robust scaling | $(x - \text{median})/\text{IQR}$ | quanh $0$ | **Thấp** |

---

# 4. Đặc trưng số: biến đổi phân phối

Chuẩn hóa thay đổi thang đo nhưng **không** thay đổi hình dạng phân phối. Nhiều biến thực tế bị **lệch (skewed)** nặng — như thu nhập, lượt xem — khiến mô hình tuyến tính khó học. Ta cần biến đổi phi tuyến để "nắn" lại hình dạng.

## 4.1. Biến đổi log

Phép biến đổi log nén đuôi phải dài của phân phối lệch dương, đưa nó gần đối xứng hơn:

$$x' = \log(x), \qquad \text{hoặc} \qquad x' = \log(1 + x)$$

Dạng $\log(1+x)$ (gọi là `log1p`) xử lý được $x = 0$. Sau biến đổi log, quan hệ **nhân tính** (ví dụ "tăng 10%") trở thành **cộng tính** trong không gian mới, rất hợp với giả định tuyến tính của [hồi quy tuyến tính](#/linear-regression).

## 4.2. Biến đổi Box-Cox

Box-Cox là một họ biến đổi tham số hóa bởi $\lambda$, tổng quát hóa cả log và lũy thừa, áp dụng cho dữ liệu **dương**:

$$
x^{(\lambda)} =
\begin{cases}
\dfrac{x^{\lambda} - 1}{\lambda}, & \lambda \neq 0 \\[2mm]
\log(x), & \lambda = 0
\end{cases}
$$

Tham số $\lambda$ được chọn (bằng hợp lý cực đại) sao cho dữ liệu sau biến đổi **gần Gauss nhất**. Khi $\lambda = 0$ ta thu lại biến đổi log; khi $\lambda = 1$ về cơ bản là tuyến tính. Mở rộng **Yeo-Johnson** cho phép cả giá trị âm.

## 4.3. Rời rạc hóa (binning)

Binning chia một biến liên tục thành các **khoảng (bin)** rồi coi mỗi khoảng như một hạng mục. Ví dụ tuổi $\mapsto \{\text{trẻ em}, \text{thanh niên}, \text{trung niên}, \text{cao tuổi}\}$. Hai chiến lược thường gặp:

* **Equal-width**: chia trục giá trị thành các đoạn bằng nhau về độ rộng.
* **Equal-frequency (quantile)**: mỗi bin chứa cùng số lượng mẫu.

Binning giúp mô hình tuyến tính nắm bắt **quan hệ phi tuyến** (sau khi one-hot các bin, mỗi khoảng có hệ số riêng) và giảm ảnh hưởng outlier, nhưng đánh đổi bằng việc **mất thông tin chi tiết** trong từng khoảng.

---

# 5. Xử lý giá trị thiếu (missing values)

Dữ liệu thực tế hầu như luôn có ô khuyết. Hầu hết thuật toán không chấp nhận `NaN`, nên ta phải xử lý. Lựa chọn phụ thuộc vào **cơ chế** gây thiếu (thiếu ngẫu nhiên hay thiếu có hệ thống).

## 5.1. Các chiến lược lấp giá trị (imputation)

* **Xóa (deletion)**: bỏ hàng hoặc cột có giá trị thiếu. Đơn giản nhưng phí dữ liệu, và **lệch (bias)** nếu thiếu không ngẫu nhiên.
* **Lấp bằng thống kê**: thay bằng **trung bình (mean)** / **trung vị (median)** cho số, **mode** cho hạng mục. Median bền với outlier hơn mean.
* **Lấp bằng mô hình**: dùng kNN hoặc hồi quy để dự đoán giá trị thiếu từ các đặc trưng khác — chính xác hơn nhưng tốn kém.

## 5.2. Biến chỉ báo thiếu (missing indicator)

Bản thân việc **một giá trị bị thiếu** đôi khi đã mang thông tin (ví dụ khách không điền thu nhập có thể tương quan với hành vi). Vì vậy nên thêm một đặc trưng nhị phân:

$$m_i = \begin{cases} 1 & \text{nếu } x_i \text{ thiếu} \\ 0 & \text{ngược lại} \end{cases}$$

bên cạnh việc lấp giá trị. Mô hình tự quyết định xem cờ thiếu có hữu ích không.

---

# 6. Đặc trưng cho văn bản

Văn bản là dữ liệu phi cấu trúc; ta cần số hóa nó thành vector. Cách kinh điển là coi mỗi tài liệu như một **túi từ (bag-of-words)**, bỏ qua thứ tự từ.

## 6.1. Bag-of-words (BoW)

Xây một **từ vựng (vocabulary)** gồm $V$ từ. Mỗi tài liệu $d$ được biểu diễn bằng vector $V$ chiều, thành phần thứ $t$ là **số lần xuất hiện** của từ $t$ trong $d$, ký hiệu $\text{tf}(t, d)$ (term frequency). Đơn giản, hiệu quả cho [phân loại Naive Bayes](#/naive-bayes), nhưng có hai điểm yếu: vector cực thưa, và các từ phổ biến vô nghĩa (như "là", "và", "the") áp đảo tần suất.

## 6.2. TF-IDF

TF-IDF (Term Frequency – Inverse Document Frequency) khắc phục điểm yếu thứ hai bằng cách **hạ trọng số** những từ xuất hiện trong quá nhiều tài liệu (ít phân biệt) và **nâng trọng số** từ hiếm, đặc trưng. Trọng số gồm hai thừa số.

**Term frequency** — mức độ quan trọng của từ trong một tài liệu (một dạng phổ biến là chuẩn hóa theo độ dài):

$$\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}$$

trong đó $f_{t,d}$ là số lần từ $t$ xuất hiện trong $d$.

**Inverse document frequency** — mức độ hiếm của từ trên toàn tập $N$ tài liệu:

$$\text{idf}(t) = \log \frac{N}{1 + |\{d : t \in d\}|}$$

mẫu số là số tài liệu chứa $t$ (cộng $1$ để tránh chia cho $0$). Từ có mặt khắp nơi thì $\text{idf} \to 0$. Trọng số cuối là tích:

$$\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \text{idf}(t)$$

Một từ được điểm cao khi nó **xuất hiện nhiều trong tài liệu này** nhưng **hiếm trong toàn tập** — đúng tiêu chí của một từ khóa đặc trưng.

```python
import numpy as np
def tfidf(corpus_counts):          # corpus_counts: list[dict] đếm từ mỗi tài liệu
    N = len(corpus_counts)
    df = {}
    for doc in corpus_counts:
        for t in doc:
            df[t] = df.get(t, 0) + 1
    out = []
    for doc in corpus_counts:
        total = sum(doc.values())
        out.append({t: (c/total) * np.log(N/(1+df[t]))
                    for t, c in doc.items()})
    return out
```

---

# 7. Đặc trưng cho ảnh và tương tác

## 7.1. Bag-of-features cho ảnh

Ý tưởng bag-of-words chuyển được sang ảnh thành **bag-of-features** (hay bag-of-visual-words). Quy trình cổ điển (tiền học sâu):

1. Trích các **điểm đặc trưng cục bộ** (local descriptor) như SIFT từ nhiều ảnh.
2. **Phân cụm (clustering)** toàn bộ descriptor bằng k-means để tạo một "từ điển thị giác (visual vocabulary)" gồm $k$ cụm.
3. Mỗi ảnh được biểu diễn bằng **histogram** đếm số descriptor rơi vào từng cụm — y hệt vector tần suất từ của văn bản.

Cách này biến ảnh kích thước tùy ý thành vector cố định $k$ chiều, dùng được cho mọi bộ phân loại chuẩn.

## 7.2. Tương tác đặc trưng và đặc trưng đa thức

Mô hình tuyến tính chỉ học quan hệ cộng tính. Để nắm **tương tác (interaction)** giữa các biến, ta tạo đặc trưng mới bằng tích hoặc lũy thừa. Với hai đặc trưng $x_1, x_2$, biểu diễn **đa thức bậc 2 (polynomial features)** là:

$$\phi(x_1, x_2) = (x_1,\ x_2,\ x_1^2,\ x_2^2,\ x_1 x_2)$$

Số hạng tích $x_1 x_2$ cho phép mô hình tuyến tính học quan hệ kiểu "ảnh hưởng của $x_1$ phụ thuộc vào $x_2$". Đây chính là cách hồi quy tuyến tính học được đường cong: tuyến tính trong **không gian đặc trưng mở rộng**, phi tuyến trong không gian gốc. Cảnh báo: số đặc trưng tăng nhanh theo bậc, dễ gây [quá khớp](#/overfitting), nên cần kết hợp với điều chuẩn (regularization).

---

# 8. Chuẩn hóa và rò rỉ dữ liệu

## 8.1. Chuẩn hóa giúp gradient descent hội tụ nhanh hơn

Đây là một lý do **toán học** quan trọng để chuẩn hóa, liên hệ trực tiếp với [hàm mất mát và tối ưu](#/ham-mat-mat-toi-uu). Khi các đặc trưng có thang đo lệch nhau, mặt mất mát (loss surface) trở thành một "thung lũng" **rất dẹt và dài** — các đường đồng mức là ellipse cực kỳ thuôn.

Gradient descent cập nhật theo hướng dốc nhất:

$$\theta \leftarrow \theta - \eta\, \nabla_\theta L(\theta)$$

Trên mặt ellipse thuôn, gradient gần như **vuông góc** với hướng đi tới cực tiểu, khiến quỹ đạo **dao động zíc-zắc** và hội tụ chậm. Sau khi chuẩn hóa, mọi đặc trưng cùng thang, các đường đồng mức trở nên **tròn hơn**, gradient chỉ thẳng về tâm, cho phép dùng learning rate $\eta$ lớn hơn và hội tụ nhanh hơn nhiều. Đây cũng là lý do các thuật toán dựa trên khoảng cách như [kNN](#/knn) và [PCA](#/pca) **bắt buộc** phải chuẩn hóa trước.

> **Trực giác.** Hình dung lăn một viên bi xuống thung lũng. Thung lũng tròn (đã chuẩn hóa) thì bi lăn thẳng xuống đáy; thung lũng méo (chưa chuẩn hóa) thì bi nảy qua nảy lại hai vách rất lâu mới tới đáy.

## 8.2. Rò rỉ dữ liệu khi chuẩn hóa sai

Rò rỉ dữ liệu (data leakage) xảy ra khi thông tin từ tập kiểm tra (test) "lọt" vào quá trình huấn luyện, khiến đánh giá lạc quan giả tạo. Một lỗi cực phổ biến là **tính tham số chuẩn hóa trên toàn bộ dữ liệu** trước khi chia tập:

* **SAI**: tính $\mu, \sigma$ trên toàn bộ dữ liệu (gồm cả test) rồi mới chia train/test. Khi đó $\mu, \sigma$ đã "biết" về tập test.
* **ĐÚNG**: tính $\mu, \sigma$ **chỉ trên tập train**, rồi dùng đúng các tham số đó để biến đổi cả train và test.

$$x'_{\text{train}} = \frac{x_{\text{train}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \qquad x'_{\text{test}} = \frac{x_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

Nguyên tắc vàng: mọi phép biến đổi học tham số từ dữ liệu (chuẩn hóa, imputation, chọn đặc trưng, mã hóa hạng mục, IDF...) phải **fit trên train, transform trên test**. Trong thực hành nên gói toàn bộ vào một **pipeline** để đảm bảo điều này tự động đúng, đặc biệt khi dùng kiểm định chéo (cross-validation) — xem thêm [đánh giá phân loại](#/danh-gia-phan-loai).

---

# 9. Ưu điểm

* **Hiệu quả vượt trội với chi phí thấp** — một đặc trưng tốt được thiết kế bằng tri thức miền (domain knowledge) thường cải thiện nhiều hơn việc tinh chỉnh siêu tham số mô hình.
* **Giúp mô hình đơn giản hoạt động tốt** — sau feature engineering, hồi quy tuyến tính/logistic có thể sánh ngang mô hình phức tạp mà vẫn **giải thích được (interpretable)**.
* **Phổ quát** — kỹ thuật áp dụng cho mọi loại dữ liệu: bảng, văn bản, ảnh, chuỗi thời gian.

---

# 10. Nhược điểm

* **Tốn công và phụ thuộc chuyên gia** — feature engineering thủ công đòi hỏi hiểu sâu bài toán và nhiều thử nghiệm.
* **Khó mở rộng** — với dữ liệu phi cấu trúc cực lớn (ảnh, âm thanh), đặc trưng thủ công thường thua **học biểu diễn (representation learning)** của học sâu, nơi mạng tự học đặc trưng.
* **Rủi ro rò rỉ** — nếu không cẩn thận với quy tắc fit-trên-train, ta dễ tạo ra mô hình "đẹp trên giấy" nhưng vô dụng thực tế.

---

# 11. Tổng kết

Feature engineering là cầu nối giữa dữ liệu thô và thuật toán học. Toàn bộ tri thức của mô hình chảy qua vector đặc trưng, nên **biểu diễn dữ liệu tốt quan trọng ngang — nếu không muốn nói là hơn — việc chọn mô hình**. Các kỹ thuật cốt lõi cần nắm:

* **Hạng mục**: one-hot (mặc định, không thứ tự giả), label (chỉ cho ordinal/cây), hashing (số chiều khổng lồ).
* **Số**: chuẩn hóa thang (z-score, min-max), biến đổi phân phối (log, Box-Cox), binning.
* **Thiếu**: lấp bằng thống kê/mô hình, kèm cờ chỉ báo thiếu.
* **Phi cấu trúc**: BoW/TF-IDF cho văn bản, bag-of-features cho ảnh, đặc trưng đa thức cho tương tác.

Và hai nguyên tắc kỹ thuật không bao giờ được quên: **chuẩn hóa để gradient descent hội tụ nhanh**, và **luôn fit trên train để tránh rò rỉ dữ liệu**.

> Bài tiếp theo sẽ đi sâu vào **chọn lọc và giảm chiều đặc trưng** — khi đặc trưng đã tốt, ta cần loại bỏ phần dư thừa và nén thông tin, mở đầu bằng [phân tích thành phần chính (PCA)](#/pca).

# Đánh giá hệ phân loại

> Một mô hình phân loại (classifier) tốt **không** đơn thuần là mô hình "đoán đúng nhiều". Khi dữ liệu mất cân bằng (class imbalance), độ chính xác (accuracy) có thể đánh lừa ta một cách nguy hiểm. Bài này xây dựng bộ công cụ đánh giá đúng đắn: từ ma trận nhầm lẫn (confusion matrix), qua precision/recall và F1, đến đường cong ROC, AUC và precision-recall — và quan trọng nhất, **chọn độ đo nào cho bài toán nào**.

---

# 1. Vì sao accuracy không đủ

Cách đo tự nhiên nhất của một bộ phân loại là **độ chính xác (accuracy)** — tỉ lệ mẫu được dự đoán đúng trên tổng số mẫu. Trực giác rất hấp dẫn: càng đoán đúng nhiều càng tốt. Nhưng trực giác này sụp đổ khi các lớp **không cân bằng**.

Hãy lấy một ví dụ y khoa kinh điển: sàng lọc một loại ung thư hiếm. Giả sử trong 1000 người đi khám, chỉ có **10** người thực sự mắc bệnh (lớp dương — positive), còn **990** người khỏe (lớp âm — negative). Bây giờ xét một mô hình "lười biếng" tuyệt đối: **luôn luôn dự đoán khỏe mạnh** cho mọi người, bất kể đầu vào.

Mô hình ngớ ngẩn này đạt accuracy:

$$\text{accuracy} = \frac{990}{1000} = 99\%$$

Một con số nghe **tuyệt vời**. Nhưng nó đã bỏ sót **toàn bộ 10 bệnh nhân** — đúng những người mà hệ thống sinh ra để tìm. Một mô hình 99% chính xác nhưng vô dụng hoàn toàn. Đây là **nghịch lý accuracy (accuracy paradox)**: khi một lớp áp đảo về số lượng, accuracy bị chi phối bởi lớp đa số và che giấu thất bại trên lớp thiểu số — vốn thường là lớp ta quan tâm nhất (bệnh nhân, giao dịch gian lận, email spam).

Vấn đề cốt lõi: accuracy gộp **mọi loại lỗi** vào một con số duy nhất, trong khi các loại lỗi có **chi phí rất khác nhau**. Bỏ sót một ca ung thư (để bệnh nhân ra về) nguy hiểm hơn nhiều so với báo động giả (gọi người khỏe quay lại xét nghiệm thêm). Ta cần một công cụ **tách bạch** các loại lỗi này — đó chính là ma trận nhầm lẫn.

> Cùng một hiện tượng xuất hiện ở [hồi quy logistic](#/logistic-regression) khi dữ liệu lệch lớp, và là một biểu hiện của [quá khớp](#/overfitting) khi mô hình "học vẹt" lớp đa số.

---

# 2. Ma trận nhầm lẫn (confusion matrix)

Với bài toán phân loại nhị phân (binary), mỗi dự đoán rơi vào đúng một trong **bốn ô** tùy theo nó đúng/sai và dự đoán dương/âm:

* **TP (True Positive)** — dương thật: thực sự dương, mô hình đoán dương. *(Phát hiện đúng bệnh nhân.)*
* **FP (False Positive)** — dương giả: thực sự âm, mô hình đoán dương. *(Báo động giả — lỗi loại I.)*
* **FN (False Negative)** — âm giả: thực sự dương, mô hình đoán âm. *(Bỏ sót bệnh nhân — lỗi loại II.)*
* **TN (True Negative)** — âm thật: thực sự âm, mô hình đoán âm. *(Loại đúng người khỏe.)*

Sắp lại thành bảng (hàng = nhãn thật, cột = dự đoán):

| | Dự đoán: Dương | Dự đoán: Âm |
| --- | --- | --- |
| **Thật: Dương** | TP | FN |
| **Thật: Âm** | FP | TN |

Đây là **ma trận nhầm lẫn**. Tên gọi rất gợi: đường chéo (TP, TN) là các dự đoán đúng, còn ngoài đường chéo (FP, FN) là nơi mô hình **nhầm lẫn**. Mọi độ đo phân loại nhị phân khác đều được tính ra từ bốn con số này.

Mẹo nhớ tên: từ **thứ hai** (Positive/Negative) là cái mô hình **dự đoán**; từ **đầu** (True/False) cho biết dự đoán đó đúng hay sai. "False Negative" = mô hình nói "âm" nhưng nói sai.

---

# 3. Accuracy, precision và recall

## 3.1. Accuracy

Viết lại accuracy theo bốn ô:

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Tử số là các dự đoán đúng (đường chéo), mẫu số là tổng tất cả. Như mục 1 đã chỉ ra, khi $TN$ áp đảo (lớp âm chiếm đa số) thì accuracy gần như chỉ phản ánh $TN$, làm mờ hoàn toàn hiệu năng trên lớp dương.

## 3.2. Precision — độ chính xác của dự đoán dương

**Precision** trả lời: *trong số những mẫu mô hình **gắn cờ dương**, bao nhiêu phần là dương thật?*

$$\text{precision} = \frac{TP}{TP + FP}$$

Mẫu số là **tất cả** những gì mô hình đoán là dương. Precision cao nghĩa là **ít báo động giả** — khi mô hình nói "dương", ta tin được. Đây là độ đo quan trọng khi **chi phí của FP cao**: ví dụ lọc spam (gắn nhầm email quan trọng vào spam khiến người dùng mất thư), hay hệ thống gợi ý (gợi ý sai làm mất lòng tin).

## 3.3. Recall (sensitivity) — độ phủ trên lớp dương

**Recall**, còn gọi là **độ nhạy (sensitivity)** hay **TPR (True Positive Rate)**, trả lời: *trong số những mẫu **thực sự dương**, mô hình tìm được bao nhiêu?*

$$\text{recall} = \frac{TP}{TP + FN}$$

Mẫu số là **tất cả** mẫu dương thật. Recall cao nghĩa là **ít bỏ sót** — đây là độ đo sống còn khi **chi phí của FN cao**: chẩn đoán ung thư (bỏ sót bệnh nhân là thảm họa), phát hiện gian lận. Trong ví dụ mục 1, mô hình "luôn đoán khỏe" có recall $= 0/10 = 0\%$ — phơi bày ngay sự vô dụng mà accuracy 99% đã che giấu.

## 3.4. Specificity — độ phủ trên lớp âm

Đối xứng với sensitivity, **specificity** (độ đặc hiệu) đo tỉ lệ mẫu âm được nhận diện đúng:

$$\text{specificity} = \frac{TN}{TN + FP} = 1 - \text{FPR}$$

trong đó **FPR (False Positive Rate)** $= FP / (FP + TN)$ là tỉ lệ báo động giả trên lớp âm. Cặp (sensitivity, specificity) rất hay dùng trong y khoa và là hai trục ngầm của đường cong ROC ở mục 5.

> **Tóm gọn trực giác.** Precision = "khi tôi báo dương thì tôi đúng bao nhiêu". Recall = "tôi tóm được bao nhiêu phần trong số dương thật". Hai câu hỏi **khác nhau** và thường **mâu thuẫn** nhau.

---

# 4. F1: hòa giải precision và recall

Một mô hình có thể "ăn gian" precision bằng cách **chỉ** báo dương ở những ca chắc chắn nhất (precision cao, recall thấp), hoặc ăn gian recall bằng cách **báo dương cho gần hết** (recall cao, precision thấp). Báo cáo riêng lẻ một con số dễ gây hiểu lầm. Ta cần một độ đo **gộp** cả hai.

**F1-score** là **trung bình điều hòa (harmonic mean)** của precision và recall:

$$F_1 = \frac{2}{\dfrac{1}{\text{precision}} + \dfrac{1}{\text{recall}}} = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

Vì sao **điều hòa** chứ không phải trung bình cộng? Trung bình cộng quá "dễ dãi": nếu precision $= 1.0$ nhưng recall $= 0.0$, trung bình cộng vẫn là $0.5$ — nghe ổn dù mô hình bỏ sót sạch lớp dương. Trung bình điều hòa thì **trừng phạt sự mất cân bằng**: với cùng cặp đó, $F_1 = 0$. Trung bình điều hòa luôn **nghiêng về giá trị nhỏ hơn**, nên $F_1$ chỉ cao khi **cả hai** precision và recall **đều cao**.

Tổng quát hơn, $F_\beta$ cho phép đánh trọng số recall gấp $\beta$ lần precision:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}$$

$\beta > 1$ (ví dụ $F_2$) ưu tiên recall — dùng cho sàng lọc bệnh; $\beta < 1$ (ví dụ $F_{0.5}$) ưu tiên precision — dùng khi báo động giả tốn kém.

---

# 5. Đánh đổi precision–recall theo ngưỡng

Hầu hết bộ phân loại không xuất ra trực tiếp nhãn "dương/âm" mà xuất ra một **điểm số (score)** hoặc **xác suất** $\hat{p} \in [0, 1]$. Ta chuyển thành nhãn bằng một **ngưỡng (threshold)** $\tau$:

$$\hat{y} = 1 \iff \hat{p} \ge \tau$$

Ngưỡng mặc định thường là $0.5$, nhưng đó chỉ là một lựa chọn. Thay đổi $\tau$ làm **dịch chuyển toàn bộ ma trận nhầm lẫn**:

* Hạ $\tau$ → mô hình "rộng lượng", báo dương nhiều hơn → $TP \uparrow, FN \downarrow$ (recall tăng) nhưng $FP \uparrow$ (precision thường giảm).
* Nâng $\tau$ → mô hình "khắt khe", báo dương ít hơn → precision thường tăng nhưng recall giảm.

Đây là **đánh đổi precision–recall (precision-recall trade-off)** cốt lõi: không có bữa trưa miễn phí. Một bộ phân loại không phải *một* điểm hiệu năng, mà là *một họ* điểm hiệu năng — mỗi ngưỡng cho một điểm. Vì vậy thay vì chốt một ngưỡng rồi báo cáo một con số, ta nên đánh giá mô hình **trên toàn dải ngưỡng**. Đó là động lực cho hai đường cong ở mục tiếp theo.

---

# 6. Đường cong ROC và AUC

## 6.1. Đường cong ROC

**Đường cong ROC (Receiver Operating Characteristic)** vẽ quan hệ giữa hai đại lượng khi $\tau$ quét từ $1$ về $0$:

* Trục tung: **TPR** (recall/sensitivity) $= TP/(TP+FN)$.
* Trục hoành: **FPR** $= FP/(FP+TN) = 1 - \text{specificity}$.

Mỗi giá trị $\tau$ cho một điểm $(\text{FPR}, \text{TPR})$; nối lại được một đường cong từ góc $(0,0)$ (ngưỡng cực cao, không báo dương gì) đến $(1,1)$ (ngưỡng cực thấp, báo dương tất cả).

Cách đọc đường cong:

* Đường chéo $y = x$ là **đoán ngẫu nhiên** — không có khả năng phân biệt.
* Đường cong càng **kéo về góc trên-trái** $(0, 1)$ càng tốt: TPR cao trong khi FPR thấp.
* Một bộ phân loại hoàn hảo chạm góc $(0, 1)$.

Một ưu điểm quan trọng: ROC **tương đối ổn định** với thay đổi tỉ lệ lớp, vì TPR và FPR mỗi cái chuẩn hóa **trong nội bộ một lớp**.

## 6.2. AUC và ý nghĩa xác suất

**AUC (Area Under the Curve)** là **diện tích dưới đường cong ROC**, một con số duy nhất trong $[0, 1]$ tóm tắt toàn bộ đường cong:

* $\text{AUC} = 1.0$: phân loại hoàn hảo.
* $\text{AUC} = 0.5$: ngẫu nhiên (bằng tung đồng xu).
* $\text{AUC} < 0.5$: tệ hơn ngẫu nhiên (thường do đảo nhãn).

Điều đẹp đẽ nhất là AUC có một **ý nghĩa xác suất** rất sáng tỏ:

> **AUC = xác suất mô hình xếp hạng một mẫu dương ngẫu nhiên cao hơn một mẫu âm ngẫu nhiên.**

Hình thức hóa, với $x^+$ là mẫu dương lấy ngẫu nhiên, $x^-$ là mẫu âm lấy ngẫu nhiên, và $s(\cdot)$ là điểm số mô hình gán:

$$\text{AUC} = \Pr\big[\, s(x^+) > s(x^-) \,\big]$$

Nghĩa là AUC đo **khả năng xếp hạng (ranking)** của mô hình — nó không quan tâm giá trị tuyệt đối của điểm số hay ngưỡng cụ thể, chỉ quan tâm mô hình có đặt các ca dương **trên** các ca âm hay không. Đây là lý do AUC độc lập với ngưỡng và rất hữu ích để so sánh mô hình tổng thể.

```python
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# y_true: nhãn thật {0,1}; y_score: xác suất dự đoán; y_pred: nhãn theo ngưỡng
auc = roc_auc_score(y_true, y_score)          # độc lập với ngưỡng
f1  = f1_score(y_true, y_pred)                 # phụ thuộc ngưỡng
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
```

## 6.3. Precision-recall curve

Khi dữ liệu **cực kỳ mất cân bằng** (lớp dương rất hiếm), ROC có thể trông **lạc quan giả tạo**: vì lớp âm quá đông, FPR vẫn nhỏ ngay cả khi số FP tuyệt đối lớn. Trong trường hợp đó, **đường cong precision-recall (PR curve)** — vẽ precision (trục tung) theo recall (trục hoành) khi quét ngưỡng — nhạy hơn nhiều.

PR curve **không** dùng đến $TN$, nên nó tập trung hoàn toàn vào lớp dương hiếm. Diện tích dưới nó, **AUC-PR** (hay average precision, AP), là tóm tắt phù hợp. Đường cơ sở (baseline) của PR curve **không** cố định ở $0.5$ như ROC, mà bằng **tỉ lệ lớp dương** trong dữ liệu — một mô hình hữu ích phải vượt rõ mức này.

> **Quy tắc ngón tay cái.** Dữ liệu cân bằng → ROC/AUC; dữ liệu mất cân bằng nặng và quan tâm lớp dương hiếm → PR curve/AUC-PR.

---

# 7. Đánh giá đa lớp (multiclass)

Khi có nhiều hơn hai lớp, ma trận nhầm lẫn mở rộng thành ma trận $C \times C$ (hàng = nhãn thật, cột = dự đoán). Precision/recall được tính **cho từng lớp** theo lối **một-đối-phần-còn-lại (one-vs-rest)**: với lớp $c$, coi $c$ là dương và mọi lớp khác là âm. Sau đó cần **gộp** các con số per-class thành một số tổng. Có ba cách trung bình:

| Cách gộp | Công thức (ví dụ precision) | Đặc điểm |
| --- | --- | --- |
| **Macro** | $\frac{1}{C}\sum_{c} \text{precision}_c$ | Trung bình **đều** mọi lớp; lớp hiếm và lớp đông **ngang quyền**. Tốt khi quan tâm lớp thiểu số. |
| **Weighted** | $\sum_{c} \frac{n_c}{N}\,\text{precision}_c$ | Trung bình có trọng số theo **số mẫu** $n_c$ mỗi lớp; lớp đông có tiếng nói lớn hơn. |
| **Micro** | $\dfrac{\sum_c TP_c}{\sum_c (TP_c + FP_c)}$ | Gộp **tất cả** TP/FP/FN toàn cục rồi mới tính; bị lớp đông chi phối. Với đa lớp, micro-precision = micro-recall = accuracy tổng. |

Lựa chọn quan trọng: **macro** làm nổi bật hiệu năng trên lớp hiếm (vì mỗi lớp một phiếu bầu), còn **micro/weighted** phản ánh hiệu năng tổng thể nghiêng theo phân bố thực tế. Nếu bạn quan tâm "mô hình làm tốt với *mọi* lớp, kể cả lớp ít dữ liệu" thì dùng macro-F1.

## 7.1. Top-k accuracy

Trong các bài toán nhiều lớp (phân loại ảnh ImageNet với 1000 lớp, gợi ý sản phẩm), đòi hỏi đoán **đúng tuyệt đối lớp top-1** đôi khi quá khắt khe. **Top-k accuracy** coi một dự đoán là đúng nếu nhãn thật nằm trong **$k$ lớp được mô hình chấm điểm cao nhất**:

$$\text{top-}k = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\big[\, y_i \in \text{top-}k\big(\hat{p}_i\big)\,\big]$$

Top-5 accuracy là chuẩn báo cáo kinh điển của ImageNet: nó khoan dung với những lớp **dễ nhầm lẫn về ngữ nghĩa** (các giống chó khác nhau) mà ta vẫn coi là "gần đúng".

---

# 8. Tổng kết

Không có **một** độ đo vạn năng. Đánh giá đúng nghĩa là chọn độ đo **khớp với chi phí lỗi** của bài toán thực:

* **Dữ liệu cân bằng, mọi lỗi như nhau** → accuracy là đủ và dễ hiểu.
* **Dữ liệu mất cân bằng** → bỏ accuracy, dùng precision/recall/F1 và PR curve.
* **Chi phí FN cao** (sàng lọc ung thư, phát hiện gian lận) → ưu tiên **recall** (hoặc $F_2$).
* **Chi phí FP cao** (lọc spam, gợi ý) → ưu tiên **precision** (hoặc $F_{0.5}$).
* **Cần một số tóm tắt cân bằng cả hai** → **F1** (trung bình điều hòa).
* **So sánh khả năng xếp hạng, độc lập ngưỡng** → **AUC-ROC**; nếu lớp dương cực hiếm → **AUC-PR**.
* **Đa lớp** → chọn macro (đề cao lớp hiếm) hoặc weighted/micro (phản ánh phân bố); với không gian lớp lớn cân nhắc **top-k**.

Mọi độ đo ấy đều nảy sinh từ **bốn con số** TP/FP/FN/TN của ma trận nhầm lẫn — hãy luôn bắt đầu bằng việc nhìn vào ma trận này trước khi báo cáo bất kỳ con số gộp nào.

> Hiểu các độ đo này áp dụng được cho mọi bộ phân loại — từ [hồi quy logistic](#/logistic-regression) và [Naive Bayes](#/naive-bayes) đến [SVM](#/svm) — và là tuyến phòng thủ đầu tiên giúp ta phát hiện [quá khớp](#/overfitting): khi điểm trên tập huấn luyện đẹp nhưng các độ đo trên tập kiểm tra (đặc biệt recall của lớp hiếm) sụp đổ.

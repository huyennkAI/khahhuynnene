# Độ đo đánh giá theo bài toán (Evaluation Metrics by Task)

> Không có **một** độ đo vạn năng cho mọi bài toán học máy. Một con số đẹp trên màn hình chỉ có giá trị khi nó **khớp với mục tiêu thực** của hệ thống. Bài này là bản đồ tổng quan: với mỗi họ bài toán — hồi quy (regression), xếp hạng/gợi ý (ranking), phát hiện vật thể (object detection), phân vùng (segmentation), sinh ngôn ngữ (NLG), sinh ảnh, phân cụm (clustering) — ta chọn độ đo nào và vì sao. Phần phân loại đã có bài riêng sâu hơn, nên ở đây chỉ tóm tắt và trỏ sang [Đánh giá hệ phân loại](#/danh-gia-phan-loai).

---

# 1. Nguyên tắc chung

## 1.1. Độ đo phải khớp bài toán và mục tiêu thực

Một mô hình được tối ưu cho **đúng cái ta đo**. Nếu độ đo lệch khỏi mục tiêu kinh doanh hay khoa học, ta sẽ nhận được một mô hình "tốt theo con số" nhưng vô dụng trong thực tế. Vì vậy bước đầu tiên của mọi dự án không phải là chọn mô hình, mà là **chọn độ đo** — và trả lời câu hỏi: *thành công thực sự trông như thế nào?*

Ví dụ: một hệ gợi ý phim được tối ưu cho sai số dự đoán điểm số (rating) có thể đạt RMSE rất thấp mà vẫn đề xuất một danh sách nhàm chán, vì cái người dùng quan tâm là **thứ tự** các phim ở đầu danh sách chứ không phải con số điểm. Bài toán này cần độ đo **xếp hạng** (mục 4), không phải độ đo **hồi quy** (mục 3).

## 1.2. Loss để tối ưu, metric để đánh giá

Cần phân biệt rạch ròi hai khái niệm hay bị lẫn:

* **Hàm mất mát (loss)** là thứ thuật toán **tối ưu** trong lúc huấn luyện. Nó phải **khả vi** (differentiable) và dễ lấy gradient — ví dụ cross-entropy, MSE.
* **Độ đo (metric)** là thứ **con người** dùng để **đánh giá** mô hình. Nó không cần khả vi và nên phản ánh mục tiêu thực — ví dụ accuracy, F1, BLEU, mAP.

Hai thứ này thường **khác nhau**. Ta huấn luyện một bộ phân loại bằng cross-entropy (loss khả vi) nhưng đánh giá bằng F1 (metric không khả vi). Ta huấn luyện một mô hình dịch máy bằng cross-entropy theo từng token nhưng đánh giá bằng BLEU trên cả câu. Sự lệch pha này là bình thường và chấp nhận được; điều nguy hiểm là khi ta nhầm loss thấp với chất lượng cao.

## 1.3. Train / validation / test

Mọi độ đo chỉ có ý nghĩa khi tính trên dữ liệu **mô hình chưa từng thấy**. Quy ước chuẩn:

* **Tập huấn luyện (train)** — để khớp tham số.
* **Tập kiểm định (validation)** — để chọn siêu tham số (hyperparameter) và dừng sớm (early stopping).
* **Tập kiểm tra (test)** — chỉ chạm **một lần duy nhất** ở cuối để báo cáo con số trung thực.

Báo cáo độ đo trên tập train là vô nghĩa: nó chỉ cho biết mô hình **thuộc bài** đến đâu, và độ chênh lệch lớn giữa train và test chính là dấu hiệu của [quá khớp](#/overfitting).

---

# 2. Phân loại (classification) — tóm tắt

Đây là họ bài toán có bộ độ đo được nghiên cứu kỹ nhất; bài [Đánh giá hệ phân loại](#/danh-gia-phan-loai) trình bày đầy đủ. Ở đây chỉ điểm lại các trụ cột:

* **Accuracy** — tỉ lệ đoán đúng. Trực quan nhưng **đánh lừa** khi dữ liệu mất cân bằng (class imbalance).
* **Precision / Recall / F1** — tách bạch hai loại lỗi từ ma trận nhầm lẫn (confusion matrix). Precision $= \frac{\text{TP}}{\text{TP}+\text{FP}}$ (ít báo động giả), recall $= \frac{\text{TP}}{\text{TP}+\text{FN}}$ (ít bỏ sót), $F_1$ là trung bình điều hòa của chúng.
* **ROC-AUC** — diện tích dưới đường cong ROC, đo khả năng **xếp hạng** dương trên âm, độc lập ngưỡng.
* **PR-AUC** — diện tích dưới đường cong precision-recall.

> **Khi nào ROC-AUC, khi nào PR-AUC?** ROC ổn định với tỉ lệ lớp nhưng có thể **lạc quan giả tạo** khi lớp dương cực hiếm, vì FPR vẫn nhỏ dù số FP tuyệt đối lớn. Khi dữ liệu **mất cân bằng nặng** và ta quan tâm lớp dương hiếm, hãy ưu tiên **PR-AUC** (vì nó bỏ qua TN, tập trung hoàn toàn vào lớp dương). Chi tiết xem [Đánh giá hệ phân loại](#/danh-gia-phan-loai).

---

# 3. Hồi quy (regression)

Với [hồi quy tuyến tính](#/linear-regression) và mọi mô hình dự đoán giá trị liên tục, độ đo so sánh giá trị dự đoán $\hat{y}_i$ với giá trị thật $y_i$ trên $n$ mẫu.

## 3.1. MAE, MSE, RMSE

**Sai số tuyệt đối trung bình (MAE — Mean Absolute Error):**

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert$$

MAE cùng đơn vị với $y$ và dễ diễn giải ("trung bình lệch 3 nghìn đồng").

**Sai số bình phương trung bình (MSE — Mean Squared Error):**

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Vì bình phương, MSE **trừng phạt nặng các sai số lớn** — một mẫu lệch gấp đôi đóng góp gấp bốn. Đơn vị của MSE là bình phương đơn vị $y$, khó diễn giải, nên ta thường lấy căn:

**Căn bậc hai MSE (RMSE — Root Mean Squared Error):**

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

RMSE đưa về cùng đơn vị với $y$ nhưng vẫn giữ tính nhạy với sai số lớn của MSE.

## 3.2. Độ bền với outlier

Đây là khác biệt then chốt khi chọn giữa MAE và MSE/RMSE. MSE/RMSE **rất nhạy với điểm ngoại lai (outlier)**: vì bình phương, một mẫu lệch cực lớn có thể chi phối toàn bộ con số. MAE **bền hơn** vì lỗi đóng góp tuyến tính.

> **Quy tắc.** Nếu outlier là lỗi dữ liệu cần bỏ qua → dùng **MAE**. Nếu sai số lớn thực sự nghiêm trọng và cần bị phạt nặng (ví dụ dự báo tải điện vượt ngưỡng) → dùng **RMSE**.

## 3.3. Hệ số xác định $R^2$

MAE và RMSE phụ thuộc thang đo của $y$, khó nói "thế là tốt hay tệ". **Hệ số xác định (coefficient of determination)** $R^2$ chuẩn hóa bằng cách so mô hình với một baseline tầm thường — luôn đoán giá trị trung bình $\bar{y}$:

$$R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

Cách đọc: $R^2$ là **tỉ lệ phương sai của $y$ mà mô hình giải thích được**.

* $R^2 = 1$: dự đoán hoàn hảo.
* $R^2 = 0$: mô hình **không hơn** việc luôn đoán trung bình.
* $R^2 < 0$: mô hình **tệ hơn** cả baseline trung bình (hoàn toàn có thể xảy ra trên tập test).

## 3.4. MAPE — sai số phần trăm

**Sai số phần trăm tuyệt đối trung bình (MAPE — Mean Absolute Percentage Error)** cho con số **không thứ nguyên**, dễ so sánh giữa các bài toán khác thang đo:

$$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left\lvert \frac{y_i - \hat{y}_i}{y_i}\right\rvert$$

MAPE rất trực quan ("dự báo doanh thu sai trung bình 8%") nhưng có hai cạm bẫy: **vô định khi $y_i = 0$**, và **bất đối xứng** (phạt dự đoán cao hơn thực tế nhẹ hơn dự đoán thấp hơn). Khi $y$ có giá trị gần $0$, hãy tránh MAPE.

---

# 4. Xếp hạng và gợi ý (ranking / retrieval)

Với [lọc cộng tác](#/collaborative-filtering) và [phân rã ma trận](#/matrix-factorization), thứ quan trọng **không** phải dự đoán đúng điểm số tuyệt đối, mà **thứ tự** của các mục ở đầu danh sách. Người dùng chỉ nhìn vài kết quả đầu, nên độ đo phải tập trung vào **top-$k$**.

## 4.1. Precision@k và Recall@k

Gọi tập $k$ mục mô hình xếp đầu là kết quả, và tập mục thực sự liên quan (relevant) là chuẩn vàng:

$$\text{Precision@}k = \frac{\#\{\text{mục liên quan trong top-}k\}}{k}, \qquad \text{Recall@}k = \frac{\#\{\text{mục liên quan trong top-}k\}}{\#\{\text{tổng mục liên quan}\}}$$

Precision@k hỏi "trong $k$ thứ tôi gợi ý, bao nhiêu trúng?"; Recall@k hỏi "trong tất cả thứ đáng gợi ý, tôi tóm được bao nhiêu ở top-$k$?". Hạn chế: cả hai **không quan tâm vị trí** trong top-$k$ — trúng ở hạng 1 hay hạng $k$ đều như nhau.

## 4.2. MAP — Mean Average Precision

Để thưởng cho việc đặt mục liên quan **lên cao**, ta tính **Average Precision (AP)**: trung bình của Precision@k tại **mỗi vị trí có một mục liên quan**:

$$\text{AP} = \frac{1}{\#\text{liên quan}}\sum_{k} \text{Precision@}k \cdot \mathbb{1}[\text{mục thứ } k \text{ liên quan}]$$

**MAP (Mean Average Precision)** là trung bình AP trên tất cả truy vấn (query) hay người dùng. MAP nhạy với thứ tự: đưa mục đúng lên đầu cho điểm cao hơn.

## 4.3. MRR — Mean Reciprocal Rank

Khi ta chỉ quan tâm **mục đúng đầu tiên** xuất hiện ở đâu (ví dụ hỏi-đáp, tìm kiếm điều hướng), dùng **Reciprocal Rank** $= 1/\text{rank}_{\text{first}}$ và lấy trung bình:

$$\text{MRR} = \frac{1}{Q}\sum_{q=1}^{Q}\frac{1}{\text{rank}_q}$$

trong đó $\text{rank}_q$ là vị trí của kết quả đúng đầu tiên cho truy vấn $q$. Đúng ở hạng 1 cho $1$, hạng 2 cho $0.5$, hạng 3 cho $0.33$...

## 4.4. NDCG — suy dẫn từ DCG với chiết khấu log

Khi mức độ liên quan có **nhiều bậc** (rất liên quan, hơi liên quan, không liên quan) chứ không nhị phân, ta cần **NDCG**. Ý tưởng: mục có điểm liên quan $\text{rel}_i$ ở vị trí càng thấp thì giá trị càng bị **chiết khấu (discount)** theo hàm log.

**Discounted Cumulative Gain (DCG)** tại top-$k$:

$$\text{DCG@}k = \sum_{i=1}^{k}\frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

Mẫu số $\log_2(i+1)$ là **chiết khấu vị trí**: hạng 1 chia $\log_2 2 = 1$, hạng 2 chia $\log_2 3 \approx 1.58$... — càng xuống dưới càng bị giảm giá, phản ánh việc người dùng ít chú ý kết quả phía sau. Tử số $2^{\text{rel}_i}-1$ khuếch đại các mục rất liên quan.

Để so sánh được giữa các truy vấn có số mục khác nhau, ta **chuẩn hóa** bằng DCG lý tưởng (IDCG — DCG của thứ tự sắp xếp hoàn hảo):

$$\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k} \in [0, 1]$$

NDCG $= 1$ nghĩa là mô hình xếp hạng đúng y như thứ tự lý tưởng.

---

# 5. Phát hiện vật thể (object detection)

Bài toán này vừa phải **định vị** (vẽ hộp bao — bounding box) vừa phải **phân loại** vật thể, nên độ đo phải đo cả hai.

## 5.1. IoU — Intersection over Union

Để biết một hộp dự đoán có "trúng" hộp thật không, ta đo độ chồng lấp bằng **IoU (Intersection over Union)**:

$$\text{IoU} = \frac{\text{diện tích phần giao}}{\text{diện tích phần hợp}} = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}$$

IoU $\in [0,1]$: $1$ là trùng khít, $0$ là không chạm nhau. Ta chọn một **ngưỡng** (thường $0.5$): nếu IoU với một hộp thật vượt ngưỡng và đúng lớp, dự đoán được tính là TP; ngược lại là FP.

## 5.2. mAP — mean Average Precision

Sau khi quy mỗi dự đoán về TP/FP nhờ IoU, ta tính **Average Precision (AP)** cho từng lớp như diện tích dưới đường cong precision-recall của lớp đó (quét theo điểm tin cậy). **mAP** là trung bình AP trên **tất cả các lớp**:

$$\text{mAP} = \frac{1}{C}\sum_{c=1}^{C}\text{AP}_c$$

Các benchmark còn tinh chỉnh thêm: **mAP@0.5** dùng một ngưỡng IoU duy nhất, còn **mAP@[0.5:0.95]** (chuẩn COCO) lấy trung bình mAP qua nhiều ngưỡng IoU từ $0.5$ đến $0.95$ — khắt khe hơn nhiều vì đòi hỏi định vị chính xác.

---

# 6. Phân vùng ảnh (segmentation)

Phân vùng gán nhãn cho **từng điểm ảnh (pixel)**, nên độ đo so sánh **mặt nạ (mask)** dự đoán với mặt nạ thật.

## 6.1. IoU / Jaccard

Vẫn là IoU như mục 5.1 nhưng tính trên **tập điểm ảnh** thay vì hộp; trong phân vùng nó còn gọi là **hệ số Jaccard**:

$$\text{IoU} = \frac{\lvert P \cap G \rvert}{\lvert P \cup G \rvert}$$

với $P$ là tập pixel dự đoán thuộc lớp, $G$ là tập pixel thật. Lấy trung bình IoU trên các lớp cho **mIoU** — độ đo chuẩn của các benchmark phân vùng.

## 6.2. Hệ số Dice

**Hệ số Dice (Dice coefficient)**, còn gọi F1 ở mức pixel, nhấn mạnh phần giao gấp đôi:

$$\text{Dice} = \frac{2\lvert P \cap G \rvert}{\lvert P \rvert + \lvert G \rvert}$$

Dice và IoU đo cùng một thứ và đơn điệu theo nhau ($\text{Dice} = \frac{2\,\text{IoU}}{1+\text{IoU}}$), nhưng Dice **rộng lượng hơn** với các vùng nhỏ, nên rất phổ biến trong ảnh y khoa (phân vùng khối u nhỏ), nơi IoU thuần có thể quá khắt khe.

---

# 7. Sinh ngôn ngữ (NLG)

Đánh giá văn bản do máy sinh khó hơn nhiều vì có **nhiều đáp án đúng**. Với các mô hình [seq2seq](#/seq2seq) (dịch máy, tóm tắt), ta so văn bản sinh với một hoặc nhiều bản tham chiếu (reference) của con người.

## 7.1. BLEU — n-gram precision và brevity penalty

**BLEU (Bilingual Evaluation Understudy)** — chuẩn của dịch máy — đo **precision của n-gram**: trong văn bản sinh, bao nhiêu phần các cụm $n$ từ cũng xuất hiện trong bản tham chiếu. Nó kết hợp precision của nhiều bậc $n$ (thường $1$ đến $4$):

$$\text{BLEU} = \text{BP}\cdot\exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

trong đó $p_n$ là precision n-gram bậc $n$ và $w_n$ là trọng số (thường đều nhau). Vì precision không phạt câu **quá ngắn** (một câu một từ trúng dễ đạt precision cao), BLEU nhân thêm **phạt độ ngắn (brevity penalty)**:

$$\text{BP} = \begin{cases} 1 & \text{nếu } c > r \\ \exp(1 - r/c) & \text{nếu } c \le r \end{cases}$$

với $c$ là độ dài câu sinh, $r$ là độ dài tham chiếu. BP kéo điểm xuống khi câu sinh ngắn hơn tham chiếu.

## 7.2. ROUGE — thiên về recall

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** — chuẩn của **tóm tắt văn bản** — ngược lại nhấn mạnh **recall**: bản tóm tắt có bắt được bao nhiêu nội dung của tham chiếu. **ROUGE-N** đếm n-gram trùng (kiểu recall), **ROUGE-L** đo chuỗi con chung dài nhất (longest common subsequence). Dịch máy quan tâm "đừng bịa thêm" → BLEU (precision); tóm tắt quan tâm "đừng bỏ sót ý" → ROUGE (recall).

## 7.3. Perplexity

**Perplexity (độ bối rối)** đánh giá trực tiếp một mô hình ngôn ngữ (language model), không cần tham chiếu: nó đo mô hình "ngạc nhiên" đến đâu trước văn bản thật. Là lũy thừa của cross-entropy trung bình mỗi token:

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(w_i \mid w_{<i})\right)$$

Trực giác: perplexity là **số lựa chọn trung bình** mô hình phân vân ở mỗi từ. Càng thấp càng tốt (mô hình càng chắc chắn về văn bản thật). Lưu ý perplexity đo độ khớp mô hình ngôn ngữ, **không** đo chất lượng văn bản sinh ra cho một tác vụ cụ thể.

---

# 8. Sinh ảnh (image generation)

Với [GAN](#/gan) và các mô hình sinh ảnh, không có "đáp án đúng" để so từng ảnh, nên ta đo ở mức **phân phối**: tập ảnh sinh có *giống* tập ảnh thật về mặt thống kê không, và có *đa dạng* không.

## 8.1. FID — Fréchet Inception Distance

**FID (Fréchet Inception Distance)** đưa cả ảnh thật và ảnh sinh qua một mạng Inception đã huấn luyện, lấy đặc trưng (feature) ở một tầng sâu, rồi xấp xỉ mỗi tập bằng một Gauss đa biến và đo **khoảng cách Fréchet** giữa hai Gauss:

$$\text{FID} = \lVert \mu_r - \mu_g \rVert^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

với $(\mu_r, \Sigma_r)$, $(\mu_g, \Sigma_g)$ là trung bình và hiệp phương sai đặc trưng của ảnh thật và ảnh sinh. **FID càng thấp càng tốt** ($0$ là phân phối trùng khít). Vì so cả trung bình lẫn hiệp phương sai, FID bắt được cả **chất lượng** lẫn **độ đa dạng** — nếu mô hình sụp đổ chế độ (mode collapse), $\Sigma_g$ co lại và FID tăng vọt.

## 8.2. Inception Score (IS)

**Inception Score** dùng phân bố lớp do Inception dự đoán: ảnh tốt thì phân bố lớp cho từng ảnh **nhọn** (rõ là vật gì — chất lượng cao), nhưng tổng hợp trên cả tập thì **trải đều** mọi lớp (đa dạng). Hai điều này được gộp qua một KL trung bình; **IS càng cao càng tốt**. Nhược điểm của IS là **không** so với ảnh thật và bị chi phối bởi tập lớp của Inception — nên thực tế FID được tin dùng hơn.

---

# 9. Phân cụm (clustering)

Với [K-Means](#/kmeans) và các thuật toán phân cụm, có hai tình huống: không có nhãn thật (đánh giá nội tại) và có nhãn thật để đối chiếu (đánh giá ngoại lai).

## 9.1. Silhouette — đánh giá nội tại

**Hệ số Silhouette** không cần nhãn thật; nó đo mỗi điểm "hợp" với cụm của nó đến đâu so với cụm gần nhất khác. Với điểm $i$, gọi $a(i)$ là khoảng cách trung bình tới các điểm **cùng cụm**, $b(i)$ là khoảng cách trung bình tới các điểm của **cụm láng giềng gần nhất**:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}} \in [-1, 1]$$

Gần $1$: điểm nằm sâu trong cụm đúng; gần $0$: nằm ở ranh giới; âm: có lẽ bị gán nhầm cụm. Lấy trung bình $s(i)$ trên toàn bộ cho điểm silhouette — hay dùng để **chọn số cụm $k$**.

## 9.2. ARI và NMI — đánh giá ngoại lai

Khi có nhãn thật để đối chiếu (dù thuật toán không dùng tới khi huấn luyện):

* **ARI (Adjusted Rand Index)** đếm các cặp điểm được hai cách phân (cụm dự đoán vs nhãn thật) **xử lý nhất quán** (cùng cụm hoặc khác cụm ở cả hai), rồi **hiệu chỉnh theo cơ may (chance)**. ARI $= 1$ là trùng khớp hoàn hảo, $\approx 0$ là ngẫu nhiên.
* **NMI (Normalized Mutual Information)** đo lượng thông tin chung (mutual information) giữa phân cụm và nhãn thật, chuẩn hóa về $[0,1]$. $1$ nghĩa là biết cụm xác định hoàn toàn nhãn và ngược lại.

Cả hai **bất biến với việc đổi tên nhãn cụm** — đúng bản chất phân cụm, nơi tên cụm là tùy ý.

---

# 10. Những cạm bẫy thường gặp

## 10.1. Metric thay thế (proxy) lệch mục tiêu

Khi mục tiêu thật khó đo (sự hài lòng người dùng), ta hay đo một **proxy** dễ đo hơn (số lần nhấp chuột — click). Nhưng tối ưu proxy quá mức có thể **phản tác dụng**: tối ưu click sinh ra tiêu đề giật gân (clickbait) làm người dùng khó chịu. Đây là biểu hiện của định luật Goodhart: *khi một độ đo trở thành mục tiêu, nó thôi là một độ đo tốt.* Hãy luôn tự hỏi proxy có còn khớp mục tiêu thực không.

## 10.2. Rò rỉ dữ liệu (data leakage)

Nếu thông tin từ tập test (hay từ tương lai) lọt vào lúc huấn luyện — ví dụ chuẩn hóa (normalize) trên toàn bộ dữ liệu **trước khi** chia, hay một đặc trưng vô tình mã hóa nhãn — thì độ đo trên test sẽ **cao giả tạo** rồi sụp đổ khi triển khai. Mọi phép biến đổi phải khớp **chỉ trên train** rồi áp lên test.

## 10.3. Dịch chuyển phân phối (distribution shift)

Độ đo trên test chỉ đáng tin nếu dữ liệu thực tế **cùng phân phối** với test. Khi phân phối dịch chuyển theo thời gian (hành vi người dùng đổi, mùa vụ, khái niệm trôi — concept drift), con số đẹp trong phòng thí nghiệm không còn đúng. Cần theo dõi độ đo **liên tục** sau khi triển khai.

## 10.4. Trung bình micro và macro

Khi gộp độ đo qua nhiều lớp hay nhiều truy vấn, cách trung bình thay đổi câu chuyện:

* **Micro-average** gộp tất cả mẫu lại rồi mới tính — bị **lớp/truy vấn đông chi phối**, phản ánh hiệu năng tổng thể.
* **Macro-average** tính cho từng lớp rồi lấy trung bình đều — mỗi lớp **một phiếu bầu**, làm nổi bật hiệu năng trên lớp hiếm.

Hai con số có thể chênh nhau rất lớn khi dữ liệu mất cân bằng; báo cáo nhầm loại trung bình dễ gây hiểu lầm về chất lượng thực.

---

# 11. Bảng tổng hợp theo bài toán

| Bài toán | Độ đo nên dùng | Ghi chú |
| --- | --- | --- |
| Phân loại (cân bằng) | Accuracy, F1 | Xem [danh-gia-phan-loai](#/danh-gia-phan-loai) |
| Phân loại (mất cân bằng) | Precision/Recall/F1, **PR-AUC** | PR-AUC khi lớp dương hiếm |
| Hồi quy | RMSE, MAE, $R^2$ | MAE bền outlier; $R^2$ chuẩn hóa |
| Dự báo phần trăm | MAPE | Tránh khi $y \approx 0$ |
| Gợi ý / lọc cộng tác | Precision@k, Recall@k, **NDCG**, MAP | NDCG khi liên quan nhiều bậc |
| Tìm kiếm / hỏi-đáp | MRR, MAP | MRR khi quan tâm kết quả đúng đầu tiên |
| Phát hiện vật thể | **mAP** (mAP@0.5, mAP@[0.5:0.95]) | Dựa trên IoU |
| Phân vùng ảnh | **mIoU** (Jaccard), **Dice** | Dice cho vùng nhỏ (y khoa) |
| Dịch máy | **BLEU** | n-gram precision + brevity penalty |
| Tóm tắt văn bản | **ROUGE** | Thiên về recall |
| Mô hình ngôn ngữ | Perplexity | Càng thấp càng tốt |
| Sinh ảnh | **FID** (thấp tốt), IS | FID bắt cả chất lượng và đa dạng |
| Phân cụm (không nhãn) | Silhouette | Chọn số cụm $k$ |
| Phân cụm (có nhãn) | ARI, NMI | Bất biến đổi tên cụm |

---

# 12. Tổng kết

Chọn độ đo là một quyết định **thiết kế**, không phải thủ tục máy móc. Vài nguyên tắc đọng lại:

* **Khớp bài toán.** Mỗi họ bài toán có cấu trúc đầu ra riêng — giá trị liên tục, thứ hạng, hộp bao, mặt nạ pixel, văn bản, phân phối ảnh, cụm — và độ đo phải đo đúng cấu trúc đó.
* **Khớp mục tiêu.** Trong cùng một họ vẫn phải chọn theo chi phí lỗi thực: RMSE hay MAE tùy outlier; precision hay recall tùy chi phí FP/FN; BLEU hay ROUGE tùy dịch hay tóm tắt.
* **Phân biệt loss và metric.** Tối ưu cái khả vi, đánh giá cái phản ánh mục tiêu — hai thứ thường khác nhau và đó là điều bình thường.
* **Đo trên dữ liệu chưa thấy** và cảnh giác với rò rỉ dữ liệu, proxy lệch mục tiêu, dịch chuyển phân phối, và cách trung bình micro/macro.

Một con số duy nhất không bao giờ kể hết câu chuyện. Hãy báo cáo vài độ đo bổ trợ nhau, luôn kèm baseline để so sánh, và nhớ rằng độ đo tốt nhất là độ đo khiến mô hình tốt theo nó cũng là mô hình tốt cho người dùng.

> Bài này là bản đồ tổng quan; mỗi ô trong bảng đều có thể đào sâu — bắt đầu từ [Đánh giá hệ phân loại](#/danh-gia-phan-loai) để nắm chắc nền tảng confusion matrix, rồi áp dụng cùng tư duy "khớp chi phí lỗi" cho mọi bài toán còn lại.

# Các hàm mất mát theo bài toán (Loss functions)

> Hàm mất mát (loss function) là cách ta **biến mục tiêu học thành một con số duy nhất** để thuật toán tối ưu kéo xuống. Chọn sai loss thì dù mô hình mạnh đến đâu cũng học sai thứ ta muốn.
>
> $$\theta^{*} = \arg\min_{\theta}\; \frac{1}{n}\sum_{i=1}^{n} \ell\big(y_i,\, \hat{y}_i\big)$$
>
> Một sự thật đẹp đẽ xuyên suốt bài: **rất nhiều loss quen thuộc chính là negative log-likelihood (NLL) của một giả định phân phối** — đổi giả định nhiễu là đổi loss.

---

# 1. Loss đến từ đâu?

Trước khi liệt kê công thức, cần hiểu vì sao chúng có dạng như vậy. Hầu hết loss không phải bịa ra một cách tùy tiện, mà rơi ra một cách tự nhiên từ nguyên lý hợp lý cực đại.

## 1.1. Mọi loss là một thước đo sai

Một hàm mất mát $\ell(y, \hat{y})$ định lượng "dự đoán $\hat{y}$ sai so với sự thật $y$ bao nhiêu". Quy ước: $\ell \ge 0$, càng nhỏ càng tốt, và $\ell = 0$ khi dự đoán hoàn hảo. Toàn bộ việc huấn luyện chỉ là tìm $\theta$ làm trung bình $\ell$ trên dữ liệu nhỏ nhất (xem [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu)).

## 1.2. Loss = NLL của một giả định phân phối

Đây là sợi chỉ đỏ. Giả sử ta mô hình hóa $p(y \mid x; \theta)$. Hợp lý cực đại (maximum likelihood) cực đại hóa $\sum_i \log p(y_i \mid x_i)$, tương đương cực tiểu hóa:

$$\mathcal{L}(\theta) = -\sum_{i=1}^{n} \log p(y_i \mid x_i; \theta)$$

Mỗi giả định về dạng $p$ sinh ra một loss. Gauss cho ra MSE, Laplace cho ra MAE, Bernoulli cho ra cross-entropy. Hiểu điều này (xem [MLE & MAP](#/mle-map)) giúp ta **biết khi nào một loss phù hợp**: nó phù hợp khi giả định nhiễu của nó khớp thực tế.

## 1.3. Hai họ lớn

Ta chia theo bản chất đầu ra: **hồi quy** (regression, $y$ liên tục) và **phân loại** (classification, $y$ rời rạc), rồi mở rộng sang học biểu diễn, so khớp phân phối, và phân vùng ảnh.

---

# 2. Loss cho hồi quy (regression)

Đầu ra là số thực. Câu hỏi cốt lõi: ta phạt sai số lớn nặng cỡ nào, và ta lo ngại điểm ngoại lai (outlier) đến đâu?

## 2.1. MSE / L2 — suy dẫn từ nhiễu Gauss

Mean Squared Error (sai số toàn phương trung bình):

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - \hat{y}_i\big)^2$$

**Suy dẫn.** Giả định $y = \hat{y} + \varepsilon$ với nhiễu Gauss $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. Khi đó

$$-\log p(y_i \mid x_i) = \frac{(y_i - \hat{y}_i)^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)$$

Bỏ hằng số và hệ số, cực tiểu NLL **chính là** cực tiểu MSE. Vậy MSE ngầm giả định nhiễu Gauss đối xứng. Đây cũng là loss của [hồi quy tuyến tính](#/linear-regression).

**Tính chất.** MSE khả vi mọi nơi, gradient $\propto (\hat{y}-y)$ tỉ lệ với sai số nên hội tụ mượt. Nhưng vì phạt **bình phương**, một outlier có sai số gấp 10 lần sẽ đóng góp gấp 100 lần — MSE **rất nhạy với outlier**.

## 2.2. MAE / L1 — suy dẫn từ nhiễu Laplace

Mean Absolute Error (sai số tuyệt đối trung bình):

$$\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}\big\lvert y_i - \hat{y}_i\big\rvert$$

**Suy dẫn.** Nếu giả định nhiễu phân phối Laplace, $p(\varepsilon) \propto \exp(-\lvert\varepsilon\rvert/b)$, thì NLL tỉ lệ với $\lvert y - \hat{y}\rvert$. MAE chính là NLL Laplace.

**Tính chất.** Vì phạt **tuyến tính**, một outlier chỉ đóng góp tuyến tính — MAE **bền với outlier** hơn hẳn MSE. Nghiệm tối ưu của MAE là **trung vị (median)**, còn của MSE là **trung bình (mean)**. Nhược điểm: $\lvert\cdot\rvert$ **không khả vi tại $0$**, và gradient có độ lớn hằng (dấu của sai số) nên gần nghiệm vẫn nhảy, hội tụ kém mượt.

## 2.3. Huber / smooth-L1 — lai L1 và L2

Huber loss lấy cái tốt của cả hai: bình phương khi sai số nhỏ (mượt, gradient có ý nghĩa), tuyến tính khi sai số lớn (bền outlier). Với ngưỡng $\delta$ và $r = y - \hat{y}$:

$$\ell_\delta(r) = \begin{cases} \tfrac{1}{2} r^2 & \text{nếu } \lvert r\rvert \le \delta \\[4pt] \delta\big(\lvert r\rvert - \tfrac{1}{2}\delta\big) & \text{nếu } \lvert r\rvert > \delta \end{cases}$$

Hai nhánh được ghép sao cho **giá trị và đạo hàm liên tục** tại $\lvert r\rvert = \delta$, nên Huber khả vi mọi nơi. Biến thể smooth-L1 (dùng nhiều trong phát hiện vật thể, ví dụ Faster R-CNN) là Huber với $\delta = 1$. $\delta$ điều khiển "ranh giới outlier": $\delta$ nhỏ thiên về L1 (bền hơn), $\delta$ lớn thiên về L2 (mượt hơn).

## 2.4. Pinball / quantile loss — hồi quy phân vị

Đôi khi ta không muốn dự đoán giá trị trung tâm mà muốn một **phân vị (quantile)** $\tau \in (0,1)$ — ví dụ phân vị 0.9 để dự báo "cận trên". Pinball loss phạt **bất đối xứng**:

$$\ell_\tau(r) = \begin{cases} \tau\, r & \text{nếu } r \ge 0 \\ (\tau - 1)\, r & \text{nếu } r < 0 \end{cases} = \max\big(\tau r,\, (\tau-1) r\big), \qquad r = y - \hat{y}$$

Khi $\tau = 0.5$, đây đúng bằng $\tfrac{1}{2}$ MAE (dự đoán trung vị). Với $\tau = 0.9$, dự đoán quá thấp ($r>0$) bị phạt nặng gấp 9 lần dự đoán quá cao, nên nghiệm tối ưu chính là phân vị 0.9. Đây là nền tảng của **dự báo khoảng (prediction interval)**.

---

# 3. Loss cho phân loại (classification)

Đầu ra là nhãn rời rạc. Ta để mô hình xuất **xác suất** và phạt theo mức độ "ngạc nhiên" khi thấy nhãn thật.

## 3.1. Binary cross-entropy (BCE) — NLL Bernoulli

Với bài toán hai lớp, mô hình xuất $\hat{p} = \sigma(z) \in (0,1)$ là xác suất lớp dương. Nhãn thật $y \in \{0,1\}$ tuân theo Bernoulli, nên

$$p(y \mid \hat{p}) = \hat{p}^{\,y}(1-\hat{p})^{1-y} \;\Longrightarrow\; \ell_{\text{BCE}} = -\big[y\log\hat{p} + (1-y)\log(1-\hat{p})\big]$$

Chính là NLL của Bernoulli, và là loss của [hồi quy logistic](#/logistic-regression). Trực giác: nếu nhãn là $1$ mà mô hình nói $\hat{p}\to 0$, thì $-\log\hat{p}\to\infty$ — phạt vô cùng nặng cho sự tự tin sai.

## 3.2. Categorical cross-entropy — NLL Categorical

Với $K$ lớp, mô hình xuất vector xác suất $\hat{p} = \mathrm{softmax}(z)$, và nhãn thật ở dạng one-hot $y$. NLL của phân phối Categorical:

$$\ell_{\text{CE}} = -\sum_{c=1}^{K} y_c \log \hat{p}_c = -\log \hat{p}_{\,y}$$

tức chỉ tính $-\log$ xác suất gán cho **đúng lớp thật**. Đây là loss của [hồi quy softmax](#/softmax-regression). Gradient cực gọn: $\partial \ell / \partial z = \hat{p} - y$, lý do người ta luôn ghép softmax với cross-entropy.

## 3.3. Label smoothing — chống quá tự tin

Cross-entropy thúc mô hình đẩy $\hat{p}_y \to 1$, dễ dẫn tới **quá tự tin (overconfidence)** và logit phình to. Label smoothing thay nhãn cứng one-hot bằng nhãn mềm: lớp đúng nhận $1-\epsilon$, mỗi lớp còn lại nhận $\epsilon/(K-1)$:

$$y_c^{\text{LS}} = (1-\epsilon)\, y_c + \frac{\epsilon}{K}$$

Hiệu ứng: mô hình không bao giờ bị ép đạt xác suất tuyệt đối, nên **hiệu chỉnh (calibration)** tốt hơn và tổng quát hóa tốt hơn — một dạng điều chuẩn nhẹ (xem [Overfitting](#/overfitting)).

## 3.4. Focal loss — đối phó mất cân bằng lớp

Khi lớp dương cực hiếm (ví dụ phát hiện vật thể, gian lận), vô số mẫu nền **dễ** lấn át gradient. Focal loss hạ trọng số mẫu đã phân đúng tự tin:

$$\ell_{\text{focal}} = -\,\alpha\,(1 - \hat{p}_y)^{\gamma}\,\log \hat{p}_y$$

Số hạng $(1-\hat{p}_y)^\gamma$ là **bộ điều biến**: mẫu dễ có $\hat{p}_y \to 1$ nên $(1-\hat{p}_y)^\gamma \to 0$, gần như bị tắt; mẫu khó giữ gần như nguyên trọng số. $\gamma$ (thường $2$) điều khiển mức "tập trung vào mẫu khó", $\alpha$ cân bằng lớp.

## 3.5. Hinge loss — tối đa hóa lề

Thay vì xác suất, ta có thể phạt theo **lề (margin)**. Với nhãn $y \in \{-1, +1\}$ và điểm số $z$:

$$\ell_{\text{hinge}} = \max\big(0,\; 1 - y\,z\big)$$

Khi $yz \ge 1$ (phân đúng và đủ xa ranh giới), loss bằng $0$ — không phạt nữa. Đây là loss của [máy vector hỗ trợ (SVM)](#/svm), dẫn tới biên rộng (max-margin). Khác cross-entropy ở chỗ hinge **không thúc xác suất tới $1$**, chỉ cần vượt lề là dừng.

---

# 4. Loss cho học biểu diễn (metric learning)

Mục tiêu không phải dự đoán nhãn mà học một **không gian nhúng (embedding)** sao cho điểm giống nhau gần nhau, khác nhau xa nhau.

## 4.1. Contrastive loss — kéo gần, đẩy xa theo cặp

Cho một cặp $(a, b)$ với nhãn $s = 1$ nếu cùng lớp, $s = 0$ nếu khác, và khoảng cách nhúng $d = \lVert f(a) - f(b)\rVert$:

$$\ell = s\, d^2 + (1-s)\,\max\big(0,\; m - d\big)^2$$

Cặp giống bị phạt theo $d^2$ (kéo lại gần $0$); cặp khác chỉ bị phạt khi còn **gần hơn lề $m$** (đẩy ra ít nhất khoảng cách $m$). Hết lề thì thôi đẩy, tránh phá vỡ cấu trúc không gian.

## 4.2. Triplet loss — so sánh tương đối

Triplet loss dùng bộ ba: mỏ neo $a$, mẫu dương $p$ (cùng lớp), mẫu âm $n$ (khác lớp). Ta đòi neo gần dương hơn gần âm một khoảng lề $m$:

$$\ell = \max\Big(0,\; \lVert f(a)-f(p)\rVert^2 - \lVert f(a)-f(n)\rVert^2 + m\Big)$$

Ưu điểm so với contrastive: phạt theo **quan hệ tương đối** chứ không ép khoảng cách tuyệt đối, linh hoạt hơn. Chìa khóa thực hành là **khai thác mẫu khó (hard mining)** — chọn các bộ ba vi phạm lề để học hiệu quả.

---

# 5. Loss giữa các phân phối

Khi cả mục tiêu lẫn dự đoán đều là **phân phối**, ta cần thước đo khoảng cách giữa phân phối.

## 5.1. KL divergence và quan hệ với cross-entropy

Phân kỳ Kullback–Leibler đo "tốn thêm bao nhiêu bit khi dùng $q$ thay cho $p$":

$$D_{\mathrm{KL}}(p \,\Vert\, q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = \underbrace{-\sum_x p(x)\log q(x)}_{\text{cross-entropy } H(p,q)} - \underbrace{\Big(-\sum_x p(x)\log p(x)\Big)}_{\text{entropy } H(p)}$$

Vậy **cross-entropy $=$ KL $+$ entropy của $p$**. Khi $p$ là nhãn thật cố định, $H(p)$ là hằng số, nên cực tiểu cross-entropy **chính là** cực tiểu KL. Đó là lý do sâu xa vì sao cross-entropy là loss phân loại chuẩn. Lưu ý KL **bất đối xứng**: $D_{\mathrm{KL}}(p\Vert q) \ne D_{\mathrm{KL}}(q\Vert p)$.

## 5.2. Vai trò trong VAE và GAN

Trong [Autoencoder & VAE](#/autoencoder-vae), mục tiêu ELBO gồm một số hạng tái tạo cộng một số hạng KL kéo hậu nghiệm xấp xỉ về tiên nghiệm:

$$\mathcal{L}_{\text{VAE}} = \underbrace{-\,\mathbb{E}_{q}\big[\log p(x\mid z)\big]}_{\text{tái tạo}} + \underbrace{D_{\mathrm{KL}}\big(q(z\mid x)\,\Vert\,p(z)\big)}_{\text{điều chuẩn không gian ẩn}}$$

Trong [GAN](#/gan), mục tiêu đối kháng gốc tương đương cực tiểu hóa **Jensen–Shannon divergence** giữa phân phối thật và phân phối sinh; các biến thể như Wasserstein GAN thay bằng khoảng cách Wasserstein để ổn định hơn. Chọn "khoảng cách phân phối" nào chính là chọn linh hồn của mô hình sinh.

---

# 6. Loss cho phân vùng ảnh và chuỗi

## 6.1. Dice loss và IoU loss

Trong phân vùng ảnh (segmentation), vùng cần tách thường nhỏ hơn nhiều so với nền, nên cross-entropy theo từng điểm ảnh bị nền lấn át. Dice loss tối ưu trực tiếp độ trùng khớp giữa mặt nạ dự đoán $\hat{p}$ và mặt nạ thật $y$:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i \hat{p}_i\, y_i + \epsilon}{\sum_i \hat{p}_i + \sum_i y_i + \epsilon}$$

Số hạng phân số là hệ số Dice (tương đương F1 trên pixel); $\epsilon$ tránh chia $0$. IoU loss (Jaccard) tương tự nhưng dùng tỉ lệ giao trên hợp. Cả hai **bất biến với mất cân bằng lớp** vì chỉ quan tâm phần chồng lấp, thường được kết hợp với cross-entropy.

## 6.2. CTC loss cho chuỗi không căn chỉnh

Khi đầu ra là chuỗi nhưng **không có căn chỉnh (alignment)** giữa đầu vào và nhãn — như nhận dạng tiếng nói hay chữ viết — Connectionist Temporal Classification (CTC) cộng dồn xác suất trên **mọi cách căn chỉnh hợp lệ** (chèn ký tự rỗng) sinh ra nhãn đích, rồi lấy NLL. Nó cho phép huấn luyện chuỗi mà không cần gán nhãn từng khung.

---

# 7. Lưu ý thực hành

## 7.1. Trọng số lớp (class weighting)

Với dữ liệu mất cân bằng, nhân mỗi số hạng loss với trọng số nghịch với tần suất lớp giúp mô hình không bỏ quên lớp hiếm:

$$\mathcal{L} = -\sum_{c} w_c\, y_c \log \hat{p}_c, \qquad w_c \propto \frac{1}{\text{tần suất lớp } c}$$

Đây là giải pháp đơn giản và thường nên thử trước focal loss.

## 7.2. Kết hợp nhiều loss

Thực tế hay cộng nhiều loss có trọng số, ví dụ Dice $+$ cross-entropy cho phân vùng, hay tái tạo $+$ KL trong VAE:

$$\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_2 + \cdots$$

Cần cân chỉnh $\lambda$ để các số hạng có cùng cỡ độ lớn, tránh một loss áp đảo phần còn lại.

## 7.3. Liên hệ với điều chuẩn (regularization)

Số hạng điều chuẩn như $\ell_2$ ($\lVert\theta\rVert^2$) hay $\ell_1$ ($\lVert\theta\rVert_1$) cộng vào loss cũng có gốc xác suất: chúng là **NLL của tiên nghiệm** trên tham số (Gauss cho $\ell_2$, Laplace cho $\ell_1$), biến cực đại hợp lý thành cực đại hậu nghiệm (xem [MLE & MAP](#/mle-map)). Vì thế loss thực tế thường là "loss dữ liệu $+$ loss điều chuẩn", và điều chuẩn là vũ khí chính chống [overfitting](#/overfitting).

---

# 8. Chọn loss theo bài toán

| Bài toán | Loss khuyến nghị | Vì sao |
| --- | --- | --- |
| Hồi quy, nhiễu sạch | MSE / L2 | NLL Gauss, gradient mượt |
| Hồi quy, có outlier | MAE/L1 hoặc Huber | bền với ngoại lai |
| Dự báo khoảng / phân vị | Pinball / quantile | phạt bất đối xứng theo $\tau$ |
| Phân loại nhị phân | BCE | NLL Bernoulli |
| Phân loại đa lớp | Categorical cross-entropy | NLL Categorical |
| Đa lớp, hay quá tự tin | CE $+$ label smoothing | hiệu chỉnh tốt hơn |
| Mất cân bằng lớp nặng | Focal loss / class weighting | nhấn mẫu khó / lớp hiếm |
| Biên rộng, không cần xác suất | Hinge loss | max-margin (SVM) |
| Học nhúng theo cặp | Contrastive loss | kéo gần / đẩy xa |
| Học nhúng tương đối | Triplet loss | so sánh dương vs âm |
| So khớp phân phối | KL / cross-entropy | nền tảng VAE, sinh |
| Phân vùng ảnh | Dice / IoU ($+$ CE) | bất biến mất cân bằng |
| Chuỗi không căn chỉnh | CTC loss | cộng dồn mọi alignment |

---

# 9. Tổng kết

Hàm mất mát là **bản hợp đồng giữa con người và thuật toán**: ta khai báo "thế nào là sai" bằng một con số, và phần còn lại của học máy chỉ là kéo con số đó xuống. Sợi chỉ đỏ thống nhất gần như mọi loss là negative log-likelihood: chọn loss thực chất là **chọn một giả định phân phối** về dữ liệu (xem [MLE & MAP](#/mle-map)).

- Hồi quy: MSE (Gauss, nhạy outlier) → MAE (Laplace, bền) → Huber (lai) → pinball (phân vị).
- Phân loại: cross-entropy (NLL Bernoulli/Categorical), tinh chỉnh bằng label smoothing, focal loss, hoặc đổi sang hinge khi cần lề rộng.
- Học biểu diễn, so khớp phân phối, phân vùng ảnh, chuỗi: mỗi miền có loss riêng phản ánh đúng cấu trúc bài toán.

> Nguyên tắc vàng: trước khi chọn loss, hãy hỏi **"giả định nhiễu / phân phối nào hợp với dữ liệu của tôi?"** và **"tôi quan tâm sai kiểu nào nhất?"**. Trả lời được hai câu đó, loss gần như tự lộ diện.

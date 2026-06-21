# Phân tích biệt thức tuyến tính (Linear Discriminant Analysis)

> Phân tích biệt thức tuyến tính (Linear Discriminant Analysis - LDA), hay **biệt thức Fisher (Fisher's discriminant)**, là một phương pháp **giảm chiều có giám sát (supervised)**. Khác với [PCA](#/pca) — vốn chỉ tìm hướng giữ lại nhiều **phương sai** nhất mà bỏ qua nhãn — LDA dùng chính nhãn lớp để tìm hướng chiếu $\mathbf{w}$ sao cho:
>
> $$\text{các lớp tách xa nhau, nhưng mỗi lớp lại co cụm}$$
>
> Mục tiêu được gói gọn trong **tiêu chí Fisher**:
>
> $$J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}} \;\longrightarrow\; \max$$
>
> với $\mathbf{S}_B$ đo độ tách **giữa các lớp** (between-class) và $\mathbf{S}_W$ đo độ tản mạn **trong từng lớp** (within-class). LDA biến bài toán này thành một bài toán **trị riêng tổng quát** giải được tường minh.

---

# 1. Vì sao cần giảm chiều có giám sát?

[PCA](#/pca) là phương pháp giảm chiều **không giám sát (unsupervised)**: nó chỉ nhìn vào phân bố hình học của dữ liệu $\mathbf{x}$ và giữ lại các trục có **phương sai (variance)** lớn nhất, hoàn toàn **không biết nhãn** $y$. Điều này tốt cho việc nén thông tin tổng quát, nhưng có một điểm yếu chí mạng cho bài toán **phân loại (classification)**.

Hãy tưởng tượng dữ liệu hai lớp nằm dọc theo một dải dài (phương sai lớn theo chiều dọc dải), nhưng hai lớp lại **xếp song song nhau** và chỉ tách biệt theo chiều **ngang** (phương sai nhỏ). PCA sẽ chọn hướng dọc dải vì phương sai ở đó lớn nhất — nhưng chiếu lên hướng đó, hai lớp **chồng lên nhau** hoàn toàn, không còn phân biệt được. Hướng thật sự "đáng giá" để phân loại lại là hướng ngang, nơi PCA vứt bỏ.

> Bài học: **phương sai lớn nhất không đồng nghĩa với khả năng phân biệt lớp tốt nhất.** Khi đã có nhãn, ta nên dùng nó.

LDA khắc phục bằng cách đặt câu hỏi khác: tìm hướng chiếu $\mathbf{w}$ sao cho sau khi chiếu, **tâm các lớp xa nhau** mà **mỗi lớp vẫn gọn**. Đây chính là ý tưởng giảm chiều **có giám sát**.

---

# 2. Ý tưởng: chiếu xuống một hướng tốt cho phân loại

Xét bài toán **hai lớp** với dữ liệu $\mathbf{x} \in \mathbb{R}^D$. Ta chiếu mỗi điểm xuống một đường thẳng theo hướng $\mathbf{w}$:

$$z = \mathbf{w}^\top \mathbf{x}$$

Sau phép chiếu, mỗi điểm $D$ chiều trở thành một số thực $z$. Câu hỏi: chọn $\mathbf{w}$ thế nào để các giá trị $z$ của hai lớp **tách biệt** nhất?

Gọi $\mathbf{m}_1, \mathbf{m}_2 \in \mathbb{R}^D$ là **vector trung bình (mean)** của hai lớp trong không gian gốc:

$$\mathbf{m}_k = \frac{1}{N_k} \sum_{\mathbf{x} \in \mathcal{C}_k} \mathbf{x}, \qquad k = 1, 2$$

Một ý tưởng thô sơ: chọn $\mathbf{w}$ để khoảng cách giữa hai tâm sau khi chiếu là lớn nhất, tức cực đại $\big(\mathbf{w}^\top \mathbf{m}_1 - \mathbf{w}^\top \mathbf{m}_2\big)^2$. Nhưng ý này **thiếu sót**: nó bỏ qua **độ tản mạn nội bộ** của mỗi lớp. Hai tâm có thể xa nhau, nhưng nếu mỗi lớp "loe ra" quá rộng theo hướng chiếu thì chúng vẫn chồng lấn.

> Trực giác cốt lõi của Fisher: **chuẩn hóa** khoảng cách giữa hai tâm bằng độ tản mạn trong lớp. Ta muốn *cách biệt giữa lớp lớn* **so với** *độ phân tán trong lớp nhỏ* — đúng tinh thần "tín hiệu trên nhiễu" (signal-to-noise).

---

# 3. Hai ma trận phân tán (scatter matrices)

Để định lượng "tách xa" và "co cụm", ta định nghĩa hai ma trận phân tán.

## 3.1. Ma trận phân tán trong lớp $\mathbf{S}_W$

Ma trận **phân tán trong lớp (within-class scatter)** đo tổng độ tản mạn của các điểm quanh tâm lớp của chúng:

$$\mathbf{S}_W = \sum_{k} \sum_{\mathbf{x} \in \mathcal{C}_k} (\mathbf{x} - \mathbf{m}_k)(\mathbf{x} - \mathbf{m}_k)^\top$$

Đây là tổng các ma trận hiệp phương sai (chưa chuẩn hóa) của từng lớp. $\mathbf{S}_W$ càng "nhỏ" nghĩa là các lớp càng **co cụm** quanh tâm.

## 3.2. Ma trận phân tán giữa lớp $\mathbf{S}_B$

Ma trận **phân tán giữa lớp (between-class scatter)** đo độ tách của các tâm lớp so với tâm chung. Với hai lớp, dạng đơn giản và đủ dùng là:

$$\mathbf{S}_B = (\mathbf{m}_1 - \mathbf{m}_2)(\mathbf{m}_1 - \mathbf{m}_2)^\top$$

Đây là một ma trận **hạng một (rank-one)**, đo phương "nối hai tâm". $\mathbf{S}_B$ càng "lớn" theo hướng $\mathbf{w}$ nghĩa là hai tâm càng **xa nhau** sau khi chiếu.

> **Liên hệ với phép chiếu.** Sau khi chiếu lên $\mathbf{w}$, phương sai giữa lớp trở thành $\mathbf{w}^\top \mathbf{S}_B \mathbf{w}$, còn phương sai trong lớp trở thành $\mathbf{w}^\top \mathbf{S}_W \mathbf{w}$. Thật vậy, $\big(\mathbf{w}^\top(\mathbf{m}_1-\mathbf{m}_2)\big)^2 = \mathbf{w}^\top (\mathbf{m}_1-\mathbf{m}_2)(\mathbf{m}_1-\mathbf{m}_2)^\top \mathbf{w} = \mathbf{w}^\top \mathbf{S}_B \mathbf{w}$, và tổng bình phương sai lệch trong lớp $\sum_k \sum_{\mathbf{x}\in\mathcal{C}_k}\big(\mathbf{w}^\top(\mathbf{x}-\mathbf{m}_k)\big)^2 = \mathbf{w}^\top \mathbf{S}_W \mathbf{w}$.

---

# 4. Tiêu chí Fisher

Ghép hai đại lượng trên, ta được **tiêu chí Fisher (Fisher criterion)** — tỉ số giữa độ tách giữa lớp và độ tản trong lớp sau khi chiếu:

$$\boxed{\; J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}} \;}$$

Bài toán LDA là:

$$\mathbf{w}^{*} = \arg\max_{\mathbf{w}} \; J(\mathbf{w})$$

Hai ý nghĩa trực quan:

* **Tử số** $\mathbf{w}^\top \mathbf{S}_B \mathbf{w}$ lớn ⇔ hai tâm lớp xa nhau theo hướng $\mathbf{w}$ (tách lớp tốt).
* **Mẫu số** $\mathbf{w}^\top \mathbf{S}_W \mathbf{w}$ nhỏ ⇔ mỗi lớp co cụm theo hướng $\mathbf{w}$ (ít chồng lấn).

Một điểm quan trọng: $J(\mathbf{w})$ **bất biến theo độ dài** của $\mathbf{w}$. Nếu thay $\mathbf{w} \to c\mathbf{w}$ thì cả tử và mẫu đều nhân $c^2$ nên $J$ không đổi. Vậy ta chỉ quan tâm **hướng** của $\mathbf{w}$, không quan tâm độ lớn — điều này cho phép áp một ràng buộc chuẩn hóa tùy ý để giải.

---

# 5. Suy dẫn nghiệm bằng bài toán trị riêng tổng quát

## 5.1. Lấy đạo hàm tiêu chí Fisher

Đặt $a(\mathbf{w}) = \mathbf{w}^\top \mathbf{S}_B \mathbf{w}$ và $b(\mathbf{w}) = \mathbf{w}^\top \mathbf{S}_W \mathbf{w}$. Vì $\mathbf{S}_B, \mathbf{S}_W$ đối xứng, gradient của các dạng toàn phương là $\nabla_\mathbf{w} a = 2\mathbf{S}_B \mathbf{w}$ và $\nabla_\mathbf{w} b = 2\mathbf{S}_W \mathbf{w}$. Tại cực trị, $\nabla_\mathbf{w} J = 0$. Dùng quy tắc thương:

$$\nabla_\mathbf{w} J = \frac{(\nabla_\mathbf{w} a)\, b - a\,(\nabla_\mathbf{w} b)}{b^2} = \frac{2\mathbf{S}_B\mathbf{w}\, b - a\, 2\mathbf{S}_W\mathbf{w}}{b^2} = 0$$

Cho tử số bằng $0$ và chia cho $b$:

$$\mathbf{S}_B \mathbf{w} - \frac{a}{b}\, \mathbf{S}_W \mathbf{w} = 0$$

Nhận ra $\dfrac{a}{b} = J(\mathbf{w}) \equiv \lambda$ chính là giá trị của tiêu chí Fisher tại nghiệm, ta thu được **bài toán trị riêng tổng quát (generalized eigenvalue problem)**:

$$\boxed{\; \mathbf{S}_B \mathbf{w} = \lambda\, \mathbf{S}_W \mathbf{w} \;}$$

Nếu $\mathbf{S}_W$ khả nghịch, đây tương đương với bài toán trị riêng thường:

$$\mathbf{S}_W^{-1} \mathbf{S}_B\, \mathbf{w} = \lambda\, \mathbf{w}$$

Nghiệm $\mathbf{w}^{*}$ là **vector riêng (eigenvector)** ứng với **trị riêng (eigenvalue)** lớn nhất $\lambda$ — và đáng chú ý, $\lambda$ đó **chính bằng** giá trị cực đại của $J$. Vậy tối ưu Fisher quy gọn về việc tìm vector riêng trội của $\mathbf{S}_W^{-1}\mathbf{S}_B$.

## 5.2. Trường hợp hai lớp: nghiệm tường minh

Với hai lớp, ta không cần giải trị riêng mà có **công thức đóng (closed-form)**. Thay $\mathbf{S}_B = (\mathbf{m}_1 - \mathbf{m}_2)(\mathbf{m}_1 - \mathbf{m}_2)^\top$ vào phương trình $\mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$:

$$(\mathbf{m}_1 - \mathbf{m}_2)\underbrace{(\mathbf{m}_1 - \mathbf{m}_2)^\top \mathbf{w}}_{\text{vô hướng}} = \lambda\, \mathbf{S}_W \mathbf{w}$$

Đại lượng $(\mathbf{m}_1 - \mathbf{m}_2)^\top \mathbf{w}$ là một **số vô hướng (scalar)**. Vậy vế trái là một bội số của vector $(\mathbf{m}_1 - \mathbf{m}_2)$. Gọi vô hướng đó là $\alpha$, ta có $\alpha(\mathbf{m}_1 - \mathbf{m}_2) = \lambda \mathbf{S}_W \mathbf{w}$, suy ra:

$$\mathbf{w} = \frac{\alpha}{\lambda}\, \mathbf{S}_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$$

Vì $J(\mathbf{w})$ bất biến theo độ dài (mục 4), ta **bỏ qua mọi hệ số vô hướng** $\alpha/\lambda$ và thu được nghiệm:

$$\boxed{\; \mathbf{w} \;\propto\; \mathbf{S}_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2) \;} \qquad \blacksquare$$

> **Trực giác.** Nếu các lớp có hiệp phương sai **đẳng hướng** ($\mathbf{S}_W \propto \mathbf{I}$), thì $\mathbf{w} \propto \mathbf{m}_1 - \mathbf{m}_2$ — đúng đường nối hai tâm, như trực giác thô sơ. Nhân thêm $\mathbf{S}_W^{-1}$ chính là phần "khôn ngoan" của Fisher: nó **xoay** hướng nối tâm để **bù trừ** cho hình dạng tản mạn của dữ liệu, kéo lệch hướng chiếu ra khỏi những trục mà các lớp đang "loe rộng".

---

# 6. Mở rộng đa lớp (multi-class LDA)

Khi có $C$ lớp, ta cần định nghĩa lại $\mathbf{S}_B$ qua tâm chung $\mathbf{m} = \frac{1}{N}\sum_\mathbf{x}\mathbf{x}$:

$$\mathbf{S}_W = \sum_{k=1}^{C} \sum_{\mathbf{x}\in\mathcal{C}_k}(\mathbf{x}-\mathbf{m}_k)(\mathbf{x}-\mathbf{m}_k)^\top, \qquad \mathbf{S}_B = \sum_{k=1}^{C} N_k (\mathbf{m}_k - \mathbf{m})(\mathbf{m}_k - \mathbf{m})^\top$$

Bây giờ ta chiếu xuống một **không gian con** nhiều chiều qua ma trận $\mathbf{W} = [\mathbf{w}_1, \dots, \mathbf{w}_d]$, và cực đại hóa dạng tổng quát của tiêu chí Fisher (dùng vết — trace):

$$J(\mathbf{W}) = \frac{\operatorname{tr}(\mathbf{W}^\top \mathbf{S}_B \mathbf{W})}{\operatorname{tr}(\mathbf{W}^\top \mathbf{S}_W \mathbf{W})}$$

Nghiệm là $d$ vector riêng ứng với $d$ trị riêng lớn nhất của $\mathbf{S}_W^{-1}\mathbf{S}_B$.

> **Giới hạn $C-1$ chiều.** Đây là một hệ quả tinh tế nhưng quan trọng. Ma trận $\mathbf{S}_B$ là tổng của $C$ ma trận hạng một $(\mathbf{m}_k - \mathbf{m})(\mathbf{m}_k - \mathbf{m})^\top$. Nhưng $C$ vector $(\mathbf{m}_k - \mathbf{m})$ **không độc lập tuyến tính**: chúng thỏa $\sum_k N_k(\mathbf{m}_k - \mathbf{m}) = \mathbf{0}$ (vì $\mathbf{m}$ là trung bình có trọng số của các $\mathbf{m}_k$). Một ràng buộc tuyến tính làm hạng giảm một, nên $\operatorname{rank}(\mathbf{S}_B) \le C-1$. Do đó $\mathbf{S}_W^{-1}\mathbf{S}_B$ có **tối đa $C-1$ trị riêng khác $0$**, và LDA chỉ chiếu được xuống tối đa $C-1$ chiều có ý nghĩa.

Ví dụ: bài toán 10 lớp (như nhận dạng chữ số) chỉ cho ra tối đa $9$ chiều biệt thức — một sự giảm chiều rất mạnh, hoàn toàn do **cấu trúc nhãn** quyết định.

---

# 7. Giả định thống kê và liên hệ với phân loại

LDA, ngoài vai trò giảm chiều, còn là một **bộ phân loại sinh (generative classifier)** dưới một giả định cụ thể:

> **Giả định.** Mỗi lớp tuân theo phân phối **Gauss (Gaussian)** $\mathcal{N}(\mathbf{m}_k, \boldsymbol{\Sigma})$ với **cùng một ma trận hiệp phương sai (shared covariance)** $\boldsymbol{\Sigma}$ cho mọi lớp.

Khi viết quy tắc Bayes $\arg\max_k \Pr(\mathcal{C}_k \mid \mathbf{x})$ với giả định này, các số hạng bậc hai $\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}$ **triệt tiêu** giữa các lớp (vì $\boldsymbol{\Sigma}$ chung), để lại một **biên quyết định tuyến tính (linear decision boundary)** — đây chính là nguồn gốc chữ "tuyến tính" (linear) trong tên gọi. Hướng tách lớp tối ưu Bayes lúc này trùng đúng với nghiệm Fisher $\mathbf{w} \propto \boldsymbol{\Sigma}^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$, với $\boldsymbol{\Sigma}$ đóng vai trò của $\mathbf{S}_W$ (đã chuẩn hóa).

Nếu **bỏ** giả định hiệp phương sai chung (cho mỗi lớp một $\boldsymbol{\Sigma}_k$ riêng), số hạng bậc hai không triệt tiêu nữa và ta được **biệt thức bậc hai (Quadratic Discriminant Analysis - QDA)** với biên quyết định cong.

---

# 8. So sánh PCA và LDA

Cả [PCA](#/pca) và LDA đều là phép chiếu tuyến tính giảm chiều và đều quy về bài toán trị riêng, nhưng triết lý hoàn toàn khác nhau.

| Tiêu chí | PCA | LDA |
| --- | --- | --- |
| Dùng nhãn? | **Không** (unsupervised) | **Có** (supervised) |
| Tối đa hóa | Phương sai tổng $\mathbf{w}^\top \mathbf{S} \mathbf{w}$ | Tỉ số Fisher $\dfrac{\mathbf{w}^\top\mathbf{S}_B\mathbf{w}}{\mathbf{w}^\top\mathbf{S}_W\mathbf{w}}$ |
| Mục tiêu | Giữ thông tin / nén | **Tách lớp** tốt nhất |
| Bài toán trị riêng | $\mathbf{S}\mathbf{w}=\lambda\mathbf{w}$ | $\mathbf{S}_B\mathbf{w}=\lambda\mathbf{S}_W\mathbf{w}$ |
| Số chiều tối đa | $\min(D, N-1)$ | $C-1$ |
| Giả định | Không cần | Gauss, hiệp phương sai chung |
| Rủi ro | Có thể vứt hướng tách lớp | Phụ thuộc nhãn, kém khi $N$ nhỏ |

> Một thực hành phổ biến: **PCA trước, LDA sau.** Khi $D$ rất lớn so với số mẫu $N$, $\mathbf{S}_W$ dễ **suy biến (singular)** và không nghịch đảo được. Người ta thường chạy PCA để hạ chiều xuống mức $\mathbf{S}_W$ khả nghịch, rồi mới áp LDA để tối ưu khả năng tách lớp.

---

# 9. Ưu điểm

* **Tận dụng nhãn** — vì có giám sát, LDA tìm đúng hướng phân biệt lớp, thường vượt PCA cho bài toán phân loại tuyến tính tách được.
* **Nghiệm đóng, rẻ** — chỉ cần một bài toán trị riêng (hoặc công thức $\mathbf{w}\propto\mathbf{S}_W^{-1}(\mathbf{m}_1-\mathbf{m}_2)$ cho hai lớp), không cần tối ưu lặp.
* **Giảm chiều rất mạnh** — đa lớp nén thẳng về tối đa $C-1$ chiều, gọn và dễ trực quan hóa.
* **Vừa giảm chiều vừa phân loại** — dưới giả định Gauss, nghiệm Fisher trùng với bộ phân loại Bayes tối ưu (biên tuyến tính).

---

# 10. Nhược điểm

* **Phụ thuộc giả định Gauss + hiệp phương sai chung** — nếu các lớp có hình dạng tản mạn khác nhau rõ rệt, LDA tối ưu kém (nên dùng QDA).
* **$\mathbf{S}_W$ dễ suy biến** khi số chiều $D$ lớn hơn số mẫu $N$ — phải PCA trước hoặc dùng dạng chính quy hóa (regularized LDA).
* **Chỉ tuyến tính** — không tách được các lớp lồng nhau phi tuyến (cần kernel LDA cho trường hợp đó).
* **Trần $C-1$ chiều** — đôi khi quá ít chiều để giữ đủ thông tin, đặc biệt khi số lớp nhỏ (hai lớp ⇒ chỉ một chiều duy nhất).

---

# 11. Tổng kết

LDA trả lời câu hỏi mà [PCA](#/pca) không quan tâm: **khi đã biết nhãn, hướng chiếu nào tốt nhất để phân biệt các lớp?** Câu trả lời gói trong tiêu chí Fisher — cực đại tỉ số giữa độ tách giữa lớp $\mathbf{S}_B$ và độ tản trong lớp $\mathbf{S}_W$:

$$J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}} \;\longrightarrow\; \mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}$$

Toàn bộ phương pháp quy về một bài toán **trị riêng tổng quát**, với trường hợp hai lớp cho nghiệm đẹp đẽ $\mathbf{w} \propto \mathbf{S}_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$. Phép nhân $\mathbf{S}_W^{-1}$ — bù trừ hình dạng tản mạn của dữ liệu — chính là điểm tinh tế phân biệt Fisher với cách "nối hai tâm" ngây thơ.

LDA đẹp vì **đơn giản, có nghiệm đóng, và có nền tảng xác suất** (bộ phân loại Bayes Gauss). Cái giá là các giả định mạnh và trần $C-1$ chiều. Khi giả định bị vi phạm, ta nới sang QDA (hiệp phương sai riêng) hay kernel LDA (phi tuyến) — nhưng tinh thần "tối đa hóa tỉ số tách-lớp trên tản-lớp" vẫn là sợi chỉ đỏ xuyên suốt.

> Bài tiếp theo sẽ bàn về **máy vector hỗ trợ (Support Vector Machine - SVM)** — thay vì giả định phân phối Gauss, nó tìm biên quyết định trực tiếp bằng cách cực đại hóa **lề (margin)** giữa các lớp.

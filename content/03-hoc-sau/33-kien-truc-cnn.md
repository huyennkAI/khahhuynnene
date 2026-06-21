# Các kiến trúc CNN (CNN Architectures)

> Lịch sử mạng nơ-ron tích chập là một câu chuyện về việc **xếp chồng các lớp ngày một sâu hơn** mà vẫn huấn luyện được. Mỗi kiến trúc lớn ra đời để gỡ một nút thắt của kiến trúc trước đó.
>
> $$\text{LeNet} \rightarrow \text{AlexNet} \rightarrow \text{VGG} \rightarrow \text{NiN} \rightarrow \text{GoogLeNet} \rightarrow \text{ResNet} \rightarrow \text{DenseNet}$$
>
> Đây là xương sống của thị giác máy tính hiện đại, và nhiều ý tưởng ở đây (khối lặp, tích chập $1\times 1$, kết nối tắt) còn lan sang cả Transformer.

---

# 1. Vì sao cần "kiến trúc"?

Một mạng tích chập về cơ bản chỉ gồm vài thành phần nguyên thủy: tầng [tích chập](#/tich-chap), tầng [pooling](#/kenh-pooling), hàm kích hoạt phi tuyến, và vài tầng kết nối đầy đủ (fully connected) ở cuối. Câu hỏi cốt lõi không phải là "dùng phép toán nào" mà là:

> Xếp chồng các thành phần đó **theo cấu trúc nào** để mạng vừa sâu (biểu diễn mạnh) vừa **huấn luyện được** (gradient không tan biến, tham số không bùng nổ)?

Trực giác chủ đạo xuyên suốt lịch sử: càng lên cao, đặc trưng càng **trừu tượng** (cạnh → kết cấu → bộ phận → vật thể) trong khi độ phân giải không gian **giảm dần** còn số kênh **tăng dần**. Pooling và stride thu nhỏ chiều rộng–cao; số bộ lọc tăng lên để bù lại lượng thông tin.

Mỗi kiến trúc dưới đây là một câu trả lời cho cùng một câu hỏi trên, trong bối cảnh phần cứng và lý thuyết của thời nó ra đời.

---

# 2. LeNet — bản thiết kế gốc

LeNet (LeCun và cộng sự, 1998) là CNN thực tế đầu tiên, dùng để nhận dạng chữ số viết tay trên séc ngân hàng. Dù chỉ vài nghìn tham số, nó đã định hình **khuôn mẫu** mà mọi kiến trúc sau kế thừa:

$$\underbrace{(\text{Conv} \rightarrow \text{Sigmoid} \rightarrow \text{Pool})}_{\text{thân trích đặc trưng}} \times 2 \;\rightarrow\; \underbrace{\text{FC} \rightarrow \text{FC} \rightarrow \text{FC}}_{\text{đầu phân loại}}$$

Ý tưởng cốt lõi: thay vì làm phẳng ảnh rồi đưa vào mạng dày đặc (như perceptron), hãy dùng **tích chập có chia sẻ trọng số** để khai thác cấu trúc không gian và tính bất biến tịnh tiến (translation invariance). Khối thân gồm hai tầng tích chập xen kẽ pooling rút trích đặc trưng; khối đầu gồm các tầng FC để phân loại.

Đóng góp: chứng minh rằng đặc trưng **học được** từ dữ liệu vượt trội đặc trưng thủ công. Hạn chế: dùng sigmoid (dễ bão hòa) và quá nông cho ảnh tự nhiên độ phân giải cao — phần cứng năm 1998 chưa cho phép đi sâu hơn.

---

# 3. AlexNet — đột phá ImageNet 2012

AlexNet (Krizhevsky, Sutskever, Hinton, 2012) về **cấu trúc** chỉ là một LeNet phình to, nhưng nó thắng cuộc thi ImageNet 2012 với cách biệt khổng lồ (top-5 error rớt từ ~26% xuống ~15%), mở màn kỷ nguyên học sâu. Bí quyết không nằm ở một phát kiến đơn lẻ mà ở việc kết hợp đúng lúc nhiều yếu tố:

- **Sâu và rộng hơn nhiều**: 5 tầng tích chập + 3 tầng FC, với số kênh lớn.
- **ReLU thay sigmoid**: $\sigma(x)=\max(0,x)$ không bão hòa ở miền dương, gradient không tan biến, huấn luyện nhanh hơn nhiều lần. Đây có lẽ là thay đổi quan trọng nhất.
- **Dropout** ở các tầng FC: ngẫu nhiên tắt nơ-ron khi huấn luyện để chống quá khớp (overfitting) cho mô hình hàng chục triệu tham số.
- **Tăng cường dữ liệu (data augmentation)**: lật, cắt, dịch màu để mở rộng tập huấn luyện.
- **Huấn luyện trên GPU**: cho phép xử lý mạng lớn trên ImageNet trong thời gian khả thi.

> Bài học của AlexNet: không phải ý tưởng mới, mà là **quy mô** (dữ liệu + tính toán + độ sâu) đúng thời điểm. Nó biến học sâu từ một ý tưởng học thuật thành công cụ thống trị.

Hạn chế: thiết kế còn **thủ công và không đều** — kích thước bộ lọc khác nhau ($11\times 11$, $5\times 5$, $3\times 3$), khó nhân rộng một cách có hệ thống.

---

# 4. VGG — sức mạnh của khối lặp đều đặn

VGG (Simonyan & Zisserman, 2014) đặt câu hỏi: thay vì thiết kế từng tầng một cách thủ công, sao không định nghĩa một **khối (block)** rồi lặp lại nó?

**Khối VGG** = nhiều tầng tích chập $3\times 3$ (giữ nguyên kích thước nhờ padding) nối tiếp, kết thúc bằng một tầng max-pooling $2\times 2$ để giảm nửa độ phân giải. Mạng VGG chỉ đơn giản là xếp chồng các khối này, mỗi lần qua một khối thì độ phân giải giảm nửa và số kênh tăng gấp đôi.

**Vì sao chỉ dùng $3\times 3$?** Đây là suy dẫn then chốt. Hai tầng $3\times 3$ chồng lên nhau có cùng **vùng tiếp nhận (receptive field)** $5\times 5$ như một tầng $5\times 5$ đơn lẻ, nhưng:

$$\underbrace{2\times(3\times 3 \times C^2)}_{\text{hai tầng }3\times3} = 18\,C^2 \quad < \quad \underbrace{5\times 5 \times C^2}_{\text{một tầng }5\times5} = 25\,C^2$$

(với $C$ kênh vào = $C$ kênh ra). Hai tầng nhỏ vừa **ít tham số hơn**, vừa chèn thêm **một phi tuyến** ở giữa, nên biểu diễn giàu hơn. Suy ra: **nhiều tầng nhỏ tốt hơn ít tầng lớn**.

Đóng góp: chứng minh thiết kế **đều đặn, mô-đun hóa** có thể đẩy độ sâu lên 16–19 tầng và vẫn dễ mở rộng. Hạn chế: rất tốn bộ nhớ và tham số (phần lớn nằm ở tầng FC cuối).

---

# 5. NiN — tích chập $1\times 1$ và global average pooling

Network in Network (Lin và cộng sự, 2013) tấn công hai vấn đề: (1) tầng FC khổng lồ ở cuối VGG/AlexNet, và (2) mỗi tầng tích chập chỉ là một biến đổi tuyến tính rồi phi tuyến điểm.

**Tích chập $1\times 1$.** Một bộ lọc $1\times 1$ không nhìn vùng lân cận không gian — nó chỉ trộn **giữa các kênh** tại từng vị trí điểm ảnh. Hãy nghĩ nó như một tầng kết nối đầy đủ áp dụng độc lập cho từng pixel:

$$y_{i,j} = \sigma\!\big(\mathbf{W}\,\mathbf{x}_{i,j} + \mathbf{b}\big), \qquad \mathbf{x}_{i,j} \in \mathbb{R}^{C_{\text{in}}},\ \mathbf{W}\in\mathbb{R}^{C_{\text{out}}\times C_{\text{in}}}$$

Nó cho phép tăng/giảm số kênh và thêm phi tuyến **mà không tốn thêm phép tính không gian** — một "MLP nhỏ trên mỗi pixel", từ đó cái tên "mạng trong mạng".

**Global Average Pooling (GAP).** Thay vì làm phẳng bản đồ đặc trưng rồi nối vào FC nặng nề, NiN tính **trung bình toàn cục** mỗi kênh thành một số, cho ra vector dài đúng bằng số lớp:

$$z_c = \frac{1}{H\cdot W}\sum_{i=1}^{H}\sum_{j=1}^{W} x_{c,i,j}$$

Lợi ích: gần như **không có tham số** ở đầu phân loại (chống quá khớp mạnh), và mỗi kênh được diễn giải tự nhiên như "mức độ hiện diện của một lớp". Hai ý tưởng này — tích chập $1\times 1$ và GAP — trở thành thành phần chuẩn của mọi kiến trúc sau.

---

# 6. GoogLeNet / Inception — đa tỉ lệ song song

GoogLeNet (Szegedy và cộng sự, 2014) đặt câu hỏi: nên dùng bộ lọc kích thước nào — $3\times 3$, $5\times 5$, hay pooling? Câu trả lời táo bạo: **dùng tất cả cùng lúc**, rồi để mạng tự học cách phối hợp.

**Khối Inception** chạy nhiều nhánh song song trên cùng đầu vào rồi **nối kết quả theo chiều kênh (concatenate)**:

- nhánh tích chập $1\times 1$,
- nhánh $1\times 1 \rightarrow 3\times 3$,
- nhánh $1\times 1 \rightarrow 5\times 5$,
- nhánh max-pool $\rightarrow 1\times 1$.

Trực giác: vật thể xuất hiện ở **nhiều tỉ lệ** khác nhau; nhánh $1\times 1$ bắt chi tiết nhỏ, nhánh $5\times 5$ bắt cấu trúc lớn. Mạng không phải chọn một tỉ lệ — nó nhìn mọi tỉ lệ và kết hợp.

Mấu chốt hiệu năng: các tầng [tích chập](#/tich-chap) $1\times 1$ (mượn từ NiN) đặt **trước** các nhánh tốn kém để **giảm số kênh** (bottleneck), nhờ đó khối đa nhánh vẫn rẻ. GoogLeNet đạt độ chính xác ngang VGG nhưng với **ít hơn ~12 lần tham số** nhờ GAP và bottleneck.

Đóng góp: chứng minh kiến trúc **không nhất thiết tuần tự** — phân nhánh và hợp nhất là một công cụ thiết kế mạnh.

---

# 7. ResNet — học ánh xạ phần dư

Đây là kiến trúc quan trọng nhất của cả chương, vì nó gỡ nút thắt khiến mọi mạng trước đó không thể đi sâu hơn vài chục tầng.

## 7.1. Nghịch lý suy thoái (degradation)

Trực giác ngây thơ nói rằng mạng càng sâu càng mạnh: một mạng 56 tầng ít nhất phải làm được mọi điều một mạng 20 tầng làm được. Nhưng thực nghiệm cho điều ngược lại — mạng sâu hơn cho **sai số huấn luyện cao hơn**. Đây không phải quá khớp (training error tăng chứ không phải test error), mà là **khó tối ưu**: gradient tan biến/bùng nổ khiến các tầng sâu không học nổi.

> Lập luận then chốt: nếu các tầng thêm vào chỉ cần học **ánh xạ đồng nhất (identity)** thì mạng sâu không thể tệ hơn mạng nông. Vậy mà tối ưu hóa lại không tìm ra được ánh xạ đồng nhất đó. Vấn đề nằm ở **dạng hàm ta bắt mạng học**.

## 7.2. Lớp hàm lồng nhau (nested function classes)

Gọi $\mathcal{F}_k$ là tập các hàm mà một mạng $k$ tầng biểu diễn được. Với kiến trúc thường, $\mathcal{F}_k$ và $\mathcal{F}_{k+1}$ **không nhất thiết lồng nhau**: thêm tầng có thể đẩy ta tới một họ hàm khác hẳn, **xa hơn** hàm tối ưu $f^\star$. Đó là lý do sâu hơn lại tệ hơn.

Ý tưởng của ResNet: thiết kế sao cho **$\mathcal{F}_k \subseteq \mathcal{F}_{k+1}$** — họ hàm lớn hơn luôn **chứa** họ hàm nhỏ hơn. Khi đó thêm tầng không bao giờ làm hại, vì trong trường hợp xấu nhất tầng mới chỉ cần trở thành đồng nhất. Muốn vậy, ta phải khiến việc **học hàm đồng nhất trở nên dễ dàng**.

## 7.3. Khối phần dư (residual block)

Thay vì bắt một chồng tầng học trực tiếp ánh xạ mong muốn $H(\mathbf{x})$, ResNet để nó học phần **lệch** so với đầu vào, gọi là phần dư (residual):

$$\mathcal{F}(\mathbf{x}) := H(\mathbf{x}) - \mathbf{x} \qquad\Longrightarrow\qquad \mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

trong đó $\mathcal{F}$ là vài tầng tích chập (thường hai tầng $3\times 3$ kèm [batch normalization](#/batch-norm) và ReLU), còn $+\,\mathbf{x}$ là **kết nối tắt (skip / shortcut connection)** cộng thẳng đầu vào vào đầu ra.

**Vì sao dễ hơn?** Nếu ánh xạ tối ưu gần với đồng nhất (điều rất thường gặp ở các tầng sâu, nơi chỉ cần tinh chỉnh nhẹ), thì mạng chỉ cần **đẩy $\mathcal{F}(\mathbf{x})$ về $0$** — tức ép các trọng số về gần $0$, điều mà weight decay và khởi tạo vốn đã thiên về. Học "không làm gì" (residual = 0) dễ hơn nhiều so với học một phép đồng nhất phức tạp qua các phép phi tuyến chồng chất. Đây chính là cơ chế bảo đảm lớp hàm lồng nhau ở mục 7.2.

## 7.4. Gradient chảy qua kết nối tắt

Lợi ích thứ hai mang tính cơ học. Xét lan truyền ngược qua một khối phần dư. Vì $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}} + \mathbf{I}$$

Gọi $L$ là hàm mất mát, theo quy tắc chuỗi gradient truyền về đầu vào là:

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}}\left(\mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right) = \underbrace{\frac{\partial L}{\partial \mathbf{y}}}_{\text{đường tắt}} + \underbrace{\frac{\partial L}{\partial \mathbf{y}}\frac{\partial \mathcal{F}}{\partial \mathbf{x}}}_{\text{đường tích chập}}$$

Số hạng $\mathbf{I}$ tạo ra một **đường cao tốc cho gradient**: ngay cả khi $\tfrac{\partial \mathcal{F}}{\partial \mathbf{x}}$ rất nhỏ (gradient tan biến trong nhánh tích chập), gradient vẫn **chảy nguyên vẹn** về tầng trước qua $\mathbf{I}$. Khi xếp chồng nhiều khối, đạo hàm trở thành **tổng** các đường đi thay vì **tích** dài các Jacobian — nên không bị triệt tiêu theo cấp số nhân.

> Đây là lý do ResNet huấn luyện được mạng **152 tầng** (và sâu hơn nữa), điều bất khả thi trước đó. Kết nối tắt vừa làm họ hàm lồng nhau, vừa giữ gradient sống.

Khi số kênh hoặc độ phân giải thay đổi giữa hai bên phép cộng, ta dùng một tầng tích chập $1\times 1$ ở nhánh tắt để khớp chiều. Biến thể **bottleneck** ($1\times1 \rightarrow 3\times3 \rightarrow 1\times1$) giảm chi phí cho các mạng rất sâu.

---

# 8. DenseNet — nối kênh thay vì cộng

DenseNet (Huang và cộng sự, 2017) đẩy ý tưởng kết nối tắt đi xa hơn. ResNet **cộng** đầu vào vào đầu ra; DenseNet **nối (concatenate)** mọi đầu ra của các tầng trước theo chiều kênh:

$$\mathbf{x}_\ell = H_\ell\big([\,\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{\ell-1}\,]\big)$$

trong đó $[\cdot]$ là phép nối kênh. Mỗi tầng nhận **toàn bộ** đặc trưng của mọi tầng trước làm đầu vào.

Sự khác biệt tinh tế nhưng quan trọng: phép cộng của ResNet **trộn** thông tin (có thể chồng lấp/triệt tiêu), còn phép nối của DenseNet **bảo toàn** mọi đặc trưng nguyên gốc, để các tầng sau tự chọn dùng. Triển khai khai triển công thức trên cho thấy mỗi tầng đều có đường nối thẳng tới hàm mất mát, nên gradient chảy còn mượt hơn ResNet, và đặc trưng được **tái sử dụng** tối đa.

Mỗi tầng chỉ thêm $k$ kênh mới (gọi là **tốc độ tăng trưởng, growth rate**), nên dù kết nối dày đặc, tổng tham số lại **ít hơn** ResNet ở cùng độ chính xác. Giữa các khối dày (dense block) là **tầng chuyển tiếp (transition layer)** dùng tích chập $1\times 1$ + [pooling](#/kenh-pooling) để nén số kênh và giảm độ phân giải, tránh bùng nổ chiều.

---

# 9. Bảng so sánh kiến trúc

| Kiến trúc | Năm | Ý tưởng cốt lõi | Đóng góp chính | Độ sâu điển hình |
| --- | --- | --- | --- | --- |
| **LeNet** | 1998 | Conv + pooling + FC | CNN thực tế đầu tiên, đặc trưng học được | ~5–7 tầng |
| **AlexNet** | 2012 | LeNet phình to | ReLU, dropout, GPU, thắng ImageNet | 8 tầng |
| **VGG** | 2014 | Khối lặp $3\times 3$ | Thiết kế đều, mô-đun hóa độ sâu | 16–19 tầng |
| **NiN** | 2013 | Tích chập $1\times 1$ + GAP | Trộn kênh, bỏ FC nặng nề | ~12 tầng |
| **GoogLeNet** | 2014 | Khối Inception đa nhánh | Đa tỉ lệ song song, bottleneck | 22 tầng |
| **ResNet** | 2015 | Khối phần dư $\mathbf{y}=\mathcal{F}(\mathbf{x})+\mathbf{x}$ | Kết nối tắt, huấn luyện mạng rất sâu | 18–152 tầng |
| **DenseNet** | 2017 | Nối kênh mọi tầng trước | Tái dùng đặc trưng, ít tham số | 121–264 tầng |

Đọc bảng theo chiều dọc cột "Độ sâu" để thấy mạch tiến hóa: mỗi kiến trúc gỡ một rào cản để tầng có thể chồng cao hơn — từ ReLU (chống bão hòa), tới khối đều (mở rộng có hệ thống), tới kết nối tắt (chống gradient tan biến).

---

# 10. Mạch logic xuyên suốt

Nhìn lại toàn cảnh, ba sợi chỉ đỏ nối các kiến trúc:

- **Mô-đun hóa.** Từ thiết kế thủ công (LeNet, AlexNet) sang khối lặp lại (VGG), khối đa nhánh (Inception), rồi khối phần dư (ResNet). Tư duy "thiết kế khối, lặp khối" thắng tư duy "thiết kế từng tầng".
- **Giảm tham số mà không giảm sức biểu diễn.** Tích chập $1\times 1$, GAP (NiN), bottleneck (GoogLeNet, ResNet), growth rate (DenseNet) đều nhắm vào cùng mục tiêu: sâu và mạnh nhưng gọn.
- **Giữ cho gradient sống.** ReLU rồi [batch normalization](#/batch-norm), và đỉnh cao là kết nối tắt — yếu tố quyết định cho phép độ sâu vượt 100 tầng.

> Kết nối tắt của ResNet không chỉ là một thủ thuật CNN: cùng cơ chế (residual + chuẩn hóa) là thành phần lõi của mỗi khối Transformer. Hiểu sâu mục 7 là hiểu nửa kiến trúc của mọi mô hình hiện đại.

---

# 11. Tổng kết

Lịch sử kiến trúc CNN là quá trình **học cách đi sâu**. LeNet đặt khuôn mẫu Conv–Pool–FC. AlexNet chứng minh quy mô + ReLU + GPU mở ra kỷ nguyên học sâu. VGG cho thấy khối $3\times 3$ lặp đều dễ mở rộng. NiN giới thiệu tích chập $1\times 1$ và GAP để bỏ FC nặng nề. GoogLeNet phân nhánh đa tỉ lệ và dùng bottleneck cho hiệu quả.

Bước ngoặt là **ResNet**: bằng cách để mạng học **phần dư** thay vì ánh xạ trực tiếp,

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x},$$

nó vừa bảo đảm **lớp hàm lồng nhau** (thêm tầng không bao giờ hại), vừa tạo **đường cao tốc cho gradient** (đạo hàm có số hạng $\mathbf{I}$ chống tan biến). Nhờ đó mạng hàng trăm tầng trở nên huấn luyện được. DenseNet tinh chỉnh ý tưởng này bằng phép **nối kênh**, đẩy việc tái sử dụng đặc trưng và dòng chảy gradient lên cao hơn nữa.

Cốt lõi cần nhớ: chiều sâu chỉ hữu ích khi ta **kiểm soát được dòng gradient và dạng họ hàm**. Mọi kiến trúc lớn đều là một cách trả lời câu hỏi đó — và câu trả lời hay nhất, kết nối tắt, đã vượt khỏi CNN để trở thành nền tảng của cả Transformer.

> Bài tiếp theo sẽ đào sâu vào **[batch normalization](#/batch-norm)** — kỹ thuật chuẩn hóa kích hoạt từng lô, một mảnh ghép không thể thiếu để các khối phần dư hoạt động ổn định trong các mạng rất sâu.

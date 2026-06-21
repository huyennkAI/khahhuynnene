# Chuẩn hóa theo batch (Batch Normalization)

> Chuẩn hóa theo batch (batch normalization, BN) chèn vào giữa các tầng một bước **chuẩn hóa kích hoạt (activation)** rồi **co giãn lại bằng tham số học**. Nhờ đó mạng sâu huấn luyện nhanh hơn nhiều, ổn định hơn, và cho phép dùng tốc độ học (learning rate) lớn.
>
> $$\mathbf{x} \;\rightarrow\; \hat{\mathbf{x}} = \frac{\mathbf{x} - \hat{\mu}_B}{\sqrt{\hat{\sigma}_B^2 + \epsilon}} \;\rightarrow\; \mathbf{y} = \gamma \odot \hat{\mathbf{x}} + \beta$$
>
> Đây là một trong những kỹ thuật đã giúp việc huấn luyện mạng rất sâu trở nên khả thi trong thực tế.

---

# 1. Vì sao cần chuẩn hóa giữa các tầng?

Một mạng sâu là một chồng nhiều phép biến đổi nối tiếp. Mỗi tầng nhận đầu vào là **đầu ra của tầng phía trước**. Khi huấn luyện, ta cập nhật mọi trọng số cùng lúc, nên phân phối đầu ra của tầng dưới **thay đổi liên tục** qua từng bước. Tầng phía trên do đó luôn phải đuổi theo một mục tiêu đang dịch chuyển.

Hiện tượng này thường được gọi là **dịch chuyển hiệp biến nội tại (internal covariate shift)**: phân phối của đầu vào ở mỗi tầng cứ trôi đi trong lúc tham số bên dưới còn đang học. Khi phân phối đó trôi quá xa, các kích hoạt dễ rơi vào vùng bão hòa của hàm phi tuyến hoặc làm độ lớn gradient dao động mạnh, khiến quá trình tối ưu (xem [Tối ưu trong học sâu](#/toi-uu-hoc-sau)) trở nên chậm và nhạy cảm với lựa chọn learning rate.

Trực giác của BN rất đơn giản:

> Nếu phân phối đầu vào trôi đi gây khó, thì hãy **cố định lại các thống kê bậc thấp** (trung bình và phương sai) của đầu vào mỗi tầng. Sau khi chuẩn hóa, mỗi tầng luôn nhìn thấy đầu vào có cùng "thước đo" — trung bình $0$, phương sai $1$ — bất kể tầng dưới đang thay đổi ra sao.

**Một tranh luận hiện đại.** Cách giải thích bằng internal covariate shift là động cơ ban đầu (Ioffe & Szegedy, 2015), nhưng các nghiên cứu sau (Santurkar và cộng sự, 2018) cho thấy lý do thực sự có lẽ khác: BN **làm mượt mặt mất mát (loss landscape)**. Cụ thể, nó làm cho gradient ổn định và có tính Lipschitz tốt hơn — độ lớn và hướng của gradient ít thay đổi đột ngột — nên mỗi bước cập nhật đáng tin cậy hơn và ta được phép bước dài hơn. Dù lời giải thích nào đúng, hiệu quả thực nghiệm là không bàn cãi.

---

# 2. Công thức chuẩn hóa theo batch

Xét một minibatch $B = \{\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(m)}\}$ gồm $m$ mẫu đi vào một tầng. BN gồm hai bước: **chuẩn hóa** rồi **co giãn–dịch chuyển**.

## 2.1. Bước chuẩn hóa

Trước hết tính trung bình và phương sai **theo batch**:

$$\hat{\mu}_B = \frac{1}{m}\sum_{i=1}^{m} \mathbf{x}^{(i)}, \qquad \hat{\sigma}_B^2 = \frac{1}{m}\sum_{i=1}^{m} \left(\mathbf{x}^{(i)} - \hat{\mu}_B\right)^2$$

Rồi chuẩn hóa từng mẫu về trung bình $0$, phương sai $1$:

$$\hat{\mathbf{x}}^{(i)} = \frac{\mathbf{x}^{(i)} - \hat{\mu}_B}{\sqrt{\hat{\sigma}_B^2 + \epsilon}}$$

Hằng số nhỏ $\epsilon$ (thường $10^{-5}$) đặt trong căn để tránh chia cho $0$ khi phương sai rất nhỏ — đây là một tiểu xảo về [ổn định số học](#/on-dinh-so) quan trọng, vì $\hat{\sigma}_B^2$ có thể tiến gần $0$ trên một batch mà các mẫu gần như giống nhau.

## 2.2. Bước co giãn và dịch chuyển

Nếu chỉ dừng ở chuẩn hóa, ta đã **ép mọi tầng phải có đầu ra trung bình $0$, phương sai $1$**, làm mất sức biểu diễn (representational power): chẳng hạn một sigmoid bị giam trong vùng gần như tuyến tính quanh gốc. Để khôi phục, BN thêm hai tham số **học được** $\gamma$ và $\beta$ (cùng số chiều với $\hat{\mathbf{x}}$):

$$\mathbf{y}^{(i)} = \gamma \odot \hat{\mathbf{x}}^{(i)} + \beta$$

trong đó $\odot$ là phép nhân theo từng thành phần. Cặp $(\gamma, \beta)$ cho phép mạng tự chọn lại trung bình và độ lệch chuẩn phù hợp cho mỗi đặc trưng. Điểm tinh tế: nếu $\gamma = \sqrt{\hat{\sigma}_B^2 + \epsilon}$ và $\beta = \hat{\mu}_B$ thì BN khôi phục **đúng** đầu vào gốc. Nghĩa là phép biến đổi đồng nhất nằm trong không gian biểu diễn của BN — chuẩn hóa không bao giờ làm giảm sức mạnh của mạng, nó chỉ đổi cách tham số hóa để dễ tối ưu hơn.

> Tóm lại: BN trước hết tẩy đi trung bình và phương sai "ngẫu nhiên" do tầng dưới gây ra, sau đó để mạng **học lại** đúng mức trung bình và phương sai mà nó thực sự cần qua $\gamma, \beta$.

---

# 3. Khác biệt giữa huấn luyện và suy luận

Đây là chi tiết dễ gây nhầm lẫn nhất của BN.

**Khi huấn luyện (training).** $\hat{\mu}_B$ và $\hat{\sigma}_B^2$ được tính trực tiếp **trên batch hiện tại**. Vì batch là ngẫu nhiên, các thống kê này nhiễu nhẹ từ bước này sang bước khác — và chính nhiễu đó là một nguồn điều chuẩn (mục 5).

**Khi suy luận (inference).** Lúc dự đoán, ta có thể chỉ có **một mẫu** hoặc muốn đầu ra **tất định** (không phụ thuộc các mẫu khác tình cờ nằm chung batch). Không thể dùng thống kê batch nữa. Giải pháp: trong lúc huấn luyện, duy trì một **trung bình trượt (running mean/variance)** ước lượng thống kê của toàn bộ dữ liệu:

$$\mu_{\text{run}} \leftarrow (1 - \rho)\,\mu_{\text{run}} + \rho\,\hat{\mu}_B, \qquad \sigma_{\text{run}}^2 \leftarrow (1 - \rho)\,\sigma_{\text{run}}^2 + \rho\,\hat{\sigma}_B^2$$

với $\rho$ là hệ số động lượng (momentum) nhỏ. Khi suy luận, BN dùng cặp cố định này thay cho thống kê batch:

$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu_{\text{run}}}{\sqrt{\sigma_{\text{run}}^2 + \epsilon}}, \qquad \mathbf{y} = \gamma \odot \hat{\mathbf{x}} + \beta$$

> Hệ quả thực hành: tầng BN có hai chế độ khác nhau. Quên chuyển mô hình sang chế độ đánh giá khi suy luận (ví dụ `model.eval()` trong PyTorch) là một lỗi kinh điển khiến kết quả test tệ bất thường, vì khi đó nó vẫn dùng thống kê của batch test.

---

# 4. BN cho tầng kết nối đầy đủ và tầng tích chập

Câu hỏi mấu chốt: **chuẩn hóa theo chiều nào?** Câu trả lời khác nhau giữa hai loại tầng.

## 4.1. Tầng kết nối đầy đủ (fully-connected)

Với tầng FC, đầu vào là ma trận kích thước $m \times d$ ($m$ mẫu, $d$ đặc trưng). Ta chuẩn hóa **theo từng đặc trưng**: lấy trung bình và phương sai dọc theo chiều batch (cộng qua $m$ mẫu), cho mỗi cột một cặp $(\hat{\mu}_B, \hat{\sigma}_B^2)$. Do đó $\gamma, \beta$ là các vector dài $d$ — mỗi đặc trưng một cặp tham số. BN thường được đặt **ngay sau phép biến đổi tuyến tính, trước hàm kích hoạt**:

$$\mathbf{y} = \phi\big(\text{BN}(\mathbf{W}\mathbf{x} + \mathbf{b})\big)$$

(Khi đã có BN, hạng tử thiên lệch $\mathbf{b}$ trở nên thừa vì $\beta$ đã đảm nhiệm việc dịch chuyển.)

## 4.2. Tầng tích chập (convolution)

Với tầng tích chập (xem [Kiến trúc CNN](#/kien-truc-cnn)), đầu ra có dạng $m \times c \times h \times w$ (batch, kênh, cao, rộng). Nguyên tắc: mọi vị trí không gian dùng **chung một bộ lọc**, nên phải được chuẩn hóa **cùng một cách**. Vì thế BN chuẩn hóa **theo kênh (per-channel)**: với mỗi kênh, gộp tất cả các phần tử trên cả $m$ mẫu và toàn bộ $h \times w$ vị trí không gian thành một tập, rồi tính một $\hat{\mu}_B$ và một $\hat{\sigma}_B^2$ cho kênh đó.

$$\hat{\mu}_c = \frac{1}{m\,h\,w}\sum_{i,j,k} x_{i,c,j,k}$$

Như vậy có $c$ cặp $(\gamma, \beta)$ — mỗi **kênh** (chứ không phải mỗi pixel) một cặp. Cách gộp này vừa tôn trọng tính bất biến tịnh tiến của tích chập, vừa cho số mẫu thống kê lớn ($m \cdot h \cdot w$) nên ước lượng phương sai rất ổn định.

| Loại tầng | Hình dạng | Chuẩn hóa theo | Số cặp $(\gamma,\beta)$ |
| --- | --- | --- | --- |
| Kết nối đầy đủ | $m \times d$ | mỗi đặc trưng (gộp $m$) | $d$ |
| Tích chập | $m \times c \times h \times w$ | mỗi kênh (gộp $m,h,w$) | $c$ |

---

# 5. Vì sao BN giúp huấn luyện tốt hơn?

## 5.1. Cho phép learning rate lớn hơn và ổn định hơn

Khi đầu vào mỗi tầng được giữ ở thang đo cố định, độ lớn của gradient không còn bị thổi phồng hay teo lại do thang đo của tầng dưới. Một tính chất đẹp: BN khiến tầng **bất biến với việc co giãn trọng số**. Nếu nhân trọng số $\mathbf{W}$ với một hằng số $a$, thì $\mathbf{W}\mathbf{x}$ phóng to $a$ lần nhưng bước chuẩn hóa chia lại đúng cho độ lệch chuẩn (cũng phóng $a$ lần), nên đầu ra $\hat{\mathbf{x}}$ **không đổi**:

$$\text{BN}(a\mathbf{W}\mathbf{x}) = \text{BN}(\mathbf{W}\mathbf{x})$$

Hệ quả là gradient theo $\mathbf{W}$ tự động giảm tỉ lệ $1/a$, tạo một cơ chế **tự điều chỉnh bước đi**. Mặt mất mát trở nên mượt hơn, nên bước lớn ít gây bùng nổ — ta được phép dùng learning rate lớn và hội tụ nhanh hơn nhiều. Đây cũng là lý do BN gắn chặt với chủ đề [ổn định số học](#/on-dinh-so) và [tối ưu trong học sâu](#/toi-uu-hoc-sau).

## 5.2. Hiệu ứng điều chuẩn nhẹ

Vì $\hat{\mu}_B, \hat{\sigma}_B^2$ tính trên **một batch ngẫu nhiên**, mỗi mẫu được chuẩn hóa bởi những thống kê hơi khác nhau tùy nó rơi vào batch nào. Đây là một dạng **nhiễu ngẫu nhiên** thêm vào kích hoạt, tác dụng tương tự một bộ điều chuẩn (xem [Điều chuẩn trong học sâu](#/dieu-chuan-hoc-sau)) — giống tinh thần của dropout. Nhờ đó nhiều mạng dùng BN có thể giảm hoặc bỏ hẳn dropout. Lưu ý: hiệu ứng này **mạnh lên khi batch nhỏ** (nhiễu lớn hơn) và **yếu đi khi batch lớn**.

---

# 6. Layer normalization và họ hàng

BN có một điểm yếu cố hữu: nó **phụ thuộc vào kích thước và thành phần batch**. Khi batch quá nhỏ (ví dụ huấn luyện trên ảnh độ phân giải cao, mỗi GPU chỉ chứa vài mẫu), thống kê batch nhiễu đến mức gây hại; và với dữ liệu chuỗi có độ dài thay đổi, việc gộp theo batch trở nên vụng về.

**Chuẩn hóa theo tầng (layer normalization, LN)** sửa điều này bằng cách đổi chiều chuẩn hóa: thay vì gộp qua batch, LN chuẩn hóa **qua các đặc trưng của riêng từng mẫu**.

$$\hat{\mu} = \frac{1}{d}\sum_{j=1}^{d} x_j, \qquad \hat{\sigma}^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j - \hat{\mu})^2$$

Vì thống kê tính trong nội bộ một mẫu, LN **không phụ thuộc batch** và hoạt động **giống hệt nhau khi huấn luyện và suy luận** (không cần running mean/var). Đó là lý do LN trở thành lựa chọn chuẩn trong **Transformer** và các mô hình ngôn ngữ, nơi độ dài chuỗi thay đổi và batch hiệu dụng theo từng vị trí thường nhỏ.

```python
import torch.nn as nn

# BN cho ảnh: chuẩn hóa theo kênh, cần theo dõi running stats
net_cnn = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),   # 16 cặp (gamma, beta), một cặp mỗi kênh
    nn.ReLU(),
)

# LN cho chuỗi: chuẩn hóa theo đặc trưng của từng token
ln = nn.LayerNorm(512)    # không phụ thuộc batch, train = eval
```

---

# 7. Ưu điểm và lưu ý

**Ưu điểm.**

* **Huấn luyện nhanh và ổn định** — cho phép learning rate lớn, giảm độ nhạy với khởi tạo, rút ngắn đáng kể số epoch cần thiết.
* **Mặt mất mát mượt hơn** — gradient ổn định, mỗi bước cập nhật đáng tin cậy hơn (mục 5.1).
* **Điều chuẩn nhẹ miễn phí** — nhiễu thống kê batch hoạt động như một bộ điều chuẩn, đôi khi thay được dropout.
* **Dễ áp dụng** — chỉ là một tầng chèn vào, có sẵn cho cả FC và tích chập.

**Lưu ý.**

* **Phụ thuộc batch** — batch quá nhỏ làm thống kê nhiễu và giảm hiệu quả; khi đó cân nhắc LN, GroupNorm hoặc InstanceNorm.
* **Hai chế độ train/inference** — phải nhớ chuyển sang chế độ đánh giá khi suy luận để dùng running mean/var (mục 3).
* **Tương tác với chính quy hóa khác** — nên xem lại cường độ dropout/weight decay vì BN đã mang sẵn hiệu ứng điều chuẩn.

---

# 8. Tổng kết

Chuẩn hóa theo batch chèn vào giữa các tầng một thao tác tưởng đơn giản mà hiệu quả sâu sắc: chuẩn hóa kích hoạt về thống kê cố định rồi cho mạng học lại thang đo cần thiết.

$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \hat{\mu}_B}{\sqrt{\hat{\sigma}_B^2 + \epsilon}}, \qquad \mathbf{y} = \gamma \odot \hat{\mathbf{x}} + \beta$$

Động cơ ban đầu là chống dịch chuyển hiệp biến nội tại, nhưng góc nhìn hiện đại nhấn mạnh rằng BN chủ yếu **làm mượt mặt mất mát**, nhờ đó cho phép learning rate lớn và huấn luyện ổn định. Nó chuẩn hóa theo từng đặc trưng cho tầng FC và theo từng kênh cho tầng tích chập, dùng thống kê batch khi huấn luyện và running mean/var khi suy luận, đồng thời mang lại một hiệu ứng điều chuẩn nhẹ.

Khi sự phụ thuộc vào batch trở thành trở ngại — như trong Transformer — **layer normalization** thay thế bằng cách chuẩn hóa trong nội bộ từng mẫu. Cùng nhau, hai kỹ thuật này là những viên gạch nền giúp mọi mạng sâu hiện đại huấn luyện được trơn tru.

> Tinh thần cốt lõi: đừng để mỗi tầng phải đuổi theo một phân phối đầu vào đang trôi. Cố định thang đo bằng chuẩn hóa, rồi trả lại sức biểu diễn bằng $\gamma, \beta$ — đó là lý do một mẹo nhỏ lại thay đổi cách ta huấn luyện học sâu.

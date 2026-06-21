# Tích chập (Convolutions)

> Tích chập (convolution) là phép toán đưa hai nguyên lý của thị giác — **bất biến tịnh tiến (translation invariance)** và **tính cục bộ (locality)** — thẳng vào kiến trúc mạng, biến một lớp kết nối đầy đủ khổng lồ thành một bộ lọc nhỏ trượt khắp ảnh.
>
> $$\underbrace{H \times W \times C}_{\text{ảnh đầu vào}} \;\xrightarrow{\;\text{bộ lọc trượt}\;}\; \underbrace{\text{feature map}}_{\text{bản đồ đặc trưng}}$$
>
> Đây là nền tảng của mọi mạng nơ-ron tích chập (CNN), từ LeNet tới ResNet.

---

# 1. Vì sao ảnh không hợp với lớp kết nối đầy đủ?

Hãy thử áp [mạng nơ-ron](#/mang-no-ron) thông thường lên ảnh. Một lớp kết nối đầy đủ (fully-connected layer) coi đầu vào là một vector phẳng và nối **mọi** nơ-ron đầu vào với **mọi** nơ-ron đầu ra.

Lấy một ảnh màu khiêm tốn $1000 \times 1000 \times 3$, tức ba triệu giá trị đầu vào. Nếu lớp ẩn cũng có một nghìn nơ-ron, số trọng số là

$$3 \times 10^6 \times 10^3 = 3 \times 10^9$$

tức **ba tỉ tham số** chỉ cho một lớp. Con số này vượt quá bộ nhớ thực tế, đòi hỏi lượng dữ liệu khổng lồ để khỏi quá khớp (overfitting), và phần lớn là lãng phí.

Lãng phí ở đâu? Lớp kết nối đầy đủ học một trọng số riêng cho mỗi cặp vị trí. Nhưng một con mèo ở góc trên bên trái vẫn là con mèo khi nó dịch xuống góc dưới bên phải. Lớp đầy đủ không hề biết điều đó — nó phải học lại đặc trưng "mèo" một cách độc lập tại từng vị trí. Đây chính là chỗ hai nguyên lý dưới đây can thiệp.

---

# 2. Hai nguyên lý dẫn tới tích chập

Ta sẽ **suy dẫn** tích chập từ một lớp kết nối đầy đủ, bằng cách áp hai ràng buộc hợp lý lên nó.

## 2.1. Bất biến tịnh tiến (translation invariance)

> Một mẫu hình (pattern) nên được phát hiện **giống nhau** dù nó xuất hiện ở đâu trong ảnh.

Bộ dò cạnh, bộ dò góc, bộ dò mắt mèo... đều không nên phụ thuộc vào vị trí tuyệt đối. Nếu ta dịch ảnh đầu vào đi một chút, đầu ra cũng chỉ nên dịch theo y hệt.

## 2.2. Tính cục bộ (locality)

> Để nhận ra một đặc trưng tại vị trí $(i, j)$, ta chỉ cần nhìn vào **vùng lân cận** quanh nó, không cần toàn bộ ảnh.

Các điểm ảnh ở rất xa hầu như không liên quan đến việc có một cạnh tại đây hay không. Lớp đầu sẽ chỉ tổng hợp thông tin trong một cửa sổ nhỏ; muốn nhìn xa hơn thì xếp chồng nhiều lớp lại (mục 7).

## 2.3. Suy dẫn: từ MLP có ràng buộc tới chia sẻ trọng số

Viết một lớp kết nối đầy đủ ở dạng tổng quát nhất, với đầu vào $X$ và đầu ra $H$ đều là **ma trận hai chiều** (giữ cấu trúc không gian của ảnh thay vì làm phẳng). Khi đó mỗi đầu ra là tổng có trọng số trên mọi điểm ảnh:

$$[H]_{i,j} = [U]_{i,j} + \sum_{k}\sum_{l} [\mathsf{W}]_{i,j,k,l}\, [X]_{k,l}$$

Tensor trọng số $\mathsf{W}$ có bốn chỉ số — to khổng lồ. Đổi biến cho tiện, đặt $k = i + a$ và $l = j + b$ (lệch tương đối so với vị trí đầu ra):

$$[H]_{i,j} = [U]_{i,j} + \sum_{a}\sum_{b} [\mathsf{V}]_{i,j,a,b}\, [X]_{i+a,\, j+b}$$

Bây giờ áp hai nguyên lý.

**Áp bất biến tịnh tiến.** Trọng số không được phụ thuộc vào vị trí tuyệt đối $(i, j)$, chỉ phụ thuộc độ lệch $(a, b)$. Vậy bỏ chỉ số $(i,j)$: $[\mathsf{V}]_{i,j,a,b} = [\mathbf{V}]_{a,b}$, và $U$ thành hằng số $u$. Đây chính là **chia sẻ trọng số (weight sharing)** — cùng một bộ trọng số $\mathbf{V}$ dùng lại ở mọi vị trí:

$$[H]_{i,j} = u + \sum_{a}\sum_{b} [\mathbf{V}]_{a,b}\, [X]_{i+a,\, j+b}$$

**Áp tính cục bộ.** Ta chỉ nhìn cửa sổ nhỏ: khi $|a| > \Delta$ hoặc $|b| > \Delta$ thì cho $[\mathbf{V}]_{a,b} = 0$. Tổng thu gọn về một vùng $(2\Delta+1) \times (2\Delta+1)$:

$$[H]_{i,j} = u + \sum_{a=-\Delta}^{\Delta}\sum_{b=-\Delta}^{\Delta} [\mathbf{V}]_{a,b}\, [X]_{i+a,\, j+b}$$

Đây **đúng là một lớp tích chập**. Từ tensor bốn chỉ số khổng lồ, hai ràng buộc đã rút nó về một ma trận nhỏ $\mathbf{V}$ vài chục phần tử, dùng chung khắp ảnh. Số tham số không còn phụ thuộc kích thước ảnh nữa — đây là lý do CNN tiết kiệm tham số đến kinh ngạc.

---

# 3. Tương quan chéo và tích chập toán học

## 3.1. Phép tương quan chéo (cross-correlation)

Trong thực hành, phép mà các thư viện gọi là "convolution" thật ra là **tương quan chéo (cross-correlation)**. Cho đầu vào $X$ kích thước $n_h \times n_w$ và một **nhân (kernel)** $K$ kích thước $k_h \times k_w$, đầu ra tại $(i,j)$ là:

$$[Y]_{i,j} = \sum_{a=0}^{k_h-1}\sum_{b=0}^{k_w-1} [X]_{i+a,\, j+b}\, [K]_{a,b}$$

Trực giác: đặt nhân chồng lên một vùng của ảnh, nhân từng phần tử rồi cộng lại được một số; trượt nhân sang phải/xuống dưới một bước rồi lặp lại. Toàn bộ các số thu được tạo thành **bản đồ đặc trưng (feature map)**.

Ví dụ với $X$ là $3 \times 3$ và $K$ là $2 \times 2$, đầu ra $Y$ là $2 \times 2$:

$$[Y]_{0,0} = [X]_{0,0}[K]_{0,0} + [X]_{0,1}[K]_{0,1} + [X]_{1,0}[K]_{1,0} + [X]_{1,1}[K]_{1,1}$$

## 3.2. So với tích chập toán học

Tích chập đúng nghĩa trong toán học có dấu **trừ** ở chỉ số — tức nhân $K$ bị **lật (flip)** theo cả hai chiều trước khi trượt:

$$[Y]_{i,j} = \sum_{a}\sum_{b} [X]_{i-a,\, j-b}\, [K]_{a,b}$$

Khác biệt duy nhất là chiều của nhân. Nhưng vì $K$ được **học từ dữ liệu**, mạng hoàn toàn tự do học ra phiên bản lật nếu cần. Do đó dùng tương quan chéo hay tích chập thật cho **cùng kết quả huấn luyện**, chỉ là nhân học được sẽ là bản lật của nhau. Vì vậy cả tài liệu lẫn mã nguồn đều gọi chung là "tích chập" cho gọn — ta cũng theo quy ước đó.

| Tiêu chí | Lớp kết nối đầy đủ | Lớp tích chập |
| --- | --- | --- |
| Kết nối | mọi-tới-mọi (toàn cục) | cục bộ (cửa sổ $k_h \times k_w$) |
| Trọng số | riêng cho từng vị trí | chia sẻ khắp ảnh |
| Số tham số (ảnh $n\times n$) | $\mathcal{O}(n^2 \cdot m)$ | $\mathcal{O}(k_h k_w)$, độc lập $n$ |
| Bất biến tịnh tiến | không | có |

---

# 4. Nhân, bộ lọc và bản đồ đặc trưng

Vài thuật ngữ cần phân biệt rõ:

* **Nhân / bộ lọc (kernel / filter)**: ma trận trọng số nhỏ $K$ được học. Một bộ lọc dò một loại đặc trưng (ví dụ cạnh dọc).
* **Bản đồ đặc trưng (feature map)**: đầu ra $Y$ sau khi trượt một bộ lọc khắp ảnh — cho biết đặc trưng đó xuất hiện mạnh ở đâu.
* Một lớp tích chập thường có **nhiều** bộ lọc, mỗi bộ tạo một feature map riêng. Khái niệm này gắn với **kênh (channel)**, sẽ bàn ở [bài kênh & pooling](#/kenh-pooling).

---

# 5. Ví dụ: phát hiện cạnh

Hãy xem tích chập "phát hiện" thật sự nghĩa là gì. Lấy nhân ngang đơn giản $K = \begin{bmatrix} 1 & -1 \end{bmatrix}$.

Đặt nhân này lên hai điểm ảnh kề nhau theo chiều ngang: đầu ra bằng **hiệu** của chúng. Ở vùng phẳng (hai điểm bằng nhau) hiệu là $0$; tại **ranh giới** giữa vùng sáng và vùng tối, hiệu khác $0$. Vậy nhân này làm nổi bật **cạnh dọc**.

Cho một ảnh có dải trắng giữa nền đen, tích chập với $K$ cho đầu ra bằng $0$ ở mọi nơi trừ đúng hai cột nơi màu chuyển — một dương, một âm. Mạng học chính là học ra các nhân như thế (và phức tạp hơn) một cách tự động.

```python
import torch

def corr2d(X, K):                       # tương quan chéo 2D
    h, w = K.shape
    Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

X = torch.ones(6, 8); X[:, 2:6] = 0     # dải đen giữa nền trắng
K = torch.tensor([[1.0, -1.0]])         # nhân dò cạnh dọc
print(corr2d(X, K))                      # khác 0 đúng tại hai cạnh
```

---

# 6. Đệm và bước nhảy (padding & stride)

Hai siêu tham số điều khiển kích thước feature map.

## 6.1. Bài toán co kích thước

Với đầu vào $n_h \times n_w$ và nhân $k_h \times k_w$ (bước nhảy 1, không đệm), feature map có kích thước

$$(n_h - k_h + 1) \times (n_w - k_w + 1)$$

Mỗi lớp tích chập làm ảnh **co lại** $k-1$ theo mỗi chiều. Xếp chồng nhiều lớp thì ảnh teo dần, lại còn làm "mất" thông tin ở viền (các điểm ảnh biên bị nhân quét qua ít lần hơn).

## 6.2. Đệm (padding)

Khắc phục bằng cách **viền thêm** $p_h$ hàng và $p_w$ cột giá trị $0$ quanh ảnh trước khi tích chập. Kích thước đầu ra trở thành:

$$(n_h - k_h + p_h + 1) \times (n_w - k_w + p_w + 1)$$

Muốn đầu ra **bằng** đầu vào (gọi là "same padding"), chọn $p_h = k_h - 1$. Vì thế nhân thường có cạnh **lẻ** ($1, 3, 5$): khi đó đệm chia đều hai phía.

## 6.3. Bước nhảy (stride)

Thay vì trượt từng ô một, ta có thể trượt $s_h$ ô mỗi lần theo chiều dọc và $s_w$ ô theo chiều ngang. Bước nhảy lớn làm feature map **nhỏ đi nhanh** (giảm chiều, giảm tính toán). Gộp cả đệm lẫn bước nhảy, công thức kích thước đầu ra tổng quát là:

$$\left\lfloor \frac{n_h - k_h + p_h + s_h}{s_h} \right\rfloor \times \left\lfloor \frac{n_w - k_w + p_w + s_w}{s_w} \right\rfloor$$

trong đó $\lfloor \cdot \rfloor$ là phép lấy phần nguyên (vì cửa sổ phải nằm trọn trong ảnh). Khi $p_h = k_h - 1$ và $s_h = 1$, công thức rút về đúng "same padding" ở mục 6.2.

| Cấu hình | Hiệu ứng lên kích thước đầu ra |
| --- | --- |
| Không đệm, $s = 1$ | co lại $k-1$ mỗi chiều |
| $p = k-1$, $s = 1$ | giữ nguyên ("same") |
| $s > 1$ | giảm khoảng $s$ lần (lấy mẫu thưa) |

---

# 7. Trường tiếp nhận (receptive field)

> Trường tiếp nhận của một phần tử trong feature map là **tập các điểm ảnh đầu vào** ảnh hưởng tới giá trị của nó.

Ở lớp đầu, mỗi đầu ra chỉ "nhìn thấy" một cửa sổ $k_h \times k_w$ — trường tiếp nhận nhỏ, đúng với tính cục bộ (mục 2.2). Nhưng khi **xếp chồng** nhiều lớp, trường tiếp nhận **lớn dần**: một phần tử ở lớp hai nhìn thấy nhiều phần tử lớp một, mỗi phần tử đó lại nhìn một cửa sổ ở ảnh gốc.

Ví dụ hai lớp nhân $3 \times 3$ liên tiếp cho trường tiếp nhận $5 \times 5$ trên ảnh gốc. Bước nhảy $>1$ cũng nở trường tiếp nhận nhanh hơn. Đây là cách CNN dung hòa nghịch lý: **mỗi lớp cục bộ, nhưng mạng sâu thì nhìn toàn cục** — các lớp đầu bắt cạnh và kết cấu, các lớp sau ghép thành bộ phận rồi vật thể. Chi tiết cách lắp ghép thành mạng hoàn chỉnh nằm ở [kiến trúc CNN](#/kien-truc-cnn).

---

# 8. Ưu điểm

* **Tiết kiệm tham số (parameter efficiency)** — chia sẻ trọng số khiến số tham số chỉ là $k_h k_w$, độc lập với kích thước ảnh; so với ba tỉ ở mục 1 là một trời một vực.
* **Bất biến tịnh tiến sẵn trong kiến trúc** — học một đặc trưng một lần là dùng được khắp ảnh, nên hiệu quả dữ liệu (data efficiency) cao và khái quát tốt.
* **Tính cục bộ phù hợp dữ liệu thị giác** — khớp với cấu trúc tự nhiên của ảnh; xếp chồng lớp cho trường tiếp nhận lớn dần một cách có kiểm soát.

---

# 9. Hạn chế

* **Thiên kiến quy nạp cứng** — bất biến tịnh tiến và tính cục bộ là giả định *cài sẵn*; với dữ liệu không có cấu trúc lưới (đồ thị, văn bản dài) chúng có thể không phù hợp.
* **Khó nắm quan hệ tầm xa** — vì cục bộ, để nối hai vùng ở rất xa cần xếp rất nhiều lớp; đây là chỗ cơ chế chú ý (attention) của Transformer vượt trội.
* **Không bất biến với xoay/co giãn** — tích chập chỉ bất biến với *tịnh tiến*, không tự nhiên bất biến với phép quay hay đổi tỉ lệ; thường phải bù bằng tăng cường dữ liệu (data augmentation).

---

# 10. Tổng kết

Tích chập không phải một phát minh từ trên trời rơi xuống, mà là **kết quả tất yếu** khi áp hai nguyên lý của thị giác — bất biến tịnh tiến và tính cục bộ — lên một lớp kết nối đầy đủ. Hai ràng buộc đó biến tensor trọng số bốn chỉ số khổng lồ thành một nhân nhỏ dùng chung khắp ảnh:

$$[Y]_{i,j} = \sum_{a=0}^{k_h-1}\sum_{b=0}^{k_w-1} [X]_{i+a,\, j+b}\, [K]_{a,b}$$

Đệm và bước nhảy điều khiển kích thước feature map qua công thức $\left\lfloor (n_h - k_h + p_h + s_h)/s_h \right\rfloor$, còn việc xếp chồng nhiều lớp làm trường tiếp nhận lớn dần, cho phép một mạng gồm toàn các phép cục bộ rốt cuộc lại nhìn được toàn cục.

> Bài tiếp theo — **[kênh & pooling](#/kenh-pooling)** — mở rộng tích chập sang nhiều kênh đầu vào/đầu ra và thêm tầng gộp (pooling) để tóm tắt đặc trưng, hoàn thiện những viên gạch cuối cùng trước khi lắp thành [kiến trúc CNN](#/kien-truc-cnn) hoàn chỉnh.

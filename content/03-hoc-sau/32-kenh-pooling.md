# Kênh & Gộp (Channels and Pooling)

> Phép [tích chập](#/tich-chap) cơ bản chỉ làm việc với một ảnh xám đơn lẻ. Nhưng ảnh thật có **nhiều kênh** (đỏ, lục, lam), và một lớp tích chập cần học **nhiều bộ lọc** cùng lúc. Bài này mở rộng tích chập sang nhiều chiều kênh, giới thiệu tích chập $1\times 1$ như một bộ trộn kênh, rồi khép lại bằng **gộp (pooling)** — thao tác giúp mạng bớt nhạy với vị trí chính xác và thu nhỏ bản đồ đặc trưng.
>
> Đây là những viên gạch còn thiếu để dựng nên một [kiến trúc CNN](#/kien-truc-cnn) hoàn chỉnh.

---

# 1. Nhiều kênh đầu vào (multiple input channels)

Một ảnh màu không phải là một ma trận $2$ chiều, mà là một **khối** (tensor) ba chiều với hình dạng $c_i \times h \times w$: $c_i$ là số kênh đầu vào (input channels), $h$ và $w$ là chiều cao và rộng. Ảnh RGB có $c_i = 3$.

Khi đầu vào có nhiều kênh, bộ lọc (kernel) cũng phải có **đúng bấy nhiêu kênh** để khớp. Một bộ lọc lúc này là một khối $c_i \times k_h \times k_w$.

## 1.1. Cộng dồn theo kênh

Quy tắc cốt lõi: ta thực hiện tích chập $2$ chiều **độc lập trên từng kênh**, rồi **cộng tất cả các kết quả** lại thành một bản đồ đặc trưng (feature map) duy nhất. Với phần tử đầu ra tại vị trí $(i, j)$:

$$y_{i,j} = \sum_{c=1}^{c_i} \sum_{p=1}^{k_h} \sum_{q=1}^{k_w} \mathbf{X}_{c,\, i+p,\, j+q} \cdot \mathbf{W}_{c,\, p,\, q}$$

Hãy chú ý ý nghĩa của tổng theo $c$: dù đầu vào có bao nhiêu kênh, một bộ lọc vẫn **trộn chúng lại** và cho ra **một** kênh. Vì các kết quả được cộng dồn, chiều kênh "biến mất" sau phép tích chập.

> Trực giác: mỗi kênh đầu vào nhìn vào một khía cạnh khác của dữ liệu (ví dụ một kênh màu). Bộ lọc học cách **kết hợp** thông tin từ mọi kênh để phát hiện một mẫu (pattern) cụ thể — chẳng hạn "cạnh màu vàng nghiêng".

```python
def corr2d_multi_in(X, K):
    # X: (c_i, h, w),  K: (c_i, k_h, k_w)
    return sum(corr2d(x, k) for x, k in zip(X, K))  # cộng theo kênh
```

---

# 2. Nhiều kênh đầu ra (multiple output channels)

Một bộ lọc chỉ cho ra một bản đồ đặc trưng, tức chỉ phát hiện **một** loại mẫu. Nhưng mỗi lớp tích chập cần dò nhiều mẫu khác nhau cùng lúc: cạnh ngang, cạnh dọc, góc, vùng màu... Giải pháp là dùng **nhiều bộ lọc**, mỗi bộ tạo ra một kênh đầu ra (output channel).

Nếu muốn $c_o$ kênh đầu ra, ta cần $c_o$ bộ lọc, mỗi bộ là một khối $c_i \times k_h \times k_w$. Gộp lại, trọng số của cả lớp là một khối $4$ chiều:

$$\mathbf{W} \in \mathbb{R}^{c_o \times c_i \times k_h \times k_w}$$

Kênh đầu ra thứ $o$ được tính bằng bộ lọc thứ $o$ áp lên toàn bộ đầu vào (theo công thức cộng dồn ở mục 1.1):

$$\mathbf{Y}_{o} = \sum_{c=1}^{c_i} \mathbf{X}_{c} \star \mathbf{W}_{o, c}, \qquad o = 1, \dots, c_o$$

trong đó $\star$ là phép tương quan chéo $2$ chiều. Kết quả $\mathbf{Y}$ có hình dạng $c_o \times h' \times w'$.

## 2.1. Số tham số của một lớp tích chập

Đây là điểm quan trọng để hiểu vì sao CNN tiết kiệm tham số. Số trọng số của một lớp là:

$$\#\text{tham số} = c_i \times c_o \times k_h \times k_w \;\; (+\, c_o \text{ bias})$$

Điều đáng chú ý: con số này **không phụ thuộc vào kích thước ảnh** $h, w$. Một bộ lọc $3\times 3$ với $c_i = 64$, $c_o = 128$ chỉ cần $64 \times 128 \times 3 \times 3 = 73{,}728$ trọng số, dùng được cho ảnh ở bất kỳ độ phân giải nào. So sánh với một lớp [mạng nơ-ron](#/mang-no-ron) fully-connected nối mọi điểm ảnh: số tham số sẽ bùng nổ theo kích thước ảnh. Đây chính là sức mạnh của **chia sẻ trọng số (weight sharing)**.

| Đại lượng | Hình dạng |
| --- | --- |
| Đầu vào $\mathbf{X}$ | $c_i \times h \times w$ |
| Trọng số $\mathbf{W}$ | $c_o \times c_i \times k_h \times k_w$ |
| Bias $\mathbf{b}$ | $c_o$ |
| Đầu ra $\mathbf{Y}$ | $c_o \times h' \times w'$ |

---

# 3. Tích chập $1\times 1$ (1×1 convolution)

Một bộ lọc $1\times 1$ thoạt nhìn có vẻ vô nghĩa: với $k_h = k_w = 1$, nó không nhìn vào vùng lân cận không gian nào cả, không thể phát hiện cạnh hay kết cấu. Vậy nó dùng để làm gì?

## 3.1. Trộn kênh như một lớp fully-connected theo từng vị trí

Mấu chốt nằm ở **chiều kênh**. Một lớp tích chập $1\times 1$ có trọng số hình dạng $c_o \times c_i \times 1 \times 1$, tức thực chất là một ma trận $\mathbf{W} \in \mathbb{R}^{c_o \times c_i}$. Tại **mỗi vị trí không gian** $(i, j)$, nó lấy vector $c_i$ kênh và biến thành vector $c_o$ kênh:

$$\mathbf{y}_{i,j} = \mathbf{W}\, \mathbf{x}_{i,j}, \qquad \mathbf{x}_{i,j} \in \mathbb{R}^{c_i},\; \mathbf{y}_{i,j} \in \mathbb{R}^{c_o}$$

Đây đúng là một lớp fully-connected, nhưng **dùng chung trọng số cho mọi vị trí** và áp riêng rẽ tại từng điểm ảnh. Nói cách khác, tích chập $1\times 1$ **không trộn không gian, chỉ trộn kênh** — nó tổ hợp tuyến tính các đặc trưng tại cùng một vị trí.

> Cách nhớ: tích chập $k\times k$ thông thường trộn cả **không gian lẫn kênh**. Tích chập $1\times 1$ tách bạch ra, chỉ giữ lại phần **trộn kênh**.

## 3.2. Tăng/giảm số kênh và vai trò "nút thắt cổ chai"

Vì $c_o$ là tham số tự do, tích chập $1\times 1$ có thể:

- **Giảm số kênh** ($c_o < c_i$): nén thông tin, cắt giảm chi phí tính toán trước khi đưa vào một lớp tích chập đắt đỏ — vai trò "nút thắt cổ chai" (bottleneck) trong ResNet, Inception.
- **Tăng số kênh** ($c_o > c_i$): mở rộng biểu diễn sau một bước nén.

Chi phí cũng rất rẻ: $c_i \times c_o$ tham số, không có thừa số $k_h \times k_w$. Nhờ vậy nó trở thành một công cụ chủ lực để điều chỉnh độ sâu kênh trong các [kiến trúc CNN](#/kien-truc-cnn) hiện đại.

---

# 4. Gộp (pooling)

Sau vài lớp tích chập, ta thường chèn một lớp **gộp (pooling)** để **giảm độ phân giải không gian** của bản đồ đặc trưng. Khác với tích chập, gộp **không có tham số học** — nó chỉ là một phép tổng hợp cố định trên các cửa sổ.

Một lớp gộp trượt một cửa sổ $p_h \times p_w$ trên đầu vào (giống tích chập) và tại mỗi vị trí tính một con số tổng hợp.

## 4.1. Max pooling và average pooling

**Max pooling** lấy giá trị lớn nhất trong cửa sổ:

$$y_{i,j} = \max_{0 \le p < p_h,\; 0 \le q < p_w} \; x_{\,i \cdot s + p,\; j \cdot s + q}$$

**Average pooling** lấy trung bình:

$$y_{i,j} = \frac{1}{p_h\, p_w} \sum_{p=0}^{p_h-1} \sum_{q=0}^{p_w-1} x_{\,i \cdot s + p,\; j \cdot s + q}$$

trong đó $s$ là bước trượt (stride). Thông thường gộp dùng stride bằng kích thước cửa sổ ($s = p_h = p_w$, ví dụ $2\times 2$ với stride $2$), khiến các cửa sổ không chồng lấn và bản đồ đặc trưng co lại còn một nửa mỗi chiều.

Max pooling phổ biến hơn vì nó giữ lại **tín hiệu mạnh nhất** (đặc trưng nổi bật nhất) trong vùng, trong khi average pooling làm mượt và đôi khi pha loãng tín hiệu quan trọng.

## 4.2. Vì sao cần gộp?

Gộp đem lại ba lợi ích bổ trợ nhau:

1. **Bất biến dịch chuyển cục bộ (local translation invariance).** Nếu một đặc trưng (ví dụ cạnh) dịch đi một vài điểm ảnh, giá trị lớn nhất trong cửa sổ vẫn gần như không đổi. Mạng nhờ đó bớt nhạy với **vị trí chính xác** của đặc trưng — chỉ cần biết "có cạnh ở đâu đó quanh đây" là đủ.

2. **Giảm độ phân giải (downsampling).** Cửa sổ $2\times 2$ stride $2$ cắt số phần tử đi $4$ lần, giảm bộ nhớ và chi phí tính toán cho các lớp sau.

3. **Mở rộng vùng tiếp nhận (receptive field).** Sau khi gộp, mỗi phần tử "đại diện" cho một vùng rộng hơn của ảnh gốc. Nhờ vậy các lớp sâu hơn có thể "nhìn" được những cấu trúc lớn hơn dù bộ lọc vẫn nhỏ.

> Ba lợi ích này lý giải vì sao gộp gần như luôn xuất hiện xen kẽ giữa các khối tích chập trong [kiến trúc CNN](#/kien-truc-cnn) cổ điển (LeNet, AlexNet, VGG).

## 4.3. Gộp theo từng kênh (channel-wise pooling)

Một khác biệt quan trọng so với tích chập: lớp gộp xử lý **từng kênh đầu vào một cách độc lập** và **không cộng dồn các kênh**. Do đó số kênh đầu ra **bằng đúng** số kênh đầu vào:

$$c_o = c_i$$

Mỗi kênh là một bản đồ đặc trưng riêng (ví dụ "vị trí các cạnh dọc"); việc giảm độ phân giải được áp dụng riêng cho từng bản đồ mà không pha trộn chúng. Đây cũng là điều phân biệt rõ vai trò: tích chập **trộn kênh**, gộp **giữ nguyên kênh**.

## 4.4. Kích thước đầu ra

Với đầu vào kích thước $h$ (một chiều), cửa sổ $p$, đệm (padding) $\,P$ và bước trượt $s$, kích thước đầu ra là:

$$h' = \left\lfloor \frac{h + 2P - p}{s} \right\rfloor + 1$$

Công thức này giống hệt với tích chập (chỉ thay vai trò của kích thước bộ lọc bằng kích thước cửa sổ gộp), vì cả hai đều là phép trượt cửa sổ.

---

# 5. Tổng kết

Ba ý tưởng trong bài bổ sung những chiều còn thiếu cho phép [tích chập](#/tich-chap) để nó làm việc được trên dữ liệu thật:

- **Nhiều kênh đầu vào:** tích chập trên từng kênh rồi **cộng dồn**, biến khối $c_i$ kênh thành một bản đồ đặc trưng.
- **Nhiều kênh đầu ra:** mỗi bộ lọc một bản đồ; trọng số lớp là khối $c_o \times c_i \times k_h \times k_w$, với $c_i \times c_o \times k_h \times k_w$ tham số — **độc lập với kích thước ảnh**.
- **Tích chập $1\times 1$:** một bộ trộn kênh thuần túy, hoạt động như fully-connected theo từng vị trí, dùng để tăng/giảm số kênh với chi phí thấp.
- **Gộp:** giảm độ phân giải, tạo bất biến dịch chuyển cục bộ và mở rộng vùng tiếp nhận; **không có tham số**, áp **theo từng kênh** nên giữ nguyên số kênh.

Với tích chập đa kênh và gộp trong tay, ta đã có đủ mọi thành phần để lắp ráp một mạng tích chập hoàn chỉnh.

> Bài tiếp theo — **Kiến trúc CNN** — sẽ xếp chồng các khối "tích chập → kích hoạt → gộp" này thành những mạng kinh điển như LeNet, AlexNet và VGG, đồng thời cho thấy số kênh tăng dần còn độ phân giải không gian giảm dần như thế nào qua chiều sâu của mạng.

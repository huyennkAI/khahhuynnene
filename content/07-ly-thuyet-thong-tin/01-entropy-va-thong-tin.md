# Bài 1: Định nghĩa Entropy và Thông tin

> Lý thuyết thông tin trả lời câu hỏi nền tảng: **thông tin là gì và ta cần bao nhiêu bit để biểu diễn nó?** Claude Shannon (1948) nhận ra rằng lượng thông tin của một sự kiện tỉ lệ nghịch với xác suất xảy ra của nó — sự kiện càng bất ngờ thì càng chứa nhiều thông tin.
>
> $$I(x) = -\log p(x), \qquad H(X) = -\sum_{x} p(x) \log p(x)$$
>
> Entropy $H(X)$ là kỳ vọng của self-information, đo mức độ bất định trung bình của biến ngẫu nhiên $X$. Đây là cận dưới lý thuyết cho bất kỳ phương án mã hóa nào của nguồn tin $X$.

---

# 1. Self-information: lượng thông tin của một sự kiện

## 1.1. Trực giác

Hãy nghĩ đến hai tin nhắn:

* "Mặt trời mọc ở hướng đông sáng nay."
* "Một thiên thạch vừa đâm vào Hà Nội."

Tin thứ nhất hầu như không cung cấp thông tin gì — ta đã biết chắc điều đó sẽ xảy ra. Tin thứ hai cực kỳ bất ngờ và chứa rất nhiều thông tin. Mối quan hệ giữa xác suất và lượng thông tin là: **sự kiện xác suất thấp mang nhiều thông tin hơn sự kiện xác suất cao**.

Thêm vào đó, nếu ta nhận được tin về hai sự kiện độc lập $A$ và $B$, thông tin nhận được phải là tổng thông tin từ $A$ và từ $B$ riêng lẻ. Điều này đòi hỏi **tính cộng tính**.

## 1.2. Định nghĩa chính thức

Với sự kiện $x$ có xác suất $p(x)$, **self-information** (hay surprise, lượng thông tin riêng) được định nghĩa:

$$I(x) = -\log_2 p(x) \quad \text{(đơn vị: bit)}$$

hoặc nếu dùng logarit tự nhiên:

$$I(x) = -\ln p(x) \quad \text{(đơn vị: nat)}$$

Trong bài này ta mặc định $\log = \log_2$ và đơn vị là bit. Quy ước $0 \cdot \log 0 = 0$ (vì $\lim_{p \to 0^+} p \log p = 0$).

## 1.3. Tính chất của self-information

**Tính chất 1 — Không âm:** $I(x) = -\log p(x) \ge 0$ vì $p(x) \in (0,1]$, nên $\log p(x) \le 0$.

**Tính chất 2 — Sự kiện chắc chắn mang không thông tin:** Nếu $p(x) = 1$ thì $I(x) = -\log 1 = 0$.

**Tính chất 3 — Tính cộng tính với sự kiện độc lập:** Nếu $A$ và $B$ độc lập thì $p(A \cap B) = p(A) \cdot p(B)$, nên:
$$I(A \cap B) = -\log p(A \cap B) = -\log p(A) - \log p(B) = I(A) + I(B)$$

## 1.4. Ví dụ cụ thể

**Tung đồng xu công bằng:** $p(\text{mặt ngửa}) = 1/2$, nên $I(\text{mặt ngửa}) = -\log_2(1/2) = 1$ bit. Kết quả mỗi lần tung mang đúng 1 bit thông tin.

**Tung xúc xắc 6 mặt cân bằng:** $p(k) = 1/6$ với mọi $k \in \{1,\ldots,6\}$, nên $I(k) = -\log_2(1/6) = \log_2 6 \approx 2.585$ bit.

**Thời tiết bất cân:** Giả sử $p(\text{nắng}) = 0.9$, $p(\text{mưa}) = 0.1$. Khi đó:
$$I(\text{nắng}) = -\log_2 0.9 \approx 0.152 \text{ bit}, \qquad I(\text{mưa}) = -\log_2 0.1 \approx 3.322 \text{ bit}$$
Ngày mưa bất ngờ hơn, nên mang nhiều thông tin hơn hẳn.

---

# 2. Shannon Entropy: kỳ vọng của self-information

## 2.1. Định nghĩa

Ta quan tâm đến **trung bình** lượng thông tin nhận được khi quan sát biến ngẫu nhiên $X$ lấy giá trị trong tập $\mathcal{X}$ với phân phối $p$. Đây chính là **entropy Shannon**:

$$H(X) = \mathbb{E}[I(X)] = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x)$$

Có thể viết lại:

$$H(X) = \sum_{x \in \mathcal{X}} p(x) \log_2 \frac{1}{p(x)}$$

Entropy đo mức độ **bất định trung bình** của $X$: ta cần bao nhiêu bit trung bình để mô tả một kết quả của $X$.

## 2.2. Ví dụ: đồng xu

**Đồng xu công bằng** $p = 1/2$:
$$H = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{2}\log_2\frac{1}{2} = \frac{1}{2} \cdot 1 + \frac{1}{2} \cdot 1 = 1 \text{ bit}$$

**Đồng xu lệch** $p = 3/4$:
$$H = -\frac{3}{4}\log_2\frac{3}{4} - \frac{1}{4}\log_2\frac{1}{4} = \frac{3}{4} \cdot 0.415 + \frac{1}{4} \cdot 2 = 0.311 + 0.5 = 0.811 \text{ bit}$$

**Đồng xu hoàn toàn lệch** $p = 1$: $H = -1 \cdot \log_2 1 = 0$ bit. Không có bất định gì.

## 2.3. Ví dụ: xúc xắc

Xúc xắc 6 mặt đều, $p(k) = 1/6$:
$$H = -\sum_{k=1}^{6} \frac{1}{6} \log_2 \frac{1}{6} = -6 \cdot \frac{1}{6} \cdot (-\log_2 6) = \log_2 6 \approx 2.585 \text{ bit}$$

Ta cần trung bình khoảng 2.585 bit để mã hóa kết quả của một lần tung xúc xắc.

---

# 3. Entropy nhị phân $H_b(p)$

## 3.1. Định nghĩa

Trường hợp quan trọng nhất: biến ngẫu nhiên Bernoulli lấy giá trị $\{0, 1\}$ với $P(X=1) = p$ và $P(X=0) = 1-p$. Entropy được gọi là **entropy nhị phân**:

$$H_b(p) = -p \log_2 p - (1-p) \log_2(1-p)$$

## 3.2. Đồ thị và đặc điểm

Đồ thị $H_b(p)$ là đường cong hình chuông trên $[0,1]$:

* Tại $p = 0$: $H_b(0) = 0$ (kết quả luôn là 0, không bất định)
* Tại $p = 1/2$: $H_b(1/2) = 1$ bit (bất định tối đa)
* Tại $p = 1$: $H_b(1) = 0$ (kết quả luôn là 1, không bất định)

Hàm $H_b(p)$ **đạt cực đại tại $p = 1/2$** với giá trị 1 bit. Điều này có ý nghĩa trực quan rõ ràng: đồng xu công bằng là khó đoán nhất.

## 3.3. Giá trị cụ thể

| $p$ | $H_b(p)$ (bit) |
|-----|----------------|
| 0.1 | 0.469 |
| 0.2 | 0.722 |
| 0.3 | 0.881 |
| 0.4 | 0.971 |
| 0.5 | 1.000 |
| 0.7 | 0.881 |
| 0.9 | 0.469 |

Chú ý tính đối xứng: $H_b(p) = H_b(1-p)$.

---

# 4. Tính chất cơ bản của entropy

## 4.1. Entropy không âm

$$H(X) \ge 0$$

**Chứng minh:** Mỗi số hạng $-p(x)\log p(x) \ge 0$ vì $p(x) \in (0,1]$ nên $\log p(x) \le 0$.

Đẳng thức xảy ra khi và chỉ khi $X$ là hằng số (tập trung tại một điểm duy nhất).

## 4.2. Entropy đạt cực đại tại phân phối đều

Với biến ngẫu nhiên rời rạc $X$ lấy $|\mathcal{X}|$ giá trị:

$$H(X) \le \log_2 |\mathcal{X}|$$

Đẳng thức xảy ra khi và chỉ khi $X$ có phân phối đều: $p(x) = 1/|\mathcal{X}|$ với mọi $x$.

**Chứng minh bằng bất đẳng thức Jensen:** Hàm $f(t) = -\log t$ là hàm lồi. Áp dụng Jensen:
$$H(X) = \sum_x p(x) \log\frac{1}{p(x)} = |\mathcal{X}| \cdot \mathbb{E}\left[\frac{1}{|\mathcal{X}|}\log\frac{1}{p(X)}\right]$$

Cách trực tiếp hơn, ta dùng bất đẳng thức $\ln t \le t - 1$:
$$H(X) - \log|\mathcal{X}| = -\sum_x p(x)\log p(x) - \log|\mathcal{X}| = -\sum_x p(x)\log(|\mathcal{X}| p(x))$$
$$= \sum_x p(x)\log\frac{1}{|\mathcal{X}|p(x)} \le \sum_x p(x)\left(\frac{1}{|\mathcal{X}|p(x)} - 1\right) \cdot \frac{1}{\ln 2} = \frac{1}{\ln 2}\left(\frac{1}{|\mathcal{X}|}\sum_x 1 - 1\right) = 0$$

## 4.3. Tính liên tục

Entropy là hàm liên tục của $(p_1, \ldots, p_n)$ trên đơn hình xác suất.

## 4.4. Tính mở rộng

Thêm sự kiện xác suất 0 không thay đổi entropy: $H(p_1, \ldots, p_n, 0) = H(p_1, \ldots, p_n)$.

---

# 5. Joint entropy

## 5.1. Định nghĩa

Với hai biến ngẫu nhiên $X$ và $Y$ có phân phối kết hợp $p(x,y)$, **joint entropy** là:

$$H(X,Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log_2 p(x,y)$$

Đây là entropy của biến ngẫu nhiên kết hợp $(X,Y)$, đo tổng lượng bất định của cả hai biến cùng nhau.

## 5.2. Ví dụ

Giả sử $X, Y$ là hai lần tung đồng xu công bằng độc lập. Phân phối kết hợp:

| $(X,Y)$ | $(0,0)$ | $(0,1)$ | $(1,0)$ | $(1,1)$ |
|---------|---------|---------|---------|---------|
| $p(x,y)$ | $1/4$ | $1/4$ | $1/4$ | $1/4$ |

$$H(X,Y) = -4 \cdot \frac{1}{4} \log_2\frac{1}{4} = -4 \cdot \frac{1}{4} \cdot (-2) = 2 \text{ bit}$$

Hoàn toàn hợp lý: ta cần 2 bit để mô tả kết quả của hai lần tung độc lập.

---

# 6. Conditional entropy

## 6.1. Định nghĩa

**Conditional entropy** $H(Y|X)$ đo lượng bất định còn lại về $Y$ khi đã biết $X$:

$$H(Y|X) = \sum_{x \in \mathcal{X}} p(x) H(Y|X=x) = -\sum_{x} \sum_{y} p(x,y) \log_2 p(y|x)$$

Trong đó $H(Y|X=x) = -\sum_y p(y|x)\log p(y|x)$ là entropy của $Y$ khi biết $X=x$.

## 6.2. Tính chất cơ bản

$H(Y|X) \ge 0$. Conditioning không tăng entropy trung bình:

$$H(Y|X) \le H(Y)$$

Đẳng thức xảy ra khi và chỉ khi $X$ và $Y$ độc lập.

---

# 7. Chain rule của entropy

## 7.1. Định lý

$$H(X,Y) = H(X) + H(Y|X)$$

**Chứng minh:**
$$H(X,Y) = -\sum_{x,y} p(x,y)\log p(x,y) = -\sum_{x,y} p(x,y)\log[p(x)\cdot p(y|x)]$$
$$= -\sum_{x,y} p(x,y)\log p(x) - \sum_{x,y} p(x,y)\log p(y|x)$$
$$= -\sum_x p(x)\log p(x) - \sum_{x,y} p(x,y)\log p(y|x) = H(X) + H(Y|X)$$

## 7.2. Diễn giải

Chain rule nói rằng: **mô tả $(X,Y)$ tốn $H(X)$ bit để mô tả $X$, rồi $H(Y|X)$ bit để mô tả $Y$ khi đã biết $X$**. Tổng cộng bằng $H(X,Y)$.

## 7.3. Mở rộng cho nhiều biến

$$H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^{n} H(X_i | X_1, \ldots, X_{i-1})$$

## 7.4. Trường hợp độc lập

Nếu $X$ và $Y$ độc lập thì $H(Y|X) = H(Y)$, nên:
$$H(X,Y) = H(X) + H(Y)$$

Điều này phù hợp với tính cộng tính của thông tin cho các sự kiện độc lập.

---

# 8. Ví dụ tổng hợp: thời tiết và áo mưa

## 8.1. Thiết lập bài toán

Giả sử:
* $X$ = trời mưa hay không: $P(X=1) = 0.3$, $P(X=0) = 0.7$
* $Y$ = mang áo mưa hay không

Phân phối kết hợp:

| | $Y=0$ (không áo mưa) | $Y=1$ (có áo mưa) |
|---|---|---|
| $X=0$ (nắng) | $0.63$ | $0.07$ |
| $X=1$ (mưa) | $0.06$ | $0.24$ |

## 8.2. Tính các entropy

**Entropy của thời tiết:**
$$H(X) = -0.7\log_2 0.7 - 0.3\log_2 0.3 = 0.7 \cdot 0.515 + 0.3 \cdot 1.737 = 0.881 \text{ bit}$$

**Phân phối biên của $Y$:** $P(Y=0) = 0.69$, $P(Y=1) = 0.31$
$$H(Y) = -0.69\log_2 0.69 - 0.31\log_2 0.31 = 0.69 \cdot 0.536 + 0.31 \cdot 1.690 = 0.892 \text{ bit}$$

**Joint entropy:**
$$H(X,Y) = -(0.63\log_2 0.63 + 0.07\log_2 0.07 + 0.06\log_2 0.06 + 0.24\log_2 0.24)$$
$$= 0.63 \cdot 0.667 + 0.07 \cdot 3.837 + 0.06 \cdot 4.059 + 0.24 \cdot 2.059$$
$$\approx 0.420 + 0.269 + 0.244 + 0.494 = 1.427 \text{ bit}$$

**Kiểm tra bằng chain rule:** $H(X,Y) = H(X) + H(Y|X)$
$$H(Y|X) = H(X,Y) - H(X) = 1.427 - 0.881 = 0.546 \text{ bit}$$

Có nghĩa là khi đã biết thời tiết, ta chỉ còn 0.546 bit bất định về quyết định mang áo mưa — ít hơn nhiều so với 0.892 bit ban đầu.

---

# 9. Ý nghĩa thực tiễn: entropy là cận dưới nén dữ liệu

## 9.1. Định lý nén nguồn Shannon (giới thiệu sơ lược)

Entropy có ý nghĩa thực tiễn sâu sắc: **không thể nén một nguồn tin $X$ xuống dưới $H(X)$ bit/ký hiệu** mà không mất thông tin. Đây là định lý nén nguồn Shannon, sẽ được chứng minh đầy đủ trong các bài sau.

Intuition: nếu ta cần biểu diễn mỗi kết quả $x$ bằng một chuỗi bit, và các chuỗi phải phân biệt nhau (để giải mã được), thì trung bình ta cần ít nhất $H(X)$ bit.

## 9.2. Ví dụ: văn bản tiếng Anh

Văn bản tiếng Anh có khoảng 26 chữ cái + dấu cách. Nếu phân phối đều thì entropy là $\log_2 27 \approx 4.75$ bit/ký tự. Nhưng thực ra tiếng Anh có cấu trúc: chữ 'e' xuất hiện $\sim 13\%$, chữ 'z' xuất hiện $\sim 0.07\%$. Shannon ước tính entropy thực của tiếng Anh khoảng $0.6$–$1.3$ bit/ký tự — đây là lý do tại sao nén văn bản hoạt động tốt.

## 9.3. Đơn vị khác nhau

| Cơ số logarit | Đơn vị |
|---------------|--------|
| 2 | bit (hay shannon) |
| $e$ | nat |
| 10 | hartley (hay ban) |

Trong khoa học máy tính thường dùng bit. Trong lý thuyết xác suất và học máy thường dùng nat. Các kết quả đều tương đương, chỉ khác nhau một hằng số nhân.

---

# 10. Tổng kết

Shannon entropy là nền tảng của toàn bộ lý thuyết thông tin:

$$H(X) = -\sum_{x} p(x) \log p(x) = \mathbb{E}\left[\log\frac{1}{p(X)}\right]$$

Các tính chất then chốt:

* $H(X) \ge 0$, đẳng thức khi $X$ là hằng số
* $H(X) \le \log|\mathcal{X}|$, đẳng thức khi $X$ phân phối đều
* Entropy nhị phân $H_b(p)$ đạt max $= 1$ bit tại $p = 1/2$
* Chain rule: $H(X,Y) = H(X) + H(Y|X)$
* Nếu độc lập: $H(X,Y) = H(X) + H(Y)$

Entropy không chỉ là một con số trừu tượng mà là **cận dưới cứng** cho bất kỳ phương án mã hóa không mất thông tin nào. Các bài tiếp theo sẽ xây dựng trên nền tảng này để hiểu khi nào ta có thể đạt đến cận đó.

> Bài tiếp theo — **Tính chất cơ bản của thông tin** — sẽ giới thiệu conditional entropy đầy đủ, mutual information $I(X;Y)$ — thước đo thông tin chia sẻ giữa hai biến, và KL divergence — khoảng cách giữa hai phân phối. Ba công cụ này tạo thành bộ ngôn ngữ hoàn chỉnh của lý thuyết thông tin.

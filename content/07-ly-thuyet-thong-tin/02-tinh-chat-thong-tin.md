# Bài 2: Tính chất cơ bản của thông tin

> Bài trước xây dựng entropy $H(X)$ và chain rule. Bài này đi sâu vào ba công cụ cốt lõi còn lại: **conditional entropy** $H(Y|X)$, **mutual information** $I(X;Y)$ — lượng thông tin chung giữa hai biến, và **KL divergence** $D_{KL}(p\|q)$ — khoảng cách giữa hai phân phối. Mọi bất đẳng thức trong lý thuyết thông tin đều bắt nguồn từ một sự thật đơn giản: $D_{KL} \ge 0$.
>
> $$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$

---

# 1. Conditional entropy: bất định còn lại

## 1.1. Nhắc lại định nghĩa

Conditional entropy $H(Y|X)$ đo lượng bất định trung bình còn lại về $Y$ sau khi đã biết $X$:

$$H(Y|X) = \sum_{x} p(x) H(Y|X=x) = -\sum_{x} \sum_{y} p(x,y) \log p(y|x)$$

## 1.2. Conditioning không tăng entropy

**Định lý:** $H(Y|X) \le H(Y)$, đẳng thức khi và chỉ khi $X$ và $Y$ độc lập.

**Chứng minh:** Từ chain rule $H(X,Y) = H(X) + H(Y|X)$ và tính đối xứng $H(X,Y) = H(Y) + H(X|Y)$, ta có:
$$H(Y|X) = H(X,Y) - H(X) \le H(Y) + H(X) - H(X) = H(Y)$$

Bất đẳng thức ở trên dùng $H(X,Y) \le H(X) + H(Y)$ (sẽ chứng minh qua $D_{KL} \ge 0$).

**Diễn giải:** Thông tin thêm không bao giờ *tăng* sự bất định trung bình. Biết thêm $X$ có thể không giúp ích gì (nếu $X,Y$ độc lập), nhưng không bao giờ làm ta mù hơn về $Y$.

## 1.3. Ví dụ minh họa

**Chẩn đoán y tế:** $X$ = kết quả xét nghiệm (dương/âm), $Y$ = có bệnh hay không.

Giả sử:
* $P(Y=1) = 0.01$ (tỉ lệ bệnh trong dân số 1%)
* Xét nghiệm có độ nhạy 99%, độ đặc hiệu 99%

Trước khi xét nghiệm: $H(Y) = H_b(0.01) \approx 0.081$ bit. Rất thấp vì bệnh hiếm.

Sau khi xét nghiệm dương tính (xác suất hậu nghiệm $P(Y=1|\text{dương}) \approx 0.5$):
$$H(Y|X=\text{dương}) = H_b(0.5) = 1 \text{ bit}$$

Trong trường hợp này $H(Y|X=\text{dương}) > H(Y)$! Điều này không mâu thuẫn vì $H(Y|X) = H(Y|X=+)P(+) + H(Y|X=-)P(-)$ là trung bình có trọng số, và:
$$H(Y|X) = 1 \cdot 0.0198 + H_b(0.0001) \cdot 0.9802 \approx 0.02 + 0.001 = 0.021 \text{ bit} < H(Y)$$

Vậy *trung bình* xét nghiệm vẫn giảm bất định.

---

# 2. Mutual information: thông tin chung

## 2.1. Trực giác

Mutual information $I(X;Y)$ đo **lượng thông tin mà $X$ và $Y$ chia sẻ**. Cụ thể hơn, nó đo entropy của $X$ giảm đi bao nhiêu khi ta biết $Y$ (hoặc ngược lại):

$$I(X;Y) = H(X) - H(X|Y)$$

## 2.2. Các biểu diễn tương đương

$$I(X;Y) = H(X) - H(X|Y)$$
$$= H(Y) - H(Y|X)$$
$$= H(X) + H(Y) - H(X,Y)$$
$$= \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

**Chứng minh sự tương đương của hai dòng đầu:**
Từ chain rule: $H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$, nên:
$$H(X) - H(X|Y) = H(Y) - H(Y|X) \checkmark$$

**Biểu diễn thứ ba:** $I(X;Y) = H(X) + H(Y) - H(X,Y)$ — tổng entropy trừ đi joint entropy.

## 2.3. Tính chất cơ bản

**Không âm:** $I(X;Y) \ge 0$, đẳng thức khi và chỉ khi $X$ và $Y$ độc lập.

**Đối xứng:** $I(X;Y) = I(Y;X)$ — thông tin $X$ mang về $Y$ bằng thông tin $Y$ mang về $X$.

**Self-information:** $I(X;X) = H(X)$ — $X$ chứa toàn bộ thông tin về chính nó.

## 2.4. Sơ đồ Venn thông tin

Quan hệ giữa các đại lượng entropy được hiểu trực quan qua sơ đồ Venn:

```
    H(X)              H(Y)
 ┌──────────────┐  ┌──────────────┐
 │              │  │              │
 │   H(X|Y)  ╔═╧══╧═╗  H(Y|X)   │
 │           ║ I(X;Y)║           │
 │           ╚═╤══╤═╝            │
 │              │  │              │
 └──────────────┘  └──────────────┘
         └─────────────────┘
               H(X,Y)
```

Trong đó:
* $H(X) = H(X|Y) + I(X;Y)$
* $H(Y) = H(Y|X) + I(X;Y)$
* $H(X,Y) = H(X|Y) + I(X;Y) + H(Y|X)$

## 2.5. Ví dụ cụ thể

**Hai lần tung xúc xắc độc lập:** $I(X;Y) = 0$ vì biết kết quả lần này không cho biết gì về lần kia.

**$Y = X$:** $I(X;X) = H(X)$ — hoàn toàn phụ thuộc.

**$Y = f(X)$ là hàm tất định của $X$:** $H(Y|X) = 0$ nên $I(X;Y) = H(Y)$.

**Ví dụ số:** $X$ là tung đồng xu công bằng, $Y = X$ với xác suất 0.8 và $Y = 1-X$ với xác suất 0.2.

Phân phối kết hợp:

| | $Y=0$ | $Y=1$ |
|---|---|---|
| $X=0$ | $0.4$ | $0.1$ |
| $X=1$ | $0.1$ | $0.4$ |

$H(X) = 1$ bit, $H(Y) = 1$ bit.

$H(X,Y) = -(2 \cdot 0.4\log 0.4 + 2 \cdot 0.1\log 0.1) = -(0.8 \cdot(-1.322) + 0.2 \cdot (-3.322))$
$= 1.058 + 0.664 = 1.722$ bit.

$I(X;Y) = H(X) + H(Y) - H(X,Y) = 1 + 1 - 1.722 = 0.278$ bit.

---

# 3. KL divergence: khoảng cách giữa hai phân phối

## 3.1. Định nghĩa

**Kullback-Leibler divergence** (hay relative entropy) từ $p$ đến $q$ là:

$$D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

Quy ước $0 \cdot \log(0/q) = 0$ và $p(x)\log(p(x)/0) = +\infty$ nếu $p(x) > 0$.

## 3.2. Trực giác

$D_{KL}(p\|q)$ đo **cái giá phải trả khi dùng phân phối $q$ để mã hóa dữ liệu thực sự đến từ $p$**. Nếu ta thiết kế mã tối ưu cho $q$ nhưng dữ liệu thực có phân phối $p$, ta tốn thêm $D_{KL}(p\|q)$ bit/ký hiệu so với mã tối ưu cho $p$.

## 3.3. Bất đẳng thức thông tin: $D_{KL} \ge 0$

**Định lý (Information Inequality):** $D_{KL}(p\|q) \ge 0$, đẳng thức khi và chỉ khi $p = q$.

**Chứng minh bằng bất đẳng thức log:**

Bất đẳng thức cơ bản $\ln t \le t - 1$ với mọi $t > 0$, đẳng thức khi $t = 1$.

$$-D_{KL}(p\|q) = \sum_x p(x)\log\frac{q(x)}{p(x)} = \frac{1}{\ln 2}\sum_x p(x)\ln\frac{q(x)}{p(x)}$$
$$\le \frac{1}{\ln 2}\sum_x p(x)\left(\frac{q(x)}{p(x)} - 1\right) = \frac{1}{\ln 2}\left(\sum_x q(x) - \sum_x p(x)\right) = \frac{1}{\ln 2}(1-1) = 0$$

Vậy $-D_{KL}(p\|q) \le 0$, tức là $D_{KL}(p\|q) \ge 0$.

Đẳng thức xảy ra khi $q(x)/p(x) = 1$ với mọi $x$ có $p(x) > 0$, tức là $p = q$.

## 3.4. KL divergence không đối xứng

$D_{KL}(p\|q) \ne D_{KL}(q\|p)$ nói chung! Do đó KL không phải là khoảng cách (metric) theo nghĩa toán học.

**Ví dụ:** $p = (1/2, 1/2)$, $q = (1/4, 3/4)$.
$$D_{KL}(p\|q) = \frac{1}{2}\log\frac{1/2}{1/4} + \frac{1}{2}\log\frac{1/2}{3/4} = \frac{1}{2}\log 2 + \frac{1}{2}\log\frac{2}{3} = 0.5 - 0.292 = 0.208 \text{ bit}$$
$$D_{KL}(q\|p) = \frac{1}{4}\log\frac{1/4}{1/2} + \frac{3}{4}\log\frac{3/4}{1/2} = \frac{1}{4}(-1) + \frac{3}{4} \cdot 0.585 = -0.25 + 0.439 = 0.189 \text{ bit}$$

## 3.5. Mutual information là KL divergence

$$I(X;Y) = D_{KL}(p(x,y) \| p(x)p(y))$$

**Chứng minh:**
$$I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)} = D_{KL}(p(x,y)\|p(x)p(y))$$

Vì $D_{KL} \ge 0$, ta có $I(X;Y) \ge 0$, đẳng thức khi và chỉ khi $p(x,y) = p(x)p(y)$, tức khi $X, Y$ độc lập. Đây là chứng minh tự nhiên nhất cho $I(X;Y) \ge 0$.

## 3.6. Ứng dụng trong học máy

**Cross-entropy loss:** Khi huấn luyện mô hình phân loại, cross-entropy $H(p,q) = -\sum_x p(x)\log q(x)$ thực ra là:
$$H(p,q) = H(p) + D_{KL}(p\|q)$$

Cực tiểu hóa cross-entropy tương đương cực tiểu hóa KL divergence (khi $H(p)$ cố định vì $p$ là phân phối thực).

---

# 4. Bất đẳng thức $H(X) \le \log|\mathcal{X}|$

## 4.1. Phát biểu chính thức

**Định lý:** Với biến ngẫu nhiên rời rạc $X$ lấy $n = |\mathcal{X}|$ giá trị:
$$H(X) \le \log_2 n$$
Đẳng thức khi và chỉ khi $X$ có phân phối đều.

## 4.2. Chứng minh bằng KL divergence

Gọi $u$ là phân phối đều: $u(x) = 1/n$ với mọi $x$.

$$D_{KL}(p\|u) = \sum_x p(x)\log\frac{p(x)}{1/n} = \sum_x p(x)\log p(x) + \log n = -H(X) + \log n \ge 0$$

Vậy $H(X) \le \log n$. Đẳng thức khi $p = u$, tức phân phối đều.

## 4.3. Ý nghĩa thực tiễn

Nếu $X$ có $n$ khả năng, ta cần tối đa $\log_2 n$ bit để mô tả nó (với phân phối đều). Nhưng nếu phân phối không đều — có một số giá trị xảy ra thường xuyên hơn — thì ta có thể nén tốt hơn, dùng ít hơn $\log_2 n$ bit trung bình.

---

# 5. Chain rules cho mutual information

## 5.1. Mutual information có chain rule

Tương tự entropy, mutual information cũng có chain rule:

$$I(X_1, X_2; Y) = I(X_1; Y) + I(X_2; Y | X_1)$$

Trong đó **conditional mutual information** là:
$$I(X;Y|Z) = H(X|Z) - H(X|Y,Z) = \sum_z p(z) I(X;Y|Z=z)$$

## 5.2. Chain rule tổng quát

$$I(X_1, \ldots, X_n; Y) = \sum_{i=1}^{n} I(X_i; Y | X_1, \ldots, X_{i-1})$$

## 5.3. Ý nghĩa: thêm đặc trưng trong học máy

Trong học máy, chain rule của mutual information giải thích feature selection:

* $I(X_1; Y)$ = thông tin đặc trưng đầu tiên mang về nhãn
* $I(X_2; Y|X_1)$ = thông tin **thêm** từ đặc trưng thứ hai sau khi đã có đặc trưng đầu

Một đặc trưng có $I(X_2; Y|X_1) \approx 0$ là dư thừa (đặc trưng này không cho thêm gì sau khi đã có $X_1$).

---

# 6. Tổng hợp: sơ đồ quan hệ các đại lượng

## 6.1. Quan hệ tất cả các đại lượng

$$H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$
$$I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$
$$D_{KL}(p\|q) = -H(p) - \sum_x p(x)\log q(x) = H(p,q) - H(p)$$
$$I(X;Y) = D_{KL}(p_{XY}\|p_X \cdot p_Y) \ge 0$$

## 6.2. Bảng tổng kết

| Đại lượng | Công thức | Ý nghĩa |
|-----------|-----------|---------|
| $H(X)$ | $-\sum p(x)\log p(x)$ | Bất định của $X$ |
| $H(Y\|X)$ | $-\sum_{x,y} p(x,y)\log p(y\|x)$ | Bất định còn lại về $Y$ khi biết $X$ |
| $H(X,Y)$ | $-\sum_{x,y} p(x,y)\log p(x,y)$ | Bất định kết hợp |
| $I(X;Y)$ | $H(X)-H(X\|Y)$ | Thông tin chung |
| $D_{KL}(p\|q)$ | $\sum p(x)\log(p(x)/q(x))$ | Giá của việc dùng sai phân phối |

## 6.3. Chuỗi bất đẳng thức

Từ $D_{KL}(p\|q) \ge 0$ và $I(X;Y) \ge 0$, ta dẫn ra:

1. $H(X,Y) \le H(X) + H(Y)$ — bất đẳng thức dưới cộng (subadditivity of entropy)
2. $H(X|Y) \le H(X)$ — conditioning không tăng entropy
3. $H(X) \le \log|\mathcal{X}|$ — entropy bị chặn bởi log của kích thước alphabet
4. $I(X;Y) \ge 0$ — mutual information không âm

Tất cả đều là hệ quả của một bất đẳng thức duy nhất: $\ln t \le t - 1$.

---

# 7. Ví dụ tổng hợp: kênh truyền tin nhị phân đối xứng (BSC)

## 7.1. Mô hình

**Binary Symmetric Channel (BSC)** với tham số lỗi $\epsilon$:
* Đầu vào $X \in \{0,1\}$ (bit gửi đi)
* Đầu ra $Y \in \{0,1\}$ (bit nhận được)
* $P(Y \ne X) = \epsilon$ (bit bị lật với xác suất $\epsilon$)

## 7.2. Tính mutual information $I(X;Y)$

Giả sử $X$ phân phối đều: $P(X=0) = P(X=1) = 1/2$.

**Phân phối kết hợp:**

| | $Y=0$ | $Y=1$ |
|---|---|---|
| $X=0$ | $(1-\epsilon)/2$ | $\epsilon/2$ |
| $X=1$ | $\epsilon/2$ | $(1-\epsilon)/2$ |

**Entropy đầu ra:** $P(Y=0) = P(Y=1) = 1/2$, nên $H(Y) = 1$ bit.

**Conditional entropy:**
$$H(Y|X=0) = H_b(\epsilon), \quad H(Y|X=1) = H_b(\epsilon)$$
$$H(Y|X) = \frac{1}{2}H_b(\epsilon) + \frac{1}{2}H_b(\epsilon) = H_b(\epsilon)$$

**Mutual information:**
$$I(X;Y) = H(Y) - H(Y|X) = 1 - H_b(\epsilon)$$

## 7.3. Diễn giải

* $\epsilon = 0$: không có lỗi, $I(X;Y) = 1$ bit — ta biết chính xác bit gửi đi.
* $\epsilon = 1/2$: lỗi ngẫu nhiên hoàn toàn, $I(X;Y) = 0$ — kênh vô dụng.
* $\epsilon = 1$: bit luôn bị lật, $I(X;Y) = 1$ bit — ta vẫn biết chính xác (chỉ cần lật ngược).

Giá trị $C = 1 - H_b(\epsilon)$ là **dung lượng kênh** của BSC, sẽ được khám phá sâu hơn trong lý thuyết mã hóa kênh.

---

# 8. Tổng kết

Bài này xây dựng bộ công cụ hoàn chỉnh của lý thuyết thông tin:

$$D_{KL}(p\|q) \ge 0 \implies I(X;Y) \ge 0 \implies H(Y|X) \le H(Y) \implies H(X) \le \log|\mathcal{X}|$$

Mọi bất đẳng thức quan trọng đều bắt nguồn từ $\ln t \le t-1$. Mutual information $I(X;Y)$ vừa đo thông tin chung vừa bằng KL divergence giữa phân phối kết hợp và tích phân phối biên. Sơ đồ Venn thông tin tóm gọn trực quan quan hệ giữa $H(X)$, $H(Y)$, $H(X,Y)$, $H(X|Y)$, $H(Y|X)$, và $I(X;Y)$.

> Bài tiếp theo — **Giới thiệu về mã (Codes)** — sẽ biến những đại lượng trừu tượng này thành câu hỏi cụ thể: làm thế nào để thiết kế một phương án mã hóa tối ưu? Bất đẳng thức Kraft và định lý McMillan sẽ cho ta biết ranh giới của thế giới mã hóa tiền tố.

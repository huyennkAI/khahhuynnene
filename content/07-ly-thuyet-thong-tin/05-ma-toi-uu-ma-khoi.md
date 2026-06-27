# Bài 5: Mã tối ưu và mã khối

> Bài trước chứng minh rằng ta có thể nén đến $H(X)$ bit/ký hiệu khi $n \to \infty$. Bài này hỏi cụ thể hơn: với một nguồn $X$ cố định, **mã prefix-free tối ưu** có độ dài trung bình là bao nhiêu? Câu trả lời là định lý hai phía của Shannon: $L^*$ nằm trong khoảng $[H/\log D, H/\log D + 1)$. Và khi mã hóa $n$ ký hiệu cùng nhau (mã khối), cận này thu hẹp lại thành $[H/\log D, H/\log D + 1/n)$.
>
> $$\frac{H(X)}{\log_2 D} \le L^* < \frac{H(X)}{\log_2 D} + 1$$

---

# 1. Mã prefix-free tối ưu

## 1.1. Định nghĩa

Với nguồn $X$ phân phối $p$ và bảng chữ mã $D$, **mã tối ưu** là mã prefix-free $C^*$ có độ dài trung bình nhỏ nhất:

$$L^* = \min_{C \text{ prefix-free}} L(C) = \min_{C \text{ prefix-free}} \sum_i p_i l_i$$

Theo định lý McMillan, ta có thể giới hạn tìm kiếm trong lớp prefix-free code mà không mất tính tổng quát.

## 1.2. Bài toán tối ưu hóa

Tìm $(l_1, \ldots, l_m) \in \mathbb{Z}_{>0}^m$ để cực tiểu hóa $\sum p_i l_i$ subject to $\sum D^{-l_i} \le 1$.

Đây là bài toán tối ưu nguyên (integer optimization). Nghiệm liên tục (không cần nguyên) là $l_i = \log_D(1/p_i)$, cho $L = H(X)/\log D$. Khi làm tròn lên nguyên gần nhất, ta mất tối đa 1 đơn vị.

---

# 2. Định lý Shannon về mã tối ưu

## 2.1. Phát biểu

**Định lý:** Với mã prefix-free tối ưu $C^*$ trên bảng chữ $D$:

$$\frac{H(X)}{\log_2 D} \le L^* < \frac{H(X)}{\log_2 D} + 1$$

## 2.2. Chứng minh cận dưới

Từ bài 3, ta đã chứng minh: với mọi mã giải mã duy nhất, $L \ge H(X)/\log D$. Vì $L^*$ là tối ưu, $L^* \ge H/\log D$.

## 2.3. Chứng minh cận trên: xây dựng mã cụ thể

Gán $l_i = \lceil \log_D(1/p_i) \rceil$ (làm tròn lên).

**Bước 1 — Thỏa Kraft:**

$$D^{-l_i} = D^{-\lceil\log_D(1/p_i)\rceil} \le D^{-\log_D(1/p_i)} = p_i$$

$$\sum_i D^{-l_i} \le \sum_i p_i = 1 \checkmark$$

Vậy tồn tại prefix-free code với bộ độ dài này.

**Bước 2 — Cận trên độ dài:**

$$l_i = \lceil\log_D(1/p_i)\rceil < \log_D(1/p_i) + 1$$

$$L^* \le \sum_i p_i l_i < \sum_i p_i \left(\log_D \frac{1}{p_i} + 1\right) = \frac{H(X)}{\log D} + 1$$

Kết hợp: $H/\log D \le L^* < H/\log D + 1$.

## 2.4. Mức độ "không hiệu quả"

Khoảng cách tối đa là $1$ bit (với $D=2$). Trong thực tế:

* Khi $H$ lớn (nhiều ký hiệu): sai số $1/H$ tương đối nhỏ.
* Khi $H$ nhỏ (phân phối rất lệch): sai số $1$ bit là đáng kể.

---

# 3. Tính chất của mã tối ưu

## 3.1. Ký hiệu có xác suất cao hơn được codeword ngắn hơn

**Bổ đề:** Trong mã tối ưu, nếu $p_i > p_j$ thì $l_i \le l_j$.

**Chứng minh bằng exchange argument:** Nếu $p_i > p_j$ nhưng $l_i > l_j$, ta đổi độ dài của chúng:

$$\Delta L = p_i l_j + p_j l_i - (p_i l_i + p_j l_j) = (p_i - p_j)(l_j - l_i) < 0$$

vì $p_i > p_j$ và $l_j < l_i$. Vậy ta giảm được độ dài trung bình, mâu thuẫn với tính tối ưu.

## 3.2. Hai ký hiệu có xác suất thấp nhất có cùng độ dài

**Bổ đề:** Trong mã tối ưu, hai ký hiệu có xác suất nhỏ nhất có cùng độ dài, và đó là độ dài lớn nhất.

**Chứng minh:** Giả sử $x_i$ và $x_j$ là hai ký hiệu xác suất thấp nhất. Nếu $l_i < l_j$, ta có thể rút ngắn $l_j$ xuống $l_j - 1$ (bằng cách cắt một bit và lấy prefix của codeword $x_j$). Điều này giảm $L$ mà vẫn thỏa Kraft.

Điều này chứng minh ràng codeword dài nhất là của ký hiệu xác suất thấp nhất, và chúng chia sẻ độ dài.

## 3.3. Các nút anh em ở độ sâu lớn nhất

Trong cây biểu diễn mã tối ưu, hai nút lá ở độ sâu lớn nhất phải là **anh em** (cùng cha). Nếu không, ta có thể thay thế chúng bằng cha của chúng (codeword ngắn hơn 1 bit) mà vẫn giữ tính prefix-free và giảm độ dài trung bình.

---

# 4. Mã khối (Block Coding)

## 4.1. Ý tưởng

Thay vì mã hóa từng ký hiệu một, ta mã hóa **$n$ ký hiệu cùng nhau** như một "siêu ký hiệu".

Nguồn mở rộng: $X^n = (X_1, \ldots, X_n)$ lấy giá trị trong $\mathcal{X}^n$ với phân phối:
$$p(x^n) = p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i) \quad \text{(với nguồn i.i.d.)}$$

## 4.2. Entropy của nguồn mở rộng

$$H(X^n) = H(X_1, \ldots, X_n) = nH(X) \quad \text{(với i.i.d.)}$$

## 4.3. Định lý: mã khối đạt tốc độ $H(X)$

Gọi $L_n$ là độ dài trung bình tối ưu khi mã hóa $n$ ký hiệu cùng nhau (bằng prefix-free code trên bảng chữ $D$). Thì:

$$\frac{H(X)}{\log D} \le \frac{L_n}{n} < \frac{H(X)}{\log D} + \frac{1}{n}$$

**Chứng minh:** Áp dụng định lý Shannon cho nguồn $X^n$ (có $m^n$ ký hiệu và entropy $nH(X)$):

$$nH(X)/\log D \le L_n < nH(X)/\log D + 1$$

Chia cho $n$:

$$H(X)/\log D \le L_n/n < H(X)/\log D + 1/n$$

## 4.4. Hội tụ về $H(X)$

Khi $n \to \infty$: $L_n/n \to H(X)/\log D$.

Với $D=2$: tốc độ bit/ký hiệu tiến về $H(X)$ từ trên.

---

# 5. Trade-off giữa complexity và hiệu suất

## 5.1. Block length và hiệu suất

| Block length $n$ | Sai số $\le$ | Kích thước codebook |
|-----------------|-------------|---------------------|
| $1$ | $1$ bit/symbol | $|\mathcal{X}|$ |
| $2$ | $0.5$ bit/symbol | $|\mathcal{X}|^2$ |
| $n$ | $1/n$ bit/symbol | $|\mathcal{X}|^n$ |

**Chi phí:** Kích thước codebook tăng theo hàm mũ! Mã hóa $n = 100$ ký hiệu cùng nhau với $|\mathcal{X}| = 27$ (tiếng Anh + khoảng trắng) đòi hỏi codebook $27^{100} \approx 10^{142}$ phần tử — hoàn toàn không khả thi.

## 5.2. Giải pháp thực tế

Trong thực tế, các phương án sau được dùng thay cho block coding ngây thơ:

1. **Arithmetic Coding:** Đạt hiệu suất của block coding với $O(1)$ bộ nhớ.
2. **LZ compression (LZW, DEFLATE):** Không cần biết phân phối trước.
3. **Huffman theo từng ký hiệu:** Đơn giản, hiệu quả gần tối ưu cho phân phối đặc biệt.

---

# 6. Ví dụ tính toán: so sánh mã $n=1$ và $n=2$

## 6.1. Bài toán

Nguồn $\mathcal{X} = \{a, b\}$ với $p(a) = 0.9$, $p(b) = 0.1$.

$H(X) = H_b(0.9) = -0.9\log_2 0.9 - 0.1\log_2 0.1 = 0.469$ bit.

## 6.2. Mã $n=1$

Mã tối ưu duy nhất (theo thứ tự Kraft):

| | Codeword | Độ dài |
|---|---|---|
| $a$ | $0$ | $1$ |
| $b$ | $1$ | $1$ |

$L_1 = 0.9 \cdot 1 + 0.1 \cdot 1 = 1$ bit. Sai số: $1 - 0.469 = 0.531$ bit.

## 6.3. Mã $n=2$ (mã hóa 2 ký hiệu cùng nhau)

Nguồn $X^2$ có 4 ký hiệu:

| $x^2$ | $p(x^2)$ | Độ dài tối ưu $\lceil -\log p \rceil$ |
|-------|---------|------|
| $aa$ | $0.81$ | $1$ |
| $ab$ | $0.09$ | $4$ |
| $ba$ | $0.09$ | $4$ |
| $bb$ | $0.01$ | $7$ |

Một mã tốt (không nhất thiết tối ưu):

| $x^2$ | Codeword |
|-------|---------|
| $aa$ | $0$ |
| $ab$ | $10$ |
| $ba$ | $110$ |
| $bb$ | $111$ |

Kiểm tra Kraft: $1/2 + 1/4 + 1/8 + 1/8 = 1$ ✓

$L_2 = 0.81\cdot 1 + 0.09\cdot 2 + 0.09\cdot 3 + 0.01\cdot 3 = 0.81 + 0.18 + 0.27 + 0.03 = 1.29$ bit cho 2 ký hiệu.

$L_2/2 = 0.645$ bit/ký hiệu. Sai số: $0.645 - 0.469 = 0.176$ bit (giảm đáng kể so với $n=1$).

Cận lý thuyết: $H(X^2)/2 = 2H(X)/2 = H(X) = 0.469$, và mã khối tối ưu cho $n=2$ đạt $L_2/2 < 0.469 + 0.5 = 0.969$. Mã trên đạt $0.645$, tốt hơn cận này nhiều.

---

# 7. Nguồn không đồng đều và hiệu quả mã

## 7.1. Khái niệm hiệu suất mã

Hiệu suất (efficiency) của mã $C$ được định nghĩa là:

$$\eta = \frac{H(X)/\log D}{L(C)} \in (0, 1]$$

Mã tốt hơn có $\eta$ gần 1.

## 7.2. Bảng so sánh

| Phân phối $p$ | $H(X)$ (bit) | $L^*$ (bit) | $\eta$ |
|--------------|-------------|-----------|--------|
| $(1/2, 1/2)$ | $1.0$ | $1.0$ | $100\%$ |
| $(3/4, 1/4)$ | $0.811$ | $1.0$ | $81.1\%$ |
| $(0.9, 0.1)$ | $0.469$ | $1.0$ | $46.9\%$ |
| $(0.99, 0.01)$ | $0.081$ | $1.0$ | $8.1\%$ |

Khi phân phối rất lệch, mã hóa từng ký hiệu rất kém hiệu quả. Block coding giải quyết vấn đề này.

---

# 8. Tổng kết

Bài này xây dựng nền tảng cho mã tối ưu:

**Mã tối ưu (single symbol):**
$$\frac{H(X)}{\log D} \le L^* < \frac{H(X)}{\log D} + 1$$

**Mã khối (block length $n$):**
$$\frac{H(X)}{\log D} \le \frac{L_n}{n} < \frac{H(X)}{\log D} + \frac{1}{n}$$

**Trade-off:** Block length $n$ lớn → hiệu suất tốt hơn → nhưng codebook kích thước $|\mathcal{X}|^n$ là bất khả thi với $n$ lớn.

Các tính chất của mã tối ưu:
* Ký hiệu xác suất cao → codeword ngắn hơn
* Hai ký hiệu xác suất thấp nhất → cùng độ dài (và là độ dài lớn nhất)
* Codeword $l_i = \lceil\log_D(1/p_i)\rceil$ thỏa Kraft và đạt cận $L^* < H/\log D + 1$

> Bài tiếp theo — **Mã Shannon** — sẽ xây dựng chi tiết mã Shannon (Shannon code): cách gán codeword dựa trên xác suất tích lũy. Đây là phương pháp tường minh để đạt cận $H/\log D + 1$, và là tiền thân của Arithmetic Coding.

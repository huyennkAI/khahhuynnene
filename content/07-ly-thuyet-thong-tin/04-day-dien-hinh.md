# Bài 4: Dãy điển hình (Typical Sequences)

> Khi tung đồng xu công bằng 1000 lần, kết quả "điển hình" là khoảng 500 mặt ngửa và 500 mặt sấp — không phải chính xác 500, nhưng gần đó. Các kết quả "không điển hình" (như 1000 mặt ngửa liên tiếp) có thể xảy ra nhưng cực kỳ hiếm. Định lý AEP (Asymptotic Equipartition Property) nắm bắt chính xác ý tưởng này và là nền tảng của mọi kết quả mã hóa nguồn.
>
> $$\left|-\frac{1}{n}\log p(x^n) - H(X)\right| < \epsilon \quad \text{với xác suất} \to 1 \text{ khi } n \to \infty$$

---

# 1. Định lý AEP

## 1.1. Phát biểu

**Định lý AEP (Asymptotic Equipartition Property):** Nếu $X_1, X_2, \ldots$ là i.i.d. (độc lập đồng phân phối) từ phân phối $p(x)$ với $H(X) < \infty$, thì:

$$-\frac{1}{n}\log p(X_1, X_2, \ldots, X_n) \xrightarrow{P} H(X)$$

(hội tụ theo xác suất khi $n \to \infty$).

## 1.2. Chứng minh

Vì $X_i$ i.i.d.:
$$p(X_1, X_2, \ldots, X_n) = \prod_{i=1}^{n} p(X_i)$$

$$-\frac{1}{n}\log p(X_1, \ldots, X_n) = -\frac{1}{n}\sum_{i=1}^{n}\log p(X_i) = \frac{1}{n}\sum_{i=1}^{n} I(X_i)$$

Mỗi $I(X_i) = -\log p(X_i)$ là i.i.d. với kỳ vọng $\mathbb{E}[I(X_i)] = H(X)$.

Áp dụng **Luật số lớn (Law of Large Numbers):**
$$\frac{1}{n}\sum_{i=1}^{n} I(X_i) \xrightarrow{P} \mathbb{E}[I(X)] = H(X)$$

## 1.3. Diễn giải

AEP nói rằng: với chuỗi dài $n$ ký hiệu, log-xác suất chuẩn hóa $-\frac{1}{n}\log p(x^n)$ hội tụ về $H(X)$. Điều này có nghĩa là xác suất của một chuỗi điển hình tiến về $2^{-nH(X)}$ (với log cơ số 2).

---

# 2. Tập điển hình

## 2.1. Định nghĩa chính thức

**Tập $\epsilon$-điển hình** $A_\epsilon^{(n)}$ là tập tất cả các chuỗi $(x_1, \ldots, x_n) \in \mathcal{X}^n$ thỏa mãn:

$$\left|-\frac{1}{n}\log_2 p(x_1, \ldots, x_n) - H(X)\right| < \epsilon$$

Tương đương:

$$2^{-n(H(X)+\epsilon)} \le p(x_1, \ldots, x_n) \le 2^{-n(H(X)-\epsilon)}$$

## 2.2. Cách đọc

Chuỗi $x^n = (x_1,\ldots,x_n)$ là "điển hình" nếu xác suất của nó xấp xỉ $2^{-nH(X)}$, tức là xác suất bằng với xác suất của một chuỗi được chọn đồng đều từ tập kích thước $2^{nH(X)}$.

---

# 3. Ba tính chất chính của tập điển hình

## 3.1. Tính chất 1: Xác suất của tập điển hình tiến về 1

**Định lý:** Với mọi $\epsilon > 0$:
$$P(X^n \in A_\epsilon^{(n)}) \ge 1 - \delta$$
với $\delta \to 0$ khi $n \to \infty$.

**Chứng minh:** Trực tiếp từ AEP — vì $-\frac{1}{n}\log p(X^n) \xrightarrow{P} H(X)$, với bất kỳ $\epsilon > 0$, xác suất rằng $|-\frac{1}{n}\log p(X^n) - H(X)| \ge \epsilon$ tiến về 0.

**Ý nghĩa:** Gần như mọi chuỗi dài đều là điển hình. Các chuỗi không điển hình có xác suất gần như bằng 0.

## 3.2. Tính chất 2: Mỗi chuỗi điển hình có xác suất xấp xỉ $2^{-nH}$

Theo định nghĩa: với $x^n \in A_\epsilon^{(n)}$:

$$2^{-n(H+\epsilon)} \le p(x^n) \le 2^{-n(H-\epsilon)}$$

Đây là lý do tên gọi "equipartition" — các chuỗi điển hình có xác suất xấp xỉ **bằng nhau** (trong phạm vi $\epsilon$).

## 3.3. Tính chất 3: Số chuỗi điển hình xấp xỉ $2^{nH}$

**Cận trên:**
$$1 \ge P(A_\epsilon^{(n)}) = \sum_{x^n \in A_\epsilon^{(n)}} p(x^n) \ge |A_\epsilon^{(n)}| \cdot 2^{-n(H+\epsilon)}$$
$$\implies |A_\epsilon^{(n)}| \le 2^{n(H+\epsilon)}$$

**Cận dưới:** Với đủ $n$ lớn, $P(A_\epsilon^{(n)}) \ge 1-\delta$:
$$1-\delta \le P(A_\epsilon^{(n)}) = \sum_{x^n \in A_\epsilon^{(n)}} p(x^n) \le |A_\epsilon^{(n)}| \cdot 2^{-n(H-\epsilon)}$$
$$\implies |A_\epsilon^{(n)}| \ge (1-\delta) \cdot 2^{n(H-\epsilon)}$$

**Kết hợp:** $(1-\delta) \cdot 2^{n(H-\epsilon)} \le |A_\epsilon^{(n)}| \le 2^{n(H+\epsilon)}$

---

# 4. Minh họa: đồng xu công bằng

## 4.1. Bài toán

Tung đồng xu công bằng $n = 1000$ lần, $H(X) = 1$ bit.

**Câu hỏi:** Tập điển hình trông như thế nào?

## 4.2. Phân tích

Một chuỗi $x^{1000} \in \{0,1\}^{1000}$ là điển hình với $\epsilon = 0.01$ nếu:

$$\left|-\frac{1}{1000}\log_2 p(x^{1000}) - 1\right| < 0.01$$

Với chuỗi có $k$ số 1 trong $n = 1000$ lần tung:
$$p(x^{1000}) = (1/2)^{1000} = 2^{-1000}$$

$$-\frac{1}{1000}\log_2 2^{-1000} = 1$$

Vì đồng xu công bằng, **mọi** chuỗi đều có xác suất $2^{-1000}$, nên $-\frac{1}{n}\log p(x^n) = 1 = H(X)$ đúng với mọi chuỗi. Tức là $A_\epsilon^{(1000)} = \{0,1\}^{1000}$ với mọi $\epsilon > 0$.

Tổng số chuỗi: $2^{1000} = 2^{nH}$ ✓

## 4.3. Đồng xu lệch

Bây giờ tung đồng xu lệch với $p = 0.8$ (xác suất ngửa) và $n = 1000$.

$H(X) = H_b(0.8) = -0.8\log_2 0.8 - 0.2\log_2 0.2 \approx 0.722$ bit.

Chuỗi điển hình có khoảng $800$ số 1 và $200$ số 0.

Xác suất của chuỗi điển hình (với đúng $k \approx 800$ số 1):
$$p(x^{1000}) = 0.8^{800} \cdot 0.2^{200} = 2^{800\log 0.8 + 200\log 0.2}$$
$$= 2^{800(-0.322) + 200(-2.322)} = 2^{-257.6 - 464.4} = 2^{-722} \approx 2^{-nH}$$

Số chuỗi điển hình $\approx 2^{722}$ — ít hơn nhiều so với tổng số $2^{1000}$ chuỗi! Các chuỗi điển hình chỉ chiếm tỉ lệ nhỏ trong không gian, nhưng chiếm gần hết xác suất.

---

# 5. Ứng dụng: chứng minh nén được tới $H(X)$ bit/ký hiệu

## 5.1. Phương án mã hóa dựa trên tập điển hình

**Ý tưởng:** Mã hóa $n$ ký hiệu cùng một lúc. Chia chuỗi thành điển hình và không điển hình:

* **Chuỗi điển hình:** Gán $n(H+\epsilon) + 1$ bits (dùng bit đầu là $0$ để đánh dấu, rồi địa chỉ trong tập điển hình)
* **Chuỗi không điển hình:** Mã hóa bằng $n\log|\mathcal{X}| + 1$ bits (bit đầu là $1$, rồi mã ngây thơ)

## 5.2. Phân tích độ dài trung bình

**Xác suất lỗi:** $P_e^{(n)} = P(X^n \notin A_\epsilon^{(n)}) \to 0$.

**Độ dài mã trung bình:**
$$\mathbb{E}[l(X^n)] \le P(A) \cdot (n(H+\epsilon)+2) + P(A^c) \cdot (n\log|\mathcal{X}|+2)$$
$$\le n(H+\epsilon) + 2 + P(A^c) \cdot n\log|\mathcal{X}|$$

Với $n$ đủ lớn, $P(A^c) < \delta$ nhỏ tùy ý:
$$\frac{\mathbb{E}[l(X^n)]}{n} \le H + \epsilon + \delta\log|\mathcal{X}| + \frac{2}{n}$$

Với $n$ đủ lớn, tỉ số này tiến về $H+\epsilon$ (tùy ý nhỏ). Vậy tốc độ nén tiến về $H(X)$ bit/ký hiệu.

## 5.3. Cận dưới: không thể nén xuống dưới $H(X)$

**Định lý:** Bất kỳ phương án mã hóa không mất thông tin nào cũng cần ít nhất $H(X)$ bit/ký hiệu trong giới hạn $n \to \infty$.

**Ý tưởng chứng minh:**

Giả sử ta có mã hoàn toàn không mất thông tin (prefix-free) cho $n$ ký hiệu. Ta cần ít nhất $|\mathcal{X}^n|_{\text{need}}$ codeword, và mỗi codeword dài ít nhất $\log|\mathcal{X}^n|_{\text{need}}$ bit.

Vì phải giải mã đúng với xác suất ít nhất $1-\delta$, và các chuỗi điển hình có tổng xác suất $\ge 1-\delta$, ta cần ít nhất $|A_\epsilon^{(n)}| \ge (1-\delta)2^{n(H-\epsilon)}$ codeword riêng biệt.

Mã prefix-free cho một tập có $M$ phần tử cần trung bình ít nhất $\log_2 M$ bit. Suy ra:
$$\frac{L_n}{n} \ge \frac{\log_2((1-\delta)2^{n(H-\epsilon)})}{n} = H - \epsilon + \frac{\log(1-\delta)}{n}$$

Cho $n \to \infty$ và $\epsilon \to 0$: $L_n/n \ge H(X)$.

---

# 6. Định lý mã hóa nguồn Shannon

## 6.1. Phát biểu chính thức

**Định lý (Source Coding Theorem - Shannon 1948):**

Cho nguồn i.i.d. $X$ với entropy $H(X)$.

1. **(Nén được)** Với mọi $R > H(X)$, tồn tại phương án mã hóa tốc độ $R$ bit/ký hiệu sao cho xác suất lỗi giải mã $P_e^{(n)} \to 0$ khi $n \to \infty$.

2. **(Không thể nén hơn)** Với mọi $R < H(X)$ và mọi phương án mã hóa tốc độ $R$ bit/ký hiệu, xác suất lỗi $P_e^{(n)} \to 1$ khi $n \to \infty$.

## 6.2. Ý nghĩa

Định lý này nói rằng $H(X)$ là **ngưỡng nén**: trên ngưỡng thì nén được hoàn toàn, dưới ngưỡng thì chắc chắn mất thông tin.

Điều đáng kinh ngạc là kết quả này không phụ thuộc vào phương án mã hóa cụ thể. Dù ta dùng phương pháp thông minh đến đâu, không thể vượt qua cận này.

## 6.3. Tốc độ hội tụ

AEP không chỉ cho ta giới hạn asymptotic mà còn cho biết tốc độ hội tụ. Bằng bất đẳng thức Chebyshev:

$$P\left(\left|-\frac{1}{n}\log p(X^n) - H(X)\right| > \epsilon\right) \le \frac{\text{Var}(\log p(X))}{\epsilon^2 n}$$

Vậy xác suất lỗi giảm như $O(1/n)$ với $n$.

---

# 7. Điều kiện yếu hơn: nguồn ergodic

## 7.1. Mở rộng

Cho đến nay ta giả sử nguồn i.i.d. Thực tế, định lý AEP mở rộng cho các nguồn có nhớ (stationary ergodic sources):

$$-\frac{1}{n}\log p(X_1, \ldots, X_n) \xrightarrow{P} H(\mathcal{X})$$

Trong đó $H(\mathcal{X}) = \lim_{n\to\infty} H(X_n | X_1, \ldots, X_{n-1})$ là **entropy rate** của nguồn.

## 7.2. Entropy rate

Với nguồn có nhớ, entropy rate định nghĩa là:
$$H(\mathcal{X}) = \lim_{n\to\infty} \frac{1}{n} H(X_1, \ldots, X_n)$$

Giới hạn này tồn tại với nguồn dừng. Với nguồn i.i.d., $H(\mathcal{X}) = H(X_1)$.

## 7.3. Ví dụ: Markov chain

Xét chuỗi Markov 2 trạng thái với ma trận chuyển tiếp:
$$P = \begin{pmatrix} 0.9 & 0.1 \\ 0.3 & 0.7 \end{pmatrix}$$

Phân phối dừng: $\pi = (0.75, 0.25)$.

Entropy rate:
$$H(\mathcal{X}) = -\sum_i \pi_i \sum_j P_{ij}\log P_{ij}$$
$$= -0.75(0.9\log 0.9 + 0.1\log 0.1) - 0.25(0.3\log 0.3 + 0.7\log 0.7)$$
$$= 0.75 \cdot H_b(0.1) + 0.25 \cdot H_b(0.3)$$
$$= 0.75 \cdot 0.469 + 0.25 \cdot 0.881 = 0.352 + 0.220 = 0.572 \text{ bit}$$

So sánh với entropy biên $H(\pi) = H_b(0.25) = 0.811$ bit — entropy rate thấp hơn nhiều do cấu trúc Markov.

---

# 8. Thực hành: nhận dạng dãy điển hình

## 8.1. Thuật toán kiểm tra

Cho chuỗi $x^n$, kiểm tra xem có trong $A_\epsilon^{(n)}$ không:

1. Tính $\hat{p}(a) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}[x_i = a]$ (tần suất thực nghiệm)
2. Tính $s = -\frac{1}{n}\log p(x^n) = -\sum_{a} \hat{p}(a) \log p(a)$
3. Kiểm tra: nếu $|s - H(X)| < \epsilon$ thì $x^n$ là điển hình.

## 8.2. Ví dụ số

Nguồn: $P(0) = 0.8$, $P(1) = 0.2$, $H(X) = 0.722$ bit.

Chuỗi $n=10$: $x^{10} = 0001100000$ (có 8 số 0 và 2 số 1).

$$-\frac{1}{10}\log p(x^{10}) = -\frac{1}{10}\log(0.8^8 \cdot 0.2^2)$$
$$= -\frac{1}{10}(8\log 0.8 + 2\log 0.2) = -\frac{1}{10}(8\cdot(-0.322) + 2\cdot(-2.322))$$
$$= -\frac{1}{10}(-2.576 - 4.644) = \frac{7.22}{10} = 0.722$$

Chính xác bằng $H(X)$! Đây là vì chuỗi có đúng tần suất $8/10 = 0.8$ và $2/10 = 0.2$.

Chuỗi $y^{10} = 0000000000$ (10 số 0):
$$-\frac{1}{10}\log p(y^{10}) = -\frac{1}{10}\log(0.8^{10}) = -\log 0.8 = 0.322$$

$|0.322 - 0.722| = 0.4 > \epsilon$ với $\epsilon$ nhỏ → chuỗi toàn 0 **không điển hình**.

---

# 9. Tổng kết

AEP và tập điển hình là nền tảng của toàn bộ lý thuyết mã hóa nguồn:

**Định lý AEP:** $-\frac{1}{n}\log p(X^n) \xrightarrow{P} H(X)$

**Tập điển hình $A_\epsilon^{(n)}$:**
* $P(X^n \in A_\epsilon^{(n)}) \ge 1 - \delta$ (hầu hết chuỗi là điển hình)
* $p(x^n) \approx 2^{-nH}$ với mọi $x^n \in A_\epsilon^{(n)}$ (xác suất gần đều)
* $|A_\epsilon^{(n)}| \approx 2^{nH}$ (chỉ cần $2^{nH}$ codeword)

**Định lý nguồn Shannon:** Tốc độ tối thiểu để nén hoàn hảo là $H(X)$ bit/ký hiệu — đây là ngưỡng cứng không thể vượt qua.

Sức mạnh của kết quả này là ở tính **phổ quát**: không phụ thuộc vào cấu trúc phân phối cụ thể, chỉ cần i.i.d. là đủ.

> Bài tiếp theo — **Mã tối ưu và mã khối** — sẽ chuyển từ kết quả asymptotic sang câu hỏi thực tế hơn: với $n$ ký hiệu cố định (không cần $n \to \infty$), làm thế nào để thiết kế mã prefix-free tối ưu? Và block coding đạt tốc độ $H(X)$ nhanh đến đâu khi $n$ tăng?

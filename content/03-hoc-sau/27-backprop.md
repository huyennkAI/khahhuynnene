# Lan truyền ngược (Backpropagation)

> Lan truyền ngược (backpropagation) là cách tính **gradient của hàm mất mát theo mọi tham số** của một mạng nơ-ron một cách hiệu quả — chỉ trong **một lượt duyệt ngược** đồ thị tính toán, thay vì đạo hàm từng tham số một.
>
> $$\text{loss} \;\xrightarrow{\;\text{lan truyền ngược}\;}\; \frac{\partial L}{\partial \mathbf{W}^{(L)}},\; \dots,\; \frac{\partial L}{\partial \mathbf{W}^{(1)}}$$
>
> Cần nhấn mạnh ngay: backprop **chỉ tính gradient**. Việc dùng gradient đó để cập nhật tham số là phần việc của bộ tối ưu (xem [Tối ưu trong học sâu](#/toi-uu-hoc-sau)).

---

# 1. Vì sao cần một thuật toán riêng để tính gradient?

Mọi quá trình huấn luyện đều quy về một bài toán: tìm $\theta$ làm nhỏ hàm mất mát $L(\theta)$ (xem [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu)). Phương pháp chủ đạo là gradient descent, cần tới $\nabla_\theta L$.

Vấn đề là một mạng hiện đại có hàng triệu đến hàng tỉ tham số. Nếu tính đạo hàm theo từng tham số **một cách độc lập** — chẳng hạn bằng sai phân hữu hạn $\frac{L(\theta + \epsilon) - L(\theta)}{\epsilon}$ — thì mỗi tham số cần ít nhất một lần lan truyền xuôi. Với $n$ tham số, chi phí là $O(n)$ lần forward, tức **không khả thi**.

Trực giác của backprop xuất phát từ một quan sát đơn giản:

> Các tham số ở những lớp khác nhau **dùng chung** rất nhiều phép tính trung gian. Nếu khéo léo tái sử dụng các kết quả trung gian theo quy tắc dây chuyền, ta tính được **toàn bộ** gradient chỉ trong chi phí xấp xỉ **một** lần lan truyền xuôi.

Đây chính là điểm khiến học sâu trở nên khả thi về mặt tính toán.

---

# 2. Lan truyền xuôi (forward propagation)

Lan truyền xuôi là quá trình tính tuần tự từ đầu vào tới hàm mất mát, lưu lại các đại lượng trung gian. Xét một perceptron đa tầng (MLP) một tầng ẩn, đầu vào $\mathbf{x} \in \mathbb{R}^{d}$, không vẽ bias cho gọn (bias xử lý y hệt một cột trọng số với đầu vào hằng $1$).

$$
\begin{aligned}
\mathbf{z}^{(1)} &= \mathbf{W}^{(1)} \mathbf{x} && \text{(tổ hợp tuyến tính tầng ẩn)} \\
\mathbf{h} &= \sigma\!\left(\mathbf{z}^{(1)}\right) && \text{(kích hoạt phi tuyến)} \\
\mathbf{z}^{(2)} &= \mathbf{W}^{(2)} \mathbf{h} && \text{(tổ hợp tuyến tính tầng ra)} \\
\hat{\mathbf{y}} &= \operatorname{softmax}\!\left(\mathbf{z}^{(2)}\right) && \text{(xác suất dự đoán)} \\
L &= \ell(\hat{\mathbf{y}}, \mathbf{y}) && \text{(hàm mất mát, ví dụ cross-entropy)}
\end{aligned}
$$

trong đó $\sigma$ là kích hoạt theo từng phần tử (element-wise), $\mathbf{W}^{(1)} \in \mathbb{R}^{m \times d}$, $\mathbf{W}^{(2)} \in \mathbb{R}^{k \times m}$.

Điểm cốt lõi cần nhớ: **mỗi đại lượng trung gian** $\mathbf{z}^{(1)}, \mathbf{h}, \mathbf{z}^{(2)}$ đều được lưu lại. Lát nữa ta sẽ thấy lan truyền ngược cần chính những giá trị này, và đó là nguồn gốc của đánh đổi bộ nhớ–tính toán (mục 8).

---

# 3. Đồ thị tính toán (computational graph)

Mọi biểu thức ở mục 2 có thể vẽ thành một **đồ thị có hướng không chu trình** (DAG): mỗi nút là một phép toán, mỗi cạnh mang một tensor từ phép toán này sang phép toán sau.

```text
x ──▶ [·W⁽¹⁾] ──▶ z⁽¹⁾ ──▶ [σ] ──▶ h ──▶ [·W⁽²⁾] ──▶ z⁽²⁾ ──▶ [softmax,ℓ] ──▶ L
        ▲                                   ▲
       W⁽¹⁾                                W⁽²⁾
```

Lan truyền xuôi là duyệt đồ thị **theo chiều topo từ đầu vào tới $L$**. Lan truyền ngược là duyệt **đúng chiều ngược lại**, từ $L$ về các tham số. Mọi mạng — dù là CNN, RNN hay Transformer — đều rút về cùng một bức tranh: một đồ thị tính toán, và gradient chảy ngược trên đồ thị đó. Đây là lý do một khung tính toán duy nhất (autograd) có thể vi phân mọi kiến trúc.

---

# 4. Quy tắc dây chuyền nhiều biến

Backprop chỉ là quy tắc dây chuyền (chain rule) áp dụng có hệ thống. Nhắc lại dạng một biến:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Nhưng trong mạng, một đại lượng trung gian thường **ảnh hưởng tới $L$ qua nhiều đường**. Dạng nhiều biến cộng dồn đóng góp của mọi đường. Nếu $L$ phụ thuộc các biến $u_1, \dots, u_p$, và mỗi $u_j$ lại phụ thuộc $x$:

$$\frac{\partial L}{\partial x} = \sum_{j=1}^{p} \frac{\partial L}{\partial u_j}\,\frac{\partial u_j}{\partial x}$$

**Quy ước ký hiệu.** Với một tensor $\mathbf{a}$, viết tắt gradient của loss theo nó là

$$\nabla_{\mathbf{a}} L \equiv \frac{\partial L}{\partial \mathbf{a}}, \qquad (\nabla_{\mathbf{a}} L)_i = \frac{\partial L}{\partial a_i}$$

Với một phép toán tuyến tính $\mathbf{u} = \mathbf{W}\mathbf{v}$, quy tắc dây chuyền nhiều biến cho dạng ma trận gọn gàng:

$$\nabla_{\mathbf{v}} L = \mathbf{W}^{\top}\, \nabla_{\mathbf{u}} L$$

Đây là viên gạch then chốt: **gradient đi ngược qua một lớp tuyến tính chính là nhân với ma trận chuyển vị**. Ta chứng minh nhanh: $u_i = \sum_j W_{ij} v_j$ nên $\frac{\partial u_i}{\partial v_j} = W_{ij}$, do đó

$$(\nabla_{\mathbf{v}} L)_j = \sum_i \frac{\partial L}{\partial u_i}\frac{\partial u_i}{\partial v_j} = \sum_i W_{ij}\,(\nabla_{\mathbf{u}} L)_i = (\mathbf{W}^{\top}\nabla_{\mathbf{u}} L)_j \qquad \blacksquare$$

---

# 5. Suy dẫn gradient cho MLP một tầng ẩn

Giờ ta áp dụng mục 4 để suy ra gradient theo từng lớp, đi **ngược** từ $L$ về $\mathbf{x}$. Ý tưởng là định nghĩa **vector lỗi lan ngược (error term)** $\boldsymbol{\delta}$ tại mỗi tầng, rồi truyền nó qua từng phép toán.

## 5.1. Lỗi tại tầng ra

Định nghĩa lỗi tại đầu vào của softmax:

$$\boldsymbol{\delta}^{(2)} \equiv \nabla_{\mathbf{z}^{(2)}} L = \frac{\partial L}{\partial \mathbf{z}^{(2)}}$$

Với cặp **softmax + cross-entropy**, có một kết quả đẹp: lỗi chính là hiệu giữa dự đoán và nhãn.

**Mệnh đề.** $\boldsymbol{\delta}^{(2)} = \hat{\mathbf{y}} - \mathbf{y}$.

**Chứng minh.** Với nhãn one-hot $\mathbf{y}$, loss là $L = -\sum_c y_c \log \hat{y}_c$ và $\hat{y}_c = \frac{e^{z^{(2)}_c}}{\sum_j e^{z^{(2)}_j}}$. Đạo hàm softmax: $\frac{\partial \hat{y}_c}{\partial z^{(2)}_i} = \hat{y}_c(\mathbb{1}[c=i] - \hat{y}_i)$. Do đó

$$
\begin{aligned}
\frac{\partial L}{\partial z^{(2)}_i}
&= -\sum_c \frac{y_c}{\hat{y}_c}\,\hat{y}_c(\mathbb{1}[c=i] - \hat{y}_i)
= -\sum_c y_c(\mathbb{1}[c=i] - \hat{y}_i) \\
&= -y_i + \hat{y}_i \sum_c y_c = \hat{y}_i - y_i
\end{aligned}
$$

vì $\sum_c y_c = 1$. Vậy $\boldsymbol{\delta}^{(2)} = \hat{\mathbf{y}} - \mathbf{y}$. $\blacksquare$

## 5.2. Gradient theo $\mathbf{W}^{(2)}$

Vì $\mathbf{z}^{(2)} = \mathbf{W}^{(2)}\mathbf{h}$ nên $z^{(2)}_i = \sum_j W^{(2)}_{ij} h_j$, suy ra $\frac{\partial z^{(2)}_i}{\partial W^{(2)}_{ab}} = \mathbb{1}[a=i]\,h_b$. Áp quy tắc dây chuyền:

$$\frac{\partial L}{\partial W^{(2)}_{ab}} = \sum_i \delta^{(2)}_i \frac{\partial z^{(2)}_i}{\partial W^{(2)}_{ab}} = \delta^{(2)}_a h_b$$

Viết gọn dưới dạng **tích ngoài (outer product)**:

$$\boxed{\;\nabla_{\mathbf{W}^{(2)}} L = \boldsymbol{\delta}^{(2)} \mathbf{h}^{\top}\;}$$

Cấu trúc này lặp lại ở mọi lớp tuyến tính: **gradient theo trọng số = (lỗi đi ra) $\times$ (đầu vào đi vào)$^\top$**.

## 5.3. Lan lỗi về tầng ẩn

Lỗi tiếp tục chảy ngược qua lớp tuyến tính $\mathbf{W}^{(2)}$. Dùng công thức chuyển vị ở mục 4:

$$\nabla_{\mathbf{h}} L = \big(\mathbf{W}^{(2)}\big)^{\top} \boldsymbol{\delta}^{(2)}$$

Tiếp đó lỗi đi qua kích hoạt $\mathbf{h} = \sigma(\mathbf{z}^{(1)})$. Vì $\sigma$ tác động theo **từng phần tử**, $\frac{\partial h_i}{\partial z^{(1)}_j} = \sigma'(z^{(1)}_i)\,\mathbb{1}[i=j]$ là ma trận đường chéo. Nhân với một ma trận đường chéo chính là **nhân từng phần tử (Hadamard)** với $\odot$:

$$\boxed{\;\boldsymbol{\delta}^{(1)} \equiv \nabla_{\mathbf{z}^{(1)}} L = \Big(\big(\mathbf{W}^{(2)}\big)^{\top}\boldsymbol{\delta}^{(2)}\Big) \odot \sigma'\!\big(\mathbf{z}^{(1)}\big)\;}$$

Đây là **trái tim của backprop**: lỗi tầng sau nhân chuyển vị trọng số để lùi một lớp tuyến tính, rồi nhân $\odot$ với đạo hàm kích hoạt để lùi qua phi tuyến. Lưu ý ta cần $\mathbf{z}^{(1)}$ đã lưu ở forward để tính $\sigma'$.

## 5.4. Gradient theo $\mathbf{W}^{(1)}$

Hoàn toàn đối xứng với mục 5.2, với đầu vào của lớp là $\mathbf{x}$:

$$\boxed{\;\nabla_{\mathbf{W}^{(1)}} L = \boldsymbol{\delta}^{(1)} \mathbf{x}^{\top}\;}$$

Tổng hợp lại, **toàn bộ** gradient của mạng được tính bằng một chuỗi ngắn các phép nhân ma trận và nhân $\odot$:

$$
\boldsymbol{\delta}^{(2)} = \hat{\mathbf{y}} - \mathbf{y}
\;\longrightarrow\;
\nabla_{\mathbf{W}^{(2)}} L = \boldsymbol{\delta}^{(2)}\mathbf{h}^{\top}
\;\longrightarrow\;
\boldsymbol{\delta}^{(1)} = \big((\mathbf{W}^{(2)})^{\top}\boldsymbol{\delta}^{(2)}\big)\odot\sigma'(\mathbf{z}^{(1)})
\;\longrightarrow\;
\nabla_{\mathbf{W}^{(1)}} L = \boldsymbol{\delta}^{(1)}\mathbf{x}^{\top}
$$

So sánh chi phí: mỗi lớp chỉ tốn cùng cỡ phép nhân ma trận như khi forward. Tổng chi phí backward $\approx$ chi phí forward, tức $O(1)$ lần forward chứ không phải $O(n)$.

---

# 6. Tổng quát cho mạng $L$ tầng

Quy luật ở mục 5 mở rộng nguyên vẹn cho mạng sâu $\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)}$, $\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$ với $\mathbf{a}^{(0)} = \mathbf{x}$. Hai phương trình truy hồi (recurrence) sau là toàn bộ thuật toán:

$$
\begin{aligned}
\boldsymbol{\delta}^{(l)} &= \Big(\big(\mathbf{W}^{(l+1)}\big)^{\top}\boldsymbol{\delta}^{(l+1)}\Big) \odot \sigma'\!\big(\mathbf{z}^{(l)}\big) && \text{(lan lỗi ngược một tầng)} \\
\nabla_{\mathbf{W}^{(l)}} L &= \boldsymbol{\delta}^{(l)}\,\big(\mathbf{a}^{(l-1)}\big)^{\top} && \text{(gradient trọng số tầng } l\text{)}
\end{aligned}
$$

khởi đầu từ $\boldsymbol{\delta}^{(L)}$ ở tầng ra. Mỗi tầng dùng lại $\boldsymbol{\delta}^{(l+1)}$ của tầng kế sau — đây chính là sự **tái sử dụng kết quả trung gian** mà mục 1 đã hứa hẹn.

| Đại lượng | Dùng ở forward | Dùng ở backward |
| --- | --- | --- |
| $\mathbf{a}^{(l-1)}$ (đầu vào tầng) | tính $\mathbf{z}^{(l)}$ | tính $\nabla_{\mathbf{W}^{(l)}} L = \boldsymbol{\delta}^{(l)}(\mathbf{a}^{(l-1)})^{\top}$ |
| $\mathbf{z}^{(l)}$ (trước kích hoạt) | tính $\mathbf{a}^{(l)}$ | tính $\sigma'(\mathbf{z}^{(l)})$ trong $\boldsymbol{\delta}^{(l)}$ |
| $\mathbf{W}^{(l+1)}$ (trọng số trên) | tính $\mathbf{z}^{(l+1)}$ | lan lỗi $\big(\mathbf{W}^{(l+1)}\big)^{\top}\boldsymbol{\delta}^{(l+1)}$ |

---

# 7. Thuật toán backprop tổng quát: duyệt ngược đồ thị

Không cần suy dẫn tay cho từng kiến trúc. Với một đồ thị tính toán bất kỳ, mỗi nút chỉ cần biết **đạo hàm cục bộ** (local Jacobian) của đầu ra theo đầu vào. Thuật toán tổng quát:

1. **Forward:** duyệt đồ thị theo thứ tự topo từ đầu vào tới $L$, tính và **lưu** giá trị mỗi nút.
2. Khởi tạo $\nabla_L L = 1$.
3. **Backward:** duyệt đồ thị theo thứ tự topo **ngược lại**. Tại mỗi nút $u$ với đầu vào $v_1, \dots, v_p$, lấy gradient $\nabla_u L$ đã tích lũy, rồi đẩy về từng đầu vào:
   $$\nabla_{v_j} L \mathrel{+}= \left(\frac{\partial u}{\partial v_j}\right)^{\top}\nabla_u L$$
   Dấu cộng dồn ($\mathrel{+}=$) hiện thực hóa **quy tắc dây chuyền nhiều biến** (mục 4): một nút dùng ở nhiều nơi thì gom đóng góp từ mọi đường.

```python
# backward trên đồ thị đã được sắp topo
node.grad = {output: 1.0}
for node in reversed(topo_order):
    for inp in node.inputs:
        inp.grad += node.local_jacobian(inp).T @ node.grad
```

Thứ tự topo ngược **đảm bảo** khi xử lý một nút, gradient của nó đã được mọi nút phía sau cộng dồn xong — nên chỉ cần một lượt là đủ.

---

# 8. Vì sao phải lưu activation: đánh đổi bộ nhớ–tính toán

Nhìn lại mục 6: công thức backward cho mỗi tầng **cần các giá trị forward** ($\mathbf{a}^{(l-1)}$ và $\mathbf{z}^{(l)}$). Do đó forward buộc phải **giữ lại các activation trung gian** cho tới khi backward dùng xong.

Đây chính là đánh đổi cốt lõi của backprop:

> Backprop nhanh ($O(1)$ lần forward về thời gian) **đổi lại** bằng bộ nhớ $O(\text{số activation})$ để lưu trữ. Bộ nhớ tỉ lệ với độ sâu mạng nhân kích thước batch — và đây thường là nút thắt khi huấn luyện mô hình lớn.

Có hai cực:

* **Lưu hết activation** (mặc định): backward nhanh nhất, tốn bộ nhớ nhất.
* **Gradient checkpointing:** chỉ lưu một phần activation, khi backward cần phần đã bỏ thì **tính lại** bằng forward cục bộ. Đổi thời gian lấy bộ nhớ — thường giảm bộ nhớ xuống $O(\sqrt{L})$ với chi phí khoảng một lần forward phụ.

Cũng vì lý do bộ nhớ này mà ổn định số (numerical stability) trở nên quan trọng: các activation và gradient lưu trữ phải nằm trong khoảng biểu diễn được, tránh tràn hoặc triệt tiêu (xem [Ổn định số](#/on-dinh-so)).

---

# 9. Liên hệ tự động vi phân (autograd / reverse-mode AD)

Backprop chính là một trường hợp riêng của **tự động vi phân chế độ ngược (reverse-mode automatic differentiation)**. Các khung như PyTorch, JAX, TensorFlow dựng đồ thị tính toán trong lúc bạn viết forward, rồi tự chạy đúng thuật toán ở mục 7.

Vì sao **reverse-mode** chứ không phải forward-mode? Với hàm $f: \mathbb{R}^n \to \mathbb{R}^m$:

* **Forward-mode** lan đạo hàm theo chiều xuôi, chi phí tỉ lệ số đầu vào $n$.
* **Reverse-mode** lan ngược, chi phí tỉ lệ số đầu ra $m$.

Trong học sâu, đầu ra là **một số vô hướng** ($m = 1$, chính là loss) còn đầu vào là **hàng triệu tham số** ($n$ rất lớn). Reverse-mode cho toàn bộ $\nabla_\theta L$ chỉ trong một lượt — đây đúng là lý do nó (và backprop) thống trị.

```python
# autograd lo toàn bộ mục 7 thay bạn
y_hat = model(x)          # forward, tự dựng đồ thị
loss = loss_fn(y_hat, y)
loss.backward()           # reverse-mode AD: điền .grad cho mọi tham số
```

Sau `loss.backward()`, mọi tham số đã có `.grad` — và **đến đây backprop kết thúc**.

---

# 10. Backprop chỉ tính gradient — cập nhật là việc của tối ưu

Một hiểu lầm phổ biến là gộp backprop với "huấn luyện". Cần tách bạch:

* **Backprop** trả lời câu hỏi *"gradient bằng bao nhiêu?"* — thuần túy là vi phân.
* **Bộ tối ưu (optimizer)** trả lời *"dùng gradient đó để đi bước thế nào?"* — đó là SGD, Momentum, Adam... (xem [Tối ưu trong học sâu](#/toi-uu-hoc-sau)).

Quy trình một bước huấn luyện ghép hai phần lại:

```python
y_hat = model(x)
loss = loss_fn(y_hat, y)   # lan truyền xuôi (mục 2)
optimizer.zero_grad()
loss.backward()            # lan truyền ngược: chỉ TÍNH gradient
optimizer.step()           # cập nhật tham số: việc của tối ưu
```

Cùng một gradient từ backprop, bạn có thể đưa vào SGD, Adam hay bất kỳ bộ tối ưu nào — kết quả huấn luyện khác nhau hoàn toàn, nhưng **bước tính gradient thì y hệt**. Hàm mất mát quyết định gradient nói gì (xem [Hàm mất mát & Tối ưu hóa](#/ham-mat-mat-toi-uu)); backprop chỉ tính nó cho hiệu quả.

---

# 11. Ưu điểm & Lưu ý

**Ưu điểm.**

* **Hiệu quả tối đa về thời gian** — toàn bộ gradient trong $O(1)$ lần forward, nhờ tái sử dụng kết quả trung gian qua quy tắc dây chuyền.
* **Tổng quát** — chỉ cần đạo hàm cục bộ tại mỗi nút; autograd áp dụng cho mọi kiến trúc mà không cần suy dẫn tay.
* **Chính xác** — cho gradient giải tích, không có sai số xấp xỉ như sai phân hữu hạn.

**Lưu ý.**

* **Tốn bộ nhớ** — phải lưu activation forward; là nút thắt với mô hình lớn (mục 8). Cân nhắc gradient checkpointing.
* **Phụ thuộc ổn định số** — gradient có thể bùng nổ hoặc triệt tiêu khi lan qua nhiều tầng; chọn kích hoạt và khởi tạo cẩn thận (xem [Ổn định số](#/on-dinh-so)).
* **Cần đúng giá trị forward** — sửa activation tại chỗ (in-place) sau forward có thể làm sai gradient.

---

# 12. Tổng kết

Backpropagation **không phải** một thuật toán huấn luyện hoàn chỉnh, mà là **cách tính gradient hiệu quả**. Bản chất, đúng như các suy dẫn ở mục 4–6, chỉ là quy tắc dây chuyền nhiều biến áp dụng có hệ thống lên đồ thị tính toán, đọng lại ở hai phương trình truy hồi:

$$
\boldsymbol{\delta}^{(l)} = \Big(\big(\mathbf{W}^{(l+1)}\big)^{\top}\boldsymbol{\delta}^{(l+1)}\Big) \odot \sigma'\!\big(\mathbf{z}^{(l)}\big),
\qquad
\nabla_{\mathbf{W}^{(l)}} L = \boldsymbol{\delta}^{(l)}\big(\mathbf{a}^{(l-1)}\big)^{\top}
$$

Lan truyền xuôi tính loss và lưu các activation; lan truyền ngược duyệt đồ thị theo chiều ngược, đẩy lỗi $\boldsymbol{\delta}$ qua từng lớp bằng nhân chuyển vị trọng số và nhân $\odot$ với đạo hàm kích hoạt. Cái giá là bộ nhớ lưu activation — một đánh đổi bộ nhớ–tính toán cốt lõi.

Backprop đổi **bộ nhớ** lấy **tốc độ tính gradient**, và đó là điều khiến học sâu khả thi. Khi `loss.backward()` chạy xong, công việc của nó kết thúc — phần còn lại, biến gradient thành tham số tốt hơn, thuộc về [bộ tối ưu](#/toi-uu-hoc-sau).

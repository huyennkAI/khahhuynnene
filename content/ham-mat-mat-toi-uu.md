# Hàm mất mát & Tối ưu hóa

> Một mô hình "học" bằng cách **đo độ sai** qua hàm mất mát, rồi điều chỉnh tham số để giảm độ sai đó. Đây là trái tim của mọi thuật toán học máy hiện đại.

## Hàm mất mát là gì?

Hàm mất mát (loss function) $L(\theta)$ định lượng khoảng cách giữa dự đoán $\hat{y}$ và giá trị thật $y$.

**Mean Squared Error** cho hồi quy:

$$L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2$$

**Cross-Entropy** cho phân loại:

$$L(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c} y_{i,c} \log \hat{y}_{i,c}$$

Mục tiêu huấn luyện là tìm $\theta^{*}$ sao cho:

$$\theta^{*} = \arg\min_{\theta} \; L(\theta)$$

## Gradient Descent

Ý tưởng: gradient $\nabla L(\theta)$ chỉ hướng **tăng nhanh nhất** của hàm mất mát, nên ta đi ngược lại để giảm:

$$\theta \leftarrow \theta - \eta \, \nabla_\theta L(\theta)$$

trong đó $\eta$ là **tốc độ học** (learning rate).

- $\eta$ quá lớn → dao động, không hội tụ.
- $\eta$ quá nhỏ → hội tụ chậm.

### Các biến thể

| Biến thể | Đặc điểm |
| --- | --- |
| Batch GD | Dùng toàn bộ dữ liệu mỗi bước, ổn định nhưng chậm |
| Stochastic GD | Dùng 1 mẫu mỗi bước, nhanh nhưng nhiễu |
| Mini-batch GD | Dùng một lô nhỏ, cân bằng tốc độ và ổn định |

Các bộ tối ưu nâng cao như **Momentum**, **RMSProp**, **Adam** điều chỉnh bước đi thông minh hơn dựa trên lịch sử gradient.

## Lan truyền ngược (Backpropagation)

Trong mạng nơ-ron, gradient được tính hiệu quả nhờ **quy tắc chuỗi**:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Thuật toán lan truyền sai số từ lớp đầu ra ngược về các lớp trước, cập nhật mọi trọng số trong một lượt.

```python
for epoch in range(num_epochs):
    y_hat = model.forward(x)
    loss = loss_fn(y_hat, y)
    grads = loss.backward()
    for p in model.parameters():
        p -= learning_rate * p.grad
```

> Hiểu được gradient descent là hiểu được vì sao mô hình cần dữ liệu, vì sao huấn luyện tốn thời gian, và vì sao learning rate lại quan trọng đến vậy.

# 预测乳腺癌,使用pandas对数据集合进行加载
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

df = pd.read_csv("C:\\Users\\ASUS\\Desktop\\机器学习\\大作业\\data.csv")
# print(df)
df_class={'B':0,'M':1}
df['diagnosis']=df['diagnosis'].map(df_class)

# 每行数据有33个乳腺病理特征, 第3列表示是否患有乳腺癌
X = df[df.columns[0:-1]].values
Y = df[df.columns[1]].values
# print(X.shape,Y.shape)

#共有569条数据，其中训练数据455条、测试数据114条。
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, 
                             test_size=0.2,random_state=1234)
print(f'X_train.shape={X_train.shape}')
print(f'Y_train.shape={Y_train.shape}')
print(f'X_test.shape={X_test.shape}')
print(f'Y_test.shape={Y_test.shape}')

# 对特征进行标准化,其中，fit_transform()的功能是对数据进行某种统一处理，
# 将数据缩放(映射)到某个固定区间。实现数据的标准化、归一化等等。作用：保留特征，去除异常值。
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Y_train = sc.fit_transform(Y_train)
# Y_test = sc.fit_transform(Y_test)
#最后, 为了使用PyTorch进一步处理，将数据封装到PyTorch的张量对象中
X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))

X_test = torch.from_numpy(X_test.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

# 将标记集合 Y_train 和 Y_test 转成2维
Y_train = Y_train.view(Y_train.shape[0],1)
Y_test = Y_test.view(Y_test.shape[0],1)
print(Y_train.size(),Y_test.size())

"""
采用线性模型f(x)= wx+b 预测乳腺癌，只有两种结果，是或否（通常用1表示是，0表示否）。该模型是一种二分类模型。
如果预测概率值小于0.5，则表示没有患乳腺癌（f(x)<0.5，输出 0）
如果预测概率值大于等于0.5，则表示患有乳腺癌（f(x)>=0.5，则输出 1）。
"""
class MyModel(torch.nn.Module):
    def __init__(self,in_features):
        super(MyModel,self).__init__()   #调用父类的构造函数！
        # 搭建自己的神经网络
        # 1.构建线性层
        self.linear = torch.nn.Linear(in_features,1)
        # 2.构建激活函数层
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """重写了父类的forward函数，正向传播"""
        pred = self.linear(x)
        out = self.sigmoid(pred)
        return out

"""神经网络定义完成后，需进行：
1.确定代价（损失）函数、学习率
2.构建神经网络模型对象
3.构建优化器对象，并为优化器指定模型参数和学习率
"""
# 损失函数公式定义
loss = torch.nn.BCELoss()

# 学习率，迭代次数
learning_rate = 0.01
num_epochs = 10000

# 获取样本量和特征数，创建模型
n_samples,n_features = X.shape
model = MyModel(n_features)

# 创建优化器，
optimizer = torch.optim.SGD(
                        model.parameters(),lr=learning_rate)

#打印模型、打印模型参数
print(model)
print(list(model.parameters()))

def check_input_range(input_tensor):
    if torch.min(input_tensor) < 0 or torch.max(input_tensor) > 1:
        # 对输入数据进行处理，确保在0和1之间
        input_tensor = input_tensor.clamp(0, 1)
    return input_tensor

threshold_value = 0.00001
Y_train = check_input_range(Y_train)
for epoch in range(num_epochs):
    pred = model(X_train)  # 正向传播，调用forward()方法
    ls = loss(pred,Y_train)   # 计算损失（标量值）
    # ls =  0 * sum([x.sum() for x in model.parameters()])
    ls.backward()   # 反向传播
    optimizer.step()     # 更新权重
    optimizer.zero_grad()    # 清空梯度
    # if epoch%500 == 0:
    #     print(f"epoch:{epoch},loss={ls.item():.4f}")
    if ls.item() <= threshold_value:
        break;
print("模型训练完成! loss={0}".format(ls))

#使用区别于训练数据的另外数据集（测试集），计算模型预测的准确率
with torch.no_grad():       # 无需向后传播（非训练过程）
    y_pred = model(X_test)
    # 上面计算出来的结果是0-1之间的数,将数据进行四舍五入,得到0或1
    y_pred_cls = np.round(y_pred)
    # print(y_pred)
    # 统计结果
    acc = torch.eq(Y_test, y_pred_cls).sum().numpy() / float(Y_test.shape[0])
    # print(torch.eq(Y_test, y_pred_cls).sum().numpy())
    print(f"准确率:{acc.item() * 100 :.2f}"+'%')
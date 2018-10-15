---
- [逻辑回归分类器](## 逻辑回归分类器)
- [逻辑斯蒂回归](##逻辑回归分类器)

## 逻辑回归分类器
1. 逻辑回归分类器是在线性回归的基础上构建，为了使得输入取值仅为0、1，因此采用了sigmoid函数映射
2. 逻辑回归模型：ln y/(1-y) = w*x+b
3. 逻辑回归得到P(Y=1|X) 和 P(Y=0|X)后，采用极大似然估计求解模型参数w、b
4. 极大似然估计思想：在样本已知的情况下，求解模型参数，使得样本出现的概率最大
   其中用似然函数表示样本出现的概率，因此极大似然估计即为使得似然函数最大化的参数
5. logistic回归的损失函数为对数损失   
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;L(w)&=-\log\left&space;(&space;\prod_{i=1}^N&space;[{\color{Red}&space;\sigma(x_i)}]^{{\color{Blue}&space;y_i}}&space;[{\color{Red}&space;1-&space;\sigma(x_i)}]^{{\color{Blue}&space;1-y_i}}&space;\right&space;)\\&space;&=-\sum_{i=1}^N&space;\left&space;[&space;y_i\log\sigma(x_i)&plus;(1-y_i)\log(1-\sigma(x_i))&space;\right&space;]\\&space;&=-\sum_{i=1}^N&space;\left&space;[&space;y_i\log\frac{\sigma(x_i)}{1-\sigma(x_i)}&plus;\log(1-\sigma(x_i))&space;\right&space;]&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;L(w)&=-\log\left&space;(&space;\prod_{i=1}^N&space;[{\color{Red}&space;\sigma(x_i)}]^{{\color{Blue}&space;y_i}}&space;[{\color{Red}&space;1-&space;\sigma(x_i)}]^{{\color{Blue}&space;1-y_i}}&space;\right&space;)\\&space;&=-\sum_{i=1}^N&space;\left&space;[&space;y_i\log\sigma(x_i)&plus;(1-y_i)\log(1-\sigma(x_i))&space;\right&space;]\\&space;&=-\sum_{i=1}^N&space;\left&space;[&space;y_i\log\frac{\sigma(x_i)}{1-\sigma(x_i)}&plus;\log(1-\sigma(x_i))&space;\right&space;]&space;\end{aligned}" title="\begin{aligned} L(w)&=-\log\left ( \prod_{i=1}^N [{\color{Red} \sigma(x_i)}]^{{\color{Blue} y_i}} [{\color{Red} 1- \sigma(x_i)}]^{{\color{Blue} 1-y_i}} \right )\\ &=-\sum_{i=1}^N \left [ y_i\log\sigma(x_i)+(1-y_i)\log(1-\sigma(x_i)) \right ]\\ &=-\sum_{i=1}^N \left [ y_i\log\frac{\sigma(x_i)}{1-\sigma(x_i)}+\log(1-\sigma(x_i)) \right ] \end{aligned}" /></a>
5. 参数优化: 针对上述损失函数，利用梯度下降法求解模型参数，梯度下降分为以下几步：
- 计算下降方向，即计算梯度
- 选择步长，更新参数（沿梯度负方向）
- 重复以上两步，直到两次迭代的差值小于某一阈值，则停止更新  
ps: 随机梯度下降与普通梯度下降的区别在于前者更新参数时只使用一个样本，而后者却使用所有训练样本
5. 多分类
- 多项式逻辑回归模型（类似于softmax回归）   

<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;P(Y=k|x)&=\frac{\exp(w_kx)}{1&plus;\sum_{k=1}^{K-1}&space;\exp(w_kx)}&space;\quad&space;k=1,2,..,K-1&space;\\&space;P(Y=K|x)&=\frac{1}{1&plus;\sum_{k=1}^{K-1}\exp(w_kx)}&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;P(Y=k|x)&=\frac{\exp(w_kx)}{1&plus;\sum_{k=1}^{K-1}&space;\exp(w_kx)}&space;\quad&space;k=1,2,..,K-1&space;\\&space;P(Y=K|x)&=\frac{1}{1&plus;\sum_{k=1}^{K-1}\exp(w_kx)}&space;\end{aligned}" title="\begin{aligned} P(Y=k|x)&=\frac{\exp(w_kx)}{1+\sum_{k=1}^{K-1} \exp(w_kx)} \quad k=1,2,..,K-1 \\ P(Y=K|x)&=\frac{1}{1+\sum_{k=1}^{K-1}\exp(w_kx)} \end{aligned}" /></a>
- 多分类逻辑回归采用[softmax回归](https://tech.meituan.com/intro_to_logistic_regression.html)   
- 也可采用构建多个分类器的方法，如one-vs-one、one-vs-all, 
  当类别之间互斥时，则需采用softmax回归
   

## 支持向量机
1. 基本思想：在特征空间中找到一个分割超平面，能够正确划分样本数据集，同时间隔最大化，间隔是指样本数据点到超平面的最短距离
2. 函数间隔：样本数据点（x,y）关于超平面 w*x + b的函数间隔定义为 <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{r}=w*x_i&plus;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{r}=w*x_i&plus;b" title="\hat{r}=w*x_i+b" /></a>   
   几何间隔：对w作规范化  
        <a href="https://www.codecogs.com/eqnedit.php?latex=r&space;=&space;\frac{w}{\left&space;\|&space;w&space;\right&space;\|}&space;*&space;x_i&space;&plus;&space;\frac{b}{\left&space;\|&space;w&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r&space;=&space;\frac{w}{\left&space;\|&space;w&space;\right&space;\|}&space;*&space;x_i&space;&plus;&space;\frac{b}{\left&space;\|&space;w&space;\right&space;\|}" title="r = \frac{w}{\left \| w \right \|} * x_i + \frac{b}{\left \| w \right \|}" /></a>
3. 间隔最大化的目标函数  (线性可分)
- 最大化超平面对训练集的几何间隔r（所有样本点几何间隔的最小值）   
- 同时超平面关于每个样本点的几何间隔r 需满足约束条件，至少大于等于r   
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;&{\color{Red}&space;\underset{w,b}{\max}}&space;\quad\gamma&space;\\&space;&\&space;\mathrm{s.t.}\quad,&space;y_i(\frac{w}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}}x_i&plus;\frac{b}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}})&space;\geq&space;\gamma,\quad&space;i=1,2,\cdots,N&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;&{\color{Red}&space;\underset{w,b}{\max}}&space;\quad\gamma&space;\\&space;&\&space;\mathrm{s.t.}\quad,&space;y_i(\frac{w}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}}x_i&plus;\frac{b}{{\color{Red}&space;\left&space;|&space;w&space;\right&space;|}})&space;\geq&space;\gamma,\quad&space;i=1,2,\cdots,N&space;\end{aligned}" title="\begin{aligned} &{\color{Red} \underset{w,b}{\max}} \quad\gamma \\ &\ \mathrm{s.t.}\quad, y_i(\frac{w}{{\color{Red} \left | w \right |}}x_i+\frac{b}{{\color{Red} \left | w \right |}}) \geq \gamma,\quad i=1,2,\cdots,N \end{aligned}" /></a>
- 由于函数间隔取值对目标函数最大化没有影响，因此令函数间隔为1，最大化问题转化为最小化
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;&{\color{Red}&space;\underset{w,b}{\min}&space;}&space;\&space;\frac{1}{2}{\color{Red}&space;\left&space;\|&space;w&space;\right&space;\|}&space;\\&space;&\&space;\mathrm{s.t.}\quad\,&space;y_i(wx_i&plus;b)&space;\geq&space;\&space;1,\quad&space;i=1,2,\cdots,N&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;&{\color{Red}&space;\underset{w,b}{\min}&space;}&space;\&space;\frac{1}{2}{\color{Red}&space;\left&space;\|&space;w&space;\right&space;\|}&space;\\&space;&\&space;\mathrm{s.t.}\quad\,&space;y_i(wx_i&plus;b)&space;\geq&space;\&space;1,\quad&space;i=1,2,\cdots,N&space;\end{aligned}" title="\begin{aligned} &{\color{Red} \underset{w,b}{\min} } \ \frac{1}{2}{\color{Red} \left \| w \right \|} \\ &\ \mathrm{s.t.}\quad\, y_i(wx_i+b) \geq \ 1,\quad i=1,2,\cdots,N \end{aligned}" /></a>
4.  针对上述凸二次规划问题，采用拉格朗日对偶算法求解
1. 构建**拉格朗日函数**     
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;L(w,b,{\color{Red}&space;\alpha})=&\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red}&space;\alpha_i}[y_i(w^Tx_i&plus;b)-1]\\&space;&{\color{Red}&space;\alpha_i&space;\geq&space;0},\quad&space;i=1,2,\cdots,N&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;L(w,b,{\color{Red}&space;\alpha})=&\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red}&space;\alpha_i}[y_i(w^Tx_i&plus;b)-1]\\&space;&{\color{Red}&space;\alpha_i&space;\geq&space;0},\quad&space;i=1,2,\cdots,N&space;\end{aligned}" title="\begin{aligned} L(w,b,{\color{Red} \alpha})=&\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red} \alpha_i}[y_i(w^Tx_i+b)-1]\\ &{\color{Red} \alpha_i \geq 0},\quad i=1,2,\cdots,N \end{aligned}" /></a>
1. 标准问题是求极小极大问题：    
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;{\color{Red}&space;\underset{w,b}{\min}}\&space;{\color{Blue}&space;\underset{\alpha}{\max}}\&space;L(w,b,\alpha)&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;{\color{Red}&space;\underset{w,b}{\min}}\&space;{\color{Blue}&space;\underset{\alpha}{\max}}\&space;L(w,b,\alpha)&space;\end{aligned}" title="\begin{aligned} {\color{Red} \underset{w,b}{\min}}\ {\color{Blue} \underset{\alpha}{\max}}\ L(w,b,\alpha) \end{aligned}" /></a>    
    其对偶问题为：    
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;{\color{Blue}&space;\underset{\alpha}{\max}}\&space;{\color{Red}&space;\underset{w,b}{\min}}\&space;L(w,b,\alpha)&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;{\color{Blue}&space;\underset{\alpha}{\max}}\&space;{\color{Red}&space;\underset{w,b}{\min}}\&space;L(w,b,\alpha)&space;\end{aligned}" title="\begin{aligned} {\color{Blue} \underset{\alpha}{\max}}\ {\color{Red} \underset{w,b}{\min}}\ L(w,b,\alpha) \end{aligned}" /></a>
1. 求 `L` 对 `(w,b)` 的极小    
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\mathrm{set}\quad&space;\frac{\partial&space;L}{\partial&space;w}=0&space;\;\;&\Rightarrow\;&space;w-\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i&space;x_i}=0\\&space;&\Rightarrow\;&space;w=\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i&space;x_i}&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;\mathrm{set}\quad&space;\frac{\partial&space;L}{\partial&space;w}=0&space;\;\;&\Rightarrow\;&space;w-\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i&space;x_i}=0\\&space;&\Rightarrow\;&space;w=\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i&space;x_i}&space;\end{aligned}" title="\begin{aligned} \mathrm{set}\quad \frac{\partial L}{\partial w}=0 \;\;&\Rightarrow\; w-\sum_{i=1}^N {\color{Red} \alpha_i y_i x_i}=0\\ &\Rightarrow\; w=\sum_{i=1}^N {\color{Red} \alpha_i y_i x_i} \end{aligned}" /></a>   
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\mathrm{set}\quad&space;\frac{\partial&space;L}{\partial&space;b}=0&space;\;\;&\Rightarrow\;&space;\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i}=0&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;\mathrm{set}\quad&space;\frac{\partial&space;L}{\partial&space;b}=0&space;\;\;&\Rightarrow\;&space;\sum_{i=1}^N&space;{\color{Red}&space;\alpha_i&space;y_i}=0&space;\end{aligned}" title="\begin{aligned} \mathrm{set}\quad \frac{\partial L}{\partial b}=0 \;\;&\Rightarrow\; \sum_{i=1}^N {\color{Red} \alpha_i y_i}=0 \end{aligned}" /></a>   
    结果代入`L`，有：   
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;L(w,b,{\color{Red}&space;\alpha})&space;&=\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red}&space;\alpha_i}[y_i(w^Tx_i&plus;b)-1]\\&space;&=\frac{1}{2}w^Tw-w^T\sum_{i=1}^N&space;\alpha_iy_ix_i-b\sum_{i=1}^N&space;\alpha_iy_i&plus;\sum_{i=1}^N&space;\alpha_i\\&space;&=\frac{1}{2}w^Tw-w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\\&space;&=-\frac{1}{2}w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\\&space;&=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N&space;\alpha_i\alpha_j\cdot&space;y_iy_j\cdot&space;{\color{Red}&space;x_i^Tx_j}&plus;\sum_{i=1}^N&space;\alpha_i&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;L(w,b,{\color{Red}&space;\alpha})&space;&=\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red}&space;\alpha_i}[y_i(w^Tx_i&plus;b)-1]\\&space;&=\frac{1}{2}w^Tw-w^T\sum_{i=1}^N&space;\alpha_iy_ix_i-b\sum_{i=1}^N&space;\alpha_iy_i&plus;\sum_{i=1}^N&space;\alpha_i\\&space;&=\frac{1}{2}w^Tw-w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\\&space;&=-\frac{1}{2}w^Tw&plus;\sum_{i=1}^N&space;\alpha_i\\&space;&=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N&space;\alpha_i\alpha_j\cdot&space;y_iy_j\cdot&space;{\color{Red}&space;x_i^Tx_j}&plus;\sum_{i=1}^N&space;\alpha_i&space;\end{aligned}" title="\begin{aligned} L(w,b,{\color{Red} \alpha}) &=\frac{1}{2}w^Tw-\sum_{i=1}^N{\color{Red} \alpha_i}[y_i(w^Tx_i+b)-1]\\ &=\frac{1}{2}w^Tw-w^T\sum_{i=1}^N \alpha_iy_ix_i-b\sum_{i=1}^N \alpha_iy_i+\sum_{i=1}^N \alpha_i\\ &=\frac{1}{2}w^Tw-w^Tw+\sum_{i=1}^N \alpha_i\\ &=-\frac{1}{2}w^Tw+\sum_{i=1}^N \alpha_i\\ &=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_j\cdot y_iy_j\cdot {\color{Red} x_i^Tx_j}+\sum_{i=1}^N \alpha_i \end{aligned}" /></a>  
1. 求 `L` 对 `α` 的极大，即   
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;&\underset{\alpha}{\max}&space;\quad&space;-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N&space;\alpha_i\alpha_j\cdot&space;y_iy_j\cdot&space;x_i^Tx_j&plus;\sum_{i=1}^N&space;\alpha_i\\&space;&\&space;\mathrm{s.t.}\quad\;&space;\sum_{i=1}^N&space;\alpha_i&space;y_i=0,\&space;\&space;{\color{Red}&space;\alpha_i&space;\geq&space;0},\quad&space;i=1,2,\cdots,N&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;&\underset{\alpha}{\max}&space;\quad&space;-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N&space;\alpha_i\alpha_j\cdot&space;y_iy_j\cdot&space;x_i^Tx_j&plus;\sum_{i=1}^N&space;\alpha_i\\&space;&\&space;\mathrm{s.t.}\quad\;&space;\sum_{i=1}^N&space;\alpha_i&space;y_i=0,\&space;\&space;{\color{Red}&space;\alpha_i&space;\geq&space;0},\quad&space;i=1,2,\cdots,N&space;\end{aligned}" title="\begin{aligned} &\underset{\alpha}{\max} \quad -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_j\cdot y_iy_j\cdot x_i^Tx_j+\sum_{i=1}^N \alpha_i\\ &\ \mathrm{s.t.}\quad\; \sum_{i=1}^N \alpha_i y_i=0,\ \ {\color{Red} \alpha_i \geq 0},\quad i=1,2,\cdots,N \end{aligned}" /></a>
    该问题的对偶问题为：   
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;&{\color{Red}&space;\underset{\alpha}{\min}&space;}&space;\quad\&space;\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N&space;\alpha_i\alpha_j\cdot&space;y_iy_j\cdot&space;x_i^Tx_j-\sum_{i=1}^N&space;\alpha_i\\&space;&\&space;\mathrm{s.t.}\quad\;&space;\sum_{i=1}^N&space;\alpha_i&space;y_i=0,\&space;\&space;{\color{Red}&space;\alpha_i&space;\geq&space;0},\quad&space;i=1,2,\cdots,N&space;\end{aligned})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;&{\color{Red}&space;\underset{\alpha}{\min}&space;}&space;\quad\&space;\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N&space;\alpha_i\alpha_j\cdot&space;y_iy_j\cdot&space;x_i^Tx_j-\sum_{i=1}^N&space;\alpha_i\\&space;&\&space;\mathrm{s.t.}\quad\;&space;\sum_{i=1}^N&space;\alpha_i&space;y_i=0,\&space;\&space;{\color{Red}&space;\alpha_i&space;\geq&space;0},\quad&space;i=1,2,\cdots,N&space;\end{aligned})" title="\begin{aligned} &{\color{Red} \underset{\alpha}{\min} } \quad\ \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_j\cdot y_iy_j\cdot x_i^Tx_j-\sum_{i=1}^N \alpha_i\\ &\ \mathrm{s.t.}\quad\; \sum_{i=1}^N \alpha_i y_i=0,\ \ {\color{Red} \alpha_i \geq 0},\quad i=1,2,\cdots,N \end{aligned})" /></a>    
    于是，标准问题最后等价于求解该**对偶问题**    
    > 继续求解该优化问题，有 [SMO 方法](https://blog.csdn.net/ajianyingxiaoqinghan/article/details/73087304#t11)；因为《统计学习方法》也只讨论到这里，故推导也止于此   

## 朴素贝叶斯方法
1. 基本思想：分类方法，根据条件独立性假设学习训练数据的联合概率分布，对于预测样本x，  
   结合贝叶斯定理，计算后验概率P(Y|x), 选择后验概率最大的类作为输入x的所属类别
2. 条件独立性假设是指在类别给定的情况，各分类特征条件独立
3. 贝叶斯分类器：y=argmax(P(Y=k|X=x))
4. 损失函数为0-1损失时，期望风险最小化等价于后验概率最大化
5. 贝叶斯分类器待估计参数：先验概率p(Y=k)，条件概率p(X=x|Y=k)  
   采用极大似然估计或者贝叶斯估计（后者是防止极大似然估计所估计的概率为0，因此添加常数项）
6. 贝叶斯估计因为学习了训练样本的联合概率分布，因此属于生成式模型

## 决策树
1. 决策树是一种分类与回归方法，通过构建一颗二叉树或多叉树实现分类、回归，
   包括特征选择、决策树生成、决策树剪枝
2. 三种算法：ID3算法、C4.5算法、cart算法
- ID3算法：1) 首先计算每个特征的信息增益，选择信息增益最大的特征划分数据集  
             - 信息增益表示给定特征后，对类的信息的不确定性减少的程度（比如区分化妆与否，性别与学历两个特征的信息增益不同）   
             - 信息增益的计算g(D,A) = H(D) - H(D|A) （划分前熵-划分后熵）   
             - H(D)表示原始数据集D的熵     
             <a href="http://www.codecogs.com/eqnedit.php?latex=H(D)&space;=&space;-\sum_{i=1}^{n}P_i&space;log(P_i)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?H(D)&space;=&space;-\sum_{i=1}^{n}P_i&space;log(P_i)" title="H(D) = -\sum_{i=1}^{n}P_i log(P_i)" /></a>    
             - H(D|A)表示给定特征A后的条件熵    
             <a href="http://www.codecogs.com/eqnedit.php?latex=H(D|A)&space;=&space;-\sum_{i=1}^{n}P_i&space;H(D|A=a_i)&space;\\&space;P_i&space;=&space;P(A=a_i)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?H(D|A)&space;=&space;-\sum_{i=1}^{n}P_i&space;H(D|A=a_i)&space;\\&space;P_i&space;=&space;P(A=a_i)" title="H(D|A) = -\sum_{i=1}^{n}P_i H(D|A=a_i) \\ P_i = P(A=a_i)" /></a>    
          2) 划分后的数据集若类别不同，则需在剩下的特征中寻找最大的特征继续划分数据集  
          3）直到所有特征用完或者每个分支下的样本属于同一类，则停止划分，若所有特征用完，  
             但样本类别不统一，则采用投票的方法决定所属类别     
  总结：ID3算法只使用于离散变量，同时只能用于分类，不能用于回归   
 - C4.5算法：与ID3算法的区别在于使用信息增益比选择特征，同时适用于连续变量
             1）使用信息增益比，信息增益/划分前熵，因为信息增益偏向选择取值较多的特征
             2）处理连续特征，对连续特征取值排序，任意两个值选取中间值划分数据集，尝试每种划分方法，
                然后计算划分前后的信息熵，选择信息增益最大的分裂点作为该特征的分裂点
- cart算法：通过构建二叉分类树或回归树，用于分类时，使用基尼系数选择特征，而用于回归时，则使用平方误差最小选择特征以及分割点   
1）回归二叉树   
- y为连续值，将输入x划分为m个区域，每个区域的取值为C1,C2,,,CM，因此最小二乘回归树模型为    
<a href="http://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;\sum_{m=1}^{M}C_m&space;I(x\in&space;R_m)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f(x)&space;=&space;\sum_{m=1}^{M}C_m&space;I(x\in&space;R_m)" title="f(x) = \sum_{m=1}^{M}C_m I(x\in R_m)" /></a>
- 利用平方误差最小化，寻找最优划分的特征以及分裂点，优化目标函数如下：   
<a href="http://www.codecogs.com/eqnedit.php?latex=\underset{j,s}{min}[\underset{C_1}{min}\sum_{x_i\in&space;R_1(j,s)}(y_i-C_1)^{2}&plus;\underset{C_2}{min}\sum_{x_i\in&space;R_2(j,s)}(y_i-C_2)^{2}]" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\underset{j,s}{min}[\underset{C_1}{min}\sum_{x_i\in&space;R_1(j,s)}(y_i-C_1)^{2}&plus;\underset{C_2}{min}\sum_{x_i\in&space;R_2(j,s)}(y_i-C_2)^{2}]" title="\underset{j,s}{min}[\underset{C_1}{min}\sum_{x_i\in R_1(j,s)}(y_i-C_1)^{2}+\underset{C_2}{min}\sum_{x_i\in R_2(j,s)}(y_i-C_2)^{2}]" /></a>
- 依次遍历所有特征j，以及每个特征的每个取值s，选择使得损失函数最小的（j,s）   
2) 分类二叉树
- 使用基尼指数寻找划分特征以及最优划分点，基尼指数表示集合的不确定性，基尼指数越大，则数据的不纯度越高   
<a href="http://www.codecogs.com/eqnedit.php?latex=Gini(p)&space;=&space;\sum_{k=1}^{K}P_k(1-P_k)&space;=1-&space;\sum_{k=1}^{K}P_k^{2}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Gini(p)&space;=&space;\sum_{k=1}^{K}P_k(1-P_k)&space;=1-&space;\sum_{k=1}^{K}P_k^{2}" title="Gini(p) = \sum_{k=1}^{K}P_k(1-P_k) =1- \sum_{k=1}^{K}P_k^{2}" /></a>

## 集成学习
1. 集成学习是指构建并结合多个分类器来完成学习任务，其中多个分类器可以是同质分类器，此时被称为基分类器，比如决策树集成，神经网络集成，
   集成也可包含不同分类器，如同时包含决策树和神经网络，称为组件学习器
- 集成学习为什么有效？：平均上，集成学习至少能与多分类器中一个保持一致的分类效果，而当多个分类器的误差独立时，集成学习的效果比其成员效果更好
- **集成学习的核心：个体学习器要有一定的准确率，同时分类器具有差异，即实现“好而不同”**
2. 集成学习分两类：boosting:各分类器之间相互依赖，必须串行生成的序列方法;   
   bagging: 各分类器之间不存在强依赖关系，可以并行生成
3. boosting: adaboost、GBDT、xgboost   （减少偏差）
   - boosting需解决两个问题：第一，每一轮如何改变数据的权值以及概率分布；    
                            第二，如何组合多个弱分类器
4. bagging: 随机森林（减少方差）
### AdaBoost算法（减少偏差）
1. 含义: 对样本赋予初始权重1/N，基于训练数据构建基分类器，根据分类器的错误率调整样本权重，
   然后继续构建分类器，直到达到指定T个，然后对T个分类器加权求和
2. 针对boosting的两个问题：
- 初始对每个样本赋予相同的权值，建立分类器后，提高上一轮分类错误的样本权值，降低分类正确的样本权值
- 采用加法模型，加大分类错误率低的模型的权重，使其在表决中起更大的作用
3. adaboost模型为加法模型，损失函数为指数损失，同时学习算法是前向分步算法的二分类学习方法       
   损失函数为指数函数L(y,f(x)) = exp(-y*f(x))    
   前向分步算法思想：从前向后，每一步只学习一个基函数及其系数，逐步优化目标函数   
#### 算法描述
- 输入：训练集 T={(x1,y1),..,(xN,yN)}, xi ∈ R^n, yi ∈ {-1,+1}，基学习器 G1(x)
- 输出：最终学习器 G(x)
1. 初始化样本权值   
<a href="http://www.codecogs.com/eqnedit.php?latex=D_1=(w_{1,1},\cdots,w_{1,i},\cdots,w_{1,N}),\quad&space;w_{1,i}=\frac{1}{N},\quad&space;i=1,2,\cdots,N" target="_blank"><img src="http://latex.codecogs.com/gif.latex?D_1=(w_{1,1},\cdots,w_{1,i},\cdots,w_{1,N}),\quad&space;w_{1,i}=\frac{1}{N},\quad&space;i=1,2,\cdots,N" title="D_1=(w_{1,1},\cdots,w_{1,i},\cdots,w_{1,N}),\quad w_{1,i}=\frac{1}{N},\quad i=1,2,\cdots,N" /></a>
2. 构建基分类器   
<a href="http://www.codecogs.com/eqnedit.php?latex=G_m(x):\chi&space;\rightarrow&space;{-1,&plus;1}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?G_m(x):\chi&space;\rightarrow&space;{-1,&plus;1}" title="G_m(x):\chi \rightarrow {-1,+1}" /></a>
3. 计算基分类器的分类错误率，即为分类错误样本的权值之和   
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;e_m&=P(G_m(x_i)\neq&space;y_i)\\&=\sum_{i=1}^Nw_{m,i}\cdot&space;{\color{Red}&space;I(G_m(x_i)\neq&space;y_i)}&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;e_m&=P(G_m(x_i)\neq&space;y_i)\\&=\sum_{i=1}^Nw_{m,i}\cdot&space;{\color{Red}&space;I(G_m(x_i)\neq&space;y_i)}&space;\end{aligned}" title="\begin{aligned} e_m&=P(G_m(x_i)\neq y_i)\\&=\sum_{i=1}^Nw_{m,i}\cdot {\color{Red} I(G_m(x_i)\neq y_i)} \end{aligned}" /></a>
4. 计算分类器的权重   
<a href="http://www.codecogs.com/eqnedit.php?latex=\alpha_m=\frac{1}{2}\ln\frac{1-e_m}{e_m}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\alpha_m=\frac{1}{2}\ln\frac{1-e_m}{e_m}" title="\alpha_m=\frac{1}{2}\ln\frac{1-e_m}{e_m}" /></a>
5. 更新样本的权值，分类错误与分类正确的样本权值相差exp(2a)  
<a href="http://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;D_{{\color{Red}m&plus;1}}&=(w_{m&plus;1,1},\cdots,w_{m&plus;1,i},\cdots,w_{m&plus;1,N})\\w_{{\color{Red}m&plus;1},i}&=\frac{w_{{\color{Red}m},i}\cdot\exp(-\alpha_{\color{Red}m}\cdot{\color{Blue}y_iG_m(x_i)&space;})}{Z_{\color{Red}m}}&space;\end{aligned}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\begin{aligned}&space;D_{{\color{Red}m&plus;1}}&=(w_{m&plus;1,1},\cdots,w_{m&plus;1,i},\cdots,w_{m&plus;1,N})\\w_{{\color{Red}m&plus;1},i}&=\frac{w_{{\color{Red}m},i}\cdot\exp(-\alpha_{\color{Red}m}\cdot{\color{Blue}y_iG_m(x_i)&space;})}{Z_{\color{Red}m}}&space;\end{aligned}" title="\begin{aligned} D_{{\color{Red}m+1}}&=(w_{m+1,1},\cdots,w_{m+1,i},\cdots,w_{m+1,N})\\w_{{\color{Red}m+1},i}&=\frac{w_{{\color{Red}m},i}\cdot\exp(-\alpha_{\color{Red}m}\cdot{\color{Blue}y_iG_m(x_i) })}{Z_{\color{Red}m}} \end{aligned}" /></a>
6. 重复构建m个分类器，最终线性组合，得到最终集成分类器   
<a href="http://www.codecogs.com/eqnedit.php?latex=G(x)=\mathrm{sign}(\sum_{m=1}^M\alpha_mG_m(x))" target="_blank"><img src="http://latex.codecogs.com/gif.latex?G(x)=\mathrm{sign}(\sum_{m=1}^M\alpha_mG_m(x))" title="G(x)=\mathrm{sign}(\sum_{m=1}^M\alpha_mG_m(x))" /></a>  
**总结：AdaBoost不改变训练数据的分布，但通过改变训练数据的权值分布，使得训练数据在不同分类器中起到不同的作用

### 梯度提升算法 GBDT （每棵树的深度较小）
1. 提升树：迭代生成多个模型，将每个模型的预测结果相加，后面模型基于前面模型的效果生成   
          在回归问题中，新的树是通过不断拟合残差得到    
2. 梯度提升：当损失函数为平方损失或指数损失时，残差易得到，但一般的损失函数，不易得到残差时，则利用**损失函数的负梯度作为残差的近似值**
3. GBDT：以CART回归树为基学习器的梯度提升算法
3. GBDT算法描述   
输入：训练集 T={(x1,y1),..,(xN,yN)}, xi ∈ R^n, yi ∈ R；损失函数 L(y,f(x))；   
输出：回归树 f_M(x)  
- 初始化回归树
<a href="http://www.codecogs.com/eqnedit.php?latex=f_0(x)={\color{Red}&space;\arg\underset{c}{\min}}\sum_{i=1}^NL(y_i,c)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f_0(x)={\color{Red}&space;\arg\underset{c}{\min}}\sum_{i=1}^NL(y_i,c)" title="f_0(x)={\color{Red} \arg\underset{c}{\min}}\sum_{i=1}^NL(y_i,c)" /></a>   
- 对于每一轮，计算残差/负梯度   
<a href="http://www.codecogs.com/eqnedit.php?latex=r_{m,i}=-\frac{\partial&space;L(y_i,{\color{Red}&space;f_{m-1}(x_i)}))}{\partial&space;{\color{Red}&space;f_{m-1}(x_i)}}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?r_{m,i}=-\frac{\partial&space;L(y_i,{\color{Red}&space;f_{m-1}(x_i)}))}{\partial&space;{\color{Red}&space;f_{m-1}(x_i)}}" title="r_{m,i}=-\frac{\partial L(y_i,{\color{Red} f_{m-1}(x_i)}))}{\partial {\color{Red} f_{m-1}(x_i)}}" /></a>
- 对负梯度拟合一个回归树，得到第m棵树的叶结点区域   
<a href="http://www.codecogs.com/eqnedit.php?latex=R_{m,j},\quad&space;j=1,2,..,J" target="_blank"><img src="http://latex.codecogs.com/gif.latex?R_{m,j},\quad&space;j=1,2,..,J" title="R_{m,j},\quad j=1,2,..,J" /></a>
- 估计每个叶结点区域的值   
<a href="http://www.codecogs.com/eqnedit.php?latex=c_{m,j}={\color{Red}&space;\arg\underset{c}{\min}}\sum_{x_i\in&space;R_{m,j}}L(y_i,{\color{Blue}&space;f_{m-1}(x_i)&plus;c})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?c_{m,j}={\color{Red}&space;\arg\underset{c}{\min}}\sum_{x_i\in&space;R_{m,j}}L(y_i,{\color{Blue}&space;f_{m-1}(x_i)&plus;c})" title="c_{m,j}={\color{Red} \arg\underset{c}{\min}}\sum_{x_i\in R_{m,j}}L(y_i,{\color{Blue} f_{m-1}(x_i)+c})" /></a>
- 更新回归树   
<a href="http://www.codecogs.com/eqnedit.php?latex=f_m(x)=f_{m-1}&plus;\sum_{j=1}^J&space;c_{m,j}{\color{Blue}&space;I(x\in&space;R_{m,j})}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f_m(x)=f_{m-1}&plus;\sum_{j=1}^J&space;c_{m,j}{\color{Blue}&space;I(x\in&space;R_{m,j})}" title="f_m(x)=f_{m-1}+\sum_{j=1}^J c_{m,j}{\color{Blue} I(x\in R_{m,j})}" /></a>   

### xgboost算法
1. xgboost算法是GBDT的高效实现，基分类器除了CART回归树，还可以是线性分类器
2. 损失函数添加了正则化项，包括 L2 权重衰减和对叶子数的限制
3. 使用牛顿法代替梯度下降法寻找最优解，前者使用一阶+二阶导数作为残差，后者只使用了一阶导数
4. 传统 CART树寻找最优切分点的标准是最小化均方差；
   XGBoost 通过最大化得分公式来寻找最优切分点：  
   这同时也起到了“剪枝”的作用——如果分数小于γ，则不会增加分支；  
   <a href="http://www.codecogs.com/eqnedit.php?latex=Gain&space;=&space;\frac{1}{2}&space;\left[\frac{G_L^2}{H_L&plus;\lambda}&plus;\frac{G_R^2}{H_R&plus;\lambda}-\frac{(G_L&plus;G_R)^2}{H_L&plus;H_R&plus;\lambda}\right]&space;-&space;\gamma" target="_blank"><img src="http://latex.codecogs.com/gif.latex?Gain&space;=&space;\frac{1}{2}&space;\left[\frac{G_L^2}{H_L&plus;\lambda}&plus;\frac{G_R^2}{H_R&plus;\lambda}-\frac{(G_L&plus;G_R)^2}{H_L&plus;H_R&plus;\lambda}\right]&space;-&space;\gamma" title="Gain = \frac{1}{2} \left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma" /></a>

### bagging算法（减少方差）
1. 提升集成学习的效果，就需使得个体学习器之间独立，而bagging通过对样本进行自助采样，
   使得基分类器的训练样本不同，让基分类器之间产生较大差异，从而提升预测效果
2. 流程：采样出T个含有m个样本的训练集，然后基于每个训练集构建基分类器，最后将这些基学习器组合，
   通常对于分类，采用简单投票法，而对于回归，则采用简单平均法
3. 优点：高效，与训练基学习器的复杂度同阶; 与AdaBoost只能用于二分类不同，bagging可以用于多分类或者回归；  
   由于自助采样，有部分样本并未进入任何一个训练集，因此可以作为验证集测试模型泛化性能
   
### 随机森林（每棵树的深度要求较高）
1. 在以决策树为基分类器构建bagging的基础上，每棵决策树随机选取部分特征（通过boostrap的方法随机选择k个特征子集）
2. 随机森林基学习器的多样性不仅来源于样本扰动，还来源于属性扰动，使得最终集成的泛化性能由于个体学习器之间的差异度增加而进一步提升
3. 优点：训练效率优于bagging，因为只采用了部分特征，基分类器的性能较低，但随着分类器个数增加，随机森林往往会收敛到很低的泛化误差

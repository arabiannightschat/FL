### 关于产出联邦学习创新点的重新系统学习

### 基础概念

+ #### 联邦学习（!!! 第三页）

+ #### 同态加密

  + High-efficient preparation and screening of electrocatalysts using a closed bipolar electrode array system
  + [PrivFL: Practical privacy-preserving federated regressions on high-dimensional data over mobile networks](https://dl.acm.org/doi/abs/10.1145/3338466.3358926)
    + 使⽤Paillier同态加密算法对⽤⼾的局部梯度进⾏加密来获取参数。
  + Privacy preserving deep learning via additively homomorphic encryption

+ #### 差分隐私

  + 高斯机制：
    + 计算查询结果的真实值。
    + 生成一个服从高斯分布（正态分布）的随机数。
    + 将随机数乘以敏感度并除以隐私参数 ε。
    + 将结果加到真实值上，得到最终的扰动结果。

+ #### 秘密共享

+ #### 隐私攻击

  + 模型反转攻击：使用反向传播技术来逆向计算梯度，然后推断原始数据的特征或敏感信息。

    + Privacy in Pharmacogenetics: An End-to-End Case Study of Personalized Warfarin Dosing
      + 逆向工程的方法，通过分析机器学习模型的输出梯度，推断出了个体的敏感信息

    + Inverting gradients-how easy is it to break privacy in federated learning?
      + 余弦相似度和对抗性攻击，重建⾼分辨率原始
    + Model inversion attacks that exploit confidence information and basic countermeasures
      + 利用模型输出的置信度信息来实施模型反转攻击
    + Deep models under the GAN: Information leakage from collaborative deep learning
      + 探讨了在协作深度学习中信息泄漏的问题，特别关注生成对抗网络（GAN）的应用，并研究了模型反转攻击的可能性

  + 成员推理攻击

    + Membership Inference Attacks Against Machine Learning Models via Prediction Sensitivity
      + 已知一条记录，推断记录是否被用于某一模型的训练。例如，知道某个患者的临床记录被⽤来训练与疾病相关的模型可以揭⽰患者患有这种疾
      + 影子模型：攻击者训练多个和目标模型相似的模型，比较他们对某条记录的预测值，推断出目标模型训练集中是否有这一记录

  + 投毒攻击 !!! 论文 VPPFL 第二页右侧

### 研究方向1

+ #### 同态加密和抗投毒技术
  
  + Privacy-Preserving Byzantine-Robust Federated_Learning_via_Blockchain_Systems
    + 们利⽤余弦相似性来检测⽤⼾上传的梯，全同态加密实现安全聚
  + Differentially Private Byzantine-Robust Federated Learning
    + 防⽌拜占庭参与者发起的对抗性攻击，并通过新颖的聚合协议实现差异隐私
  + ShieldFL: Mitigating Model Poisoning Attacks in Privacy-Preserving Federated Learning
    + ⽤双陷⻔同态加密，以抵抗模型中毒攻击⽽不泄露隐私
  + Privacy-Enhanced Federated Learning Against Poisoning Adversaries
    + 基于投毒的同态加密，其中的决策部分，基于孤立森林，引自论文 
  + FedDefender: Client-Side Attack-Tolerant Federated Learning
    + 容忍攻击的本地元更新
      + 提前在本地进行模拟扰动训练，使得神经网络对噪声扰动不敏感
    + 容忍攻击的全局知识
      + 在模型中毒攻击的情况下，全局模型的可信度可能会受到损害，进行全局知识蒸馏（基于梯度下降优化的固有本质，模型深层更容易过度适应噪声，所以只取全局知识的神经网络中间层，设定置信度，进行一定的全局蒸馏损失）
  
+ #### 差分隐私和抗投毒技术
  
  + VPPFL: A Verifiable Privacy-Preserving Federated Learning Scheme
    + 基于此论文的相关工作进行学习
    + 论文方法：服务器1生成高斯噪声，对称加密发送给参与方；服务器1发送差分矩阵（噪声平均值）给服务器2，用于服务器2消除噪声影响；服务器2使用聚类算法找到恶意用户
  + A Differentially Private Federated Learning Model Against Poisoning Attacks in Edge Computing
    + 基于边缘计算的针对中毒攻击的差分私有FL

### 研究方向2

设备异构下的同态加密和差分隐私

### 研究方向3

+ 简单的秘密共享
  + **[idea]** 设置两个服务器，由服务器1为每一个参与方生成随机数矩阵并发送给参与方，服务器2收到一轮参数后，向服务器1索要本轮参与方随机数矩阵的聚合结果，得到真实聚合
    + 每结束一轮聚合，服务器1都和全部参与方重新设定随机数矩阵，防止服务器2猜出随机数矩阵
    + 确保服务器1，2是不合谋的
  + **[idea]** 参与方将数据拆成n份，发送给n个服务器，n个服务器聚合后再相加
    + 服务器允许诚实且好奇，允许n-2个服务器合谋，参与方允许诚实且好奇
    + 为什么联邦学习不直接使用这种方法 // TODO

###  如何搜索论文被引用

1. 进入这个网站 http://isiknowledge.com
2. 输入一篇论文
3. 点击被引频次旁边的数字（Citations）

### 如何找到论文源码

[paperswithcode.com](https://paperswithcode.com)

### 1. 论文（秘密共享屏蔽参与方单次权重）：

Practical Secure Aggregation for Privacy-Preserving Machine Learning

https://www.cnblogs.com/20189223cjt/p/12529827.html

https://juejin.cn/post/7041816522347511816

https://zhuanlan.zhihu.com/p/341898547

#### 1.1 前置知识

1. ##### DiffieHellman

在通信渠道不安全的情况下，沟通一个密钥的方法

\1.   Alice 和 Bob 先对 p 和 g 达成一致，而且公开

\2.   Alice取一个私密的整数 a，不让任何人知道，发给 Bob 计算结果：*A*=g^a mod p. Eve 也看到了A的值。

\3.   类似 Bob 取一私密的整数 b, 发给 Alice 计算结果 *B*= g^b mod *p.* 同样Eve也会看见传递的B是什么。

\4.   Alice 计算出 *S*=*B^a* mod *p*=g^ab mod *p.*

\5.   Bob 也能计算出 *S*=*A^b* mod *p*=g^ab mod *p.*

\6.   Alice 和 Bob 现在就拥有了一个共用的密钥 *S.*

#### 1.2 从研究脉络梳理：

思路1：每两个用户，例如用户 u 和用户 v 协商一个随机数 S u,v（S 有顺序 —— 即 S u,v 和 S v,u 是两个值），每个用户计算自己的参数值时将所有和其他用户的随机向量值先加再减。

![image-20230912160646458](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230912160646458.png)

![image-20230912160607456](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230912160607456.png)

当然，在FL场景中，上面的 x 和 s 实际上都是多维向量。

两两共享的随机数可以通过 DH 密钥协商协议来实现，通过借助 pseudo random generator (PRG) 减少通信开销。协商的密钥作为伪随机数生成器 PRG 的种子。

接下来为了处理掉线问题，设计让每个客户把 S u,v 采用秘密共享的方式分发出去，只要有 t 个客户端在线，就可以还原回 S u,v

新的问题出现了，如果某个客户端掉线，又上线，这时候服务器已经知道其 mask，会暴漏数据，提出 Double-Masking：引入新的随机数 bi

![image-20230912162754176](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230912162754176.png)

![image-20230912162808351](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230912162808351.png)

聚合时，服务器会通过秘密共享重建所有在线参与方的 bi，都减掉，另外重建掉线者的 S u,v 完成聚合

#### 1.3 总的流程

Round 0 是关于 D-H 以及 PRG 的操作

Round 1 是生成 bu 和 S u,v，对称加密后借助服务器分发给大家

Round 2 计算 yu，发送给服务器

Round 3 是一步签名

Round 4 服务器发送幸存者列表，参与方为活跃用户发送 bu，为掉线用户发送 S u,v，服务器进行聚合

![image-20230912164329304](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230912164329304.png)

### 2. 论文（差分隐私和阈值同态加密）：A Hybrid Approach to Privacy-Preserving Federated Learning

https://blog.csdn.net/qq_44026293/article/details/112062798

#### 2.1 前置知识

1. ##### 差分隐私 DP：

   是⼀个严格的数学框架，其中当且仅当训练数据集中包含单个实例仅导致算法输出发⽣统计上不显着的变化时，算法才可以被描述为差分隐私。

   差分隐私两种流行机制为拉普拉斯机制和高斯机制

   **差分隐私的保护场景**


   ```
   比如医院记录了所有人是否患有该疾病的记录，那么我们可以通过一下差分的方法来获取某个人的具体信息。
   （1）先查询整个数据库内患有该病的人数Num1；
   （2）查询除某个人以外患有该病的人数Num2；
   （3）如果两者相差为1的话，再不考虑重名的情况下，极大可能知道某个人真实的患病情况；
   ```

   **中心化差分隐私（centralized differential privacy）简称CDP**

   ```
   认为第三方是可信的，因此主要保护的是数据收集分析后的结果发放过程，差分隐私保护机制运行在可信第三方上。
   ```

   **本地化差分隐私（Local Differential Privacy）简称LDP**

   ```
   原理：认为第三方是不可信的，所以本地差分隐私保护的是用户上传数据到第三方的过程，差分隐私机制运行在各个用户的本地。
   ```

2. ##### 同态加密

   Enc(m1) ◦ Enc(m2) = Enc(m1 + m2)

   ⽅案是 Paillier

   ![image](https://img2022.cnblogs.com/blog/1928790/202207/1928790-20220705145336961-637762979.png)

   ![image](https://img2022.cnblogs.com/blog/1928790/202207/1928790-20220705145348297-1840962618.png)

3. ##### 阈值同态加密

   RSA加密方案（乘法同态）以及 Paillier加密方案（加法同态）

   https://www.cnblogs.com/pam-sh/p/16446840.html

   单密钥的同态加密存在问题，即能使用同态加密计算的密文必须是相同公钥加密的，用相同私钥解密

   多方联合计算最安全的方案是各自生成，保存公私钥

   ![image](https://img2022.cnblogs.com/blog/1928790/202207/1928790-20220705150001441-882915729.png)

4. 为了保护两种潜在威胁：（1）学习过程中的推理（2）对输出的推理（包括中间输出）

   （1）使用安全多方计算

   （2）使用差分隐私

#### 2.2 论文贡献

- 提出一个新的FL系统（训练方法），在保证数据隐私的基础上有着比普通的FL系统更高的准确率  

- 该FL系统中包含一个可调的参数，我们可以通过这个参数在系统的准确性和隐私性做一个权衡

- 算法：

  ![image-20230915193853810](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230915193853810.png)

  ![image-20230915194109706](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230915194109706.png)

  ![image-20230915194450052](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230915194450052.png)

  理解如下：

  服务器发送Q给参与方 -> 参与方进行训练并加噪声再加密得到 Ri' -> 服务器聚合 Ri' 得到 R' -> 发还给参与方 R' -> 每个参与方进行解密得到 Ri'' -> 服务器聚合得到 R'' 为最终权重更新 M

#### 2.3 提出问题及思考

1. 为什么要同时使用差分隐私和阈值同态加密？

   有一种解释如下：

   模型的主体还是差分隐私，因为LDP（定义在2.1节）通常带来的精度损失太大，所以要用同态加密解决对中心服务器的信任问题，从而少添加一些噪声，在准确率和安全性上做平衡。

   另外一种解释：

   如果有超过t个非诚实参与者合谋，可以推断出其他诚实参与方的数据，所以添加差分隐私保护其他参与方的数据，比如：论文中提到如果系统中有五个参与⽅，其中⼀个参与⽅担⼼其他四个参与⽅都在串通，那么诚实的⼀⽅就没有理由继续参与。

2. 如果不使用差分隐私只使用阈值同态加密是否可行？

   我认为可行，一般场景下不会出现超过t个参与方合谋攻击的情况，需要学习其他论文和隐私攻击方式来印证。

### 3. 论文（服务端可信时，添加的模型扰动）An Accuracy-Lossless Perturbation Method for Defending Privacy Attacks in Federated Learning

#### 3.1 摘要：

https://github.com/Kira0096/PBPFL

提出了一种模型扰动方法，防止客户端获取真实的全局模型参数和局部梯度，可以防御重建和成员推理攻击。

与差分隐私或其他扰动⽅法⽆法 消除添加的噪声不同，我们的⽅法确保服务器可以通过消除添加的噪声来 恢复聚合的真实梯度。因此，我们的⽅法不会妨碍学习的准确性。

论文方法：我们详细描述了我们提出的使⽤ ReLU ⾮线性激活的隐私保护深度模型训练，它可以轻松应⽤于 ResNet 和 DenseNet 等最先进的模型，主要是在下发模型参数前添加了全局模型扰动。

评价：

这篇论文提供了横向联邦场景下高效地进行无损**[模型加密](https://www.zhihu.com/search?q=模型加密&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2574847376})**的解决方案。传统[加密算法](https://www.zhihu.com/search?q=加密算法&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2574847376})/[差分隐私算法](https://www.zhihu.com/search?q=差分隐私算法&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2574847376})在联邦场景下，大幅影响计算传输效率/模型精度，而我们的方案仅增加了常数倍的计算传输 overhead，就实现了accuracy lossless 的[加密训练](https://www.zhihu.com/search?q=加密训练&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2574847376})。

除了加密机制的创新外，在设定上我们也与以往工作不同，主要考虑横向联邦场景，在server可信的背景下，如何保护 model privacy。这也使得这篇工作有个相当有趣的潜在应用场景：在横向联邦+[多数据提供方](https://www.zhihu.com/search?q=多数据提供方&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2574847376})的设定下，传统联邦建模方案往往无法分离数据所有权与模型所有权。即在建模过程中，数据提供商同样能够access 多方共同训练的模型，而作为数据需求方，通常期望数据商仅提供数据，无法享受产生的价值。

而基于我们所提供的**模型加密**训练方案，过程中 server（数据需求方）能够获取解密后的模型，clients（数据提供商）模型迭代过程中仅能够 access 参数加密后的模型，无法感知到真实的模型梯度以及模型预测结果，从而最大程度的保证模型隐私安全，即保护数据需求方的权益。

#### 3.2 前置知识

1. ##### 联邦学习训练深度神经⽹络（DNN）

   进行三个阶段：1） 全局模型⼴播，下发模型参数 `W`

   2）局部模型训练，最后将 `∇F(W,D)` 上传到服务器

   3）全局模型更新，服务器聚合当前  `∇F(W,D)` 并下发 `W`

2. ##### Hadamard 乘积

   m x n 矩阵 A = [aij] 与矩阵 B = [bij] 的 Hadamard 积，记为 A * B 。

   新矩阵元素定义为矩阵 A、B 对应元素的乘积 (A * B)ij = aij * bij

#### 3.3 思考

1. 为什么能在添加噪声的情况下不影响模型精度？

   差分隐私的原则应该是随机数分散均匀减少对精度的影响

   论文方法会记录噪声向量r，并证明了扰动梯度和真实梯度间的关系，在聚合后再恢复准确精度

### 4 论文（使用区块链作为联邦学习激励机制）FLChain: A Blockchain for Auditable Federated Learning with Trust and Incentive

https://zhuanlan.zhihu.com/p/420633671

#### 4.1 前置知识

1. ##### 拜占庭攻击

   [https://aisigsjtu.github.io/projects/fl-attack-defense](https://aisigsjtu.github.io/projects/fl-attack-defense/)[/](https://aisigsjtu.github.io/projects/fl-attack-defense/)

   联邦学习中的拜占庭攻击是通过污染本地数据或直接修改模型梯度实现的，防御方式一般由在服务端聚合时筛选威胁大的权重实现，也有修改聚合规则，客户端添加全局权重检测等方法。

#### 4.2 论文贡献

提出FLChain来构建一个分散的、可公开审计的、健康的、有信任和激励的联邦学习生态系统。 在FLChain中，诚实的客户可以根据自己的贡献通过一个经过训练的模型获得公平分配的利润，恶意的客户可以被及时发现并受到严厉的惩罚。

实现过程：

```
梯度生成：该过程中，FL的训练节点在本地独立训练它们的模型。在每个本地机器学习迭代结束后，可验证并且加密后的本地梯度将被上传至FLChain链上； 
        备注：加密算法使用秘密共享机制。每一个加密以及解密的梯度部分都需要可审计的证明来阻止不诚实的行为。常用的秘密共享机制为加性秘密共享，便于梯度聚合。 
梯度聚合：该过程中，中心server聚合本地上传的所有梯度值并将聚合结果上传至FLChain链，当然所有server中只有leader节点才能将聚合结果最终上传至FLChain链，leader节点的选取标准是选择可信度最高的节点。 
协同解密：该过程中，本地训练节点从FLChain链下载聚合结果，并且协同解密。解密获得最新 global model 后，本地训练节点然后更新模型参数并且开始下一轮的本地模型训练。 
强制结束：该过程中，FLChain模型会结束有问题的合作模型训练。仅当诚实节点在FL内的确发现了错误行为才会发生。 
```



### 5 同态加密 针对不规则用户优化 Privacy-Preserving_Federated_Deep_Learning_With_Irregular_Users

2022 年

现有的隐私保护联邦深度学习⽅法主要是从三种底层技术发展⽽来的：差分隐私[3]、[8]、同态加密[10]、[11]和安全多⽅计算

它们都假设每个⽤⼾持有的数据是⾼质量的并且彼此相似

本文：加法同态和姚氏混淆电路技术

文中提到的不规则用户处理方式是：依据局部梯度和全局梯度方向的一致性得到用户可信度，调整梯度聚合比例

#### 5.1 前置知识

1. ##### 姚氏混淆电路

   安全多方计算，主要是指， **多个通信的参与者在保障通信和计算过程的正确性、隐私性、公平性等安全特征的基础上，联合进行某些功能函数计算，各自得到他们预定的计算结果的一类技术**

   它的核心技术是将两方参与的安全计算函数编译成布尔电路的形式，并将真值表加密打乱，从而实现电路的正常输出而又不泄露参与计算的双方私有信息。由于任何安全计算函数都可转换成对应布尔电路的形式，相较其他的安全计算方法，具有较高的通用性，因此引起了业界较高的关注度。

### 6 同态加密以及抗投毒技术 Privacy-Enhanced Federated Learning Against Poisoning Adversaries

单纯的 PPFL（preserving-privacy federated learning）方案致力于各方的模型信息不可区分来抵御推理攻击，而抗投毒攻击的方案则致力于根据异常数据与正常数据的差异来清除异常数据（也就是去寻找数据间的差异性）。显然两者的准求是有矛盾的，而如何解决这个矛盾是现在大家都热衷的问题，

本文则提出了一个新的隐私保护联邦学习框架 PEFL(Privacy-Enhanced Federated Learning) 作为桥梁来解决上述的矛盾问题。其中使用同态加密 (HE) 来解决前者，也就是抵御推理攻击，使用Median 技术来解决后者，也就是抗投毒攻击。值得一提的是本文是第一个在密文上检测异常数据的 FL 框架。实验表明 PEFL 能够有效的抵御上述两种攻击，同时训练得到的模型拥有较好的性能。

https://zhuanlan.zhihu.com/p/410806354

#### 6.1 前置知识

1. **投毒攻击(Poisoning Attack)**

   模型偏斜（Model skewing）：通过给出错误的输出结果的大量数据来达到攻击的目的

   反馈武器化（Feedback weaponization）：用户打分反馈直接作用于神经网络导致的错误

   后门攻击（backdoor attacks）：将自己选定好的某个 trigger 与特定的输出结果建立联系，同时不影响其他正常数据的预测结果

2. ##### IID

   用户持有的本地数据是独立同分布

#### 6.2 论文方案

![image-20230927131144367](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230927131144367.png)

本文提出的系统模型相比较一般的仅包含 Data Owner (Client) 和 Service Provider (SP) 的联邦学习模型而言增加了 Cloud Platform (CP)部分。此外还有一个密钥生成中心 (KGC)，KGC 用来分别生成用户 Client 的密钥对和云服务平台的密钥对 。整个系统对于 SP 和 CP 端假设半诚实模型（semi-honest Model），同时假设Client 端用户满足 honest-majority。

1. 第一步：客户端本地训练 -> 同态加密 -> 上传梯度 Gx

![image-20230928135409408](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230928135409408.png)

2. 第二步，服务器将 Gx 盲化成 Rx 并发送给云平台 CP -> CP拥有私钥可以解密 Rxi 得到 dxi -> CP 找到中位数 dmedi 并使用公钥加密发送给服务器 -> 服务器去除盲化得到 Gmedi，是梯度的中位数

![image-20230928140924020](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230928140924020.png)

   3. 第三步，以该中位数向量为基准来计算得到各个客户端上传的梯度向量是否异常

![image-20230928141014400](C:\Users\10655\AppData\Roaming\Typora\typora-user-images\image-20230928141014400.png)

   

### 论文 7 差分隐私 异步优化

PADL_Privacy-Aware_and_Asynchronous_Deep_Learning_for_IoT_Applications

分层重要性传播算法（LIP）和拉普拉斯差分隐私机制，异步优化（AAO）协议

#### 权重和梯度

权重决定了模型对输入数据的解释和预测，是模型中节点的参数

梯度是指损失函数对于权重的偏导数，表示了损失函数相对于每个权重的变化率。在机器学习中，梯度用于更新权重，以最小化损失函数。

#### 给重要的权重对应的梯度添加较少的噪声

根据权重的重要性，当重要性较⾼时，将较少的噪声注⼊到权重中以提⾼准确性。虽然重要性较低，但为了保护隐私将分配更多的噪声

权重的重要性指的是在机器学习模型中，每个特征或变量对最终预测结果的影响程度。权重越高，说明该特征对预测结果的影响越大；权重越低，说明该特征对预测结果的影响越小。

#### 异步优化

数据收集站点上传训练好的模型，并请求下⼀轮新的训练 模型。当服务器中的模型尚未初始化时，直接替换为数据采集点提交的模型，并通知 数据采集点继续⽤当前模型进⾏训练。当M 个模型被完全替换⼀次后，服务器返回 之前更新的模型，标记为m，其中m ∈ [0, M − 1]。这样，云端标记为 m+1 的模型就 被数据采集站点上传的模型所取代，其中(m + 1) ∈ [0, M − 1]。

解释：当服务器没有平均出一套权重之前，各个参与方使用当前模型权重继续进行下一次训练，以提高训练效率

异步学习：异步学习可以简单地认为是在独⽴的数据收集站点并⾏ 训练模型的多个副本。在本⽂中，每个数据收集站点都保存 DNN 模型的 副本以及⼀部分训练数据。他们利⽤当前的模型参数和私有数据库，独⽴ 训练本地模型。毫⽆疑问，我们的解决⽅案实现了异步学习

）异步更新：如图3所⽰，各数据采集点与云服务器交互开始时，各数 据采集点上传训练好的模型参数，并为下⼀次训练需要⼀个新的模型 ing。对于m ∈ [0, M−1] 的模型请求，云端将标记为m+1 的模型替 换为数据采集站上传的模型，并返回标记为m的模型。⽪胡尔等⼈。 [41] 实验证明，当服务器中的模型数量M和数据采集站点数量N满⾜： M2 = N时，可以完全实现异步

### 论文 8 差分隐私 敏感度分片

Protect_Privacy_from_Gradient_Leakage_Attack_in_Federated_Learning

#### 给梯度敏感性高的梯度增加更多噪声

提出了一种基于梯度敏感性的切片策略来量化风险，给不同分片添加了不同的差分隐私扰动；

提出了全局梯度相关性的补偿机制，全局梯度的相关性可以用来补偿扰动引入的误差，也就是说那些添加了很少扰动的片段可以补偿聚合后的梯度误差

（全局梯度意为聚合后的梯度）

#### 梯度敏感性

是梯度对输入数据的微小变化的敏感程度，梯度敏感性高的模型更容易受到梯度泄漏攻击的威胁

#### 梯度分片、梯度剪枝

梯度分片：将梯度分片，并选择任意个分片上传（一般根据敏感度分片或者按层分片）

梯度剪枝：一般按照敏感度剪掉一些梯度

分片可以添加不同的扰动，剪枝比较简单干脆

#### 渐进性扰动

这种方法通过在每个训练轮次中逐渐增加扰动的幅度，从而降低扰动的影响。随着训练的进行，模型逐渐适应于扰动，并且梯度的相关性逐渐减少。这种渐进性的扰动可以减少梯度泄漏攻击对训练准确性的影响

#### 可能能抄袭的

https://blog.csdn.net/wenzhu2333/article/details/124556920

### 论文 9 秘密共享 优化论文1

LI G H TSE CAG G: A LIGHTWEIGHT AND VERSATILE DESIGN FOR SECURE AGGREGATION IN FEDERATED LEARNING

基于论文1的设计优化

回顾论文1的内容：服务器会重建聚合数据，而不知晓每个用户的数据

#### 不依赖差分隐私

保护个⼈更新的隐私，⽽不依赖于差异隐私（Truex et al., 2020）或可信执⾏环境（TEE） （Nguyen 等⼈， 2021）。

### 论文 10 新的秘密共享方法

FastSecAgg: Scalable Secure Aggregation for Privacy-Preserving Federated Learning

#### 基本方法

每个客户端将梯度分发给N个客户端（通过服务器发送，这时需要进行对称加解密）

### 论文 11 拜占庭攻击

服务器根据根数据集以及每轮更新的方向，给参与方发来的梯度打分，称为信任分数

基线：是利⽤拜占庭稳健的聚合规则，该规则本质上是⽐较客⼾端的本地模型更新并 在使⽤它们更新全局模型之前删除统计异常值，但是重点是没有预训练小模型

### 论文 12 客户端对恶意服务器进行防御

Fusion Efficient and Secure Inference Resilient to Malicious Servers

#### 主要研究

论文使用较强的威胁模型（非诚实好奇的对手），方案是：参与方知晓样本标签，对服务器计算结果进行评判，参与方将一组公开样本和查询样本混合，服务器如果作弊则必须为查询样本的所有复制样本给出不正确但一致的结果，来保护服务器诚实性

### 论文 13 隐私保护综述论文一些知识

指出联邦学习中，不是每个参与方都有机会参与每一轮训练，通常利用采样的方式确定哪些用户可以参与训练过程

联邦学习的各参与方可以是“异质”的，即参与方软硬件 配置、持有的数据格式、数据分布、模型结构等都可不同，

FedAVG 方法允许参与方在服务器聚合参数之前多次迭代计算梯度值

常用的 FedAVG 聚合方式为加权平均

论文介绍了隐私攻击的类别和方法

#### 隐私保护方法

差分隐私：是建立在严格的数学理论基础之上的强隐私 保护模型，能保证攻击者即便在具有最大背景知识的前提 下，即已知数据库中除目标记录以外其他所有记录的信息， 也无法推测出目标记录的敏感信息。

+ 实现本地化差分隐私的机制主要是随机响应技术、混洗 模型［57］ 。混洗模型在本地差分隐私的基础上，增加了一个 可信的 shuffler 部件，将用户端发来的数据随机打散后再发 给服务器，达到匿名的效果
+ 差分隐私需要有可信的第三方数据收集 者，保证所收集的数据不会被窃取和泄露。在实际应用中， 第三方数据收集者是否真正可信很难保证。本地化差分隐 私将数据隐私化的工作转移到用户端，在数据发出用户设备 之前先进行扰动，避免了不可信第三方造成的数据泄露
+ 差分隐私就是简单的添加随机扰动

### 论文 14 联邦学习隐私综述

Federated learning as a privacy solution-an overview

#### 聚合算法

差分隐私和安全多⽅计算等概念可以轻松应⽤于 FedAvg

FedMA 该平均算法提出了对具有相似特征提取签名的隐藏元素进⾏分层匹配和平均，以构建全局模型。 FedMA ⾮常适⽤于 CNN 和 LSTM， 并解决了数据偏差

FedSGD FedPer FedDyn FedDist

### 论文 15 联邦学习隐私综述 2023-2

Review on security of federated learning and its application in

分布式机器学习：数据并行和模型并行（不理解）

边缘计算。在联邦学习设置中，通信成本通常决定计算成本。原因是设备上的数据集 较少，参与者只有在⽹络稳定的情况下才愿意进⾏模型训练。因此，在每次全局聚合之前可以 在边缘节点或终端设备上执⾏更多计算，以减少训练模型所需的通信轮数

可以通过多训练，少聚合的方式提交联邦学习通信效率

#### 联邦学习学习应用

⾕歌键盘的下⼀个单词预测是第⼀个应⽤于移动设备的联合学习系统

在医学领域得到⼴泛 应⽤：基于联邦学习对多家医院和机构的实际健康数据进⾏训练可以提⾼ 医疗诊断的质量

如：接受了来⾃多个医院的 COVID-19 图像数据的培训，输入图像，输出疾病预测

#### 基于 FedAvg 的方法：SVeriFL

SVeriFL，⼀种具有隐私保护功能的连续 可验证联邦学习。引⼊了基于BLS签名和多⽅安全的精⼼设计的协议，可以验证参 与者上传参数的完整性和服务器聚合结果的正确性；还可以证明任意多个参与者之 间从服务器接收到的聚合结果的⼀致性。

#### 联邦学习框架

TensorFlow 框架和微众银⾏ Fate 框 架。还有其他衍⽣框架，例如Flower [28]，这是⼀个全新的框架

#### 疑问及解答

1. 垂直联邦学习，样本特征不同，则神经网络的输入层不同，如何协同工作？

例如，每个参与方可以独立地处理自己的特征通过神经网络的第一层，然后在更高的层次上与其他参与方的结果进行整合

2. 垂直联邦学习如何进行样本对齐？

样本ID需要加密共享，创建一个对齐表，再使用对齐的ID进行训练，这一过程中也需要使用到加密算法

3. 迁移联邦学习只会在后面几层进行聚合吗？

是的，确实可能只有模型的后几层（通常是分类层或决策层）在参与方之间进行聚合，而模型的前面几层（特征提取层）则在本地进行训练和调整

4. 横向联邦学习中如何保证样本特征相同？

> 在实践中，为了使模型能够被共同训练，各个参与方需要在数据预处理阶段统一特征。
>
> 特征统一的主要步骤通常包括：
>
> 1. **特征对齐**：
>    - 所有参与方必须协商定义一个共同的特征集，确保所有数据在同样的特征空间中进行表达。这通常通过安全的多方计算（Secure Multi-Party Computation, SMPC）或同态加密（Homomorphic Encryption）等隐私保护技术来实现，以避免原始数据的泄露。
> 2. **特征编码**：
>    - 所有特征必须以相同的方式被编码和标准化。这意味着所有的参与方都需要使用相同的方法将原始数据转换为模型可以使用的格式。例如，分类特征通常需要进行独热编码（One-Hot Encoding），数值特征需要进行归一化或标准化。
> 3. **特征抽取**：
>    - 如果原始特征空间中的某些特征对于模型训练不是必需的，参与方可以通过特征选择（Feature Selection）来共同决定哪些特征是重要的。
> 4. **特征工程**：
>    - 除了基本的特征对齐和编码之外，参与方可以协作进行更复杂的特征工程，比如构建组合特征、应用特征转换等。
> 5. **隐私保护的特征共享**：
>    - 如果特征工程步骤需要不同参与方之间的信息交换，那么需要采用隐私保护技术确保在不直接交换原始数据的情况下进行。
> 6. **通信协议**：
>    - 参与方之间应当建立起一套通信协议，明确如何安全地交换特征信息，如何处理不匹配的数据和缺失值等问题。

### 论文 16 同态加密+客户端投毒攻击

FedDefender: Client-Side Attack-Tolerant Federated Learning

论文提到：模型的最后⼀层⽐中间层更容易因噪声更新⽽过度拟，可以按不同层进行差分隐私

基于 Fed Avg 添加对抗模型中毒攻击防御，中毒攻击依赖于向中央服务器发送损坏的本地更新，最终污染经过 训练的全局模型。

> 实现这⼀⽬标的⼀种⽅法是为全局模型聚合（即阶段•4 ）实施服务器端鲁棒聚合 策略，其⽬的是在全局聚合期间保留来⾃良性客⼾端的更新，同时丢弃来⾃恶意客⼾端的 所有错误更新阶段
>
> FedDefender 的⽬标是直接解决⽅程 1。 3 通过重新设计本地模型训练流程（阶段•2 ）并修改合法客⼾端的本地更新Δ 。详 细信息将在下⼀节中描述。

主要工作

1. 容忍攻击的本地元更新

提前在本地进行模拟扰动训练，使得神经网络对噪声扰动不敏感

1. 容忍攻击的全局知识

然⽽，在模型中毒攻击的情况下，如果全局模型的可信度可能会受到损害，进行全局知识蒸馏（基于梯度下降优化的固有本质，模型深层更容易过度适应噪声，所以只取全局知识的神经网络中间层，设定置信度，进行一定的全局蒸馏损失）

#### 阅读源码

使用简单的方法：模拟多个参与方，添加模拟扰动、知识蒸馏工作进行本地数据训练，再进行聚合，输出模型准确度

### 异构联邦学习

异构物联网下资源高效的分层协同联邦学习方法，用于解决数据分布不均匀和设备计算通信性能不同的问题

分层联邦学习：添加一层边缘层，收集来自多个设备的模型更新，并进行一定程度的聚合和处理，然后中心服务器进行进一步的聚合

### 论文 - 隐私攻击

Analyzing User-Level Privacy Attack Against Federated Learning

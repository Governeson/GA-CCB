import os
import string
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from torch import nn
from torch import Tensor
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchmetrics import Accuracy, Precision, Recall, F1Score
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 使用单个矩阵一次性计算出queries,keys,values
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 将queries，keys和values划分为num_heads
        # print("1qkv's shape: ", self.qkv(x).shape)  # 使用单个矩阵一次性计算出queries,keys,values
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上
        # print("2qkv's shape: ", qkv.shape)

        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # print("queries's shape: ", queries.shape)
        # print("keys's shape: ", keys.shape)
        # print("values's shape: ", values.shape)

        # 在最后一个维度上相加
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        # print("energy's shape: ", energy.shape)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        # print("scaling: ", scaling)
        att = F.softmax(energy, dim=-1) / scaling
        # print("att1' shape: ", att.shape)
        att = self.att_drop(att)
        # print("att2' shape: ", att.shape)

        # 在第三个维度上相加
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # print("out1's shape: ", out.shape)
        out = rearrange(out, "b h n d -> b n (h d)")
        # print("out2's shape: ", out.shape)
        out = self.projection(out)
        # print("out3's shape: ", out.shape)
        return out
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        # sequential:将图片转变为序列
        self.proj = nn.Sequential(
            # 使用一个卷积层而不是一个线性层 -> 性能增加 3通道
            # conv(kernal_size:3,768,16,16)
            #nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # 输出 [1,768,14,14] 重排变成[1,196,768]
            Rearrange('b e (h) (w) -> b (h w) e'),  # (h w) 相当于两者相乘
        )
        # 生成一个维度为emb_size的向量当做最后分类用的cls_token [1,1,768]
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # 位置编码信息，一共有(img_size // patch_size)^2: 即14*14=196个 + 1(cls token)个位置向量 共197个
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape  # 单独先将batch缓存起来
        # print("!!!!!!!!!!!!!!",x.size()) # 第一次[1,3,224,224] 第二次[2,3,224,224]
        x = self.proj(x)  # 进行卷积操作
        # print("##########",x.size()) # 第一次[1,196,768] 第二次[2,197,768]
        # 将cls_token 扩展b次
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # 将cls token在维度1扩展到输入上
        x = torch.cat([cls_tokens, x], dim=1)
        # print(x.shape, self.positions.shape)
        x += self.positions
        # print("x.size",x.size())
        return x
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.3):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
'''
class DyT(nn.Module):
    def __init__(self, dim, init_alpha=0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)  # 动态缩放因子
        self.gamma = nn.Parameter(torch.ones(dim))             # 仿射变换参数（类似LN）
        self.beta = nn.Parameter(torch.zeros(dim))            # 仿射变换参数（类似LN）

    def forward(self, x):
        x = torch.tanh(self.alpha * x)        # 动态缩放 + tanh 非线性
        return self.gamma * x + self.beta     # 仿射变换
'''
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.3,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.3,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                #DyT(emb_size),
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                #DyT(emb_size),
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p))
            ))
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

# 亮度因子列表
brightness_factors = [0.8, 1.0, 1.2]
class Graydim(torch.nn.Module):
    def __init__(self):
        super(Graydim,self).__init__()
        self.attention = SpatialAttention()
    def adjust_brightness(self, gray_image, brightness_factor):
        # 调整灰度图像的亮度
        adjusted_gray_image = cv2.convertScaleAbs(gray_image, alpha=brightness_factor, beta=0)
        return adjusted_gray_image
    def forward(self, x):
        grouped_gray_images = [[] for _ in range(len(brightness_factors))]
        for i in range(x.size(0)):  # Iterate over each image in the batch
            rgb_image = x[i].detach().cpu().numpy().transpose(1, 2, 0)
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            for j, brightness_factor in enumerate(brightness_factors):
                adjusted_gray_image = self.adjust_brightness(gray_image, brightness_factor)
                grouped_gray_images[j].append(adjusted_gray_image)
                '''
                #gray_images.append(gray_image)
                plt.subplot(1, 2, 2)
                plt.imshow(adjusted_gray_image, cmap='gray')
                plt.title('Adjusted Gray Image (Brightness Factor: {})'.format(brightness_factor))
                plt.axis('off')
            plt.show()'''
        # 将灰度图像堆叠为张量
        gray_tensor = torch.from_numpy(np.stack([np.array(grouped_gray_images[i]).astype(np.float32) for i in range(len(brightness_factors))], axis=1))
        gray_tensor = gray_tensor.to(device)  # 将tensor移动到当前默认的设备
        #print(gray_tensor.size())
        weighted_tensor = self.attention(gray_tensor)
        return weighted_tensor  # 返回处理后的张量

# 亮度因子列表
brightness_channel_factor= [0.8,1.2]
class GrayChannel(torch.nn.Module):
    def __init__(self):
        super(GrayChannel,self).__init__()
    def forward(self, x):
        # 从RGB转换为HSV
        x = x.permute(0, 2, 3, 1).cpu().numpy()  # 将通道顺序调整为(批量大小, 224, 224, 3)
        hsv_images = [cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV) for rgb_image in x]
        # 提取V通道
        v_channel = torch.stack([torch.tensor(hsv_image[:, :, 2], dtype=torch.float32) for hsv_image in hsv_images],
                                dim=0)
        # 为每个亮度因子分别乘以v_channel   # 将V通道的范围从0-255映射到0-1
        v_channel1 = v_channel * brightness_channel_factor[0] / 255.0
        v_channel2 = v_channel * brightness_channel_factor[1] / 255.0
        # 拼接这两个通道
        v_channel_combined = torch.stack([v_channel1, v_channel2], dim=1).to(device)
        return v_channel_combined
        # 将灰度图像堆叠为张量
        #gray_tensor = torch.from_numpy(np.stack([np.array(grouped_gray_images[i]).astype(np.float32) for i in range(len(brightness_factors))], axis=1))
        #gray_tensor = gray_tensor.to(device)  # 将tensor移动到当前默认的设备
        #print(gray_tensor.size())
        #weighted_tensor = self.attention(gray_tensor)
        #return weighted_tensor  # 返回处理后的张量
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 16*5*224224
        self.gray_spa = Graydim()  # 灰度空间注意力
        self.gray_channel = GrayChannel()  # 通道调整模块
        self.conv1 = torch.nn.Conv2d(5, 32, 3, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 768, 3, 1, 1)
        self.bn4 = torch.nn.BatchNorm2d(768)
        self.pool = torch.nn.MaxPool2d(2)
        self.dropout = torch.nn.Dropout(p=0.3)
        #self.flatten = nn.Flatten()
        #self.fc1 = torch.nn.Linear(768 * 14 * 14, 128)  # 全连接层，输入维度为32*14*14，输出维度为128
        #self.fc2 = torch.nn.Linear(128, 16)  # 全连接层，输入维度为128，输出维度为类别数
    def forward(self, x):
        gray_attention = self.gray_spa(x)
        gray_channel = self.gray_channel(x)
        x = torch.cat((x, gray_channel), dim=1)
        x = x * gray_attention
        #print('1',x.size())
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        #print('2',x.size())
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        #print('3', x.size())
        x = self.pool(torch.nn.functional.relu(self.bn3(self.conv3(x))))
        #print('4', x.size())
        x = self.pool(torch.nn.functional.relu(self.bn4(self.conv4(x))))
        #print('5', x.size())
        #x = self.flatten(x)
        #x = self.fc1(x)
        #print('6', x.size())
        #x = self.fc2(x)
        #print('7', x.size())
        return x

# 定义通道注意力层的类
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 通道注意力模块
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 使用1x1卷积替代全连接层，更高效且避免维度变换
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // reduction_ratio),  # 添加批归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)  # 添加批归一化
        )
        self.sigmoid = nn.Sigmoid()

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 处理平均池化和最大池化路径
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        # 特征融合
        out = avg_out + max_out  # 逐元素相加
        return self.sigmoid(out)
# 定义空间注意力层的类
class SpatialAttention(torch.nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size=7, padding=3)  # 卷积层，输入通道为2，输出通道为1，卷积核大小为7

    def forward(self, x):
        avg_out = torch.mean(x, dim=1).unsqueeze(1)  # 对输入数据在通道维度上求平均值，得到平均特征图，大小为批次大小*1*高*宽
        max_out = torch.max(x, dim=1)[0].unsqueeze(1)  # 对输入数据在通道维度上求最大值，得到最大特征图，大小为批次大小*1*高*宽
        out = torch.cat([avg_out, max_out], dim=1)  # 将平均特征图和最大特征图在通道维度上拼接起来，大小为批次大小*2*高*宽
        out = self.conv1(out)  # 卷积层 -> 输出，大小为批次大小*1*高*宽
        out = torch.sigmoid(out)  # 激活函数，得到每个位置的注意力权重，范围为[0,1]
        return out
# 定义组合注意力模块的类
class CBAM(torch.nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
    # 组合注意力
    def forward(self, x):
        temp = x
        x = self.channel_attention(x) * x  # 通道注意力乘以输入数据
        x = self.spatial_attention(x) * x  # 空间注意力乘以输入数据
        x += temp
        return x

class Unsample(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.rerange_1 = nn.Sequential(
            Rearrange('b h w e -> b e h w'), )
        self.rerange_2 = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'))
        self.cbam = CBAM(768)
    def forward(self, x: Tensor) -> Tensor:
        temp = x
        x = torch.split(x, [1, 196], dim=1)
        cls_token = x[0]
        x = x[1]
        patch = x.size()[0]
        x = x.reshape(patch, 14, 14, 768)  # [8/16, 14, 14, 768]
        x = self.rerange_1(x)  # [8/16, 768, 14, 14]
        x = self.cbam(x) * x
        x = self.rerange_2(x)  # [8/16, 196, 768]
        x = torch.cat([cls_token, x], dim=1)  # [8/16, 197, 768]
        result = x + temp
        return result

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))
class ViT(nn.Sequential):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 depth: int = 8,
                 n_classes: int = 10,
                 **kwargs):
        super().__init__(
            CNN(),
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            Unsample(),
            #TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            #Unsample(),
            ClassificationHead(emb_size, n_classes)
        )

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print("Total Trainable Params: {}".format(total_params))
    return total_params

def rgb_to_hsv(image):
    # 将PIL图像转换为NumPy数组
    image_np = np.array(image)
    # 使用OpenCV将RGB图像转换为HSV格式
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    # 转换为PIL图像
    hsv_image_pil = Image.fromarray(hsv_image)
    return hsv_image_pil

# 对图片进行预处理
transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.RandomCrop(224, padding=4),  # 随机裁剪
     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
     transforms.RandomRotation(degrees=(0, 90), expand=False, center=None),  # 随机旋转角度0～90度
     transforms.ToTensor(),  # 将图片化为张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

device = torch.device('mps')

#0为PlantVillage数据集 1为AI Challanger
dataset_choose = 0
name = ('/Users/zoulongquan/Desktop/*****论文实验/消融实验/37PV_gray_cnn_T_U')
if not os.path.exists(name):
    os.makedirs(name)
# 添加tensorboard"
writer = SummaryWriter(name)
if dataset_choose == 0:
    # 训练集存放路径
    image_train = '/Users/zoulongquan/Desktop/*****论文实验/dataset/PlantVillage37/train'
    image_test = '/Users/zoulongquan/Desktop/*****论文实验/dataset/PlantVillage37/test'
    num_classes = 10
elif dataset_choose == 1:
    image_train = '/Users/zoulongquan/Desktop/*****论文实验/dataset/AI_Challanger/train'
    image_test = '/Users/zoulongquan/Desktop/*****论文实验/dataset/AI_Challanger/test'
    num_classes = 16

#加载数据集，这里假设您的图片是jpg格式
train_dataset = torchvision.datasets.ImageFolder(root=image_train, transform=transform)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=4)
test_dataset = torchvision.datasets.ImageFolder(root=image_test, transform=transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False,num_workers=4)
'''
# 选取前10张图片
train_subset = torch.utils.data.Subset(train_dataset, range(50))
test_subset = torch.utils.data.Subset(test_dataset, range(50))

# 创建DataLoader
trainloader = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True)
testloader = torch.utils.data.DataLoader(test_subset, batch_size=16, shuffle=False)
'''
# 获取类别名称与类别索引的对应关系
class_names = train_dataset.classes

# 获取数据集中的图像数量
trainset_size = len(train_dataset)
testset_size = len(test_dataset)

if __name__ == '__main__':
    model = ViT(n_classes = 10)
    train_condition = 0
    #指定参数文件地址
    check_point_path = name+'/checkpoint.pth'

    #若train_condition = 0为从头训练，若为0则加载训练模型参数训练
    if train_condition == 0:
        learning_rate = 0.01
        start_epoch = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        try:
            checkpoint = torch.load(check_point_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            learning_rate = checkpoint['learning_rate']
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']-5

            print("成功加载模型参数")
            print("当前轮次为：",start_epoch)
            print("当前学习率为：",learning_rate)
        except FileNotFoundError:
            print("错误：未找到模型文件，请检查文件路径。")
        except Exception as e:
            print(f"错误：加载模型参数时出现未知错误：{e}")
    #summary(model, input_size=(3, 224, 224))
    model = model.to(device)
    count_parameters(model)
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    criterion = criterion.to(device)
    num_epochs = 100
    total_train_step = 0  # 记录训练的次
    total_test_step = 0  # 记录测试的次数
    class_stats = {i: {"TP": 0, "FP": 0, "FN": 0} for i in range(num_classes)}

    best_accuracy = 0.0  # 用于记录最优模型
    best_loss = 100.0


    if train_condition != 0:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
    # 训练模型
    model.train()
    for m in range(start_epoch,num_epochs):  # 遍历每个训练轮数
        fun_loss = 0  # 每个epoch初始化损失值
        print("///第{}轮训练开始///".format(m + 1))
        if(m == 15):
            learning_rate = 0.001
            optimizer.param_groups[0]['lr'] = learning_rate  # 更新优化器参数

        if m > 15:
            learning_rate = learning_rate * 0.95
            optimizer.param_groups[0]['lr'] = learning_rate  # 更新优化器参数
        correct = 0  # 初始化正确预测计数
        total = 0  # 初始化总预测计数
        print("///当前学习率为:{}///".format(learning_rate))
        # 遍历数据集中的所有图像
        for i, data in enumerate(trainloader, 0):
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # 清零梯度
            outputs = model(x)
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            # 计算准确率

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            fun_loss += loss.item()
            total_train_step+=1
            if total_train_step % 100 == 0:
                accuracy = correct / total  # 计算准确率
                print("训练次数:{}, Loss:{}, Accuracy:{}".format(total_train_step, loss.item(), accuracy))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
                writer.add_scalar("train_accuracy", accuracy, total_train_step)
                # writer.add_images('inputs', inputs[0], total_train_step, dataformats='CHW')
                # out_img = outputs.view(-1, 1, num_classes)
                # out_pic = torchvision.utils.make_grid(outputs)
                # writer.add_images('outputs', out_pic, total_train_step, dataformats='CHW')
                # fun_loss = 0.0
        # 记录学习率时使用当前轮次（修正4）
        writer.add_scalar("train_learning_rate", learning_rate, m)
         # 保存检查点
        checkpoint = {
            'epoch': m,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        torch.save(checkpoint, check_point_path)

        print("检查点已保存")
        # 测试模型
        model.eval()
        total_test_loss = 0
        correct = 0
        correct_per_class = [0] * num_classes
        total_per_class = [0] * num_classes
        test_acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        precision = Precision(task='multiclass', num_classes=num_classes).to(device)
        recall = Recall(task='multiclass', num_classes=num_classes).to(device)
        f1_score = F1Score(task='multiclass', num_classes=num_classes).to(device)

        test_acc.reset()
        precision.reset()
        recall.reset()
        f1_score.reset()

        with torch.no_grad():
            for data in testloader:
                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()  # 累加loss的标量值
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                # 统计每个类别的 TP、FP、FN
                for i in range(num_classes):
                    # TP: 预测为 i 且真实为 i
                    tp_mask = (preds == i) & (targets == i)
                    class_stats[i]["TP"] += tp_mask.sum().item()

                    # FP: 预测为 i 但真实不是 i
                    fp_mask = (preds == i) & (targets != i)
                    class_stats[i]["FP"] += fp_mask.sum().item()

                    # FN: 真实为 i 但预测不是 i
                    fn_mask = (targets == i) & (preds != i)
                    class_stats[i]["FN"] += fn_mask.sum().item()
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

        for i in range(num_classes):
            tp = class_stats[i]["TP"]
            fp = class_stats[i]["FP"]
            fn = class_stats[i]["FN"]

            # 处理除零错误
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            macro_precision += precision
            macro_recall += recall
            macro_f1 += f1

        # 计算指标
        avg_test_loss = total_test_loss / len(testloader)
        test_accuracy = correct / testset_size
        # 计算宏平均
        macro_precision /= num_classes
        macro_recall /= num_classes
        macro_f1 /= num_classes

        # 记录到TensorBoard
        writer.add_scalar("test_loss", avg_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
        writer.add_scalar("Macro Precision", macro_precision, total_test_step)
        writer.add_scalar("Macro Recall", macro_recall, total_test_step)
        writer.add_scalar("Macro F1-score", macro_f1, total_test_step)

        print(f"整体测试集Loss: {avg_test_loss:.4f}")
        print(f"整体测试正确率: {test_accuracy:.4f}")
        # 输出结果
        #print(f"整体测试集Loss: {total_test_loss / len(testloader):.4f}")
        print(f"整体测试正确率 (Accuracy): {test_accuracy:.4f}")
        print(f"宏平均 Precision: {macro_precision:.4f}")
        print(f"宏平均 Recall: {macro_recall:.4f}")
        print(f"宏平均 F1-score: {macro_f1:.4f}")


        if test_accuracy >= best_accuracy:
            print(f"测试准确率提升：{best_accuracy:.4f} → {test_accuracy:.4f}，保存最优模型...")
            torch.save(model.state_dict(), os.path.join(name, "best_acc_model.pth"))
            best_accuracy = test_accuracy  # 更新最佳准确率
        if avg_test_loss <= best_loss:
            print(f"loss下降：{best_loss:.4f} → {avg_test_loss:.4f}，保存最优模型...")
            torch.save(model.state_dict(), os.path.join(name, "best_loss_model.pth"))
            best_loss = avg_test_loss  # 更新最佳准确率

        total_test_step += 1

    torch.save(model.state_dict(), "final_model.pth")
    writer.close()
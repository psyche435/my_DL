import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import time
import datetime
from tqdm import tqdm  # 用于显示进度条

# 设置随机种子，保证结果可复现
torch.manual_seed(5)
np.random.seed(5)

start_token = 'G'
end_token = 'E'
batch_size = 64

# 检测GPU并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def process_poems(file_name):
    """
    处理诗歌文件，生成诗歌向量、词汇表和词汇到索引的映射
    """
    poems = []
    try:
        with open(file_name, "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    # 尝试解析标题和内容
                    if ':' in line:
                        title, content = line.split(':', 1)
                    else:
                        content = line

                    # 清理内容
                    content = content.replace(' ', '').replace('，', '，').replace('。', '。')

                    # 过滤不符合条件的诗歌
                    if any(special_char in content for special_char in ['_', '(', '（', '《', '[', ']', '》', '）', '：', '？', '！', '“', '”', '‘', '’']):
                        continue
                    if len(content) < 10 or len(content) > 80:  # 调整诗歌长度范围
                        continue

                    # 添加开始和结束标记
                    content = start_token + content + end_token
                    poems.append(content)
                except ValueError:
                    continue  # 忽略解析错误的行
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return [], {}, []

    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))

    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += list(poem)

    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)  # 添加空格作为填充字符

    # 创建词汇到索引的映射
    word_int_map = dict(zip(words, range(len(words))))

    # 将诗歌转换为向量
    poems_vector = [list(map(lambda w: word_int_map.get(w, word_int_map[' ']), poem)) for poem in poems]

    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    """
    生成训练批次
    """
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []

    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        x_data = poems_vec[start_index:end_index]
        y_data = []

        for row in x_data:
            # 目标序列是输入序列右移一位
            y = row[1:]
            y_data.append(y)

        # 使用 pad_sequence 对 x_data 和 y_data 进行填充
        x_tensor = pad_sequence([torch.tensor(row) for row in x_data], batch_first=True)
        y_tensor = pad_sequence([torch.tensor(row) for row in y_data], batch_first=True)

        # 将数据移到GPU
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

        print("x_tensor shape:", x_tensor.shape)
        print("y_tensor shape:", y_tensor.shape)

        x_batches.append(x_tensor)
        y_batches.append(y_tensor)

    return x_batches, y_batches


# 定义词嵌入层
class WordEmbedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(WordEmbedding, self).__init__()
        # 随机初始化词向量
        w_embedding_random_initial = np.random.uniform(-1, 1, size=(vocab_length, embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embedding_random_initial).float())

    def forward(self, input_sentence):
        """
        :param input_sentence: 包含多个词索引的张量
        :return: 词嵌入张量
        """
        return self.word_embedding(input_sentence)


# 定义权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("initialized linear weight")


# 定义改进的RNN模型 - 使用GRU替代LSTM
class ImprovedRNNModel(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, gru_hidden_dim, num_layers=2, dropout=0.2):
        super(ImprovedRNNModel, self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.gru_dim = gru_hidden_dim
        self.num_layers = num_layers

        # 定义GRU层 (多层，batch_first=True表示输入输出格式为(batch, seq, feature))
        self.gru = nn.GRU(
            embedding_dim,
            gru_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False  # 单向GRU更适合文本生成
        )

        # 定义全连接层，将GRU输出映射到词汇表大小
        self.fc = nn.Linear(gru_hidden_dim, vocab_len)

        # 应用权重初始化
        self.apply(weights_init)

    def forward(self, sentence, hidden=None):
        # 获取词嵌入
        batch_input = self.word_embedding_lookup(sentence)  # 输出形状：[batch_size, seq_len, embedding_dim]

        # 通过GRU层
        output, hidden = self.gru(batch_input, hidden)  # 输出形状：[batch_size, seq_len, gru_dim]

        # 通过全连接层
        output = self.fc(output)  # 输出形状：[batch_size, seq_len, vocab_size]

        return output, hidden


def train_model():
    """
    训练模型
    """
    # 处理数据集
    poems_vector, word_to_int, vocabularies = process_poems('./poems.txt')

    if not poems_vector:
        print("没有有效的诗歌数据！")
        return

    print(f"成功加载 {len(poems_vector)} 首诗歌")
    print(f"词汇表大小: {len(word_to_int)}")

    # 设置批次大小
    BATCH_SIZE = 64

    # 创建词嵌入层 - 增加嵌入维度
    word_embedding = WordEmbedding(vocab_length=len(word_to_int), embedding_dim=256)
    word_embedding = word_embedding.to(device)  # 移到GPU

    # 创建模型 - 增加隐藏层维度和层数
    model = ImprovedRNNModel(
        batch_sz=BATCH_SIZE,
        vocab_len=len(word_to_int),
        word_embedding=word_embedding,
        embedding_dim=256,
        gru_hidden_dim=512,  # 增加隐藏层维度
        num_layers=3,  # 增加层数
        dropout=0.3  # 添加dropout防止过拟合
    )
    model = model.to(device)  # 移到GPU

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    loss_function = nn.CrossEntropyLoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # 计算总训练步数用于进度显示
    total_epochs = 50
    batches_per_epoch = len(poems_vector) // BATCH_SIZE
    total_steps = total_epochs * batches_per_epoch

    print(f"开始训练，总轮数: {total_epochs}, 每轮批次: {batches_per_epoch}, 总步数: {total_steps}")

    # 训练开始时间
    start_time = time.time()
    step_count = 0

    # 训练循环 - 增加训练轮数
    for epoch in range(total_epochs):
        # 生成批次数据
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_batches = len(batches_inputs)
        total_loss = 0

        model.train()  # 设置为训练模式

        # 使用tqdm显示进度条
        progress_bar = tqdm(enumerate(batches_inputs), total=n_batches, desc=f'Epoch {epoch + 1}/{total_epochs}')

        for batch_idx, batch_x in progress_bar:
            batch_y = batches_outputs[batch_idx]

            # 前向传播
            model.zero_grad()
            output, _ = model(batch_x)  # 输出形状：[batch_size, seq_len, vocab_size]

            # 确保输出序列长度和目标序列长度一致
            if output.size(1) > batch_y.size(1):
                output = output[:, :batch_y.size(1), :]
            elif output.size(1) < batch_y.size(1):
                padding = torch.zeros(output.size(0), batch_y.size(1) - output.size(1), output.size(2)).to(output.device)
                output = torch.cat([output, padding], dim=1)

            # 调整输出形状以匹配损失函数的期望输入
            output = output.reshape(-1, output.size(-1))  # 形状：[batch_size * seq_len, vocab_size]
            batch_y = batch_y.reshape(-1)  # 形状：[batch_size * seq_len]

            # 计算损失
            loss = loss_function(output, batch_y)
            total_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪防止梯度爆炸
            optimizer.step()

            # 更新进度条信息
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

            # 更新全局步数和计算ETA
            step_count += 1
            elapsed_time = time.time() - start_time
            steps_per_second = step_count / elapsed_time
            remaining_steps = total_steps - step_count
            eta_seconds = remaining_steps / steps_per_second
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))

            # 更新进度条ETA信息
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'ETA': eta})

        # 每个epoch的平均损失
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{total_epochs}, Average Loss: {avg_loss:.4f}, ETA: {eta}")

        # 更新学习率
        scheduler.step(avg_loss)

        # 每个epoch保存一次模型
        torch.save(model.state_dict(), f'./poem_generator_gru_epoch_{epoch + 1}')
        print(f"Epoch {epoch + 1} 模型已保存")

    # 计算总训练时间
    total_training_time = time.time() - start_time
    print(f"训练完成！总训练时间: {str(datetime.timedelta(seconds=int(total_training_time)))}")


def temperature_sample(logits, temperature=1.0):
    """使用温度参数进行采样"""
    if temperature == 0:  # 贪婪采样
        return torch.argmax(logits, dim=-1)

    # 应用温度
    probs = torch.softmax(logits / temperature, dim=-1)

    # 采样
    sample = torch.multinomial(probs, 1).squeeze()

    return sample


def generate_poem(begin_word, model_path='./poem_generator_gru_epoch_50', temperature=0.7):
    """
    生成诗歌
    """
    # 处理数据集获取词汇表
    poems_vector, word_to_int, vocabularies = process_poems('./poems.txt')

    if not poems_vector:
        print("没有有效的诗歌数据！")
        return

    print(f"成功加载 {len(poems_vector)} 首诗歌")
    print(f"词汇表大小: {len(word_to_int)}")

    # 设置批次大小
    BATCH_SIZE = 64

    # 创建词嵌入层 - 增加嵌入维度
    word_embedding = WordEmbedding(vocab_length=len(word_to_int), embedding_dim=256)
    word_embedding = word_embedding.to(device)  # 移到GPU

    # 创建模型 - 增加隐藏层维度和层数
    model = ImprovedRNNModel(
        batch_sz=BATCH_SIZE,
        vocab_len=len(word_to_int),
        word_embedding=word_embedding,
        embedding_dim=256,
        gru_hidden_dim=512,  # 增加隐藏层维度
        num_layers=3,  # 增加层数
        dropout=0.0  # 生成时不使用dropout
    )
    model = model.to(device)  # 移到GPU

    # 加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 设置为评估模式
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 生成诗歌
    poem = begin_word
    word = begin_word
    hidden = None  # 初始化隐藏状态

    with torch.no_grad():
        while len(poem) < 100:  # 限制最大长度
            # 准备输入
            input = torch.tensor([word_to_int[word]], dtype=torch.long).unsqueeze(0).to(device)  # 添加批次维度并移到GPU

            # 预测下一个词
            output, hidden = model(input, hidden)  # 输出形状：[1, 1, vocab_size]
            output = output.squeeze(0)  # 形状：[1, vocab_size]

            # 使用温度采样
            next_idx = temperature_sample(output[0].cpu(), temperature)  # 采样在CPU上进行

            # 转换为字
            next_word = vocabularies[next_idx.item()]

            # 如果生成结束标记，停止生成
            if next_word == end_token:
                break

            # 添加到诗歌中
            poem += next_word
            word = next_word

    # 格式化输出，移除开始和结束标记
    poem = poem.replace(start_token, '').replace(end_token, '')

    # 智能格式化诗歌
    formatted_poem = format_poem(poem)

    return formatted_poem


def format_poem(poem):
    """
    智能格式化诗歌，识别五言绝句、七言绝句等常见格式
    """
    # 尝试识别常见的唐诗格式
    poem_length = len(poem)

    # 检查是否包含标点符号
    has_punctuation = any(p in poem for p in ['，', '。'])

    if not has_punctuation:
        # 尝试根据长度猜测格式
        if poem_length == 20:  # 五言绝句
            return f"{poem[:5]}，{poem[5:10]}。\n{poem[10:15]}，{poem[15:20]}。"
        elif poem_length == 28:  # 七言绝句
            return f"{poem[:7]}，{poem[7:14]}。\n{poem[14:21]}，{poem[21:28]}。"
        elif poem_length == 40:  # 五言律诗
            return f"{poem[:5]}，{poem[5:10]}。\n{poem[10:15]}，{poem[15:20]}。\n{poem[20:25]}，{poem[25:30]}。\n{poem[30:35]}，{poem[35:40]}。"
        elif poem_length == 56:  # 七言律诗
            return f"{poem[:7]}，{poem[7:14]}。\n{poem[14:21]}，{poem[21:28]}。\n{poem[28:35]}，{poem[35:42]}。\n{poem[42:49]}，{poem[49:56]}。"
        else:
            # 默认每行7个字
            lines = []
            for i in range(0, poem_length, 7):
                line = poem[i:i+7]
                if len(line) == 7 and i + 7 < poem_length:
                    line += '，'
                elif len(line) < 7:
                    line += '。'
                lines.append(line)
            return '\n'.join(lines)
    else:
        # 如果包含标点符号，根据标点符号分割
        lines = []
        current_line = ""
        for char in poem:
            current_line += char
            if char in ['，', '。']:
                lines.append(current_line)
                current_line = ""
        if current_line:
            lines.append(current_line)
        return '\n'.join(lines)


if __name__ == "__main__":
    # 训练模型
    train_model()

    # 生成诗歌示例
    for start_word in ["日", "红", "山", "夜", "湖", "君"]:#"日", "红", "山", "夜", "湖", "君"
        print(f"\n以 '{start_word}' 开头的诗歌 (温度=0.7):")
        poem = generate_poem(start_word, model_path='./poem_generator_gru_epoch_18', temperature=0.7)
        print(poem)

        print(f"\n以 '{start_word}' 开头的诗歌 (温度=0.5):")
        poem = generate_poem(start_word, model_path='./poem_generator_gru_epoch_18', temperature=0.5)
        print(poem)
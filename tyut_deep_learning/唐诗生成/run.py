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
# batch_size = 64 # Defined locally

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
                    content = line
                    # 尝试解析标题和内容 (如果格式为 标题:内容)
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) > 1 and parts[1].strip(): # Ensure content exists after colon
                            content = parts[1]
                        # If no content after colon, or no colon, the whole line is content

                    # 清理内容
                    content = content.replace(' ', '').replace('，', '，').replace('。', '。')

                    # 过滤不符合条件的诗歌
                    if any(special_char in content for special_char in ['_', '(', '（', '《', '[', ']', '》', '）', '：', '？', '！', '“', '”', '‘', '’']):
                        continue
                    if len(content) < 5 or len(content) > 80:  # 调整诗歌长度范围 (min length 5 for better training)
                        continue

                    # 添加开始和结束标记
                    content = start_token + content + end_token
                    poems.append(content)
                except ValueError:
                    # print(f"Skipping line due to ValueError: {line}") # Optional debug
                    continue  # 忽略解析错误的行
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return [], {}, []

    if not poems:
        print("警告: 未能从文件中加载任何有效的诗歌。")
        return [], {}, []

    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))

    # 统计每个字出现次数
    all_words_chars = [] # Renamed for clarity
    for poem_str in poems:
        all_words_chars.extend(list(poem_str)) # Use extend for list of chars

    counter = collections.Counter(all_words_chars)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    
    # Ensure G and E are in vocab if they exist in poems
    # Also ensure ' ' is in vocab for padding
    unique_chars, _ = zip(*count_pairs)
    vocab_list = list(unique_chars)

    if ' ' not in vocab_list:
        vocab_list.append(' ') # Add space if not present, for padding

    # 创建词汇到索引的映射
    word_int_map = {word: i for i, word in enumerate(vocab_list)}

    # 将诗歌转换为向量
    # Fallback to space index if a char is somehow not in vocab (should not happen with this logic)
    space_idx = word_int_map[' '] 
    poems_vector = [[word_int_map.get(w, space_idx) for w in poem_str] for poem_str in poems]
    
    print(f"词汇表构建完成。词汇表大小: {len(vocab_list)}")
    if start_token not in word_int_map: print(f"警告: start_token '{start_token}' 未在词汇表中!")
    if end_token not in word_int_map: print(f"警告: end_token '{end_token}' 未在词汇表中!")


    return poems_vector, word_int_map, vocab_list # Return vocab_list instead of words tuple


def generate_batch(batch_size, poems_vec, word_to_int):
    """
    生成训练批次
    """
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []

    padding_idx = word_to_int.get(' ', 0) # Get index of space for padding

    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        x_data = poems_vec[start_index:end_index]
        y_data = []

        for row in x_data:
            y = row[1:]
            y_data.append(y)

        # 使用 pad_sequence 对 x_data 和 y_data 进行填充
        # Ensure tensors are created before padding
        x_tensor_list = [torch.tensor(row, dtype=torch.long) for row in x_data]
        y_tensor_list = [torch.tensor(row, dtype=torch.long) for row in y_data]

        x_tensor = pad_sequence(x_tensor_list, batch_first=True, padding_value=padding_idx)
        y_tensor = pad_sequence(y_tensor_list, batch_first=True, padding_value=padding_idx)

        # 将数据移到GPU
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

        # print("x_tensor shape:", x_tensor.shape) # Kept for debugging as in user's code
        # print("y_tensor shape:", y_tensor.shape)

        x_batches.append(x_tensor)
        y_batches.append(y_tensor)

    return x_batches, y_batches


class WordEmbedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(WordEmbedding, self).__init__()
        w_embedding_random_initial = np.random.uniform(-1, 1, size=(vocab_length, embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embedding_random_initial).float())

    def forward(self, input_sentence):
        return self.word_embedding(input_sentence)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None: # Check if bias exists
            m.bias.data.fill_(0)
        # print("initialized linear weight") # Kept as in user's code
    elif classname.find('GRU') != -1: # Initialize GRU weights for better stability
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data) # Orthogonal initialization for recurrent weights
            elif 'bias' in name:
                param.data.fill_(0)


class ImprovedRNNModel(nn.Module):
    # batch_sz removed from __init__ as it's inferred from input
    def __init__(self, vocab_len, word_embedding, embedding_dim, gru_hidden_dim, num_layers=2, dropout=0.2):
        super(ImprovedRNNModel, self).__init__()

        self.word_embedding_lookup = word_embedding
        # self.batch_size = batch_sz # Not used
        self.vocab_length = vocab_len
        self.gru_dim = gru_hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            embedding_dim,
            gru_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0, # Dropout only if multiple layers
            bidirectional=False
        )
        self.fc = nn.Linear(gru_hidden_dim, vocab_len)
        self.apply(weights_init)

    def forward(self, sentence, hidden=None):
        batch_input = self.word_embedding_lookup(sentence)
        output, hidden = self.gru(batch_input, hidden)
        output = self.fc(output)
        return output, hidden


def train_model(continue_from_epoch=None, continue_model_path=None):
    """
    训练模型，可以从指定epoch继续训练
    """
    poems_vector, word_to_int, vocabularies = process_poems('./poems.txt')

    if not poems_vector:
        print("没有有效的诗歌数据！训练中止。")
        return

    print(f"成功加载 {len(poems_vector)} 首诗歌")
    print(f"词汇表大小: {len(word_to_int)}")

    BATCH_SIZE = 256
    EMBEDDING_DIM = 256
    GRU_HIDDEN_DIM = 512
    NUM_LAYERS = 3
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    TOTAL_EPOCHS = 200  # 总训练轮数，从开始到结束
    CLIP_GRAD_NORM = 1.0
    
    padding_idx = word_to_int.get(' ', 0) # Get padding index

    word_embedding = WordEmbedding(vocab_length=len(word_to_int), embedding_dim=EMBEDDING_DIM).to(device)
    model = ImprovedRNNModel(
        vocab_len=len(word_to_int),
        word_embedding=word_embedding,
        embedding_dim=EMBEDDING_DIM,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(ignore_index=padding_idx) # Use ignore_index

    # 修改了这里：移除了 verbose 参数
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # 从指定epoch继续训练
    start_epoch = 0
    if continue_from_epoch is not None and continue_model_path is not None:
        try:
            print(f"从第 {continue_from_epoch} 个epoch继续训练，加载模型: {continue_model_path}")
            model.load_state_dict(torch.load(continue_model_path, map_location=device))
            start_epoch = continue_from_epoch
            print(f"成功加载模型，将从第 {start_epoch+1} 个epoch开始训练")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将从头开始训练")
            start_epoch = 0
    else:
        print("从头开始训练模型")

    total_steps_approx = (TOTAL_EPOCHS - start_epoch) * (len(poems_vector) // BATCH_SIZE)
    print(f"开始训练，总轮数: {TOTAL_EPOCHS}, 从第 {start_epoch+1} 个epoch开始, 大约总步数: {total_steps_approx}")

    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        np.random.shuffle(poems_vector) # Shuffle data each epoch
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_batches = len(batches_inputs)
        if n_batches == 0:
            print(f"Epoch {epoch+1}/{TOTAL_EPOCHS}: 没有足够的批次数据，跳过此轮。")
            continue
            
        epoch_total_loss = 0 # Renamed for clarity
        model.train()
        progress_bar = tqdm(range(n_batches), total=n_batches, desc=f'Epoch {epoch + 1}/{TOTAL_EPOCHS}')

        for batch_idx in progress_bar:
            batch_x = batches_inputs[batch_idx]
            batch_y = batches_outputs[batch_idx]

            optimizer.zero_grad()
            output_logits, _ = model(batch_x) # Hidden state initialized per batch by GRU

            # Align output for loss
            # output_logits: [batch_size, seq_len, vocab_size]
            # batch_y: [batch_size, seq_len_y]
            target_len = batch_y.size(1)
            output_for_loss = output_logits[:, :target_len, :].contiguous() # Ensure output matches target length

            loss = loss_function(output_for_loss.view(-1, len(word_to_int)), batch_y.view(-1))
            epoch_total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()

            avg_batch_loss = epoch_total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_epoch_loss': f'{avg_batch_loss:.4f}'})
        
        avg_epoch_loss = epoch_total_loss / n_batches
        scheduler.step(avg_epoch_loss)

        elapsed_time_epoch = time.time() - start_time # Total time since training started
        print(f"Epoch {epoch + 1} 完成. 平均损失: {avg_epoch_loss:.4f}. 用时: {str(datetime.timedelta(seconds=int(elapsed_time_epoch)))}. LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), './best_poem_generator_gru.pth') # Save best model
            print(f"最佳模型已保存于 Epoch {epoch + 1}，损失: {best_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'./poem_generator_gru_epoch_{epoch + 1}.pth')
            print(f"Epoch {epoch + 1} 模型已保存")

    total_training_time = time.time() - start_time
    print(f"训练完成！总训练时间: {str(datetime.timedelta(seconds=int(total_training_time)))}")


def temperature_sample(logits, temperature=1.0):
    if temperature < 1e-3: # Treat near-zero temperature as greedy
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    sample = torch.multinomial(probs, 1) # Output is [1]
    return sample.squeeze() # Return scalar tensor


def format_poem(poem_text):
    """
    智能格式化诗歌，识别五言绝句、七言绝句等常见格式。
    改进了默认情况下的标点符号处理。
    """
    poem_text = poem_text.replace(start_token, '').replace(end_token, '')
    poem_length = len(poem_text)
    if poem_length == 0:
        return ""

    has_punctuation = any(p in poem_text for p in ['，', '。'])

    if not has_punctuation:
        lines = []
        if poem_length == 20:  # 五言绝句
            return f"{poem_text[:5]}，{poem_text[5:10]}。\n{poem_text[10:15]}，{poem_text[15:20]}。"
        elif poem_length == 28:  # 七言绝句
            return f"{poem_text[:7]}，{poem_text[7:14]}。\n{poem_text[14:21]}，{poem_text[21:28]}。"
        elif poem_length == 40:  # 五言律诗
            return f"{poem_text[:5]}，{poem_text[5:10]}。\n{poem_text[10:15]}，{poem_text[15:20]}。\n{poem_text[20:25]}，{poem_text[25:30]}。\n{poem_text[30:35]}，{poem_text[35:40]}。"
        elif poem_length == 56:  # 七言律诗
            return f"{poem_text[:7]}，{poem_text[7:14]}。\n{poem_text[14:21]}，{poem_text[21:28]}。\n{poem_text[28:35]}，{poem_text[35:42]}。\n{poem_text[42:49]}，{poem_text[49:56]}。"
        else:
            # 默认标点逻辑改进
            # 尝试七言
            line_len_to_use = 7
            # If poem length is small and closer to 5, could use 5. Heuristic:
            if poem_length < 20 and poem_length % 5 == 0 and poem_length % 7 != 0 :
                 line_len_to_use = 5
            
            num_poem_lines = 0
            for i in range(0, poem_length, line_len_to_use):
                line_segment = poem_text[i:i + line_len_to_use]
                if not line_segment: continue

                is_final_segment_of_poem = (i + line_len_to_use >= poem_length)

                if is_final_segment_of_poem:
                    line_segment += "。"
                else:
                    if (num_poem_lines % 2) == 0: # First line of a pair (0-indexed)
                        line_segment += "，"
                    else: # Second line of a pair
                        line_segment += "。"
                lines.append(line_segment)
                num_poem_lines += 1
            return '\n'.join(lines)
    else:
        # 如果包含标点符号，根据标点符号分割 (user's original logic)
        lines = []
        current_line = ""
        for char_idx, char in enumerate(poem_text):
            current_line += char
            if char in ['，', '。']:
                lines.append(current_line)
                current_line = ""
            # Ensure last part is added if no trailing punctuation
            elif char_idx == poem_length - 1 and current_line:
                lines.append(current_line + "。") # Add final period if missing
                current_line = "" # Clear it
        if current_line: # If loop finished and current_line has content
            lines.append(current_line + "。") # Add final period if missing
        return '\n'.join(filter(None, lines)) # Filter out potential empty strings if multiple punctuations


def generate_poem(begin_word, model_path='./best_poem_generator_gru.pth', temperature=0.7, max_len=100):
    """
    生成诗歌 (假设 begin_word 是单个起始字符)
    """
    poems_vector_data, word_to_int, vocabularies = process_poems('./poems.txt') # vocabularies is list

    if not poems_vector_data: # Check if data loaded
        print("没有有效的诗歌数据用于加载词汇表！")
        return "错误: 词汇表加载失败。"
    
    # print(f"生成诗歌：成功加载 {len(poems_vector_data)} 首诗歌用于词汇表") # Verbose
    # print(f"生成诗歌：词汇表大小: {len(word_to_int)}")

    # Model Hyperparameters (must match the trained model's architecture)
    EMBEDDING_DIM = 256
    GRU_HIDDEN_DIM = 512
    NUM_LAYERS = 3
    
    word_embedding = WordEmbedding(vocab_length=len(word_to_int), embedding_dim=EMBEDDING_DIM).to(device)
    model = ImprovedRNNModel( # batch_sz removed
        vocab_len=len(word_to_int),
        word_embedding=word_embedding,
        embedding_dim=EMBEDDING_DIM,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.0 # No dropout during generation
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"加载模型失败 '{model_path}': {e}")
        return "错误: 模型加载失败。"

    poem_chars = [start_token] # Start with G token
    
    # Prime with begin_word (if it's a single char, or handle multi-char priming if desired)
    # Assuming begin_word is a single character as per user's examples
    if not isinstance(begin_word, str) or len(begin_word) == 0:
        print("错误: `begin_word` 必须是非空字符串。")
        return "错误: 起始词无效。"

    # Prime with start_token
    current_input_char_idx = word_to_int.get(start_token)
    if current_input_char_idx is None: return "错误: start_token不在词汇表中"
    hidden = None
    input_tensor = torch.tensor([[current_input_char_idx]], dtype=torch.long).to(device)
    output_logits, hidden = model(input_tensor, hidden)

    # Prime with actual begin_word character(s)
    # This loop handles multi-character begin_word by feeding them sequentially
    last_char_of_begin_word = ''
    for char_in_begin in list(begin_word):
        poem_chars.append(char_in_begin)
        current_input_char_idx = word_to_int.get(char_in_begin, word_to_int.get(' ', 0))
        input_tensor = torch.tensor([[current_input_char_idx]], dtype=torch.long).to(device)
        output_logits, hidden = model(input_tensor, hidden) # output_logits from last char of begin_word will be used
        last_char_of_begin_word = char_in_begin
    
    # The 'word' for generation loop should be the last character processed
    current_gen_input_char = last_char_of_begin_word 

    with torch.no_grad():
        # The loop generates (max_len - len(begin_word)) more characters
        for _ in range(max_len - len(begin_word)): 
            # Use the logits from the previous step (priming or generation)
            # output_logits is [1, 1, vocab_size] from the single char input
            next_idx = temperature_sample(output_logits.squeeze(0).squeeze(0).cpu(), temperature)
            next_char = vocabularies[next_idx.item()]

            if next_char == end_token:
                break
            
            poem_chars.append(next_char)
            current_gen_input_char = next_char # Update current char for next input

            # Prepare input for next prediction step
            current_input_char_idx = word_to_int.get(current_gen_input_char, word_to_int.get(' ',0))
            input_tensor = torch.tensor([[current_input_char_idx]], dtype=torch.long).to(device)
            output_logits, hidden = model(input_tensor, hidden) # Get new logits for next sampling

    # poem_chars now contains [G, b, e, g, i, n, ..., generated_chars]
    # format_poem will remove G and E
    return format_poem("".join(poem_chars))


def generate_acrostic_poem(start_phrase, model_path='./best_poem_generator_gru.pth', line_length=7, temperature=0.7):
    """
    生成藏头诗 (Acrostic Poem)
    start_phrase: 藏头的词语或句子，例如 "春夏秋冬"
    line_length: 每行诗的字数 (不含标点)
    """
    poems_vector_data, word_to_int, vocabularies = process_poems('./poems.txt')
    if not poems_vector_data:
        print("没有有效的诗歌数据用于加载词汇表！(藏头诗)")
        return "错误: 词汇表加载失败。(藏头诗)"

    EMBEDDING_DIM = 256
    GRU_HIDDEN_DIM = 512
    NUM_LAYERS = 3
    
    word_embedding = WordEmbedding(vocab_length=len(word_to_int), embedding_dim=EMBEDDING_DIM).to(device)
    model = ImprovedRNNModel(
        vocab_len=len(word_to_int),
        word_embedding=word_embedding,
        embedding_dim=EMBEDDING_DIM,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.0
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"加载模型失败 (藏头诗) '{model_path}': {e}")
        return "错误: 模型加载失败。(藏头诗)"

    acrostic_lines = []
    unknown_char_idx = word_to_int.get(' ', 0)

    with torch.no_grad():
        for line_idx, head_char in enumerate(start_phrase):
            current_line_chars = [head_char]
            hidden_state_for_line = None # Reset hidden state for each line for more diverse lines

            # Prime with start_token (G)
            # current_input_for_model_idx = word_to_int.get(start_token, unknown_char_idx)
            # input_tensor = torch.tensor([[current_input_for_model_idx]], dtype=torch.long).to(device)
            # line_logits, hidden_state_for_line = model(input_tensor, hidden_state_for_line)
            
            # Prime with the head_char
            current_input_for_model_idx = word_to_int.get(head_char, unknown_char_idx)
            input_tensor = torch.tensor([[current_input_for_model_idx]], dtype=torch.long).to(device)
            # The output of this step (line_logits) will be used to predict the *next* char in the line
            line_logits, hidden_state_for_line = model(input_tensor, hidden_state_for_line)

            # Generate the rest of the line
            for _ in range(line_length - 1): # -1 because head_char is already one character
                # Sample from the logits produced by the previous character
                next_char_idx = temperature_sample(line_logits.squeeze(0).squeeze(0).cpu(), temperature)
                generated_char = vocabularies[next_char_idx.item()]

                if generated_char == end_token:
                    break 
                current_line_chars.append(generated_char)
                
                # Prepare for next iteration: input is the char just generated
                current_input_for_model_idx = next_char_idx.item() # Use the index directly
                input_tensor = torch.tensor([[current_input_for_model_idx]], dtype=torch.long).to(device)
                line_logits, hidden_state_for_line = model(input_tensor, hidden_state_for_line)
            
            # Add punctuation
            line_text = "".join(current_line_chars)
            if line_idx < len(start_phrase) - 1: # Not the last line of the acrostic
                if line_idx % 2 == 0: # First line of a couplet
                    line_text += "，"
                else: # Second line of a couplet
                    line_text += "。"
            else: # Last line of the acrostic
                line_text += "。"
            acrostic_lines.append(line_text)
            
    return "\n".join(acrostic_lines)


if __name__ == "__main__":
    # 步骤 1: 从第70个epoch继续训练模型
    #train_model(continue_from_epoch=70, continue_model_path='./poem_generator_gru_epoch_70.pth')

    # 步骤 2: 使用训练好的模型生成诗歌
    MODEL_FILENAME = './best_poem_generator_gru.pth' # 或者您特定的epoch模型，例如 './poem_generator_gru_epoch_100.pth'
    # 创建一个虚拟的 poems.txt 用于测试，如果真实文件不存在
    try:
        with open('./poems.txt', 'r', encoding='utf-8') as f:
            pass 
    except FileNotFoundError:
        print("警告: 'poems.txt' 未找到。正在创建一个虚拟文件用于测试。")
        with open('./poems.txt', 'w', encoding='utf-8') as f:
            f.write("李白:床前明月光，疑是地上霜。举头望明月，低头思故乡。\n")
            f.write("杜甫:国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。\n")
            f.write("王之涣:白日依山尽，黄河入海流。欲穷千里目，更上一层楼。\n")
            f.write("孟浩然:春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。\n")

    print(f"\n--- 开始生成常规诗歌 (模型: {MODEL_FILENAME}) ---")
    for start_word_char in ["日", "红", "山", "夜", "湖", "君"]:
        print(f"\n以 '{start_word_char}' 开头的诗歌 (温度=0.7):")
        poem = generate_poem(start_word_char, model_path=MODEL_FILENAME, temperature=0.7, max_len=56) # max_len for a 7-char律诗
        if poem: print(poem)

        print(f"\n以 '{start_word_char}' 开头的诗歌 (温度=0.5):")
        poem = generate_poem(start_word_char, model_path=MODEL_FILENAME, temperature=0.5, max_len=56)
        if poem: print(poem)

    print(f"\n--- 开始生成藏头诗 (模型: {MODEL_FILENAME}) ---")
    acrostic_phrases = ["春夏秋冬", "山水清音", "明月清风"]
    for phrase in acrostic_phrases:
        print(f"\n以 '{phrase}' 为开头的藏头诗 (七言, 温度=0.7):")
        acrostic = generate_acrostic_poem(phrase, model_path=MODEL_FILENAME, line_length=7, temperature=0.7)
        if acrostic: print(acrostic)

        print(f"\n以 '{phrase}' 为开头的藏头诗 (五言, 温度=0.7):")
        acrostic = generate_acrostic_poem(phrase, model_path=MODEL_FILENAME, line_length=5, temperature=0.7)
        if acrostic: print(acrostic)
import re
import random
from typing import List, Tuple, Set

def process_line_to_ground_truth(line: str) -> Tuple[str, str]:
    """
    处理单行文本，生成带有方言词标记的 Ground Truth (GT) 文本。

    该函数会解析一个由制表符分隔的行，格式为：
    "音频文件名\t标注文本\t方言词列表"

    它会忽略方言词列表中的【】{}等符号，然后在标注文本中找到
    所有这些方言词的出现位置，并用 <> 将它们括起来。

    Args:
        line (str): 从文件中读取的单行字符串。

    Returns:
        Tuple[str, str]: 一个元组，包含 (音频文件名, 带标记的GT文本)。
                         如果行格式不正确，则返回 (None, None)。
    """
    parts = line.strip().split('\t')
    if len(parts) != 3:
        # 如果行不包含三个部分，则跳过
        print(f"警告: 跳过格式不正确的行: {line.strip()}")
        return None, None

    filename, transcription, dialect_words_raw = parts
    
    # 1. 清理和提取方言词
    # 去除 【】{} 等符号
    dialect_words_cleaned = re.sub(r'[【】{}]', '', dialect_words_raw)
    # 按逗号分割，并去除空字符串
    dialect_words = [word for word in dialect_words_cleaned.split(',') if word]

    # 2. 在标注文本中找到所有方言词的区间
    intervals_to_mark = []
    for word in set(dialect_words): # 使用 set 去重，避免重复查找
        # 使用 re.finditer 找到所有不重叠的匹配项
        for match in re.finditer(re.escape(word), transcription):
            intervals_to_mark.append((match.start(), match.end()))
            
    # 如果没有找到任何区间，直接返回原始文本
    if not intervals_to_mark:
        return filename, transcription

    # 3. 根据区间构建带 <> 标记的GT文本
    # 按起始位置对区间进行排序，以便正确插入标记
    intervals_to_mark.sort()
    
    new_text_parts = []
    last_pos = 0
    for start, end in intervals_to_mark:
        # 添加上一个标记到当前标记之间的文本
        new_text_parts.append(transcription[last_pos:start])
        # 添加带标记的方言词
        new_text_parts.append(f"<{transcription[start:end]}>")
        last_pos = end
    
    # 添加最后一个标记之后剩余的文本
    new_text_parts.append(transcription[last_pos:])
    
    gt_text = "".join(new_text_parts)
    
    return filename, gt_text


def _extract_char_indices(text_with_markup: str) -> Set[int]:
    """
    [辅助函数] 从带 <> 标记的文本中提取所有被标记字符的索引。
    
    例如，对于 "你好<世界>和平"，它会识别出 "世界" 被标记。
    在无标记文本 "你好世界和平" 中，"世" 的索引是2，"界"是3。
    函数将返回 {2, 3}。
    
    Args:
        text_with_markup (str): 包含 <> 标记的字符串。

    Returns:
        Set[int]: 一个包含所有被标记字符在“纯文本”中索引的集合。
    """
    indices = set()
    plain_text_pos = 0
    in_markup = False
    
    for char in text_with_markup:
        if char == '<':
            in_markup = True
        elif char == '>':
            in_markup = False
        else:
            if in_markup:
                indices.add(plain_text_pos)
            plain_text_pos += 1
            
    return indices

def calculate_text_iou(gt_text: str, hyp_text: str) -> float:
    """
    计算 Ground Truth (GT) 和 Hypothesis (Hyp) 文本之间标记区间的交并比 (IoU)。

    该函数将两个带有 <> 标记的字符串作为输入，计算它们标记的字符集合的
    交集与并集，并返回其比率。

    Args:
        gt_text (str): Ground Truth 文本，带有 <> 标记。
        hyp_text (str): Hypothesis 文本，带有 <> 标记。

    Returns:
        float: 返回 [0.0, 1.0] 之间的 IoU 值。
               - 如果 GT 和 Hyp 的标记都为空，IoU 为 1.0。
               - 如果其中一个为空而另一个不为空，IoU 为 0.0。
    """
    # 1. 提取 GT 和 Hyp 中被标记的字符索引集合
    gt_indices = _extract_char_indices(gt_text)
    hyp_indices = _extract_char_indices(hyp_text)
    
    # 2. 计算交集和并集的大小
    intersection_size = len(gt_indices.intersection(hyp_indices))
    union_size = len(gt_indices.union(hyp_indices))
    
    # 3. 计算 IoU
    if union_size == 0:
        # 如果并集为空，意味着 GT 和 Hyp 都没有标记，可以认为是完全匹配
        return 1.0
    
    iou = intersection_size / union_size
    return iou


def dummy_multimodal_model(audio_filename: str, gt_text: str) -> str:
    """
    一个模拟的多模态语音大模型函数。

    它接收音频文件名和GT文本，并返回一个模拟的、带有错误的预测文本 (Hyp)。
    这用于演示和测试 IoU 函数。

    Args:
        audio_filename (str): 音频文件名 (当前未使用，为保持接口一致性)。
        gt_text (str): 带标记的 Ground Truth 文本。

    Returns:
        str: 一个模拟的、带标记的 Hypothesis 文本。
    """
    # 移除所有标记以获取纯文本
    plain_text = re.sub(r'[<>]', '', gt_text)
    
    # 随机选择一种错误类型来模拟
    choice = random.random()

    if choice < 0.3: # 30% 概率：完美匹配
        return gt_text
    
    elif choice < 0.5: # 20% 概率：漏掉一个标记 (False Negative)
        match = re.search(r'<(.*?)>', gt_text)
        if match:
            # 只移除第一个找到的标记
            return gt_text.replace(match.group(0), match.group(1), 1)
        return gt_text # 如果没有标记，则返回原样

    elif choice < 0.7: # 20% 概率：边界错误
        match = re.search(r'<(.*?)>', gt_text)
        if match and len(match.group(1)) > 1:
            # 将标记缩短一个字符
            content = match.group(1)
            new_content = f"<{content[:-1]}>"
            return gt_text.replace(match.group(0), new_content, 1)
        return gt_text

    elif choice < 0.9: # 20% 概率：增加一个错误标记 (False Positive)
        words = re.split(r'(\s+)', plain_text) # 按空格分割以模拟词
        words = [w for w in words if w.strip()]
        if words:
            random_word_idx = random.randint(0, len(words) - 1)
            word_to_mark = words[random_word_idx]
            # 确保只标记一次，避免无限替换
            return plain_text.replace(word_to_mark, f"<{word_to_mark}>", 1)
        return plain_text

    else: # 10% 概率：完全不匹配
        return plain_text


def main(file_path: str):
    """
    主函数，驱动整个处理流程。

    Args:
        file_path (str): `fangyan_text.txt` 文件的路径。
    """
    print("-" * 80)
    print(f"正在处理文件: {file_path}")
    print("-" * 80)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return
        
    total_iou = 0
    processed_lines = 0

    for i, line in enumerate(lines):
        if not line.strip(): # 跳过空行
            continue
            
        filename, gt_text = process_line_to_ground_truth(line)
        
        if filename is None or gt_text is None:
            continue
            
        # 调用模拟的大模型函数获取预测结果
        hyp_text = dummy_multimodal_model(filename, gt_text)
        
        # 计算 IoU
        iou = calculate_text_iou(gt_text, hyp_text)
        
        total_iou += iou
        processed_lines += 1
        
        # 打印单行结果
        print(f"文件: {filename}")
        print(f"  GT  : {gt_text}")
        print(f"  HYP : {hyp_text}")
        print(f"  IoU : {iou:.4f}\n")

    if processed_lines > 0:
        average_iou = total_iou / processed_lines
        print("-" * 80)
        print(f"处理完成！")
        print(f"总计处理行数: {processed_lines}")
        print(f"平均 IoU: {average_iou:.4f}")
        print("-" * 80)
    else:
        print("没有可处理的有效数据行。")


if __name__ == '__main__':
    # 请将 'fangyan_text.txt' 替换为您的文件的实际路径
    # 例如: '/mnt/sda/ASR/DataSets/LabelData/20250606fangyanciku_tiqu/filtered_audio_macheng/macheng/fangyan_text.txt'
    # 为了方便运行，这里假设文件在脚本同目录下
    input_file = 'fangyan_text.txt'
    
    # 为了演示，我们先创建一个示例的 fangyan_text.txt 文件
    sample_data = """95503243_input.wav_4.wav	哦哦你把电费望哈啄	【啄】
95510499_input.wav_3.wav	叫么斯名字	么【斯】
95514113_input.wav_2.wav	喂电费交哈子	【哈】子
94953590_input.wav_2.wav	哦黄泡亮了那应该是欠费了䅰自己查哈子再用微信交哈子啊要交不到再	黄【泡】,【哈】子
95079976_input.wav_1.wav	䅰踔厉查哈查哈我的上个月的电费电量啄我的那个手机那	【啄】,【踔厉】
"""
    try:
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(sample_data)
        print(f"已创建示例文件 '{input_file}' 用于演示。")
        
        # 运行主程序
        main(input_file)
        
    except Exception as e:
        print(f"运行脚本时发生错误: {e}")
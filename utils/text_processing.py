import re
from typing import Tuple, Set, List

def mark_words_in_text(transcription: str, dialect_words: List[str], left_bracket: str = "【", right_bracket: str = "】") -> str:
    """
    在原文本中用指定的括号标记方言词汇
    
    Args:
        transcription: 原文本
        dialect_words: 方言词汇列表
        left_bracket: 左括号符号，默认为 "【"
        right_bracket: 右括号符号，默认为 "】"
    
    Returns:
        str: 标记后的文本
    """
    if not dialect_words:
        return transcription
    
    intervals_to_mark = []
    for word in set(dialect_words):
        for match in re.finditer(re.escape(word), transcription):
            intervals_to_mark.append((match.start(), match.end()))
    
    if not intervals_to_mark:
        return transcription
    
    intervals_to_mark.sort()
    new_text_parts = []
    last_pos = 0
    
    for start, end in intervals_to_mark:
        new_text_parts.append(transcription[last_pos:start])
        new_text_parts.append(f"{left_bracket}{transcription[start:end]}{right_bracket}")
        last_pos = end
    
    new_text_parts.append(transcription[last_pos:])
    return "".join(new_text_parts)

def process_line_to_ground_truth(line: str, use_word_comparison: bool = False) -> Tuple[str, str, str]:
    parts = line.strip().split('\t')
    if len(parts) != 3:
        print(f"警告: 跳过格式不正确的行: {line.strip()}")
        return "", "", ""
    filename, transcription, dialect_words_raw = parts
    
    if use_word_comparison:
        # 新的比对方法：直接返回词汇列表
        dialect_words_cleaned = re.sub(r'[【】{}]', '', dialect_words_raw)
        # dialect_words = [word.strip() for word in dialect_words_cleaned.split(',') if word.strip()]
        # gt_words = ','.join(dialect_words)
        return filename, transcription, dialect_words_cleaned
    else:
        # 原来的比对方法：返回带标记的文本
        dialect_words_cleaned = re.sub(r'[【】{}]', '', dialect_words_raw)
        dialect_words = [word for word in dialect_words_cleaned.split(',') if word]
        gt_text = mark_words_in_text(transcription, dialect_words)
        return filename, gt_text, ""

def _extract_char_indices(text_with_markup: str) -> Set[int]:
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
    gt_indices = _extract_char_indices(gt_text)
    hyp_indices = _extract_char_indices(hyp_text)
    intersection_size = len(gt_indices.intersection(hyp_indices))
    union_size = len(gt_indices.union(hyp_indices))
    if union_size == 0:
        return 1.0
    return intersection_size / union_size

def calculate_word_metrics(gt_words: str, hyp_words: str) -> Tuple[float, float, float]:
    """
    计算词汇级别的召回率、准确率和F1分数
    
    Args:
        gt_words: 真实词汇，逗号分隔的字符串
        hyp_words: 预测词汇，逗号分隔的字符串
    
    Returns:
        Tuple[float, float, float]: (召回率, 准确率, F1分数)
    """
    # 将逗号分隔的字符串转换为词汇集合
    gt_word_set = set(word.strip() for word in gt_words.replace(',','，').split('，') if word.strip())
    hyp_word_set = set(word.strip() for word in hyp_words.split('，') if word.strip())
    
    # 计算交集
    intersection = gt_word_set.intersection(hyp_word_set)
    
    # 计算召回率 (Recall = TP / (TP + FN))
    if len(gt_word_set) == 0:
        recall = 1.0 if len(hyp_word_set) == 0 else 0.0
    else:
        recall = len(intersection) / len(gt_word_set)
    
    # 计算准确率 (Precision = TP / (TP + FP))
    if len(hyp_word_set) == 0:
        precision = 1.0 if len(gt_word_set) == 0 else 0.0
    else:
        precision = len(intersection) / len(hyp_word_set)
    
    # 计算F1分数
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return recall, precision, f1 
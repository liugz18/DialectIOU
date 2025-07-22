import re
from typing import Tuple, Set

def process_line_to_ground_truth(line: str) -> Tuple[str, str]:
    parts = line.strip().split('\t')
    if len(parts) != 3:
        print(f"警告: 跳过格式不正确的行: {line.strip()}")
        return "", ""
    filename, transcription, dialect_words_raw = parts
    dialect_words_cleaned = re.sub(r'[【】{}]', '', dialect_words_raw)
    dialect_words = [word for word in dialect_words_cleaned.split(',') if word]
    intervals_to_mark = []
    for word in set(dialect_words):
        for match in re.finditer(re.escape(word), transcription):
            intervals_to_mark.append((match.start(), match.end()))
    if not intervals_to_mark:
        return filename, transcription
    intervals_to_mark.sort()
    new_text_parts = []
    last_pos = 0
    for start, end in intervals_to_mark:
        new_text_parts.append(transcription[last_pos:start])
        new_text_parts.append(f"<{transcription[start:end]}>")
        last_pos = end
    new_text_parts.append(transcription[last_pos:])
    gt_text = "".join(new_text_parts)
    return filename, gt_text

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
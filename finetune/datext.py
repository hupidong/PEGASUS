import re
import json

whitespace = '\t\n\r\v\f'
punctuation = """!！。?？"""
FIRST_SEP = punctuation + whitespace
SECOND_SEP = ' ,，；;'


def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely, reference to
    "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r'([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def get_human_and_activity_context(content, human_name, activity_span, offset=20):
    human_spans = re.finditer(human_name, content)
    activity_spans = re.finditer(re.escape(activity_span), content)

    sentence_splits = cut_chinese_sent(content)
    sentence_split_spans = []
    index_end = 0
    for sentence in sentence_splits:
        try:
            index_start = content.index(sentence, index_end)
            index_end = index_start + len(sentence)
            span = (index_start, index_end)
            sentence_split_spans.append(span)
        except:
            print(json.dumps({"content": content, "sentence": sentence, "sentence_split": sentence_splits},
                             ensure_ascii=False, indent=4))
            raise RuntimeError
    assert len(sentence_split_spans) == len(sentence_splits)

    distance_min = 10000000
    try:
        human_target = list(human_spans)[0]
        activity_target = list(activity_spans)[0]
    except:
        print(json.dumps({"content": content, "activity": activity_span, "human": human_name}, ensure_ascii=False,
                         indent=4))
        return content

    for activity_span in activity_spans:
        activity_start = activity_span.span()[0]
        activity_end = activity_span.span()[1]
        for human_span in human_spans:
            human_start = human_span.span()[0]
            human_end = human_span.span()[1]
            if activity_start <= human_start <= activity_end and activity_start <= human_end <= activity_end:
                distance = 0
            else:
                distance_left = min(abs(human_start - activity_start), abs(human_start - activity_end))
                distance_right = min(abs(human_end - activity_start), abs(human_end - activity_end))
                distance = min(distance_left, distance_right)

            if distance < distance_min:
                distance_min = distance
                human_target = human_span
                activity_target = activity_span

    """
    """
    human_sentence_idx = None
    activity_sentence_idx = None

    for i, sentence_span in enumerate(sentence_split_spans):
        human_start = human_target.span()[0]
        human_end = human_target.span()[1]
        if sentence_span[0] <= human_start <= sentence_span[1] and sentence_span[0] <= human_end <= sentence_span[1]:
            human_sentence_idx = i
            break

    for i, sentence_span in enumerate(sentence_split_spans):
        activity_start = activity_target.span()[0]
        activity_end = activity_target.span()[1]
        if sentence_span[0] <= activity_start <= sentence_span[1] \
                and sentence_span[0] <= activity_end <= sentence_span[1]:
            activity_sentence_idx = i
            break

    if human_sentence_idx is None or activity_sentence_idx is None:
        start = min(human_target.span()[0], activity_target.span()[0])
        end = max(human_target.span()[1], activity_target.span()[1])
        start = start - offset if start - offset >= 0 else 0
        end = end + offset
        text = content[start:end]
        # cut
        text_cut, index = cut_text_forward(text)
        if index > offset:
            text_cut = text
        text = text_cut
        text_cut, index = cut_text_backward(text)
        if index < -offset:
            text_cut = text
        return text_cut

    if human_sentence_idx == activity_sentence_idx:
        return sentence_splits[human_sentence_idx]
    elif human_sentence_idx > activity_sentence_idx:
        return sentence_splits[activity_sentence_idx] + sentence_splits[human_sentence_idx]
    else:
        return sentence_splits[human_sentence_idx] + sentence_splits[activity_sentence_idx]


def cut_text_backward(text, first_sep=FIRST_SEP, second_sep=SECOND_SEP):
    """
    将一段文本从后往前根据标点符号进行截断
    :param text:
    :param first_sep:
    :param second_sep:
    :return:
    """
    # 从末尾往前遍历直至遇到第一个非word字符
    second_sep_index = None
    k = -1
    index = -1
    for k in range(-1, -len(text) - 1, -1):
        if not second_sep_index and text[k] in second_sep:
            second_sep_index = k
        if text[k] in first_sep:
            text = text[0: len(text) + k + 1]
            index = k - 1
            break
    if k == -len(text) and second_sep_index:
        text = text[0: len(text) + second_sep_index + 1]
        index = second_sep_index - 1
    return text, index


def cut_text_forward(text, first_sep=FIRST_SEP, second_sep=SECOND_SEP):
    """
    将一段文本从前往后根据标点符号进行截断
    :param text:
    :param first_sep:
    :param second_sep:
    :return:
    """
    second_sep_index = None
    k = 0
    index = 0
    for k in range(0, len(text)):
        if not second_sep_index and text[k] in second_sep:
            second_sep_index = k
        if text[k] in first_sep:
            text_new = text[k + 1:]
            index = k + 1
            break
    if index == len(text):
        text_new = text
        index = 0
    elif second_sep_index:
        text_new = text[second_sep_index + 1:]
        index = second_sep_index + 1

    return text_new, index

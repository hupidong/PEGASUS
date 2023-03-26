import re
import json

whitespace = '\t\n\r\v\f'
sent_end_punctuation = """!！。?？"""
FIRST_SEP = sent_end_punctuation + whitespace
SECOND_SEP = """ ,，；;"""

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


def get_human_and_activity_context(content, human_name, activity_span, offset=20, **kwargs):
    max_seq_len_char = kwargs.get("max_seq_len_char", 512)
    max_margin_of_activity_and_human = kwargs.get("max_margin_of_activity_and_human", 400)

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
        human_spans = list(human_spans)
        activity_spans = list(activity_spans)
        human_target = human_spans[0]
        activity_target = activity_spans[0]
    except:
        print(json.dumps({"content": content, "activity": activity_span, "human": human_name}, ensure_ascii=False,
                         indent=4))
        return content

    """
    search sentence index of target activity and human
    """
    activity_sentence_idx = None
    human_sentence_idx = None
    activity_start = activity_target.span()[0]
    activity_end = activity_target.span()[1]

    # search index of target activity
    for i, sentence_span in enumerate(sentence_split_spans):
        if sentence_span[0] <= activity_start <= sentence_span[1] \
                and sentence_span[0] <= activity_end <= sentence_span[1]:
            activity_sentence_idx = i
            break

    # search index of target human
    FLAG = False
    for i, sentence_span in enumerate(sentence_split_spans):
        for span in human_spans:
            human_start = span.span()[0]
            human_end = span.span()[1]
            if not (sentence_span[0] <= human_start <= sentence_span[1] and sentence_span[0] <= human_end <=
                    sentence_span[1]):
                continue

            if activity_start <= human_start <= activity_end and activity_start <= human_end <= activity_end:
                if activity_sentence_idx:
                    human_sentence_idx = activity_sentence_idx
                    human_target = span
                    FLAG=True
                    break
            elif human_start >= activity_end:
                human_sentence_idx = i
                human_target = span
                FLAG = True
                break
        if FLAG:
            break


    if human_sentence_idx and activity_sentence_idx is None:
        span_activity = activity_target.span()
        span_human_sentence = re.finditer(re.escape(sentence_splits[human_sentence_idx]), content)
        span_human = list(span_human_sentence)[0].span()
        margin = 0
        if int(sum(human_target.span()) / 2) - span_activity[1] > max_margin_of_activity_and_human:
            margin = int(sum(human_target.span()) / 2) - span_activity[1]
        if int(sum(human_target.span()) / 2) - span_activity[0] < -max_margin_of_activity_and_human:
            margin = int(sum(human_target.span()) / 2) - span_activity[0]
        if margin > max_margin_of_activity_and_human or margin < -max_margin_of_activity_and_human:
            start = span_activity[0]
            end = span_activity[1]
            print(f"margin of human and activity is too big {margin}, more than {max_margin_of_activity_and_human}.")
            print(json.dumps({"content": content, "activity": activity_span, "human": human_name}, ensure_ascii=False,
                             indent=4))
        else:
            start = min(span_human[0], span_activity[0])
            end = max(span_human[1], span_human[1])
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

    if human_sentence_idx is None or activity_sentence_idx is None:
        margin = 0
        if int(sum(human_target.span()) / 2) - activity_target.span()[1] > max_margin_of_activity_and_human:
            margin = int(sum(human_target.span()) / 2) - activity_target.span()[1]
        if int(sum(human_target.span()) / 2) - activity_target.span()[0] < -max_margin_of_activity_and_human:
            margin = int(sum(human_target.span()) / 2) - activity_target.span()[0]
        if margin > max_margin_of_activity_and_human or margin < -max_margin_of_activity_and_human:
            start = activity_target.span()[0]
            end = activity_target.span()[1]
            print(f"margin of human and activity is too big {margin}, more than {max_margin_of_activity_and_human}.")
            print(json.dumps({"content": content, "activity": activity_span, "human": human_name}, ensure_ascii=False,
                             indent=4))
        else:
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
    if human_sentence_idx > activity_sentence_idx:
        margin = 0
        if int(sum(human_target.span()) / 2) - activity_target.span()[1] > max_margin_of_activity_and_human:
            margin = int(sum(human_target.span()) / 2) - activity_target.span()[1]
        if int(sum(human_target.span()) / 2) - activity_target.span()[0] < -max_margin_of_activity_and_human:
            margin = int(sum(human_target.span()) / 2) - activity_target.span()[0]
        if margin > max_margin_of_activity_and_human or margin < -max_margin_of_activity_and_human:
            print(f"margin of human and activity is too big {margin}, more than {max_margin_of_activity_and_human}.")
            print(json.dumps({"content": content, "activity": activity_span, "human": human_name}, ensure_ascii=False,
                             indent=4))
            return sentence_splits[activity_sentence_idx]
        else:
            max_seq_len = max_seq_len_char - (activity_target.span()[1] - activity_target.span()[0]) - \
                          (human_target.span()[1] - human_target.span()[0])
            add = ""
            range_list = list(range(activity_sentence_idx + 1, human_sentence_idx))
            range_list.reverse()
            for idx in range_list:
                if len(add + sentence_splits[idx]) < max_seq_len:
                    add = sentence_splits[idx] + add
            return sentence_splits[activity_sentence_idx] + add + sentence_splits[human_sentence_idx]
    if human_sentence_idx < activity_sentence_idx:
        margin = 0
        if int(sum(human_target.span()) / 2) - activity_target.span()[1] > max_margin_of_activity_and_human:
            margin = int(sum(human_target.span()) / 2) - activity_target.span()[1]
        if int(sum(human_target.span()) / 2) - activity_target.span()[0] < -max_margin_of_activity_and_human:
            margin = int(sum(human_target.span()) / 2) - activity_target.span()[0]
        if margin > max_margin_of_activity_and_human or margin < -max_margin_of_activity_and_human:
            print(f"margin of human and activity is too big, more than {max_margin_of_activity_and_human}.")
            print(json.dumps({"content": content, "activity": activity_span, "human": human_name}, ensure_ascii=False,
                             indent=4))
            return sentence_splits[activity_sentence_idx]
        else:
            max_seq_len = max_seq_len_char - (activity_target.span()[1] - activity_target.span()[0]) - \
                          (human_target.span()[1] - human_target.span()[0])
            add = ""
            for idx in range(human_sentence_idx + 1, activity_sentence_idx):
                if len(add + sentence_splits[idx]) < max_seq_len:
                    add += sentence_splits[idx]
            return sentence_splits[human_sentence_idx] + add + sentence_splits[activity_sentence_idx]


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

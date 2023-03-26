import json
import re
import unicodedata

import jieba
from paddlenlp.transformers import AutoTokenizer

from datext import get_human_and_activity_context, cut_chinese_sent


def convert_example_human_activity(example, text_column, summary_column, tokenizer, max_source_length,
                                   max_target_length, expansion_coef=1.4, **kwargs):
    """
    Convert a example into necessary features.
    """
    max_margin_of_activity_and_human = kwargs.get("max_margin_of_activity_and_human", 300)
    example[text_column] = create_source_human_activity(example,
                                                        max_seq_len_char=int(max_source_length * expansion_coef),
                                                        title_len=0,
                                                        sep_token=tokenizer.sep_token,
                                                        use_activity_name=kwargs.get("use_activity_name", True),
                                                        activity_column=text_column,
                                                        max_margin_of_activity_and_human=max_margin_of_activity_and_human)
    inputs = example[text_column]
    targets = example[summary_column]
    inputs = unicodedata.normalize("NFKC", inputs)
    targets = unicodedata.normalize("NFKC", targets)

    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=False, truncation=True, return_attention_mask=True
    )
    labels = tokenizer(targets, max_length=max_target_length, padding=False, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def convert_example_news(example, summary_column, tokenizer, max_source_length,
                         max_target_length, truncate_func=None, expansion_coef=1.5, **kwargs):
    """
    Convert a example into necessary features.
    """
    ratio_head2tail = kwargs.get("ratio_head2tail", [2, 1])
    inputs = create_source_news(example,
                                max_char_len=int(max_source_length * expansion_coef),
                                sep_token=tokenizer.sep_token,
                                truncate_func=truncate_func,
                                ratio_head2tail=ratio_head2tail)
    targets = example[summary_column]
    inputs = unicodedata.normalize("NFKC", inputs)
    targets = unicodedata.normalize("NFKC", targets)

    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=False, truncation=True, return_attention_mask=True
    )
    labels = tokenizer(targets, max_length=max_target_length, padding=False, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def create_source_human_activity(example, max_seq_len_char=512, title_len=128, sep_token="[SEP]", **kwargs):
    use_activity_name = kwargs.get("use_activity_name", True)
    activity_column = kwargs.get("activity_column", "content")
    max_margin_of_activity_and_human = kwargs.get("max_margin_of_activity_and_human", 400)
    empty = "null"
    source_human = example.get("human_name", "")
    assert source_human, "目标人物不能为空"
    prompt = "生成" + source_human + "的活动摘要："

    source_activity = example[activity_column]

    if use_activity_name:
        max_seq_len_char_context = max_seq_len_char - len(prompt) - len(source_activity) - title_len
    else:
        max_seq_len_char_context = max_seq_len_char - len(prompt) - title_len

    if "context" not in example:
        if "text" not in example:
            source_context = ""
        else:
            text = example["text"].split("###")[-1]
            if len(unicodedata.normalize("NFKC", text)) <= max_seq_len_char_context:
                source_context = text
            else:
                source_context = get_human_and_activity_context(content=text,
                                                                human_name=source_human,
                                                                activity_span=example[activity_column],
                                                                offset=40,
                                                                max_seq_len_char=max_seq_len_char_context,
                                                                max_margin_of_activity_and_human=max_margin_of_activity_and_human)
            example["context"] = source_context
    else:
        source_context = example.get("context", "")
    source_context = source_context[0: max_seq_len_char_context]
    if not source_context:
        source_context = empty

    if use_activity_name:
        source = prompt + sep_token + source_activity + sep_token + source_context
    else:
        source = prompt + sep_token + source_context
    return source


def create_source_news(example, max_char_len=1024, sep_token="[SEP]", truncate_func=None, **kwargs):
    empty = "null"
    source_title = example.get("title", "")
    if not source_title:
        source_title = empty

    source_body = example.get("body", "")
    if not source_body:
        source_body = empty
    if truncate_func:
        ratio_head2tail = kwargs.get("ratio_head2tail", [2, 1])
        source_body = truncate_func(source_body, max_seq_len=max_char_len - len(source_title) - 15,
                                    ratio_head2tail=ratio_head2tail)
    source = "完成摘要:" + sep_token + "标题是:" + source_title + sep_token + "正文是:" + source_body
    return source


def read_file(file):
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
            yield line


def truncate_news(body, max_seq_len=1024, ratio_head2tail=[2, 1], **kwargs):
    """
    针对正文的截断：首尾按照长度比例截取
    """
    assert len(ratio_head2tail) == 2
    max_head_len = int(max_seq_len * ratio_head2tail[0] / sum(ratio_head2tail))
    max_tail_len = max_seq_len - max_head_len
    body = unicodedata.normalize("NFKC", body)
    sents = cut_chinese_sent(body)

    head = ""
    token_count = 0
    head_idx = 0
    for idx, sent in enumerate(sents):
        if token_count + len(sent) > max_head_len:
            break
        head += sent
        head_idx += 1
        token_count += len(sent)
    head = head.rstrip() + "\n"

    tail = ""
    token_count = 0
    for i in range(-1, -len(sents) + head_idx, - 1):
        if token_count + len(sents[i]) > max_tail_len:
            break
        tail = sents[i] + tail
        token_count += len(sents[i])
    return head + tail


def pre_tokenize_uppercase(text):
    """
    针对有大写字母情形时的预切分，针对do_lower_case==False的情形
    for example:
    小鹏G3i和小鹏PX5 > 小鹏G##3##i和小鹏P##X##5
    """
    text = re.sub('((?<=[a-zA-Z])[a-zA-Z0-9])', r"##\1", text)
    text = re.sub('((?<=[0-9])[a-zA-Z])', r"##\1", text)
    return text


def replace_letter_with_hash(letter, text):
    assert 'a' <= letter <= 'z' or 'A' <= letter <= 'Z', "the target replace char should be english letter"
    text = re.sub(letter, "##" + letter, text)
    return text


if __name__ == "__main__":
    user_dict = r"./data/vocab/uppercase.txt"
    jieba.load_userdict(user_dict)

    text = "O##F##w##e##e##k"
    vocab_dir = "./checkpoints/news_summary/Randeng-Pegasus-238M-Summary-Chinese/model_custom_tokenizer"
    do_lower_case = False
    tokenizer = AutoTokenizer.from_pretrained(vocab_dir, do_lower_case=do_lower_case)
    tokenized = tokenizer(text, max_length=512, padding=False, truncation=True, return_attention_mask=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized.input_ids, skip_special_tokens=False)
    string = tokenizer.convert_tokens_to_string(tokens)
    pass

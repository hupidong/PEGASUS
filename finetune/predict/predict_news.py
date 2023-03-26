import math
import random
import unicodedata

from paddlenlp import Taskflow
import sys, os

from tqdm import tqdm

sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../"))
from data_loader import create_source_news, truncate_news

from utils import load_jsonl_data, save2jsonl
from datext import cut_text_backward, sent_end_punctuation, FIRST_SEP, SECOND_SEP, cut_chinese_sent


def summary_post_process(summary):
    punctuation = sent_end_punctuation + ".、"
    summary = summary.replace("()", "").replace("（）", "")
    summary, _ = cut_text_backward(summary, first_sep=sent_end_punctuation, second_sep=FIRST_SEP)
    if summary[-1] not in sent_end_punctuation:
        if unicodedata.normalize("NFKC", summary)[-1] not in (SECOND_SEP + ".、"):
            summary += "。"
        else:
            summary = summary[0:-1] + "。"
    sents = cut_chinese_sent(summary)
    if len(sents)>=2 and "扫描二维码" in sents[-1]:
        summary = "".join(sents[0:-1])
    return summary


if __name__ == "__main__":
    do_unimo = False
    do_pegasus = True

    data_path = r"F:\\Work\kg\gen\news_summary\train_v1\test.jsonl"
    save_path = r"F:\\Work\kg\gen\news_summary\train_v1\test_pred_by_pegasus_base.jsonl"

    data_path = r"F:\\Work\kg\ie\fin_events\fin_events_20230118_pred_by_pegasus.jsonl"
    save_path = r"F:\\Work\kg\ie\fin_events\fin_events_20230118_pred_by_pegasus_v6.jsonl"

    task_path_unimo = r"F:\\Work\Learning\awesome-nlp\gen\unimo-text\checkpoints\news_summary\unimo-text-1.0-large\model_best"
    task_path_pegasus = r"F:\\Work\Learning\awesome-nlp\gen\pegasus\finetune\checkpoints\news_summary" \
                        r"\Randeng-Pegasus-238M-Summary-Chinese\model_best"

    # wsl
    # data_path = r"/mnt/f/Work/kg/gen/news_summary/train_v1/test.jsonl"
    # save_path = r"/mnt/f/Work/kg/gen/news_summary/train_v1/test_pred_by_unimo_and_pegasus_v4.jsonl"
    # task_path_unimo = r"/mnt/f/Work/Learning/awesome-nlp/gen/unimo-text/checkpoints/news_summary/unimo-text-1.0-large/model_best"
    # task_path_pegasus = r"/mnt/f/Work/Learning/awesome-nlp/gen/pegasus/finetune/checkpoints/news_summary/Randeng-Pegasus-523M-Summary-Chinese/model_best"

    print(f"model_path_unimo: {task_path_unimo}")
    print(f"model_path_pegasus: {task_path_pegasus}")

    examples = load_jsonl_data(data_path)
    print(f"example-num: {len(examples)}")
    # predict-parameters
    batch_size = 16
    device = 0
    expansion_coef = 1.5
    max_token_len = 1024
    max_seq_len = int(max_token_len * expansion_coef)
    ratio_head2tail = [3, 1]
    max_gen_length = 115
    length_penalty = 0.7
    num_beams = 4
    decode_strategy = "beam_search"
    use_fp16_decoding = True
    use_faster = False
    use_fast_tokenizer = True
    is_shuffle = False
    if is_shuffle:
        random.shuffle(examples)
    if do_pegasus:
        summarizer_pegasus = Taskflow("text_summarization",
                                      task_path=task_path_pegasus,
                                      max_seq_len=max_token_len,
                                      max_length=max_gen_length,
                                      length_penalty=length_penalty,
                                      decode_strategy=decode_strategy,
                                      num_beams=num_beams,
                                      batch_size=batch_size,
                                      use_fp16_decoding=use_fp16_decoding,
                                      device_id=device,
                                      use_faster=use_faster,
                                      use_fast_tokenizer=use_fast_tokenizer)
    if do_unimo:
        summarizer_unimo = Taskflow("text_summarization",
                                    task_path=task_path_unimo,
                                    max_seq_len=max_token_len,
                                    max_length=max_gen_length,
                                    length_penalty=length_penalty,
                                    decode_strategy=decode_strategy,
                                    num_beams=num_beams,
                                    batch_size=batch_size,
                                    use_fp16_decoding=use_fp16_decoding,
                                    device_id=device,
                                    use_faster=use_faster,
                                    use_fast_tokenizer=use_fast_tokenizer)
    batch_num = math.ceil(len(examples) / batch_size)
    batches = [examples[i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]
    results = []
    for batch in tqdm(batches):
        source_batch = [create_source_news(example,
                                           max_char_len=max_seq_len,
                                           truncate_func=truncate_news) for example in batch]
        source_batch = [unicodedata.normalize("NFKC", source) for source in source_batch]
        if do_unimo:
            result_batch_unimo = summarizer_unimo(source_batch)
        if do_pegasus:
            result_batch_pegasus = summarizer_pegasus(source_batch)
        for idx, example in enumerate(batch):
            example["input"] = source_batch[idx]
            if do_unimo:
                example["--unimo"] = result_batch_unimo[idx]
                example["--unimo_post"] = summary_post_process(result_batch_unimo[idx])
            if do_pegasus:
                example["pegasus"] = result_batch_pegasus[idx]
                example["pegasus_post"] = summary_post_process(result_batch_pegasus[idx])
        results.extend(batch)
        save2jsonl(results, save_path)

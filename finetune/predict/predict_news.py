import math
import random

from paddlenlp import Taskflow
import sys, os
import jieba

from tqdm import tqdm

sys.path.append("../../")
from finetune.data_loader import create_source_news, truncate_news

sys.path.append(r"F:\\Work\Learning\awesome-nlp")
sys.path.append("/mnt/f/Work/Learning/awesome-nlp")
from dautils.utils import load_jsonl_data, save2jsonl
from dautils.datext import cut_text_backward, FIRST_SEP


def summary_post_process(summary):
    summary = summary.replace("()", "").replace("（）", "")
    summary, _ = cut_text_backward(summary, second_sep=FIRST_SEP)
    if summary[-1] not in FIRST_SEP:
        summary += "。"
    return summary


if __name__ == "__main__":
    do_unimo = False
    do_pegasus = True

    data_path = r"F:\\Work\kg\gen\news_summary\train_v1\test.jsonl"
    save_path = r"F:\\Work\kg\gen\news_summary\train_v1\test_pred_by_pegasus_base.jsonl"

    data_path = r"F:\\Work\kg\ie\fin_events\fin_events_20230118_pred_by_pegasus.jsonl"
    save_path = r"F:\\Work\kg\ie\fin_events\fin_events_20230118_pred_by_pegasus_v4.jsonl"

    task_path_unimo = r"F:\\Work\Learning\awesome-nlp\gen\unimo-text\checkpoints\news_summary\unimo-text-1.0-large\model_best"
    # task_path_unimo = r"F:\\Work\Learning\awesome-nlp\gen\unimo-text\checkpoints\news_summary\unimo-text-1.0-summary\model_best"
    # task_path_pegasus = r"F:\\Work\Learning\awesome-nlp\gen\pegasus\finetune\checkpoints\news_summary\Randeng-Pegasus-523M-Summary-Chinese\model_best"
    task_path_pegasus = r"F:\\Work\Learning\awesome-nlp\gen\pegasus\finetune\checkpoints\news_summary" \
                        r"\Randeng-Pegasus-238M-Summary-Chinese\model_best"

    # wsl
    # data_path = r"/mnt/f/Work/kg/gen/news_summary/train_v1/test.jsonl"
    # save_path = r"/mnt/f/Work/kg/gen/news_summary/train_v1/test_pred_by_unimo_and_pegasus_v4.jsonl"
    # task_path_unimo = r"/mnt/f/Work/Learning/awesome-nlp/gen/unimo-text/checkpoints/news_summary/unimo-text-1.0-large/model_best"
    # task_path_pegasus = r"/mnt/f/Work/Learning/awesome-nlp/gen/pegasus/finetune/checkpoints/news_summary/Randeng-Pegasus-523M-Summary-Chinese/model_best"

    print(f"model_path_unimo: {task_path_unimo}")
    print(f"model_path_pegasus: {task_path_pegasus}")

    uppercase_file = r"F:\\Work\Learning\awesome-nlp\gen\pegasus\finetune\data\vocab\uppercase.txt"
    jieba.load_userdict(uppercase_file)

    examples = load_jsonl_data(data_path)
    print(f"example-num: {len(examples)}")
    # predict-parameters
    batch_size = 2
    device = 1
    expansion_coef = 1.5
    max_token_len = 1024
    max_seq_len = int(max_token_len * expansion_coef)
    ratio_head2tail = [3,1]
    max_gen_length = 90
    length_penalty = 0.7
    num_beams = 4
    decode_strategy = "beam_search"
    use_fp16_decoding = True
    use_faster = False
    use_fast_tokenizer = True
    do_lower_case = False
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
                                      use_fast_tokenizer=use_fast_tokenizer,
                                      do_lower_case=do_lower_case)
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
                                    use_fast_tokenizer=use_fast_tokenizer,
                                    do_lower_case=do_lower_case)
    batch_num = math.ceil(len(examples) / batch_size)
    batches = [examples[i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]
    results = []
    for batch in tqdm(batches):
        source_batch = [create_source_news(example,
                                           max_char_len=max_seq_len,
                                           truncate_func=truncate_news,
                                           do_lower_case=do_lower_case) for example in batch]
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

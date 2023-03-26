import json
import math
import re
import unicodedata
import random

from paddlenlp import Taskflow
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../"))
from data_loader import create_source_human_activity, get_human_and_activity_context
from utils import load_jsonl_data, save2jsonl


def summary_post_process_human_activity(summary, example):
    if "没有参加活动" in summary:
        return ""
    if example["human_name"].lower() not in summary.lower() \
            and not summary.startswith("出席") \
            and not summary.startswith("参加"):
        return ""
    summary = unicodedata.normalize("NFKC", summary)
    summary = re.sub(r'(\(\s*(简称|英文简称|下称|以下简称|或称).*?\))', "", summary)
    summary = re.sub(r'(\(\s*排名.*?\))', "", summary)
    summary = re.sub(r'(\(\s*(左|右|中间|签约台左|签约台右|.*图左|.*图右).*?\))', "", summary)
    summary = re.sub(r'(\(\s*(周(一|二|三|四|五|六|日)|星期(一|二|三|四|五|六|日|天)).*?\))', "", summary)
    return summary


if __name__ == "__main__":
    random.seed(10000)
    do_unimo = False
    do_pegasus = True
    # data_path = r"F:\\Work\Learning\awesome-nlp\gen\unimo-text\data\human_activity\test.jsonl"
    # save_path = r"F:\\Work\Learning\awesome-nlp\gen\unimo-text\data\human_activity\predict_activity_by_unimo-large.json"

    data_path = r"F:\\Work\kg\gen\human_activity\去重后清洗数据.json"
    # save_path = r"F:\\Work\kg\gen\human_activity\去重后清洗数据_pred_by_unimo.json"
    save_path = r"F:\\Work\kg\gen\human_activity\去重后清洗数据_pred_by_unimo_and_pegasus_v4.json"

    # data_path = r"F:\\Work\kg\gen\human_activity\train_v3\test.jsonl"
    # save_path = r"F:\\Work\kg\gen\human_activity\train_v3\test_pred_by_unimo.jsonl"

    task_path_unimo = r"../../../../../learning/awesome-nlp/gen/unimo-text/checkpoints/human_activity/unimo-text-1.0-large/model_best"
    task_path_pegasus = r"../../../../../learning/awesome-nlp/gen/pegasus/finetune/checkpoints/human_activity" \
                        r"/Randeng-Pegasus-523M-Summary-Chinese/release_v3"
    #task_path_pegasus = r"../../../../../Learning/awesome-nlp/gen/pegasus/finetune/checkpoints/human_activity" \
    #                    r"/Randeng-Pegasus-238M-Summary-Chinese/model_best"

    data_path = r"../../../../../kg/gen/human_activity/offline-inference/v3/person_event_待生成1_20230324.json"
    save_path = r"../../../../../kg/gen/human_activity/offline-inference/v3/person_event_待生成1_20230324_by_pegasus.json"

    print(f"model_path_unimo: {os.path.realpath(task_path_unimo)}")
    print(f"model_path_pegasus: {os.path.realpath(task_path_pegasus)}")

    start_index = -1
    activity_column = "event_name"
    examples = load_jsonl_data(data_path)
    # examples = [example for example in examples if example.get("index", 0) > start_index]
    # examples = [example for example in examples if len(example["text"])>40]
    for example in examples:
        example["content"] = example[activity_column]
    print(f"example-num: {len(examples)}")

    # predict-parameters
    batch_size = 18
    device = 0
    title_len = 0
    expansion_coef = 1.5
    max_token_len = 512
    max_seq_len = int(max_token_len * expansion_coef)
    max_margin_of_activity_and_human = 500
    max_gen_length = 180
    length_penalty = 0.5
    num_beams = 4
    decode_strategy = "beam_search"
    use_fp16_decoding = True
    use_faster = False
    use_fast_tokenizer = True
    use_activity_name = True
    activity_column = "event_name"
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
    f = open(save_path, 'a', encoding='utf8')
    for idx, batch in enumerate(tqdm(batches)):
        source_batch = [create_source_human_activity(example, max_seq_len_char=max_seq_len, title_len=title_len,
                                                     use_activity_name=use_activity_name,
                                                     max_margin_of_activity_and_human=max_margin_of_activity_and_human)
                        for example in batch]
        source_batch = [unicodedata.normalize("NFKC", source) for source in source_batch]
        if do_unimo:
            result_batch_unimo = summarizer_unimo(source_batch)
            result_batch_unimo_post = [summary_post_process_human_activity(result, example)
                                       for result, example in zip(result_batch_unimo, batch)]
        if do_pegasus:
            result_batch_pegasus = summarizer_pegasus(source_batch)
            result_batch_pegasus_post = [summary_post_process_human_activity(result, example)
                                         for result, example in zip(result_batch_pegasus, batch)]
        for idx, example in enumerate(batch):
            if do_unimo:
                example["--unimo_raw"] = result_batch_unimo[idx]
                example['--unimo'] = result_batch_unimo_post[idx]
            if do_pegasus:
                example["pegasus_raw"] = result_batch_pegasus[idx]
                example['pegasus'] = result_batch_pegasus_post[idx]
        results.extend(batch)
        for example in batch:
            f.write(json.dumps(example, ensure_ascii=False)+"\n")
    f.close()
    save2jsonl(results, save_path)


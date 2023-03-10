import math
from random import random

from paddlenlp import Taskflow
import os, sys
import jieba

from tqdm import tqdm
from finetune.data_loader import create_source_human_activity, get_human_and_activity_context

sys.path.append(r"F:\\Work\Learning\awesome-nlp")
from dautils.utils import load_jsonl_data, save2jsonl

if __name__ == "__main__":
    do_unimo = False
    do_pegasus = True
    # data_path = r"F:\\Work\Learning\awesome-nlp\gen\unimo-text\data\human_activity\test.jsonl"
    # save_path = r"F:\\Work\Learning\awesome-nlp\gen\unimo-text\data\human_activity\predict_activity_by_unimo-large.json"

    data_path = r"F:\\Work\kg\gen\human_activity\去重后清洗数据.json"
    # save_path = r"F:\\Work\kg\gen\human_activity\去重后清洗数据_pred_by_unimo.json"
    save_path = r"F:\\Work\kg\gen\human_activity\去重后清洗数据_pred_by_unimo_and_pegasus_v4.json"

    data_path = r"F:\\Work\kg\gen\human_activity\待生成数据_20230308.json"
    save_path = r"F:\\Work\kg\gen\human_activity\待生成数据_20230308_pred_by_pegasus_v1.json"

    # data_path = r"F:\\Work\kg\gen\human_activity\train_v3\test.jsonl"
    # save_path = r"F:\\Work\kg\gen\human_activity\train_v3\test_pred_by_unimo.jsonl"

    task_path_unimo = r"F:\\Work\Learning\awesome-nlp\gen\unimo-text\checkpoints\human_activity\unimo-text-1.0-large\model_best"
    task_path_pegasus = r"F:\\Work\Learning\awesome-nlp\gen\pegasus\finetune\checkpoints\human_activity" \
                        r"\Randeng-Pegasus-523M-Summary-Chinese\model_2526"
    print(f"model_path_unimo: {task_path_unimo}")
    print(f"model_path_pegasus: {task_path_pegasus}")

    jieba.load_userdict(os.path.join(task_path_pegasus, "vocab.txt"))

    start_index = -1
    activity_column = "event_name"
    examples = load_jsonl_data(data_path)
    # examples = [example for example in examples if example.get("index", 0) > start_index]
    #examples = [example for example in examples if len(example["text"])>40]
    for example in examples:
        example["content"] = example[activity_column]
    print(f"example-num: {len(examples)}")
    # predict-parameters
    batch_size = 16
    device = 0
    title_len = 0
    expansion_coef = 1.5
    max_token_len = 1024
    max_seq_len = int(max_token_len * expansion_coef)
    max_gen_length = 80
    length_penalty = 0.5
    num_beams = 4
    decode_strategy = "beam_search"
    use_fp16_decoding = True
    use_faster = False
    use_fast_tokenizer = True
    do_lower_case = True
    use_activity_name = True
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
        """
        context_batch = [
            get_human_and_activity_context(example["text"], example["human_name"], example["content"], offset=40) for
            example in batch]
        for example, context in zip(batch, context_batch):
            example["context"] = context
        """

        source_batch = [create_source_human_activity(example, max_seq_len_char=max_seq_len, title_len=title_len,
                                                     use_activity_name=use_activity_name, do_lower_case=do_lower_case)
                        for example in batch]
        if do_unimo:
            result_batch_unimo = summarizer_unimo(source_batch)
        if do_pegasus:
            result_batch_pegasus = summarizer_pegasus(source_batch)
        for idx, example in enumerate(batch):
            if do_unimo:
                example["--unimo"] = result_batch_unimo[idx]
            if do_pegasus:
                example["pegasus"] = result_batch_pegasus[idx]
        results.extend(batch)
        save2jsonl(results, save_path)

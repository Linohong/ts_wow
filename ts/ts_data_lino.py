import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from transformers import BartTokenizer
from typing import List, Tuple, Optional, Dict
from pathlib import Path, PosixPath
import json
from dataclasses import dataclass
import logging
import random
import os
# from .cecap_utils_lino import sort_candidates_by

logger = logging.getLogger(__name__)

@dataclass
class Example(object):
    src: str
    knowl: List[str]
    target: str
    label: Optional[List[int]] = None
    label_index: Optional[int] = None
    knowl_topic: List[str] = None


@dataclass
class ts_Example(object):
    src: str
    cand_topic: str
    topic_sents: str
    label: str
    is_train: bool
    original_id: Optional[int] = None

    tot_label: Optional[str] = None  #

class KATDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        cecap=False,
        num_ks_sample=74092,
        # max_num_kno=40
        # **dataset_kwargs
    ):
        super().__init__()
        multiple_data = False
        self.examples = []

        if cecap != False and type(cecap) == list:
            multiple_data = True
            for c in cecap:
                data_file = '../dataset/cecap'
                if 'cecap_reddit' in c:
                    data_file = os.path.join(os.path.abspath(data_file), c)
                else:
                    data_file = os.path.join(os.path.abspath(data_file), c + '.data')
                    print('*********************************************')
                    print('*********************************************')
                    print("********* PRINT FOR FIXING ERROR ************ ")
                    print('*********************************************')
                    print('*********************************************')
                    print(data_file)
                self.examples.extend(self.load_file(data_file))

            print('2022.05.01 -- adjust the number of examples to %d'%(num_ks_sample))
            random_indices = random.sample(range(0,len(self.examples)), k=num_ks_sample)
            self.examples = [self.examples[i] for i in random_indices]

        if '.' in type_path and cecap==False:
            data_file = Path(data_dir).joinpath(type_path)
        else:
            data_file = Path(data_dir).joinpath(type_path + ".pkl")

        if self.examples != []:
            self.examples.extend(self.load_file(data_file))
        else:
            self.examples = self.load_file(data_file)

        if n_obs is not None and n_obs >= 1:
            random.Random(42).shuffle(self.examples)
            self.examples = self.examples[:n_obs]

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else "" 
        self.pad_token_id = self.tokenizer.pad_token_id
        # self.max_num_kno = max_num_kno

    def load_file(self, filename):
        examples = None

        if type(filename) != PosixPath and 'cecap' in filename:
            temp_cnt = 0
            if 'cecap_reddit' in filename:
                cecap_files = os.listdir(filename)
                print('LOADING REDDIT DATA...', end='')
                for file in cecap_files:
                    if all([require in file for require in ['cecap_reddit', '.data']]) and \
                        not any([avoid in file for avoid in ['_all_sentences.data', '_dialogues.data']]):
                        if examples == None:
                            examples = torch.load(os.path.join(filename, file))
                        else:
                            examples.extend(torch.load(os.path.join(filename, file)))
                        temp_cnt+=1
                        # if temp_cnt == 5:
                        #     break
                print("DONE!!! Number of read files: %d" % temp_cnt)
            else:
                print(filename)
                examples = torch.load(filename)

            print('CECAP DATA REPROCESSING ...', end='')
            examples = self.cecap_process(examples, is_reddit='cecap_reddit' in filename)
            print('The number of CECAP examples: %d' % len(examples))
            print('DONE!')

            return examples

        filename = Path(filename)
        if filename.suffix in ('.pkl', '.pt'):
            examples = torch.load(filename)
            # return examples[:len(examples)//20]
            return examples

        examples = []
        with filename.open() as f:
            for line in f:
                jsonobj = json.loads(line)

                #TODO: (2022.04.05) 이 부분에서 Example 단위를 고쳐야한다.
                for e in jsonobj:
                    knowl = []
                    knowl_topic = []
                    for kno in e['knowledge_sentences']:
                        topic, k = kno.split('__knowledge__')
                        knowl.append(k.strip())
                        knowl_topic.append(topic.strip())
                    ex = Example(
                        e["context"],
                        # [e["knowledge_sentences"]] if isinstance(e["knowledge_sentences"], str) else e["knowledge_sentences"],
                        knowl,
                        # [kno_list] if isinstance(kno_list, str) else kno_list,
                        e["response"],
                        # label_index=label_index
                        knowl_topic=knowl_topic,
                    )
                    examples.append(ex)

        return examples

    def cecap_process(self, examples, last_context_len=2, is_reddit=False):
        for e in examples:
            if is_reddit:
                # (turn-num, text): ex) (0, 'hello~~~'), (0, 'how was your day?'), (1, 'it was nice!'), ...
                # initialization.
                prev_turn_num = e.src[-1][0]
                cur_count = 0
                history = [e.src[-1][1]]

                # concatenate or append.
                for i in range(2, len(e.src)):  # starting from the end.
                    if prev_turn_num == e.src[-i][0]: # if the same turn, concatenate the sentence.
                        history[0] = history[0] + ' ' + e.src[-i][1]
                    else:   # if not the same turn, append the turn to the front of the history.
                        history = [e.src[-i][1]] + history
                        prev_turn_num = e.src[-i][0]
                        cur_count += 1
                        if cur_count == last_context_len:
                            break
                e.src = history

            e.src = SEP.join(e.src[-last_context_len:])
            e.src = e.src.replace('\n', ' ')
            e.knowl = [k.replace('\n', ' ') for k in e.knowl]
            e.target = e.target.replace('\n', ' ')

        return examples

    def read_targets(self):
        return [t.target for t in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> Dict[str, str]:
        return self.examples[index]


class KATDataCollator:
    def __init__(self, tokenizer: BartTokenizer, data_args, is_train=False, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.is_train = is_train
        self.eval_dialogue_id = {'__id__':0}

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            # input_ids, attention_mask, kno_input_ids, kno_attention_mask, labels = self._encode(batch)
            batch_elems = self._encode(batch, shuffle_kno=self.data_args.shuffle_knowledge,
                                        include_true_kno=self.data_args.include_true_kno,
                                        sort_kno=self.data_args.sort_kno,
                                        do_tot=self.data_args.do_tot)

        else:
            assert 0

        batch = {
            "input_ids": batch_elems[0],
            "attention_mask": batch_elems[1],
            "labels": batch_elems[2],
            "original_ids": batch_elems[3],
            "tot_labels": batch_elems[4],
        }
        return batch

    def _encode(self, batch:List[Example], shuffle_kno=True, include_true_kno=False, sort_kno=False, do_tot=False) -> Dict[str, torch.Tensor]:  # Lino version
        SEP = self.tokenizer.sep_token
        src_texts = [x.src for x in batch]
        cand_knowls = [(SEP+SEP).join([x.cand_topic, x.topic_sents]) for x in batch]
        labels = [0 if x.label == True else 1 for x in batch]
        original_ids = [x.original_id for x in batch]
        if do_tot:
            tot_labels = [0 if x.tot_label == True else 1 for x in batch]
            tot_labels = torch.tensor(tot_labels)
        else:
            tot_labels = None

        input_texts = [[st, ct] for st, ct in zip(src_texts, cand_knowls)]
        generator_datas = self.tokenizer.batch_encode_plus(
            input_texts,
            max_length=self.data_args.max_source_length,
            truncation="longest_first",  # default option: longest_first
            padding="longest",
            return_tensors="pt",
        ).data

        input_ids, attention_mask = (
            generator_datas["input_ids"],
            generator_datas["attention_mask"],
        )
        labels = torch.tensor(labels)

        return input_ids, attention_mask, labels, original_ids, tot_labels

    def _encode_ks_neg(self, batch:List[Example], shuffle_kno=True, include_true_kno=False, sort_kno=False) -> Dict[str, torch.Tensor]:  # Lino version
        '''
            This function is designed to make true and negative case of examples
            for knowledge selection and response generation.

            Negative examples are sampled at the ratio of 1:3 (neg:pos)
        '''
        src_texts = [x.src for x in batch]
        tgt_texts = [x.target for x in batch]

        # kno_texts = []
        max_num_kno = self.data_args.max_num_kno
        nums = [len(x.knowl) for x in batch]
        max_num_kno = min(max(nums), max_num_kno)

        kno_texts = []
        # concatenate until the input exceeds the maxl.
        # make sure the true knowledge is included in the input string

        maxl = self.data_args.max_source_length
        true_knowledges = []  # for checking purpose
        true_kno_positions = []

        for i, x in enumerate(batch):  # x is an unit of one example.
            kno_text = ''
            kno = x.knowl[:max_num_kno]
            cur_true_kno = kno[0]
            true_knowledges.append(cur_true_kno)
            ran_abs_num = len(cur_true_kno) * self.data_args.seed # set a unique seed for the example as the length of the true knowledge.

            if True:  # shuffle the knowledge. include the true knowledge of 75%.
                kno = kno[1:]
                random.Random(ran_abs_num).shuffle(kno) # shuffle the knowledge

                tkn_st = self.tokenizer(src_texts[i], padding=False, truncation=False)
                cur_len = len(tkn_st['input_ids'])
                for j in range(len(kno)):  # concatenating knowledges
                    tkn_kt = self.tokenizer(SEP + kno[j], padding=False, truncation=False)
                    cur_len += len(tkn_kt['input_ids'])
                    if cur_len > maxl or j==len(kno)-1:
                        # Randomly select one position to put the true knowledge.
                        # The true knowledge is set not to be truncated.
                        if j == 0:  # sampling impossible, only one knowledge exists except the true knowledge, the first one exceeds the maxl.
                            true_kno_position = 0
                        else:
                            true_kno_position = random.Random(ran_abs_num).sample(range(j), k=1)[0]

                        # put true knowledge to the sampled position.
                        if np.random.choice(a=[True, False], p=[0.75, 0.25]):  # include true knowledge for True, do not include otherwise.
                            if len(kno) == 0:
                                kno = [cur_true_kno]
                            else:
                                kno[true_kno_position] = cur_true_kno
                            true_kno_positions.append(true_kno_position + 1)  # gather true knowledge positions for the knowledge selection task.
                            # positions are added with 1 from the true position to include the negative case.
                            break
                        else: # do not include the true knowledge
                            if len(kno) == 0:
                                kno = [cur_true_kno]  # The ture knowledge must be included.
                                true_kno_positions.append(true_kno_position + 1)
                            else:
                                true_kno_positions.append(0)  # indicating the negative case.

                for jj in range(j+1):
                    kno_text += SEP + kno[jj].strip()
                    kno_text = kno_text.strip()

            kno_texts.append(kno_text)

        input_texts = [[st, kt] for st, kt in zip(src_texts, kno_texts)]

        generator_datas = self.tokenizer.batch_encode_plus(
            input_texts,
            # tgt_texts=tgt_texts,
            max_length=self.data_args.max_source_length,
            # max_target_length=self.data_args.max_target_length,
            truncation=True,  # default option: longest_first
            padding="longest",
            return_tensors="pt",
        ).data

        with self.tokenizer.as_target_tokenizer():
            target_text = self.tokenizer(tgt_texts,
                                         padding='longest',  # batch should consists of the same length of input.
                                         truncation=True,  # default option: longest_first
                                         max_length=self.data_args.max_target_length,
                                         return_tensors='pt')

        input_ids, attention_mask, labels = (
            generator_datas["input_ids"],
            generator_datas["attention_mask"],
            target_text['input_ids'],
        )
        if self.data_args.do_ks:
            ks_labels = torch.tensor(true_kno_positions)
        else:
            ks_labels = None

        return input_ids, attention_mask, labels, ks_labels

    def _encode_original(self, batch:List[Example]) -> Dict[str, torch.Tensor]:
        src_texts = [x.src for x in batch]
        tgt_texts = [x.target for x in batch]
        kno_texts = []
        max_num_kno = self.data_args.max_num_kno
        nums = [len(x.knowl) for x in batch]
        max_num_kno = min(max(nums), max_num_kno)
        for x in batch:
            kno = x.knowl[:max_num_kno]
            if len(kno) < max_num_kno:
                kno = kno + ['pad'] * (max_num_kno - len(kno))
            kno_texts.extend(kno)
        generator_datas = self.tokenizer.prepare_seq2seq_batch(
            src_texts,
            tgt_texts=tgt_texts,
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="longest",  
            return_tensors="pt",
        ).data
        input_ids, attention_mask, labels = (
            generator_datas["input_ids"],
            generator_datas["attention_mask"],
            generator_datas["labels"],
        )
        kno_datas = self.tokenizer.batch_encode_plus(
            kno_texts,
            max_length=self.data_args.max_kno_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        kno_input_ids = kno_datas["input_ids"]
        _, dylen = kno_input_ids.shape
        kno_input_ids = kno_input_ids.reshape(-1, max_num_kno, dylen)
        kno_attention_mask = kno_datas["attention_mask"]
        kno_attention_mask = kno_attention_mask.reshape(-1, max_num_kno, dylen)
        return input_ids, attention_mask, kno_input_ids, kno_attention_mask, labels
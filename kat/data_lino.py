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
from .cecap_utils_lino import sort_candidates_by

SEP = '</s>'

logger = logging.getLogger(__name__)

@dataclass
class Example(object):
    src: str
    knowl: List[str]
    target: str
    label: Optional[List[int]] = None
    label_index: Optional[int] = None


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
        for_response_filtering=False,
        with_wow=False,
        ts_dir=None,
        # max_num_kno=40
        # **dataset_kwargs
    ):
        super().__init__()
        multiple_data = False
        self.for_response_filtering = for_response_filtering
        self.with_wow = with_wow
        self.ts_dir = ts_dir

        # Load CECAP data
        self.examples = []
        if cecap != False and type(cecap) == list:
            multiple_data = True
            for c in cecap:
                cecap_data_file = 'dataset/cecap'
                if 'cecap_reddit' in c:
                    cecap_data_file = os.path.join(cecap_data_file, c)
                else:
                    cecap_data_file = os.path.join(cecap_data_file, c + '.data')
                self.examples.extend(self.load_file(cecap_data_file))

        # Load WoW data
        if (cecap != False and self.with_wow) or cecap == False:
            if '.' in type_path:
                data_file = Path(data_dir).joinpath(type_path)
            else:
                data_file = Path(data_dir).joinpath(type_path + ".pkl")

            self.examples = self.load_file(data_file) + self.examples
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
        if type(filename) != PosixPath and 'cecap' in filename:
            temp_cnt = 0
            if 'cecap_reddit' in filename:
                cecap_files = os.listdir(filename)
                cecap_examples = None
                print('LOADING REDDIT DATA...', end='')
                for file in cecap_files:
                    if all([require in file for require in ['cecap_reddit', '.data']]) and \
                        not any([avoid in file for avoid in ['_all_sentences.data', '_dialogues.data']]):
                        if cecap_examples == None:
                            cecap_examples = torch.load(os.path.join(filename, file))
                        else:
                            cecap_examples.extend(torch.load(os.path.join(filename, file)))
                        temp_cnt+=1
                        # if temp_cnt == 5:
                        #     break
                print("DONE!!! Number of read files: %d" % temp_cnt)
            else:
                print(filename)
                cecap_examples = torch.load(filename)

            print('CECAP DATA REPROCESSING ...', end='')
            cecap_examples = self.cecap_process(cecap_examples, is_reddit='cecap_reddit' in filename)

            random.Random(42).shuffle(cecap_examples)
            cecap_examples = cecap_examples[:74092]
            print('The number of CECAP examples (in the loading phase): %d' % len(cecap_examples))
            print('DONE!')

            return cecap_examples

        filename = Path(filename)
        if filename.suffix in ('.pkl', '.pt'):
            examples = torch.load(filename)
            # return examples[:len(examples)//20]
            if self.ts_dir != None:
                examples = self.ts_process(examples, self.ts_dir)

            return examples

        examples = []
        with filename.open() as f:
            for line in f:
                jsonobj = json.loads(line)

                # kno_list = jsonobj['knowledge']
                # if 'train' in filename:
                #     if isinstance(kno_list, str):
                #         print('KNOWLEDGE IS GIVEN AS A STRING IN A FILE EXCEPTION!!!')
                #         exit(100)
                #     else:
                #         label_index = random.sample(range(min(self.max_num_kno, len(kno_list))), k=1)
                #         # swap label position
                #         kno_list[0], kno_list[label_index] = kno_list[label_index], kno_list[0]

                ex = Example(
                    jsonobj["context"], 
                    [jsonobj["knowledge"]] if isinstance(jsonobj["knowledge"], str) else jsonobj["knowledge"],
                    # [kno_list] if isinstance(kno_list, str) else kno_list,
                    jsonobj["response"],
                    # label_index=label_index
                )
                examples.append(ex)
            print('The number of WoW examples (in the loading phase): %d' % len(examples))

        return examples

    def ts_process(self, examples, ts_checkpoint, topic_nums=3):
        with open(os.path.join(ts_checkpoint, 'ts_detail_results.json_TRUE')) as f:
            data = json.load(f)
            assert len(data) == len(examples), "The number of ts results and the given examples do not match."
            for i, (tsd, ex) in enumerate(zip(data, examples)):
                topic_sents = ''
                for j in range(min(topic_nums,len(tsd['cand']))):
                    topic_sents = topic_sents + tsd['cand'][j]['topic_sents'].strip()
                examples[i].knowl = [sent.strip() for sent in topic_sents.strip('</s>').split('</s>')]

                # cut history
                his_len = 2
                examples[i].src = '</s>'.join([sent.strip() for sent in examples[i].src.strip('</s>').split('</s>')][his_len:])

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

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            # input_ids, attention_mask, kno_input_ids, kno_attention_mask, labels = self._encode(batch)
            if self.data_args.is_kat_tslf:
                batch_elems = self._encode_kat_tslf(batch)  # sinput_ids, attention_mask, kno_input_ids, kno_attention_mask, labels
            elif any([True for model_name in ['dialogpt', 'gpt2'] if
                    (model_name in self.tokenizer.name_or_path.lower())]):
                # print("2022.04.27 Lino's setting True")
                batch_elems = self._encode_gpttypes(batch, shuffle_kno=self.data_args.shuffle_knowledge,
                                           include_true_kno=self.data_args.include_true_kno,
                                           sort_kno=self.data_args.sort_kno,
                                           is_train=self.is_train)
            else:
                batch_elems = self._encode(batch, shuffle_kno=self.data_args.shuffle_knowledge,
                                            include_true_kno=self.data_args.include_true_kno,
                                            sort_kno=self.data_args.sort_kno,
                                            only_true_kno=self.data_args.only_true_kno)
            # input_ids, attention_mask, labels   (in a general case)
            # input_ids, attention_mask, labels, ks_labels  (in a ks case)

        else:
            assert 0

        if self.data_args.is_kat_tslf:
            batch = {
                "input_ids": batch_elems[0],
                "attention_mask": batch_elems[1],
                "kno_input_ids": batch_elems[2],
                "kno_attention_mask": batch_elems[3],
                "labels": batch_elems[4],
            }
        else:
            batch = {
                "input_ids": batch_elems[0],
                "attention_mask": batch_elems[1],
                # "kno_input_ids": kno_input_ids,
                # "kno_attention_mask": kno_attention_mask,
                "labels": batch_elems[2],
                "ks_labels": None if self.data_args.do_ks == False else batch_elems[3]
            }

        return batch

    def _encode(self, batch:List[Example], shuffle_kno=True, include_true_kno=False, sort_kno=False, only_true_kno=False) -> Dict[str, torch.Tensor]:  # Lino version
        src_texts = [x.src for x in batch]
        tgt_texts = [x.target for x in batch]


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

            # 03.31 lino test
            self.data_args.cand_kno_num.append(len(kno))

            if shuffle_kno and include_true_kno:
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
                        true_kno_positions.append(true_kno_position)  # gather true knowledge positions for the knowledge selection task.

                        # 03.31 lino test
                        self.data_args.real_kno_num.append(j+1)

                        # put true knowledge to the sampled position.
                        if len(kno) == 0:
                            kno = [cur_true_kno]
                        else:
                            kno[true_kno_position] = cur_true_kno
                        break


                if only_true_kno:
                    kno = [cur_true_kno]
                    j = 0
                for jj in range(j+1):
                    kno_text += SEP + kno[jj].strip()
                    kno_text = kno_text.strip()

            elif shuffle_kno and include_true_kno==False:
                # random.shuffle(kno)  # shuffle the knowledge
                if sort_kno != False:
                    if sort_kno == 'ks_pred_record':
                        kno = sort_candidates_by(method=sort_kno, dialogue_history=src_texts[i], candidates=kno,
                                                 pred_record=self.data_args.pred_record)
                    else:
                        kno = sort_candidates_by(method=sort_kno, dialogue_history=src_texts[i], candidates=kno,
                                                 recall_result=self.data_args.recall_result)
                else:
                    random.Random(ran_abs_num).shuffle(kno)  # shuffle the knowledge

                tkn_st = self.tokenizer(src_texts[i], padding=False, truncation=False)
                cur_len = len(tkn_st['input_ids'])
                for j in range(len(kno)):
                    tkn_kt = self.tokenizer(SEP + kno[j], padding=False, truncation=False)
                    cur_len += len(tkn_kt['input_ids'])
                    if cur_len > maxl or j == len(kno) - 1:  # the length exceeds @ j-th knowledge.
                        break



                if only_true_kno:
                    kno = [cur_true_kno]
                    j = 0
                for jj in range(j+1):  # include j-th knowledge to be truncated by the tokenizer.
                    kno_text += SEP + kno[jj].strip()
                    kno_text = kno_text.strip()

                # print(kno_text)
            else:  # this includes the true knowledge for sure.
                if sort_kno != False:
                    kno = sort_candidates_by(method=sort_kno, dialogue_history=src_texts[i], candidates=kno)


                if only_true_kno:
                    kno = [cur_true_kno]
                    j = 0
                for k in kno: # concatenate knowledge
                    kno_text += SEP + k.strip()
                    kno_text = kno_text.strip()
                # kno_text = kno[0].strip()

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

        # for i, d in enumerate(generator_datas['input_ids']):
        #     print("-------- %d --------" % i)
        #     print('**1. ENCODED INPUT STRING: %s' % self.tokenizer.decode(d))  # </s> </s> -- special tokens between each sequences
        #     print('**2. TRUE KNOWLEDGE: %s' % true_knowledges[i])
        #     print('**3. TARGET: %s' % self.tokenizer.decode(target_text['input_ids'][i]))

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

    def _encode_kat_tslf(self, batch:List[Example], shuffle_kno=True, include_true_kno=False, sort_kno=False, is_kat_tslf=True) -> Dict[str, torch.Tensor]:  # Lino version
        src_texts = [x.src for x in batch]
        tgt_texts = [x.target for x in batch]
        kno_texts = []
        max_num_kno = self.data_args.max_num_kno
        nums = [len(x.knowl) for x in batch]
        max_num_kno = min(max(nums), max_num_kno)

        for x in batch:
            kno = x.knowl[:max_num_kno]  # a list of knowledge of length: max_num_kno (=40)

            if len(kno) < max_num_kno:
                kno = kno + ['pad'] * (max_num_kno - len(kno))
            kno_texts.extend(kno)
        generator_datas = self.tokenizer.prepare_seq2seq_batch(
            # kno_texts are extension of all 32 batches of 40 knos, thus length of 1280.
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
            kno_texts,  # list of length 1280.
            max_length=self.data_args.max_kno_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        kno_input_ids = kno_datas["input_ids"]  # shape of 1280 x 64
        _, dylen = kno_input_ids.shape  # dylen = 64
        kno_input_ids = kno_input_ids.reshape(-1, max_num_kno, dylen)  # becomes 32 x 40 x 64
        kno_attention_mask = kno_datas["attention_mask"]  # shape: 1280 x 64
        kno_attention_mask = kno_attention_mask.reshape(-1, max_num_kno, dylen)
        return input_ids, attention_mask, kno_input_ids, kno_attention_mask, labels

        # 여기 위에서부터

    def _encode_gpttypes(self, batch:List[Example], shuffle_kno=True, include_true_kno=False, sort_kno=False, is_train=True) -> Dict[str, torch.Tensor]:  # Lino version
        EOS = self.tokenizer.eos_token

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

            # 03.31 lino test
            self.data_args.cand_kno_num.append(len(kno))

            if shuffle_kno and include_true_kno:
                kno = kno[1:]
                random.Random(ran_abs_num).shuffle(kno) # shuffle the knowledge

                tkn_st = self.tokenizer(src_texts[i], padding=False, truncation=False)
                plus = self.tokenizer(SEP + SEP + SEP)
                while len(tkn_st['input_ids'])+len(plus['input_ids']) >= self.data_args.max_source_length:
                    src_texts[i] = ' '.join(src_texts[i].split(' ')[50:]) # if the source text is so long that there is no space for the knowledge, we cut 50 words from the front.
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
                        true_kno_positions.append(true_kno_position)  # gather true knowledge positions for the knowledge selection task.

                        # 03.31 lino test
                        self.data_args.real_kno_num.append(j+1)

                        # put true knowledge to the sampled position.
                        if len(kno) == 0:
                            kno = [cur_true_kno]
                        else:
                            kno[true_kno_position] = cur_true_kno
                        break

                for jj in range(j+1):
                    kno_text += SEP + kno[jj].strip()
                    kno_text = kno_text.strip()

            elif shuffle_kno and include_true_kno==False:
                # random.shuffle(kno)  # shuffle the knowledge
                if sort_kno != False:
                    if sort_kno == 'ks_pred_record':
                        kno = sort_candidates_by(method=sort_kno, dialogue_history=src_texts[i], candidates=kno,
                                                 pred_record=self.data_args.pred_record)
                    else:
                        kno = sort_candidates_by(method=sort_kno, dialogue_history=src_texts[i], candidates=kno,
                                                 recall_result=self.data_args.recall_result)
                else:
                    random.Random(ran_abs_num).shuffle(kno)  # shuffle the knowledge

                tkn_st = self.tokenizer(src_texts[i], padding=False, truncation=False)
                plus = self.tokenizer(SEP + SEP + SEP)
                while len(tkn_st['input_ids'])+len(plus['input_ids']) >= self.data_args.max_source_length:
                    src_texts[i] = ' '.join(src_texts[i].split(' ')[50:]) # if the source text is so long that there is no space for the knowledge, we cut 50 words from the front.
                    tkn_st = self.tokenizer(src_texts[i], padding=False, truncation=False)

                cur_len = len(tkn_st['input_ids'])
                for j in range(len(kno)):
                    tkn_kt = self.tokenizer(SEP + kno[j], padding=False, truncation=False)
                    cur_len += len(tkn_kt['input_ids'])
                    if cur_len > maxl or j == len(kno) - 1:  # the length exceeds @ j-th knowledge.
                        break

                for jj in range(j+1):  # include j-th knowledge to be truncated by the tokenizer.
                    kno_text += SEP + kno[jj].strip()
                    kno_text = kno_text.strip()

                # print(kno_text)
            else:
                if sort_kno != False:
                    kno = sort_candidates_by(method=sort_kno, dialogue_history=src_texts[i], candidates=kno)
                for k in kno: # concatenate knowledge
                    kno_text += SEP + k.strip()
                    kno_text = kno_text.strip()
            kno_texts.append(kno_text)

        if any([True for model_name in ['dialogpt', 'gpt2'] if
                (model_name in self.tokenizer.name_or_path.lower())]):
            # print("2022.04.27 Lino's setting True")
            # if is_train:
            #     input_texts = [kt + SEP + SEP + st + SEP + tt + EOS for st, kt, tt in zip(src_texts, kno_texts, tgt_texts)]
            # else:
            input_texts = [[kt, SEP + SEP + st + SEP] for st, kt in zip(src_texts, kno_texts)]
        else:
            print('SOMETHING IS WRONG HERE.')

        generator_datas = self.tokenizer.batch_encode_plus(
            input_texts,
            max_length=self.data_args.max_source_length,
            truncation='only_first',  # default option: longest_first
            padding="longest",
            return_tensors="pt",
        ).data  # ['attention_mask'] field to 0 for padding.

        resps = []
        amasks = []
        for i, tgt_text in enumerate(tgt_texts):
            resp = self.tokenizer.encode(tgt_text,
                                          padding=False,
                                          max_length=self.data_args.max_target_length,
                                          return_tensors='pt')
            resps.append(resp.view(-1, 1))
            amasks.append(torch.ones(resp.view(-1, 1).shape))
        resps = nn.utils.rnn.pad_sequence(resps, padding_value=self.tokenizer.pad_token_id, batch_first=True)  # to be appended
        amasks = nn.utils.rnn.pad_sequence(amasks, padding_value=0, batch_first=True)  # to be appended

        generator_datas['input_ids'] = torch.cat((generator_datas['input_ids'], resps.squeeze(-1)), dim=1)
        generator_datas['attention_mask'] = torch.cat((generator_datas['attention_mask'], amasks.squeeze(-1)), dim=1)

        input_ids, attention_mask, labels = (
            generator_datas["input_ids"],
            generator_datas["attention_mask"],
            generator_datas['input_ids'],
        )

        if self.data_args.do_ks:
            ks_labels = torch.tensor(true_kno_positions)
        else:
            ks_labels = None

        return input_ids, attention_mask, labels, ks_labels

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
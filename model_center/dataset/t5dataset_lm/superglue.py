# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractclassmethod
import torch
import json

class SuperGLUE(torch.utils.data.Dataset):
    def __init__(self, tokenizer, device="cpu"):
        super().__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.device = device
        self.max_enc_size = 0
        self.max_dec_size = 0

    def make_input(self, tokenizer, template, max_encoder_length, max_decoder_length, label):
        input = tokenizer.encode(template)
        
        input = input[:-1] # remove <\s>

        length = len(input)

        if length > max_encoder_length:
            input = input[-max_encoder_length:]

        # input_tokens = torch.zeros((max_encoder_length,), dtype=torch.int32)
        input_tokens = torch.tensor(input).int()

        input_length = torch.tensor(length, dtype=torch.long)

        output = [tokenizer.pad_token_id, tokenizer.convert_tokens_to_ids("<extra_id_0>"), self.get_verbalizer(self.tokenizer)[label]]
        length = len(output) - 1
        # output_tokens = torch.zeros((max_decoder_length,), dtype=torch.int32)
        output_tokens = torch.tensor(output[:-1], dtype=torch.long)
        output_length = torch.tensor(length, dtype=torch.long)

        target = torch.tensor(output[1:], dtype=torch.long)

        index = torch.zeros((length,), dtype=torch.long)
        index[length - 1] = 1

        self.data.append({
            "enc_input": input_tokens,
            "enc_length": input_length,
            "dec_input": output_tokens,
            "dec_length": output_length,
            "targets": target,
            "index": index,
            "label": label
        })
        
        if input_length > self.max_enc_size:
            self.max_enc_size = input_length
        if output_length > self.max_dec_size:
            self.max_dec_size = output_length

    def read_data(self, dataset, path, split, rank, world_size):
        if split == 'test': return
        if split == 'dev': split = 'validation'
        path = f"{path}/{dataset}/cache/{split}.jsonl"
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            for i, row in enumerate(lines):
                yield json.loads(row)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, sample):
        bs = len(sample)
        batch = {
            "input_ids": torch.zeros(bs, self.max_enc_size, dtype=torch.long),
            "length": torch.zeros(bs, dtype=torch.long),
            "decoder_input_ids": torch.zeros(bs, self.max_dec_size, dtype=torch.long),
            "decoder_length": torch.zeros(bs, dtype=torch.long),
            "targets": torch.zeros(bs, self.max_dec_size, dtype=torch.long),
            "index": torch.zeros(bs, self.max_dec_size, dtype=torch.long),
            "labels": torch.zeros(bs, dtype=torch.long),
            "loss_mask": torch.zeros(bs, self.max_dec_size, dtype=torch.float)
        }

        for i in range(bs):
            batch["input_ids"][i][:len(sample[i]["enc_input"])] = sample[i]["enc_input"]
            batch["decoder_input_ids"][i][:len(sample[i]["dec_input"])] = sample[i]["dec_input"]
            batch["length"][i] = sample[i]["enc_length"]
            batch["decoder_length"][i] = sample[i]["dec_length"]
            batch["targets"][i][:len(sample[i]["targets"])] = sample[i]["targets"]
            batch["index"][i][:len(batch["index"])] = sample[i]["index"]
            batch["labels"][i] = sample[i]["label"]
            batch["loss_mask"][i][:len(sample[i]["targets"])] = 1
        
        for k in batch:
            batch[k] = batch[k].to(self.device)

        return batch


class BoolQ_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length) -> None:
        super().__init__()

        for row in self.read_data("BoolQ", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text_a = row['passage']
            text_b = row['question']

            template = f'{text_a}. {text_b}? <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]


class CB_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length, device="cpu"):
        super().__init__(device=device)

        for row in self.read_data("cb", path, split, rank, world_size):
            label = row["label"]
            text_a = row["premise"]
            text_b = row["hypothesis"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does sentence 1 entails sentence 2? <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0], tokenizer.encode("Maybe")[0]]
    

class COPA_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()

        for row in self.read_data("COPA", path, split, rank, world_size):
            label = row["label"]
            text = row["premise"]
            choice_1 = row["choice1"]
            choice_2 = row["choice2"]
            question = row["question"]

            template = f'Choice 1: {choice_1} Choice 2: {choice_2} The {question} of "{text}" was choice <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

            if split == 'train': # mirror
                label = label ^ 1
                template = f'Choice 1: {choice_2} Choice 2: {choice_1} The {question} of "{text}" was choice <extra_id_0>.'

                self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("1")[0], tokenizer.encode("2")[0]]


class MultiRC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()
        self.qids = []

        for template, label, qid in self.read_data("MultiRC", path, split, rank, world_size):
            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)
            self.qids.append(qid)

    def read_data(self, dataset, path, split, rank, world_size):
        if split == 'test': return
        if split == 'dev': split = 'val'
        path = f"{path}/{dataset}/{split}.jsonl"
        cnt = 0
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            max_id = (len(lines)) // world_size * world_size
            for i, row in enumerate(lines[:max_id]):
                row = json.loads(row)
                text = row["passage"]["text"]

                for question_json in row["passage"]["questions"]:
                    question = question_json["question"]
                    for answer_json in question_json["answers"]:
                        cnt += 1

        max_id = (cnt) // world_size * world_size
        cnt = 0
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            for i, row in enumerate(lines[:max_id]):
                row = json.loads(row)
                text = row["passage"]["text"]

                for question_json in row["passage"]["questions"]:
                    question = question_json["question"]
                    for answer_json in question_json["answers"]:
                        cnt += 1
                        if cnt > max_id: break
                        if cnt % world_size != rank: continue
                        answer = answer_json["text"]
                        label = answer_json["label"]

                        template = f'{text} Is answer "{answer}" the answer to the question "{question}"? <extra_id_0>.'

                        qid = f'{row["idx"]}-{question_json["idx"]}'

                        yield (template, label, qid)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]


class ReCoRD_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()

        for row in self.read_data("ReCoRD", path, split, rank, world_size):
            label = row["idx"]
            text = row["passage"]["text"]
            
            entities = []
            for entity_json in row['passage']['entities']:
                start = entity_json['start']
                end = entity_json['end']
                entity = text[start:end+1]
                entities.append(entity)

            text = text.replace("@highlight\n", "- ")  # we follow the GPT-3 paper wrt @highlight annotations

            for question_json in row["qas"]:
                question = question_json["query"]
                answers = []
                for answer_json in question_json["answers"]:
                    answer = answer_json["text"]
                    answers.append(answer)

                template = f'{text} Question: {question} Entities: {entities} Which entities can be filled in the placeholder? <extra_id_0>.'

                self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]


class RTE_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length, device="cpu"):
        super().__init__(tokenizer, device=device)

        for row in self.read_data("rte-full", path, split, rank, world_size):
            label = 0 if row["label"] == "not_entailment" else 1
            text_a = row["premise"]
            text_b = row["hypothesis"]

            # template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does sentence 1 entails sentence 2? <extra_id_0>.'
            template = f'{text_b}? <extra_id_0>. {text_a}'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("no")[0], tokenizer.encode("yes")[0]]


class WiC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()

        for row in self.read_data("WiC", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text_a = row["sentence1"]
            text_b = row["sentence2"]
            word = row["word"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does the word {word} in sentence 1 express the same meaning as in sentence 2? <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]


class WSC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length):
        super().__init__()

        for row in self.read_data("WSC", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text = row["text"]
            
            span_1 = row["target"]["span1_text"]
            span_2 = row["target"]["span2_text"]

            template = f'{text} Does {span_2} refers to {span_1}? <extra_id_0>.'

            self.make_input(tokenizer, template, max_encoder_length, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode("No")[0], tokenizer.encode("Yes")[0]]

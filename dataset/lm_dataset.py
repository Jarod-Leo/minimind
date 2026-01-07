import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path) # 样本数据, 写到__init__中被dataloader加载时会自动调用， 加载数据到内存

    def load_data(self, path):
        """加载 JSONL 格式的原始文本数据"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip()) # 需要一行行加载
                samples.append(data)
        return samples

    def __len__(self):
        """DataLoader 需要知道总共有多少数据，以此来计算总共有多少个 Batch。"""
        return len(self.samples)

    def __getitem__(self, index): 
        """DataLoader 内部的采样器会产生一系列索引，然后不断调用此方法来获取单个样本。"""
        sample = self.samples[index]

        # 构建输入文本
        # 将纯文本转化为Token ID
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        # loss_mask: 只有非填充（Padding）的部分才计算损失
        loss_mask = (input_ids != self.tokenizer.pad_token_id)# shape: [seq_len]
        # 构建因果语言建模任务：
        # X: [0, 1, 2, 3] -> 输入序列
        # Y: [1, 2, 3, 4] -> 预测目标（错开一位）
        X = torch.tensor(input_ids[:-1], dtype=torch.long) # shape: [seq_len - 1]
        Y = torch.tensor(input_ids[1:], dtype=torch.long) # shape: [seq_len - 1]
        # 损失掩码也要相应错开一位，因为 Y 的第一个 token 对应的是原始序列的第 2 个位置,要与Y贴合
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long) # shape: [seq_len - 1]
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # 预先编码标识符，用于定位答案的起始和结束位置
        # bos_token <im_start>
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        # eos_token <im_ends>
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip()) # 加载去掉前后空格的json数据
                samples.append(data)
        return samples

    def _create_chat_prompt(self, cs):
        """调用分词器的模板功能，将对话列表转为带角色标识的字符串"""
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def _generate_loss_mask(self, input_ids):
        """
        动态 Loss 掩码生成：
        核心逻辑：模型只需要学习『如何回答』，不需要学习『用户问了什么』。
        因此只有在 'assistant' 标识之后到 'eos_token' 之前的部分 mask 为 1。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # # 检查当前位置是否匹配 assistant 开始标识（self.bos_id，通常是 ["<|assistant|>", "\n"] 等 token 序列）
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 记录 assistant 内容起始位置（bos_id 之后）
                start = i + len(self.bos_id)
                end = start
                # 从 start 开始向后搜索 eos_id（assistant 结束标识）
                while end < len(input_ids):
                    # 匹配到回答结束
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 assistant 回答内容区域（不包括 bos_id，但包括 eos_id 之前的 token）设为 1
                # +1 是为了跳过 bos_id 后的第一个 token（有时是换行），根据实际数据调整
                # min 防止越界，self.max_length 是序列最大长度限制   
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 跳到当前 assistant 块结束位置，继续查找下一个（支持多轮对话）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                # 未匹配 bos_id，继续向前搜索
                i += 1
        # 返回长度为 seq_len 的 list，值 0 表示忽略损失，1 表示参与损失计算
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示（包含prompt和answer）
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 手动 Padding, 右padding
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # 处理换行等不可见字符
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        # # ================================
        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # DPO 数据包含：被选中的（chosen）和被拒绝的（rejected）两个序列
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        # 分别处理两个序列，流程同 SFT
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)

        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
        # DPO 需要同时返回选中的序列和拒绝的序列，以便后续计算隐含的奖励差值
        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        # 逻辑与 SFT 相同，确保只计算模型生成部分的概率
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id) 
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1 # 与labels对齐
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

# 目标：只返回提示词（Prompt），用于模型自生成答案，再通过 AI 打分进行强化
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            # messages[:-1] 表示只取到用户最后一次提问，不包含助手的回答
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        ), answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt, answer = self._create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt, # 模型需要补全的开头
            'answer': answer # 原始的正确答案（用于计算奖励或对比)
        }


if __name__ == "__main__":
    pass

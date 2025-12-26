import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        max_output_len: int = 10,
        tau: int = 1,
        k: int = 10,
        p: int = 0.5
    ) -> None:
        '''
            Initialize the TextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            tau: Temperature parameter for random sampling
            k: Top-k parameter for top-k sampling
            p: Cumulative probability threshold for nucleus sampling
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau
        self.k = k 
        self.p = p
        
        if decoding_strategy == "greedy":
            self.generator_func = self.greedy_decoding
        elif decoding_strategy == "random":
            self.generator_func = self.random_sampling
        elif decoding_strategy == "topk":
            self.generator_func = self.topk_sampling
        elif decoding_strategy == "nucleus":
            self.generator_func = self.nucleus_sampling

    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "1 in_seq_len"], 
    ) -> Int[torch.Tensor, "1 in_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "1 in_seq_len"],
    ) -> Int[torch.Tensor, "1 in_seq_len"]: 
        '''
            Implement Greedy decoding technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:
        generated_tokens = []
        device = input_ids.device  # Get the device of input_ids
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            if next_token == self.eos_token_id:
                break
            generated_tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], device=device)  # Ensure tensor is on same device
            input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
        return torch.tensor(generated_tokens, device=device)
        
    def random_sampling(
        self, 
        input_ids: Int[torch.Tensor, "1 in_seq_len"]
    ) -> Int[torch.Tensor, "1 in_seq_len"]:
        '''
            Implement Random sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:
        generated_tokens = []
        device = input_ids.device
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids.to(device))
            probabilities = torch.nn.functional.softmax(outputs.logits[:, -1, :] / self.tau, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            if next_token == self.eos_token_id:
                break
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=-1)
        return torch.tensor(generated_tokens, device=device) 

    def topk_sampling(
        self, 
        input_ids: Int[torch.Tensor, "1 in_seq_len"]
    ) -> Int[torch.Tensor, "1 in_seq_len"]:
        '''
            Implement Top-k sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:
        generated_tokens = []
        device = input_ids.device
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids.to(device))
            logits = outputs.logits[:, -1, :]
            topk_values, topk_indices = torch.topk(logits, self.k, dim=-1)
            probabilities = torch.nn.functional.softmax(topk_values, dim=-1)
            next_token = topk_indices[0, torch.multinomial(probabilities, num_samples=1)].item()
            if next_token == self.eos_token_id:
                break
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=-1)
        return torch.tensor(generated_tokens, device=device)        

    def nucleus_sampling(
        self, 
        input_ids: Int[torch.Tensor, "1 in_seq_len"]
    ) -> Int[torch.Tensor, "1 in_seq_len"]:
        '''
            Implement Nucleus sampling technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:
        generated_tokens = []
        device = input_ids.device
        for _ in range(self.max_output_len):
            outputs = self.model(input_ids.to(device))
            logits = outputs.logits[:, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            sorted_logits[sorted_indices_to_remove] = float('-inf')
            probabilities = torch.nn.functional.softmax(sorted_logits, dim=-1)
            next_token = sorted_indices[0, torch.multinomial(probabilities, num_samples=1)].item()
            if next_token == self.eos_token_id:
                break
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=-1)
        return torch.tensor(generated_tokens, device=device)

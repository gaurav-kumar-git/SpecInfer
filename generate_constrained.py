import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Set
from collections import defaultdict

warnings.filterwarnings("ignore")

class TrieNode:
    def __init__(self):
        self.children = defaultdict(lambda: None)
        self.is_end = False
        self.word = None

class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        self.model = model
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tokenizer = tokenizer

    def _build_trie(self, word_list: List[str]) -> TrieNode:
        """Build a trie from the word list"""
        root = TrieNode()
        for word in word_list:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            if not tokens:
                continue
                
            node = root
            for token in tokens:
                if node.children[token] is None:
                    node.children[token] = TrieNode()
                node = node.children[token]
            node.is_end = True
            node.word = word
        return root

    def _get_valid_tokens(self, node: TrieNode) -> List[int]:
        """Get all valid tokens from current trie node"""
        return [token for token in node.children if node.children[token] is not None]

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        # Initialize
        generated = input_ids.clone()
        trie_root = self._build_trie(word_list)
        current_node = trie_root
        generated_tokens = []
        remaining_words = set(word_list)
        
        for _ in range(self.max_output_len):
            with torch.no_grad():
                # Get model predictions
                outputs = self.model(generated)
                logits = outputs.logits[:, -1, :]  # Get last token logits
                
                # Apply constraints if we have remaining words to include
                valid_tokens = self._get_valid_tokens(current_node)
                if valid_tokens and remaining_words:
                    # Create mask to only allow valid tokens
                    mask = torch.full_like(logits, float('-inf'))
                    mask[:, valid_tokens] = logits[:, valid_tokens]
                    next_token = torch.argmax(mask, dim=-1)
                    
                    # Update trie position
                    next_token_id = next_token.item()
                    current_node = current_node.children[next_token_id]
                    
                    if current_node is None:
                        # Constraint failed, reset to root
                        current_node = trie_root
                        continue
                        
                    if current_node.is_end:
                        # Word completed successfully
                        remaining_words.discard(current_node.word)
                        current_node = trie_root
                else:
                    # No active constraints, normal greedy decoding
                    next_token = torch.argmax(logits, dim=-1)
                    current_node = trie_root
                
                # Append token and check for EOS
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
                generated_tokens.append(next_token.item())
                
                if next_token == self.eos_token_id:
                    break
        
        return torch.tensor(generated_tokens, device=input_ids.device)

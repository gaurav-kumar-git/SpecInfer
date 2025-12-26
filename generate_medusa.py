import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

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
        generated_tokens = []
        current_input = input_ids.clone()
        
        for _ in range(self.max_output_len):
            # Forward pass through the model
            with torch.no_grad():
                outputs = self.model(current_input)
            
            # Get logits from the LM head
            logits = outputs.logits[:, -1, :]  # Get last token's logits
            
            # Greedily select the most probable next token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Check for EOS token
            if next_token.item() == self.eos_token_id:
                break
                
            # Append the generated token
            generated_tokens.append(next_token.item())
            
            # Update input for next iteration
            current_input = torch.cat([current_input, next_token], dim=-1)
        
        return torch.tensor(generated_tokens, device=input_ids.device)
            

    def multi_head_decoding(
        self,
        input_ids: torch.Tensor  # shape: (1, in_seq_len)
    ) -> torch.Tensor:
        generated_tokens = []
        current_input = input_ids.clone()

        for _ in range(self.max_output_len):
            with torch.no_grad():
                medusa_outputs, base_outputs, orig_logits = self.model(
                    input_ids=current_input,
                    medusa_forward=True,
                    output_orig=True
                )

                lm_probs = torch.softmax(orig_logits[:, -1, :], dim=-1)
                medusa_probs = [
                    torch.softmax(medusa_outputs[head_idx, :, -1, :], dim=-1)
                    for head_idx in range(self.no_heads - 1)
                ]

                all_probs = [lm_probs] + medusa_probs

            # Internal dynamic beam width adjustment
            # beam_width = min(self.beam_width, max(5, len(generated_tokens) // 2 + 1))
            beam_width = self.beam_width
            candidates = [current_input.clone()]
            scores = [0.0]

            new_candidates = []
            new_scores = []

            for s in range(self.no_heads):
                log_probs = torch.log(all_probs[s][0])
                for i, (candidate, score) in enumerate(zip(candidates, scores)):
                    topk_values, topk_indices = torch.topk(log_probs, beam_width)

                    for token_score, token_idx in zip(topk_values, topk_indices):
                        new_score = score + token_score.item()
                        new_candidate = torch.cat([
                            candidate,
                            token_idx.unsqueeze(0).unsqueeze(0)
                        ], dim=-1)

                        # Early pruning
                        if len(new_scores) > 0 and new_score < min(new_scores) * 0.1:
                            continue

                        new_candidates.append(new_candidate)
                        new_scores.append(new_score)
            
            if token_idx.item() == self.eos_token_id:
                generated_tokens.extend(new_candidate[0, current_input.shape[1]:].tolist())
                return torch.tensor(generated_tokens, device=input_ids.device)
            
            if len(new_scores) > 0:
                top_indices = torch.topk(
                    torch.tensor(new_scores),
                    min(beam_width, len(new_scores))
                ).indices

                candidates = [new_candidates[i] for i in top_indices]
                scores = [new_scores[i] for i in top_indices]

            best_idx = torch.argmax(torch.tensor(scores)).item()
            best_candidate = candidates[best_idx]
            new_tokens = best_candidate[0, current_input.shape[1]:]

            generated_tokens.extend(new_tokens.tolist())
            current_input = best_candidate

            if len(generated_tokens) >= self.max_output_len:
                break

        return torch.tensor(generated_tokens, device=input_ids.device)

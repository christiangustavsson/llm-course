from typing import List, Dict, Any
import logging

def tokenize_batch(texts: List[str], tokenizer, device) -> List[int]:
    try:
        # Tokenize with truncation and padding
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            padding=False,  # Don't pad since we just want counts
            return_tensors=None,  # Don't convert to tensors since we just want counts
            return_length=True
        )
        
        # Get actual token counts (subtract 2 for special tokens)
        token_counts = [len(tokens) - 2 for tokens in encoded['input_ids']]
        
        return token_counts
    except Exception as e:
        logging.error(f"Error tokenizing batch: {str(e)}")
        return [0] * len(texts)  # Return 0 counts on error

def process_batch(batch_texts: List[str], tokenizer, device) -> List[Dict[str, Any]]:
    token_counts = tokenize_batch(batch_texts, tokenizer, device)
    return [{'text': text, 'token_count': count} 
            for text, count in zip(batch_texts, token_counts)] 
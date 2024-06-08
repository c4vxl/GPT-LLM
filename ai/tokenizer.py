from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, pretrained = None) -> None:
        self.at = AutoTokenizer.from_pretrained(pretrained) if pretrained is not None else AutoTokenizer()
        self.at.pad_token = self.at.eos_token
        self.vocab_size = self.at.vocab_size
    
    def encode(self, text: str):
        """
        Encode given text to tokens
        
        Args:
            text (str): Text to tokenize
        """

        return self.at.encode(text)
    
    def decode(self, tokens: list[int]):
        """
        Decode given tokens to text
        
        Args:
            tokens (list[int]): Tokens to textify
        """

        return self.at.decode(tokens)
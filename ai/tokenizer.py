from transformers import AutoTokenizer

class Tokenizer(AutoTokenizer):
    def __init__(self, pretrained = None) -> None:
        self.from_pretrained(pretrained if pretrained is not None else "GPT2")
        self.pad_token = self.eos_token
    
    def encode(self, text: str):
        """
        Encode given text to tokens
        
        Args:
            text (str): Text to tokenize
        """

        return self.encode(text)
    
    def decode(self, tokens: list[int]):
        """
        Decode given tokens to text
        
        Args:
            tokens (list[int]): Tokens to textify
        """

        return self.decode(tokens)
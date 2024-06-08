from .model import Model, ModelArgs
from .tokenizer import Tokenizer
import torch


class ChatModel:
    def __init__(self, model_path="model.pth", tokenizer: Tokenizer = Tokenizer()):
        self.tokenizer = tokenizer

        # Load model from file
        self.model = Model.load_from_file(model_path)

        # Verify model architecture
        expected_args = ModelArgs()
        if self.model.args.n_layer != expected_args.n_layer or self.model.args.n_embd != expected_args.n_embd:
            raise ValueError("Model architecture does not match the expected architecture.")

    def generate_from_prompt(self, prompt: str) -> str:
        """
        Generate a response from the given prompt.

        Args:
            prompt (str): The prompt provided by the user.

        Returns:
            str: The generated response.
        """
        
        encoded_prompt = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0)
        logits = self.model(input_ids)
        output_ids = torch.argmax(logits, -1)[0].tolist()

        response = self.tokenizer.decode(output_ids)

        return response

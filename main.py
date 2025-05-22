from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Force CPU usage
device = torch.device("cpu")

# Define model and tokenizer
model_name = "distilgpt2"  # Smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

class StoryBot:
    """Class StoryBot for generating text using a GPT-2 model"""
    def __init__(self, model, tokenizer):
        self.generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=1)

    def __call__(self, user_prompt: str, target_length: int) -> str:
        generated_text = ""
        while len(generated_text) < target_length:
            response = self.generator(
                user_prompt + generated_text,
                max_length=len(generated_text) + 200,  # Generate text in chunks of 200 tokens,
                temperature=0.9,
                num_return_sequences=1,
                top_k=50, # Set to an integer value (e.g., 50)
                top_p=0.85,  # Limit the possible words to top 50
                eos_token_id=50256,
                repetition_penalty=1.2,  # Add repetition penalty if supported 
            )
            new_text = response[0]["generated_text"][len(generated_text):]  # Remove the prompt part
            generated_text += new_text
        return generated_text.strip()

# Instantiate StoryBot
story_bot = StoryBot(model, tokenizer)

# Generate a story (with target length of 1000 characters)
story = story_bot("Tell me a sci-fi story about a yogi who teleports time lines.", 1000)

# Ensure the story is at least 1000 characters long
if len(story) < 1000:
    story = story.ljust(1000)  # Pad the story with spaces to make it 1000 characters long

# Print the first 1000 characters
print(f"Generated Story: {story[:1000]}")

with open("story.txt", "w", encoding="utf-8") as f:
    f.write(story)
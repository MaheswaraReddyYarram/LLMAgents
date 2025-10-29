from transformers import pipeline, set_seed

if __name__ == '__main__':
    # Initialize the text generation pipeline with GPT-2 model
    generator = pipeline('text-generation', model='gpt2')

    # Set a seed for reproducibility
    set_seed(42)

    # Generate text based on a prompt
    prompt = "In a distant future, humanity has"
    generated_text = generator(prompt, max_length=50, num_return_sequences=1, pad_token_id = generator.tokenizer.eos_token_id)

    # Print the generated text
    print(generated_text[0]['generated_text'])
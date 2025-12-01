try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    print("Transformers imported successfully")
except ImportError as e:
    print(f"Error importing transformers: {e}")
    exit(1)

try:
    # Download and save T5-base locally
    model_name = "t5-base"
    local_path = "./t5-base"  # Local directory to save the model

    print(f"Downloading {model_name}...")
    
    # Download model and tokenizer
    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    print("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Save locally
    print(f"Saving to {local_path}...")
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)

    print(f"T5-base downloaded and saved to {local_path}")
    
    # Test the saved model
    print("Testing loaded model...")
    test_model = T5ForConditionalGeneration.from_pretrained(local_path)
    test_tokenizer = T5Tokenizer.from_pretrained(local_path)
    print("✓ Model saved and loaded successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure you have installed sentencepiece: pip install sentencepiece")
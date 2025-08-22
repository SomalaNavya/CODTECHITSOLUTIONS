from transformers import T5ForConditionalGeneration, T5Tokenizer

def summarize_text(article: str) -> str:
    """
    Summarizes a lengthy article using a pre-trained T5 model.

    Args:
        article (str): The full text of the article to be summarized.

    Returns:
        str: A concise summary of the article.
    """
    # Load the tokenizer and model from Hugging Face
    # 't5-small' is a good starting point for a lightweight model
    # 't5-base' or 't5-large' can be used for higher quality summaries
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Prepend the task prefix "summarize:" as required by the T5 model
    input_text = "summarize: " + article

    # Tokenize the input text and prepare it for the model
    # truncation=True ensures the text doesn't exceed the model's max length
    # return_tensors="pt" formats the output as PyTorch tensors
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    # num_beams controls the number of sequences to search for
    # min_length and max_length set the size of the summary
    outputs = model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=30,
        max_length=150,
        early_stopping=True
    )

    # Decode the generated tokens back into a human-readable string
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary

if _name_ == "_main_":
    # Example lengthy article
    long_article ="""
    Artificial intelligence (AI) is a wide-ranging branch of computer science concerned with building smart machines capable of
    performing tasks that typically require human intelligence. AI is an interdisciplinary science with multiple approaches,
    but advancements in machine learning and deep learning are creating a paradigm shift in virtually every sector of the tech
    industry. For many years, AI was relegated to the domain of science fiction, but today, it is an integral part of our daily 
    lives, from virtual assistants to self-driving cars. AI systems work by ingesting vast amounts of labeled training data,
    analyzing it for correlations and patterns, and using these patterns to make predictions about future states. 
    For example, a chatbot might analyze millions of text conversations to learn how to generate human-like responses.
    The three main types of AI are narrow AI, general AI, and superintelligence. Narrow AI, also known as weak AI,
    is the only type of AI that has been successfully realized to date. General AI, or strong AI, is still largely theoretical
    and superintelligence is a futuristic concept.
    """

    # Get the summary
    concise_summary = summarize_text(long_article)

    # Display the results
    print("Original Article:")
    print("-" * 50)
    print(long_article)
    print("\n\nConcise Summary:")
    print("-" * 50)
    print(concise_summary)
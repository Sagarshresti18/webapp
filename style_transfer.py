from transformers import pipeline, PegasusTokenizer, PegasusForConditionalGeneration

# Load the pre-trained models and tokenizers for style transfer
style_transfer_models = {
    'formal_to_informal': {
        'model': PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase'),
        'tokenizer': PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
    },
    'informal_to_formal': {
        'model': PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase'),
        'tokenizer': PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
    }
}

# Load the style transfer pipelines
pipelines = {
    style: pipeline('text2text-generation', model=model['model'], tokenizer=model['tokenizer'])
    for style, model in style_transfer_models.items()
}

# Style transfer function
def transfer_style(text, style):
    if style not in pipelines:
        raise ValueError(f"Invalid style: {style}. Available styles: {', '.join(pipelines.keys())}")

    result = pipelines[style](text, max_length=100, do_sample=True, top_k=50)[0]['generated_text']
    return result.strip()

# Example usage
if __name__ == "__main__":
    print("Available Styles:")
    styles = list(pipelines.keys())
    for i, style in enumerate(styles, start=1):
        print(f"{i}. {style}")

    style_choice = input("Enter the style number (1: Formal to Informal / 2: Informal to Formal): ")

    if not style_choice.isdigit() or int(style_choice) < 1 or int(style_choice) > len(styles):
        print("Invalid style choice. Please try again.")
        exit()

    style = styles[int(style_choice) - 1]

    input_text = input("Enter the text to transfer style: ")

    output_text = transfer_style(input_text, style)

    print(f"\nOriginal Text: {input_text}")
    print(f"Style Transferred Text: {output_text}")

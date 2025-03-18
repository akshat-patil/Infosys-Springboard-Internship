from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

app = Flask(__name__)

# Load the saved model and tokenizer for abstractive summarization
model = BartForConditionalGeneration.from_pretrained("C:/Users/91862/Desktop/Text_Sum_Infosys/saved_model")
tokenizer = BartTokenizer.from_pretrained("C:/Users/91862/Desktop/Text_Sum_Infosys/saved_model")

# Function to generate abstractive summary
def generate_abstractive_summary(user_input):
    inputs = tokenizer(user_input, return_tensors='pt', truncation=True, padding='max_length', max_length=1024).to(model.device)
    summary_ids = model.generate(inputs['input_ids'], 
                                 attention_mask=inputs['attention_mask'], 
                                 max_length=512,
                                 min_length=100,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 length_penalty=1.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to generate extractive summary
def generate_extractive_summary(user_input):
    parser = PlaintextParser.from_string(user_input, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    num_sentences = max(2, len(user_input.split('.')) // 3)
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Route to display the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and generate summary
@app.route('/generate_summary', methods=['POST'])
def generate_summary_route():
    if request.method == 'POST':
        user_input = request.form.get('user_input', "")
        
        # Handle file upload
        if 'file' in request.files and request.files['file']:
            file = request.files['file']
            if file.filename.endswith('.txt'):
                user_input = file.read().decode('utf-8')

        if user_input:
            # Generate the requested summary type
            summary_type = request.form.get('summary_type')
            if summary_type == 'abstractive':
                summary = generate_abstractive_summary(user_input)
            elif summary_type == 'extractive':
                summary = generate_extractive_summary(user_input)
            else:
                summary = "Invalid summary type selected."

            return render_template('index.html', original_text=user_input, generated_summary=summary)

    return render_template('index.html', error="Please provide valid input.")

if __name__ == '__main__':
    app.run(debug=True)

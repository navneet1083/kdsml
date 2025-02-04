import json
import matplotlib.pyplot as plt
from collections import Counter

# Load SQuAD dataset
def load_squad_data(file_path):
    with open(file_path, 'r') as f:
        squad_data = json.load(f)
    return squad_data

# Analyze SQuAD dataset
def analyze_squad(squad_data):
    context_lengths = []
    question_lengths = []
    answer_lengths = []

    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            context_lengths.append(len(context.split()))

            for qa in paragraph['qas']:
                question = qa['question']
                question_lengths.append(len(question.split()))

                for answer in qa['answers']:
                    answer_text = answer['text']
                    answer_lengths.append(len(answer_text.split()))

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(context_lengths, bins=50, color='blue', alpha=0.7)
    axes[0].set_title('Context Lengths')
    axes[0].set_xlabel('Length')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(question_lengths, bins=50, color='green', alpha=0.7)
    axes[1].set_title('Question Lengths')
    axes[1].set_xlabel('Length')
    axes[1].set_ylabel('Frequency')

    axes[2].hist(answer_lengths, bins=50, color='red', alpha=0.7)
    axes[2].set_title('Answer Lengths')
    axes[2].set_xlabel('Length')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Example usage
squad_train = load_squad_data('train-v2.0.json')
analyze_squad(squad_train)
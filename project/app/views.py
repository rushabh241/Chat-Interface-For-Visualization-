import os
import pandas as pd
import numpy as np
import PyPDF2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from scipy.cluster.hierarchy import dendrogram, linkage
import spacy
from collections import Counter
import re

def index(request):
    return render(request, 'index.html')

def accept_prompt(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')
        response = {'status': 'success', 'ress': 'Your Request : '}

        if 'user_file' in request.FILES:
            user_file = request.FILES['user_file']
            file_path = handle_uploaded_file(user_file)
            
            keywords, action = parse_user_input(user_input)

            if action == 'summary' or action == 'explain':
                extracted_text = extract_text(file_path)
                file_summary = generate_text_summary(extracted_text)
                response['ress'] = 'Below is your request'
                response['extracted_text'] = extracted_text
                response['summary'] = {'Summary': file_summary}
            elif action == 'analyze':
                match_type = 'contains'  # Default match type, can be enhanced to detect from user_input
                result = extract_and_analyze_text(file_path, keywords, match_type)
                if isinstance(result, dict):
                    response['ress'] = 'Below is your request'
                    response['extracted_lines'] = result['extracted_lines']
                    response['word_frequency'] = result['word_frequency'].to_dict()
                else:
                    response['status'] = 'error'
                    response['ress'] = result
            elif action in ['histogram', 'scatter plot', 'pie chart', 'line chart', 'area plot', 'donut chart', 'box plot', 'bubble plot', 'heat map', 'dendrogram', 'venn diagram', 'treemap chart', '3d scatter plot']:
                image_url = generate_plot(file_path, action) 
                response['ress'] = f'{action.capitalize()} generated'
                response['image_url'] = image_url
            else:
                response['status'] = 'error'
                response['ress'] = 'Invalid request'
        
        return JsonResponse(response)

    return JsonResponse({'status': 'error', 'ress': 'Invalid request method'})

def handle_uploaded_file(f):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, f.name)
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_path

def parse_user_input(user_input):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(user_input)
    
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ']]
    action = 'extract_text'  # Default action

    if any(word in user_input.lower() for word in ['analyze', 'keywords', 'frequency']):
        action = 'analyze'
    elif any(word in user_input.lower() for word in ['histogram', 'bar chart']):
        action = 'histogram'
    elif any(word in user_input.lower() for word in ['scatter', 'scatter plot']):
        action = 'scatter plot'
    elif any(word in user_input.lower() for word in ['pie chart']):
        action = 'pie chart'
    elif any(word in user_input.lower() for word in ['line chart']):
        action = 'line chart'
    elif any(word in user_input.lower() for word in ['area plot']):
        action = 'area plot'
    elif any(word in user_input.lower() for word in ['donut chart']):
        action = 'donut chart'
    elif any(word in user_input.lower() for word in ['box plot']):
        action = 'box plot'
    elif any(word in user_input.lower() for word in ['bubble plot']):
        action = 'bubble plot'
    elif any(word in user_input.lower() for word in ['heat map']):
        action = 'heat map'
    elif any(word in user_input.lower() for word in ['dendrogram']):
        action = 'dendrogram'
    elif any(word in user_input.lower() for word in ['venn diagram']):
        action = 'venn diagram'
    elif any(word in user_input.lower() for word in ['treemap chart']):
        action = 'treemap chart'
    elif any(word in user_input.lower() for word in ['3d scatter plot']):
        action = '3d scatter plot'
    elif any(word in user_input.lower() for word in ['summary', 'brief', 'explain']):
        action = 'summary'
    
    return keywords, action

def extract_and_analyze_text(file_path, keywords, match_type='contains'):
    nlp = spacy.load('en_core_web_sm')
    extracted_lines = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if match_type == 'contains':
                    if any(keyword.lower() in line.lower() for keyword in keywords):
                        extracted_lines.append(line.strip())
                elif match_type == 'exact':
                    if any(keyword.lower() == line.lower().strip() for keyword in keywords):
                        extracted_lines.append(line.strip())
                elif match_type == 'regex':
                    if any(re.search(keyword, line, re.IGNORECASE) for keyword in keywords):
                        extracted_lines.append(line.strip())
                else:
                    raise ValueError("Invalid match_type. Choose 'contains', 'exact', or 'regex'.")
    except FileNotFoundError:
        return f"Error: The file '{file_path}' does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

    # Perform basic analysis
    doc = nlp(" ".join(extracted_lines))
    word_freq = Counter([token.text.lower() for token in doc if not token.is_stop and not token.is_punct])

    # Convert to DataFrame for better manipulation and visualization
    df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

    # Plot the top 10 most common words
    df.head(10).plot(kind='bar', x='Word', y='Frequency', legend=False, title='Top 10 Most Common Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(settings.MEDIA_ROOT, 'top_10_words.png'))
    plt.close()

    return {
        "extracted_lines": extracted_lines,
        "word_frequency": df
    }

def generate_plot(file_path, action):

    df = load_file_to_dataframe(file_path)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return generate_empty_plot_message(action)

    if action == 'histogram': #
        return generate_histogram(file_path)
    elif action == 'scatter plot':#
        return generate_scatter_plot(file_path)
    elif action == 'pie chart':#
        return generate_pie_chart(file_path)
    elif action == 'line chart':#
        return generate_line_chart(file_path)
    elif action == 'area plot':#
        return generate_area_plot(file_path)
    elif action == 'donut chart':#
        return generate_donut_chart(file_path)
    elif action == 'box plot':#
        return generate_box_plot(file_path)
    elif action == 'bubble plot':#
        return generate_bubble_plot(numeric_df)
    elif action == 'heat map':#
        return generate_heat_map(numeric_df)
    elif action == 'dendrogram':#
        return generate_dendrogram(numeric_df)
    elif action == 'venn diagram':
        return generate_venn_diagram(file_path)
    elif action == 'treemap chart':
        return generate_treemap_chart(numeric_df)
    elif action == '3d scatter plot':
        return generate_3d_scatter_plot(file_path)
    else:
        raise ValueError("Invalid action for generating plot.")
    
def generate_empty_plot_message(action):
    plt.figure()
    plt.text(0.5, 0.5, f'No numeric data available for {action}', horizontalalignment='center')
    image_path = save_plot(f'{action}_empty.png')
    return image_path

def generate_histogram(file_path):
    df = load_file_to_dataframe(file_path)
    plt.figure(figsize=(10, 6))
    df.hist(bins=30, edgecolor='black', grid=False)
    plt.suptitle('Histogram')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    image_path = save_plot('histogram.png')
    return image_path

def generate_scatter_plot(file_path):
    df = load_file_to_dataframe(file_path)
    plt.figure()
    if df.shape[1] >= 2:
        df.plot.scatter(x=df.columns[0], y=df.columns[1])
        plt.title('Scatter Plot')
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
    else:
        plt.text(0.5, 0.5, 'Not enough data for scatter plot', horizontalalignment='center')
    image_path = save_plot('scatter_plot.png')
    return image_path

def generate_pie_chart(file_path):
    df = load_file_to_dataframe(file_path)
    plt.figure()
    if df.shape[1] >= 1:
        df[df.columns[0]].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Pie Chart')
    else:
        plt.text(0.5, 0.5, 'Not enough data for pie chart', horizontalalignment='center')
    image_path = save_plot('pie_chart.png')
    return image_path

def generate_line_chart(file_path):
    df = load_file_to_dataframe(file_path)
    plt.figure()
    df.plot.line()
    plt.title('Line Chart')
    plt.xlabel('Index')
    plt.ylabel('Values')
    image_path = save_plot('line_chart.png')
    return image_path

def generate_area_plot(file_path):
    df = load_file_to_dataframe(file_path)
    plt.figure()
    df.plot.area()
    plt.title('Area Plot')
    plt.xlabel('Index')
    plt.ylabel('Values')
    image_path = save_plot('area_plot.png')
    return image_path

def generate_donut_chart(file_path):
    df = load_file_to_dataframe(file_path)
    plt.figure()
    if df.shape[1] >= 1:
        data = df[df.columns[0]].value_counts()
        plt.pie(data, labels=data.index, autopct='%1.1f%%', wedgeprops=dict(width=0.3))
        plt.title('Donut Chart')
    else:
        plt.text(0.5, 0.5, 'Not enough data for donut chart', horizontalalignment='center')
    image_path = save_plot('donut_chart.png')
    return image_path

def generate_box_plot(file_path):
    df = load_file_to_dataframe(file_path)
    plt.figure()
    df.plot.box()
    plt.title('Box Plot')
    image_path = save_plot('box_plot.png')
    return image_path

def generate_bubble_plot(df):
    plt.figure()
    if df.shape[1] >= 3:
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=df.iloc[:, 2] * 100, alpha=0.5)
        plt.title('Bubble Plot')
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
    else:
        plt.text(0.5, 0.5, 'Not enough data for bubble plot', horizontalalignment='center')
    image_path = save_plot('bubble_plot.png')
    return image_path

def generate_heat_map(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heat Map')
    image_path = save_plot('heat_map.png')
    return image_path


def generate_dendrogram(df):
    plt.figure(figsize=(10, 6))
    Z = linkage(df, 'ward')
    dendrogram(Z)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    image_path = save_plot('dendrogram.png')
    return image_path


from matplotlib_venn import venn2

def generate_venn_diagram(df):
    plt.figure()
    if isinstance(df, str):
        df = load_file_to_dataframe(df)
        
    if df.shape[1] >= 2:
        set1 = set(df.iloc[:, 0].dropna())
        set2 = set(df.iloc[:, 1].dropna())
        venn2([set1, set2], set_labels=(df.columns[0], df.columns[1]))
        plt.title('Venn Diagram')
    else:
        plt.text(0.5, 0.5, 'Not enough data for Venn diagram', horizontalalignment='center')
    image_path = save_plot('venn_diagram.png')
    return image_path


import squarify

def generate_treemap_chart(df):
    plt.figure(figsize=(10, 6))
    if df.shape[1] >= 1:
        data = df.iloc[:, 0].dropna()
        sizes = data[data > 0]  # Ensure no zero or negative sizes
        labels = data.index

        if sizes.empty:
            plt.text(0.5, 0.5, 'No valid data for Treemap chart', horizontalalignment='center')
        else:
            # Ensure sizes and labels are aligned
            labels = labels[sizes.index]

            squarify.plot(sizes=sizes, label=labels, alpha=0.8)
            plt.title('Treemap Chart')
    else:
        plt.text(0.5, 0.5, 'Not enough data for Treemap chart', horizontalalignment='center')
    image_path = save_plot('treemap_chart.png')
    return image_path




def generate_3d_scatter_plot(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if df.shape[1] >= 3:
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2])
        ax.set_title('3D Scatter Plot')
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        ax.set_zlabel(df.columns[2])
    else:
        ax.text(0.5, 0.5, 0.5, 'Not enough data for 3D scatter plot', horizontalalignment='center')
    image_path = save_plot('3d_scatter_plot.png')
    return image_path

def save_plot(filename):
    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    plt.savefig(file_path)
    plt.close()
    return os.path.join(settings.MEDIA_URL, filename)

def load_file_to_dataframe(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError('Unsupported file format')
    return df

def extract_text(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.txt', '.csv', '.json', '.xlsx']:
        return extract_text_from_file(file_path)
    else:
        raise ValueError("Unsupported file format.")

def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def generate_text_summary(text):
    sentences = sent_tokenize(text)
    if len(sentences) > 5:
        summary = ' '.join(sentences[:5]) + '...'
    else:
        summary = text
    return summary

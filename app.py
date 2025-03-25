from flask import Flask, render_template, request, send_from_directory
import hashlib
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import random
import sqlite3
from docx import Document  
from PyPDF2 import PdfReader 
from transformers import AutoTokenizer, AutoModel
import torch



app = Flask(__name__)

def hash_value(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16)

def winnowing(text, k=5, t=4):
    words = text.split()
    fingerprints = set()
    n_grams = [tuple(words[i:i + k]) for i in range(len(words) - k + 1)]
    hash_values = [hash_value(' '.join(n_gram)) for n_gram in n_grams]
    
    for i in range(len(hash_values) - t + 1):
        window = hash_values[i:i + t]
        min_hash = min(window)
        fingerprints.add(min_hash)        
    return fingerprints


def compare_texts(input_text, reference_texts, k=5, t=4):
    input_fingerprints = winnowing(input_text, k, t)
    similarity_scores = {}
    
    for ref in reference_texts:
        ref_text = ref['text']
        ref_fingerprints = winnowing(ref_text, k, t)
        intersection = input_fingerprints.intersection(ref_fingerprints)
        union = input_fingerprints.union(ref_fingerprints)
        
        similarity_score = len(intersection) / len(union) if union else 0
        similarity_scores[ref_text] = similarity_score
    
    return similarity_scores

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

def calculate_similarity(doc1_embedding, doc2_embeddings):
    similarity_scores = []
    
    doc1_embedding = np.array(doc1_embedding).reshape(1, -1)
    
    for embedding in doc2_embeddings:
        embedding = np.array(embedding).reshape(1, -1)  
        
        if doc1_embedding.shape[1] != embedding.shape[1]:
            embedding = embedding[:, :doc1_embedding.shape[1]]  

        similarity = cosine_similarity(doc1_embedding, embedding)[0][0]
        similarity_scores.append(similarity)

    
    return similarity_scores

def build_similarity_graph(documents, threshold=0.3):
    G = nx.Graph()
    
    for i, doc in enumerate(documents):
        if isinstance(doc, dict):
            doc_text = doc.get('text', '')
        else:
            doc_text = doc

        label = " ".join(doc_text.split()[:5]) + "..." if len(doc_text.split()) > 5 else doc_text
        G.add_node(i, label=label)

        for j, other_doc in enumerate(documents):
            if i != j:
                similarity = cosine_similarity([doc['embedding']], [other_doc['embedding']])[0][0]
                if similarity >= threshold:
                    G.add_edge(i, j, weight=similarity)
    
    return G

def plot_graph_density(G,similarity_scores_percentage, document_labels):
    
    top_10_labels = document_labels[:100]  # Get first 10 documents, or top 10 from your list
    top_10_scores = similarity_scores_percentage[:100]
    top_10_labels = document_labels[:10]
    top_10_scores = similarity_scores_percentage[:10]
    # Get similarity scores for top 10 documents
    
    density = nx.density(G)
    print(f"Graph Density: {density:.4f}")
    
    density_plot_path = os.path.join('static', 'plots', 'graph_density_plot.png')
    plt.figure(figsize=(10, 6))
    plt.bar(top_10_labels, top_10_scores, color='skyblue')
    plt.title("Document Similarity Percentages", fontsize=16)
    plt.xlabel("Documents", fontsize=14)
    plt.ylabel("Similarity (%)", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(density_plot_path)
    plt.close()
    
    return density_plot_path

app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def extract_text_from_file(filepath):
    if filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    elif filepath.endswith('.docx'):
        doc = Document(filepath)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    elif filepath.endswith('.pdf'):
        reader = PdfReader(filepath)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    else:
        return "Unsupported file type"

@app.route('/text_analyze', methods=['GET', 'POST'])
def text_analyze():
    if request.method == 'POST':       
        file = request.files['file']
        print(file)
        if file.filename == '':
            return "No selected file"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        extracted_text = extract_text_from_file(filepath)
        if not extracted_text.strip():
            return render_template('dashboard.html', message="❌ No text found in the file.")

        # Check if word count is less than 20
        word_count = len(extracted_text.strip().split())
        if word_count < 20:
            return render_template('dashboard.html', message=f"❌ Text is too short ({word_count} words). Minimum 20 words required. Please upload a longer content.")
        
        with open('embeddings_with_text1.json', 'r') as json_file:
            embeddings_data = json.load(json_file)
        
        similarity_results = compare_texts(extracted_text, embeddings_data)
        
        scores = {ref_text: score for ref_text, score in similarity_results.items() if score > 0}
        texts = list(scores.keys())
        scores_values = list(scores.values())
        print(scores_values)

        if all(score == 0 for score in scores_values):
            maxscore=0
        else:
            max_score = max(scores_values)
            maxscore = round((max_score),1)
            maxscore = f'{max_score:.2f}'
            max_index = scores_values.index(max_score)
            max_text = texts[max_index]
            

            print(f"Highest Similarity Score: {max_score}")
        
            print(f"Text with Highest Similarity: {max_text}")
        
        
        plot_data = pd.DataFrame({'Reference Text': texts, 'Similarity Score': scores_values})

        plot_path = os.path.join('static', 'plots', 'similarity_plot.png')
        plt.figure(figsize=(10, 5))
        plt.barh(plot_data['Reference Text'], plot_data['Similarity Score'], color='skyblue')
        plt.xlabel('Similarity Score')
        plt.title('Similarity Scores with Reference Texts')
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()


        doc1_embedding = generate_embedding(extracted_text)
        doc2_embeddings = [generate_embedding(item['text']) for item in embeddings_data]
        similarity_scores = calculate_similarity(doc1_embedding, doc2_embeddings)

        similarity_scores = calculate_similarity(doc1_embedding, doc2_embeddings)
        print(similarity_scores)
        #similarity_scores = abs(similarity_scores)
        similarity_scores_percentage = [
            max(0, 10 - score * 10) if (10 - score * 10) >= 1 else 0  # Ensure no negative values and exclude values below 4
            for score in similarity_scores
        ]
        # Generate document labels (limited to 50)
        document_labels = [f"Doc {i+1}" for i in range(len(similarity_scores))]

        G = nx.Graph()

        # Add your paragraph as the main node (in red)
        G.add_node("Your Document", color="red")

        # Add embedding paragraphs as nodes (in blue) and connect them with similarity scores
        for i, label in enumerate(document_labels):
            G.add_node(label, color="blue")  # Add dataset paragraph node
            G.add_edge("Your Document", label, weight=similarity_scores[i])  # Add edge with similarity score

        min_similarity_score = min(similarity_scores)  # Find the minimum similarity score
        min_similarity_index = similarity_scores.index(min_similarity_score)  # Index of the min score
        min_similarity_doc = document_labels[min_similarity_index]  # Document label corresponding to min score


        # Step 2: Plot the graph
        graph_path = os.path.join('static', 'plots', 'similarity_graph.png')
        pos = nx.spring_layout(G)  # Generate graph layout

        # Node colors
        node_colors = [G.nodes[node]["color"] for node in G.nodes]

        plt.figure(figsize=(12, 8))
        nx.draw(
            G, pos, with_labels=True, node_size=1000, node_color=node_colors,
            font_size=10, font_weight="bold", edge_color="gray"
        )

        for edge in G.edges(data=True):
            if edge[2]['weight'] == min_similarity_score:
                nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color='red', width=2)

        # Add edge labels (similarity scores)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={(i, j): f"{w:.2f}%" for (i, j), w in edge_labels.items()}
        )

        plt.title("Similarity Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(graph_path)
        #plt.show()
        plt.close()

        top_10_labels = [extracted_text] + [f"Doc {i+1}" for i in range(10)]  # First item is your paragraph, then top 10 docs
        top_10_scores = [similarity_scores_percentage[0]] + similarity_scores_percentage[:10]  # Your paragraph's score first, then top 10 scores

        # Plot the similarity percentages for top 10 documents
        plt.figure(figsize=(12, 6))
        plt.bar(top_10_labels, top_10_scores, color=["red"] + ["skyblue"] * len(similarity_scores_percentage[:10]))  # Red for your paragraph, blue for others
        plt.axhline(y=50, color='red', linestyle='--', label='50% Similarity Threshold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Similarity Percentage')
        plt.xlabel('Document Paragraphs')
        plt.title('Similarity Percentage Difference (Your Paragraph vs Top 10 Embedding Paragraphs)')
        plt.legend()
        plt.tight_layout()

        # Save similarity plot
        similarity_plot_path = os.path.join('static', 'plots', 'similarity_percentage_plot.png')
        plt.savefig(similarity_plot_path)
        plt.close()

        # Generate graph density plot if needed
        density_plot_path = plot_graph_density(G, similarity_scores_percentage, document_labels)

        return render_template('dashboard.html', plot_image='plots/similarity_plot.png', graph_image='plots/similarity_graph.png', density_image='plots/graph_density_plot.png', maxscore=maxscore)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(os.path.join('static', 'plots'), filename)


database = "database.db"
conn = sqlite3.connect(database)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS register (id integer primary key autoincrement, name TEXT, email TEXT, password TEXT)")
conn.commit()

@app.route('/')
def home():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(database)
        cur = conn.cursor()

        cur.execute("select * from register where email = ? ",(email,))
        existing_user = cur.fetchone()
        if existing_user:
            message='Email Already Registerd'
            return render_template('register.html', message=message)
        
        cur.execute("INSERT INTO register (name, email, password) VALUES (?, ?, ?)", (name, email, password))
       
        conn.commit()
        conn.close()
        return render_template('login.html')
    return render_template('register.html')

@app.route('/loginpage')
def loginpage():
    return render_template('login.html')

@app.route('/registerpage')
def registerpage():
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect(database)
        cur = conn.cursor()
        cur.execute("SELECT * FROM register WHERE email = ? AND password = ?", (email, password))
        data = cur.fetchone()
        if data:
            
            return render_template("dashboard.html")
        else:
            message= 'Incorrect Email or Password'
            return render_template("login.html", message=message)
    return render_template('register.html')

if __name__ == '__main__':
    app.run(port=800, threaded=False)

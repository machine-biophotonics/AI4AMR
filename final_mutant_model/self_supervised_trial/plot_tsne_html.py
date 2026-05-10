#!/usr/bin/env python3
"""
Generate interactive HTML t-SNE plot for self-supervised embeddings.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='embeddings_P1.csv', help='Input CSV')
    parser.add_argument('--output', type=str, default='tsne_plot.html', help='Output HTML')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity')
    parser.add_argument('--n_iter', type=int, default=1000, help='t-SNE iterations')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'self_supervised_trial', args.input)
    output_path = os.path.join(script_dir, 'self_supervised_trial', args.output)
    
    print(f"Loading embeddings from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Get data type column
    data_types = df['data_type'].values
    image_names = df['image_name'].values
    wells = df['well'].values
    
    # Get embedding columns
    embed_cols = [c for c in df.columns if c.startswith('emb_')]
    embeddings = df[embed_cols].values
    
    print(f"Shape: {embeddings.shape}")
    print(f"Drugs: {(data_types == 'drug').sum()}, Mutants: {(data_types == 'mutant').sum()}")
    
    # Standardize embeddings
    print("Standardizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Run t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity}, n_iter={args.n_iter})...")
    tsne = TSNE(
        n_components=2, 
        perplexity=args.perplexity, 
        max_iter=args.n_iter,
        random_state=args.random_state,
        init='pca'
    )
    embeddings_2d = tsne.fit_transform(embeddings_scaled)
    
    # Create data for HTML
    print("Generating HTML...")
    
    drug_mask = data_types == 'drug'
    mutant_mask = data_types == 'mutant'
    
    # Split data by type
    drug_x = embeddings_2d[drug_mask, 0].tolist()
    drug_y = embeddings_2d[drug_mask, 1].tolist()
    drug_names = image_names[drug_mask].tolist()
    drug_wells = wells[drug_mask].tolist()
    
    mutant_x = embeddings_2d[mutant_mask, 0].tolist()
    mutant_y = embeddings_2d[mutant_mask, 1].tolist()
    mutant_names = image_names[mutant_mask].tolist()
    mutant_wells = wells[mutant_mask].tolist()
    
    # Generate HTML with Plotly
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>t-SNE Visualization - Self-Supervised Embeddings</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 20px;
            color: #666;
        }}
        #plot {{
            width: 100%;
            height: 800px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>t-SNE Visualization - Self-Supervised Embeddings</h1>
        <div class="stats">
            <strong>Total:</strong> {len(data_types)} images | 
            <strong style="color: red;">Drugs:</strong> {drug_mask.sum()} | 
            <strong style="color: blue;">Mutants:</strong> {mutant_mask.sum()} | 
            <strong>Perplexity:</strong> {args.perplexity}
        </div>
        <div id="plot"></div>
    </div>
    
    <script>
        var drug_x = {json.dumps(drug_x)};
        var drug_y = {json.dumps(drug_y)};
        var drug_names = {json.dumps(drug_names)};
        var drug_wells = {json.dumps(drug_wells)};
        
        var mutant_x = {json.dumps(mutant_x)};
        var mutant_y = {json.dumps(mutant_y)};
        var mutant_names = {json.dumps(mutant_names)};
        var mutant_wells = {json.dumps(mutant_wells)};
        
        var drug_trace = {{
            x: drug_x,
            y: drug_y,
            mode: 'markers',
            type: 'scattergl',
            name: 'Drug',
            marker: {{
                size: 6,
                color: 'red',
                opacity: 0.6
            }},
            text: drug_names.map((name, i) => name + '<br>Well: ' + drug_wells[i]),
            hoverinfo: 'text'
        }};
        
        var mutant_trace = {{
            x: mutant_x,
            y: mutant_y,
            mode: 'markers',
            type: 'scattergl',
            name: 'Mutant',
            marker: {{
                size: 6,
                color: 'blue',
                opacity: 0.6
            }},
            text: mutant_names.map((name, i) => name + '<br>Well: ' + mutant_wells[i]),
            hoverinfo: 'text'
        }};
        
        var layout = {{
            xaxis: {{
                title: 't-SNE 1',
                showgrid: true,
                gridcolor: '#eee'
            }},
            yaxis: {{
                title: 't-SNE 2',
                showgrid: true,
                gridcolor: '#eee'
            }},
            hovermode: 'closest',
            showlegend: true,
            legend: {{
                x: 1,
                y: 1
            }},
            margin: {{t: 50, r: 50, b: 50, l: 50}}
        }};
        
        Plotly.newPlot('plot', [drug_trace, mutant_trace], layout, {{responsive: true}});
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
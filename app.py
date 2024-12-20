import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from werkzeug.utils import secure_filename
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

# Get the absolute path of the current directory
base_dir = os.path.abspath(os.path.dirname(__file__))

# Create the Flask app with explicit template and static folders
app = Flask(__name__,
            template_folder=os.path.join(base_dir, 'templates'),
            static_folder=os.path.join(base_dir, 'static'))

# Configure upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
current_df = None
numerical_columns = []
categorical_columns = []
preprocessing_history = []  # Track preprocessing steps
plot_cache = {}
dashboard_cache = {}

@app.route('/')
def index():
    app.logger.info('Index route accessed')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_df
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'})
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_df = pd.read_csv(filepath)
        return jsonify({
            'message': 'File uploaded successfully',
            'columns': current_df.columns.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/classify_columns', methods=['POST'])
def classify_columns():
    global current_df, numerical_columns, categorical_columns
    if current_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    try:
        threshold = request.json.get('threshold', 10)
        numerical_columns = []
        categorical_columns = []
        
        for column in current_df.columns:
            if current_df[column].dtype in ['int64', 'float64']:
                if current_df[column].nunique() > threshold:
                    numerical_columns.append(column)
                else:
                    categorical_columns.append(column)
            else:
                categorical_columns.append(column)
        
        return jsonify({
            'numerical_columns': numerical_columns,
            'categorical_columns': categorical_columns
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/summary')
def get_summary():
    global current_df
    if current_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    try:
        summary = {
            'dataset_info': {
                'rows': len(current_df),
                'columns': len(current_df.columns),
                'memory_usage': f"{current_df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            },
            'column_summary': {}
        }
        
        for column in current_df.columns:
            missing_count = current_df[column].isnull().sum()
            missing_percentage = (missing_count / len(current_df)) * 100
            
            col_summary = {
                'dtype': str(current_df[column].dtype),
                'missing_values': int(missing_count),
                'missing_percentage': f"{missing_percentage:.2f}%",
                'unique_values': int(current_df[column].nunique())
            }
            
            if column in numerical_columns:
                col_summary.update({
                    'mean': f"{float(current_df[column].mean()):.2f}",
                    'std': f"{float(current_df[column].std()):.2f}",
                    'min': f"{float(current_df[column].min()):.2f}",
                    'max': f"{float(current_df[column].max()):.2f}",
                    '25%': f"{float(current_df[column].quantile(0.25)):.2f}",
                    '50%': f"{float(current_df[column].quantile(0.50)):.2f}",
                    '75%': f"{float(current_df[column].quantile(0.75)):.2f}"
                })
            elif column in categorical_columns:
                value_counts = current_df[column].value_counts()
                col_summary['top_categories'] = {
                    str(k): int(v) for k, v in value_counts.head(5).items()
                }
            
            summary['column_summary'][column] = col_summary
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/plot', methods=['POST'])
def create_plot():
    global current_df, plot_cache
    if current_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    try:
        plot_type = request.json.get('plot_type')
        x_column = request.json.get('x_column')
        y_column = request.json.get('y_column')
        
        # Create cache key
        cache_key = f"{plot_type}_{x_column}_{y_column}"
        
        # Check cache first
        if cache_key in plot_cache:
            return jsonify({'plot': plot_cache[cache_key]})
        
        fig = None
        if plot_type == 'histogram':
            fig = px.histogram(current_df, x=x_column, title=f'Histogram of {x_column}')
        elif plot_type == 'bar':
            if x_column in categorical_columns:
                counts = current_df[x_column].value_counts()
                fig = px.bar(x=counts.index, y=counts.values, title=f'Bar Chart of {x_column}')
            else:
                return jsonify({'error': 'Bar charts are only for categorical columns'})
        elif plot_type == 'scatter':
            if x_column in numerical_columns and y_column in numerical_columns:
                fig = px.scatter(current_df, x=x_column, y=y_column, 
                               title=f'Scatter Plot: {x_column} vs {y_column}')
            else:
                return jsonify({'error': 'Scatter plots require numerical columns'})
        elif plot_type == 'box':
            if x_column in numerical_columns:
                fig = px.box(current_df, y=x_column, title=f'Box Plot of {x_column}')
            else:
                return jsonify({'error': 'Box plots require numerical columns'})
        
        plot_json = fig.to_json()
        plot_cache[cache_key] = plot_json
        return jsonify({'plot': plot_json})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/handle_missing', methods=['POST'])
def handle_missing_values():
    global current_df
    if current_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    try:
        columns = request.json.get('columns')
        strategy = request.json.get('strategy')
        
        if strategy == 'drop':
            current_df.dropna(subset=columns, inplace=True)
        else:
            for column in columns:
                if column in numerical_columns:
                    imputer = SimpleImputer(strategy=strategy)
                    current_df[column] = imputer.fit_transform(current_df[[column]])
                elif column in categorical_columns and strategy == 'most_frequent':
                    imputer = SimpleImputer(strategy='most_frequent')
                    current_df[column] = imputer.fit_transform(current_df[[column]])
        
        return jsonify({'message': 'Missing values handled successfully'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/handle_outliers', methods=['POST'])
def handle_outliers():
    global current_df, preprocessing_history
    if current_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    try:
        columns = request.json.get('columns')
        method = request.json.get('method')
        treatment = request.json.get('treatment', 'replace')  # 'replace' or 'cap'
        
        affected_rows = 0
        for column in columns:
            if column not in numerical_columns:
                continue
                
            if method == 'zscore':
                z_scores = np.abs((current_df[column] - current_df[column].mean()) / current_df[column].std())
                outliers = z_scores > 3
                affected_rows += outliers.sum()
                
                if treatment == 'replace':
                    current_df[column] = current_df[column].mask(outliers, current_df[column].mean())
                else:  # cap
                    mean = current_df[column].mean()
                    std = current_df[column].std()
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    current_df[column] = current_df[column].clip(lower_bound, upper_bound)
                    
            elif method == 'iqr':
                Q1 = current_df[column].quantile(0.25)
                Q3 = current_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ~current_df[column].between(lower_bound, upper_bound)
                affected_rows += outliers.sum()
                
                if treatment == 'replace':
                    current_df[column] = current_df[column].mask(outliers, current_df[column].median())
                else:  # cap
                    current_df[column] = current_df[column].clip(lower_bound, upper_bound)
        
        preprocessing_history.append({
            'action': 'outlier_treatment',
            'method': method,
            'treatment': treatment,
            'columns': columns,
            'affected_rows': int(affected_rows)
        })
        
        return jsonify({
            'message': f'Outliers handled successfully. Affected rows: {affected_rows}',
            'affected_rows': int(affected_rows)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/encode_categorical', methods=['POST'])
def encode_categorical():
    global current_df, preprocessing_history
    if current_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    try:
        columns = request.json.get('columns')
        method = request.json.get('method')  # 'label', 'ordinal', or 'onehot'
        
        encoded_cols = []
        for column in columns:
            if column not in categorical_columns:
                continue
                
            if method == 'label':
                le = LabelEncoder()
                current_df[f'{column}_encoded'] = le.fit_transform(current_df[column])
                encoded_cols.append(f'{column}_encoded')
                
            elif method == 'ordinal':
                oe = OrdinalEncoder()
                current_df[f'{column}_encoded'] = oe.fit_transform(current_df[[column]])
                encoded_cols.append(f'{column}_encoded')
                
            elif method == 'onehot':
                ohe = OneHotEncoder(sparse=False)
                encoded_data = ohe.fit_transform(current_df[[column]])
                encoded_df = pd.DataFrame(encoded_data, columns=[f'{column}_{cat}' for cat in ohe.categories_[0]])
                current_df = pd.concat([current_df, encoded_df], axis=1)
                encoded_cols.extend(encoded_df.columns.tolist())
        
        preprocessing_history.append({
            'action': 'categorical_encoding',
            'method': method,
            'columns': columns,
            'encoded_columns': encoded_cols
        })
        
        return jsonify({
            'message': f'Categorical encoding ({method}) applied successfully',
            'encoded_columns': encoded_cols
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/preprocessing_history')
def get_preprocessing_history():
    return jsonify({'history': preprocessing_history})

@app.route('/correlation')
def get_correlation():
    global current_df
    if current_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    try:
        num_df = current_df[numerical_columns]
        if len(numerical_columns) < 2:
            return jsonify({'error': 'Need at least 2 numerical columns for correlation'})
        
        corr_matrix = num_df.corr()
        fig = px.imshow(corr_matrix,
                       labels=dict(x="Features", y="Features", color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       title="Correlation Heatmap")
        
        return jsonify({'plot': fig.to_json()})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/dashboard')
def get_dashboard():
    global current_df, dashboard_cache
    if current_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    try:
        # Create dashboard with essential plots only
        dashboard_data = {
            'summary': [],
            'missing_values': {},
            'numerical_dist': {},
            'categorical_dist': {}
        }
        
        # Basic dataset info
        dashboard_data['dataset_info'] = {
            'rows': len(current_df),
            'columns': len(current_df.columns),
            'numerical_cols': len(numerical_columns),
            'categorical_cols': len(categorical_columns)
        }
        
        # Missing values summary (faster than visualization)
        missing_data = current_df.isnull().sum()
        dashboard_data['missing_values'] = {
            'columns': missing_data.index.tolist(),
            'values': missing_data.values.tolist()
        }
        
        # Process only top 5 numerical columns
        for column in numerical_columns[:5]:
            try:
                # Calculate basic stats
                stats = current_df[column].describe()
                dashboard_data['numerical_dist'][column] = {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'q1': float(stats['25%']),
                    'q2': float(stats['50%']),
                    'q3': float(stats['75%'])
                }
                
                # Create simplified histogram data
                hist_data, bin_edges = np.histogram(current_df[column].dropna(), bins=20)
                dashboard_data['numerical_dist'][column]['histogram'] = {
                    'counts': hist_data.tolist(),
                    'bins': bin_edges.tolist()
                }
            except Exception as e:
                continue
        
        # Process only top 5 categorical columns
        for column in categorical_columns[:5]:
            try:
                value_counts = current_df[column].value_counts().head(10)
                dashboard_data['categorical_dist'][column] = {
                    'categories': value_counts.index.tolist(),
                    'counts': value_counts.values.tolist()
                }
            except Exception as e:
                continue
        
        return jsonify(dashboard_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download')
def download_data():
    global current_df
    if current_df is None:
        return jsonify({'error': 'No data uploaded'})
    
    try:
        output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv')
        current_df.to_csv(output_file, index=False)
        return send_file(output_file, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Static folder:", app.static_folder)
    print("Template folder:", app.template_folder)
    print("Base directory:", base_dir)
    print("Server will be available at: http://127.0.0.1:8000")
    app.run(host='127.0.0.1', port=8000, debug=True)

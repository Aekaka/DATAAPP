import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import time
from datetime import datetime
import io
import base64
import openai
import os
from sklearn.feature_selection import mutual_info_classif
from supabase import create_client, Client
from openai import OpenAI
import json
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import tempfile

# Set page configuration
st.set_page_config(
    page_title="🔍 Advanced Data Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        margin-right: 1rem;
        border-radius: 50%;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# Supabase configuration
SUPABASE_URL = "https://bwngewxnlsiqmgffdrkz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ3bmdld3hubHNpcW1nZmZkcmt6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0NjUwNzcsImV4cCI6MjA2NTA0MTA3N30.UptU1TZyyUOl62lPdqjGPftDXt3cKqFxsOU00hcuy6A"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_message(role, content, avatar):
    """Display a chat message with the specified role and content."""
    with st.container():
        col1, col2 = st.columns([1, 11])
        with col1:
            st.image(avatar, width=50)
        with col2:
            st.markdown(f"**{role}**")
            st.markdown(content)
        st.markdown("---")

def get_csv_download_link(df, filename="data.csv"):
    """Generate a download link for the dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Processed Data</a>'
    return href

def validate_data(df):
    """Validate and clean the uploaded data."""
    issues = []
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append("Missing values detected in columns: " + 
                     ", ".join(missing[missing > 0].index.tolist()))
    
    # Check for infinite values
    inf_cols = df.select_dtypes(include=np.number).columns
    inf_count = np.isinf(df[inf_cols]).sum().sum()
    if inf_count > 0:
        issues.append(f"Found {inf_count} infinite values")
    
    return issues

@st.cache_data
def load_data(file):
    """Load and cache data from uploaded file."""
    try:
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file, low_memory=False)
        elif file_extension == 'xlsx':
            df = pd.read_excel(file)
        elif file_extension == 'json':
            df = pd.read_json(file)
        elif file_extension == 'parquet':
            df = pd.read_parquet(file)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
        
        # Basic cleaning
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Convert date columns to datetime
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # First check if the column contains any non-null values
                    if df[col].notna().any():
                        # Try to convert to datetime
                        pd.to_datetime(df[col], errors='coerce')
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    # If conversion fails, keep as object type
                    pass
        
        # Ensure all numeric columns are properly typed
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Try to convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # If more than 50% of values are numeric, convert the column
                if numeric_series.notna().mean() > 0.5:
                    df[col] = numeric_series
            except:
                pass
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess the data for analysis."""
    df_processed = df.copy()
    
    # Handle missing values
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Fill numerical missing values with median
    for col in numerical_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    return df_processed

def generate_advanced_insights(df):
    """Generate comprehensive insights about the dataset."""
    insights = []
    
    # Basic Statistics
    insights.append("## 📊 Dataset Overview")
    insights.append(f"- Records: {df.shape[0]:,}")
    insights.append(f"- Features: {df.shape[1]:,}")
    insights.append(f"- Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Data Quality
    missing_cells = df.isnull().sum().sum()
    total_cells = df.size
    data_completeness = ((total_cells - missing_cells) / total_cells) * 100
    insights.append(f"\nData Quality Score: {data_completeness:.1f}%")
    
    # Column Types Analysis
    type_counts = df.dtypes.value_counts()
    insights.append("\n## 📋 Data Types")
    for dtype, count in type_counts.items():
        insights.append(f"- {count} {dtype} columns")
    
    # Numerical Analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        insights.append("\n## 📈 Numerical Features")
        for col in numerical_cols:
            stats = df[col].describe()
            insights.append(f"\n### {col}")
            insights.append(f"- Range: {stats['min']:,.2f} to {stats['max']:,.2f}")
            insights.append(f"- Mean: {stats['mean']:,.2f}")
            insights.append(f"- Median: {stats['50%']:,.2f}")
            insights.append(f"- Standard Deviation: {stats['std']:,.2f}")
            
            # Skewness
            skew = df[col].skew()
            insights.append(f"- Skewness: {skew:.2f}")
            if abs(skew) > 1:
                insights.append("  ⚠️ Distribution is highly skewed")
    
    # Categorical Analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        insights.append("\n## 📊 Categorical Features")
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            unique_count = len(value_counts)
            insights.append(f"\n### {col}")
            insights.append(f"- {unique_count:,} unique values")
            if unique_count < 10:
                for val, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    insights.append(f"  - {val}: {count:,} ({percentage:.1f}%)")
    
    # Correlation Analysis
    if len(numerical_cols) >= 2:
        insights.append("\n## 🔄 Feature Relationships")
        corr_matrix = df[numerical_cols].corr()
        high_corr = np.where(np.abs(corr_matrix) > 0.7)
        high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
                          for x, y in zip(*high_corr) if x != y and x < y]
        
        if high_corr_pairs:
            insights.append("\nStrong correlations found:")
            for col1, col2, corr in high_corr_pairs:
                insights.append(f"- {col1} ↔️ {col2}: {corr:.2f}")
    
    return "\n".join(insights)

def create_advanced_visualizations(df):
    """Create comprehensive visualizations for the dataset."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    st.subheader("📊 Data Visualization Suite")
    
    # Distribution Analysis
    if len(numerical_cols) > 0:
        st.subheader("Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_col = st.selectbox("Select column for analysis", numerical_cols)
            fig = px.histogram(df, x=selected_col, 
                             title=f"Distribution of {selected_col}",
                             marginal="box")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Q-Q plot
            fig = px.scatter(x=np.sort(df[selected_col]),
                           y=np.quantile(np.random.normal(size=len(df)), 
                                       np.linspace(0, 1, len(df))),
                           title=f"Q-Q Plot of {selected_col}")
            fig.add_shape(type='line',
                         x0=df[selected_col].min(),
                         y0=df[selected_col].min(),
                         x1=df[selected_col].max(),
                         y1=df[selected_col].max(),
                         line=dict(color='red', dash='dash'))
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    if len(numerical_cols) >= 2:
        st.subheader("Correlation Analysis")
        corr_matrix = df[numerical_cols].corr()
        
        # Heatmap
        fig = px.imshow(corr_matrix,
                       title="Correlation Heatmap",
                       color_continuous_scale="RdBu",
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter Matrix
        if len(numerical_cols) <= 4:  # Limit to avoid overcrowding
            fig = px.scatter_matrix(df[numerical_cols],
                                  title="Scatter Matrix",
                                  dimensions=numerical_cols)
            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical Analysis
    if len(categorical_cols) > 0:
        st.subheader("Categorical Analysis")
        selected_cat = st.selectbox("Select categorical column", categorical_cols)
        
        # Bar chart with count and percentage
        value_counts = df[selected_cat].value_counts()
        percentages = (value_counts / len(df)) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            name='Count',
            text=value_counts.values,
            textposition='auto',
        ))
        
        fig.add_trace(go.Scatter(
            x=value_counts.index,
            y=percentages,
            name='Percentage',
            yaxis='y2',
            text=[f'{p:.1f}%' for p in percentages],
            mode='lines+markers+text',
            textposition='top center'
        ))
        
        fig.update_layout(
            title=f"Distribution of {selected_cat}",
            yaxis=dict(title='Count'),
            yaxis2=dict(title='Percentage', overlaying='y', side='right'),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def train_model(df, target_column, problem_type='classification', test_size=0.2):
    """Train a machine learning model on the dataset."""
    try:
        # Verify target column exists
        if target_column not in df.columns:
            return {
                'error': f"Target column '{target_column}' not found in the dataset"
            }

        # Prepare the data
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle categorical variables in features
        X_processed = X.copy()
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            X_processed[col] = label_encoders[col].fit_transform(X_processed[col].astype(str))
        
        # Convert all columns to numeric, replacing non-numeric with NaN
        for col in X_processed.columns:
            pd.to_numeric(X_processed[col], errors='coerce')
        
        # Fill NaN values with 0 (you might want to use mean or median instead)
        X_processed = X_processed.fillna(0)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=42)
        
        # Train the model
        if problem_type == 'classification':
            # Convert target to numeric for classification
            if y.dtype == 'object':
                y_encoder = LabelEncoder()
                y_train = y_encoder.fit_transform(y_train)
                y_test = y_encoder.transform(y_test)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            return {
                'success': True,
                'model': model,
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'feature_importance': pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            }
        else:  # regression
            # Verify target is numeric
            if not np.issubdtype(y.dtype, np.number):
                return {
                    'error': f"Target column '{target_column}' must contain numeric values for regression"
                }
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'success': True,
                'model': model,
                'mse': mse,
                'r2': r2,
                'feature_importance': pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            }
    except Exception as e:
        return {
            'error': f"An error occurred while training the model: {str(e)}"
        }

def perform_clustering(df, n_clusters=3):
    """Perform K-means clustering on the dataset."""
    # Prepare the data
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) < 2:
        return None
    
    X = df[numerical_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    return {
        'pca_components': X_pca,
        'clusters': clusters,
        'explained_variance': pca.explained_variance_ratio_
    }

def get_ai_response(question, df):
    """Get a response from the OpenAI GPT model based on the user's question and the dataset."""
    prompt = f"Given the following dataset:\n{df.head().to_string()}\n\nAnswer the question: {question}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Change to a model you have access to
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    # Move inputs to CPU
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    outputs = model(**inputs)
    answer_start = outputs.start_logits.argmax().item()
    answer_end = outputs.end_logits.argmax().item()
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end + 1]
        )
    )
    return answer

def main():
    st.title("🤖 Advanced Data Explorer")
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)
    st.markdown("""
    <span style='font-size:1.2em;'>Welcome to the Advanced Data Explorer! Upload your data file and let's discover insights together.<br>
    This tool provides comprehensive analysis, visualization, and machine learning capabilities.<br>
    <b>All features are accessible and user-friendly.</b></span>
    """, unsafe_allow_html=True)
    
    # Add a skip link for accessibility
    st.markdown('<a href="#main-content" class="skip-link" style="position:absolute;left:-10000px;top:auto;width:1px;height:1px;overflow:hidden;">Skip to Main Content</a>', unsafe_allow_html=True)
    
    # Sidebar for settings and options
    with st.sidebar:
        st.header("⚙️ Settings")
        show_advanced = st.checkbox("Show Advanced Options", value=False, help="Show or hide advanced data processing options.")
        if show_advanced:
            st.subheader("Data Processing")
            handle_missing = st.selectbox(
                "Handle Missing Values",
                ["median/mode", "mean", "drop", "none"],
                help="Choose how to handle missing values in the dataset."
            )
            remove_outliers = st.checkbox(
                "Remove Outliers",
                help="Automatically remove statistical outliers from numerical columns."
            )
    
    # --- Supabase Authentication UI ---
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    auth_mode = st.sidebar.radio("Authentication", ["Login", "Sign Up"], help="Sign in or create a new account.")
    email = st.sidebar.text_input("Email", help="Enter your email address.")
    password = st.sidebar.text_input("Password", type="password", help="Enter your password.")
    
    if st.session_state['user']:
        try:
            user_email = st.session_state['user'].email
            st.sidebar.success(f"Logged in as: {user_email}")
            if st.sidebar.button("Log Out"):
                st.session_state['user'] = None
                st.rerun()
        except AttributeError:
            st.sidebar.error("Session error. Please log in again.")
            st.session_state['user'] = None
            st.rerun()
    else:
        if auth_mode == "Login":
            if st.sidebar.button("Login"):
                if not email or not password:
                    st.sidebar.error("Please enter both email and password.")
                else:
                    try:
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        if res.user:
                            st.session_state['user'] = res.user
                            st.success("Logged in successfully!")
                            st.rerun()
                        else:
                            st.error("Login failed. Please check your credentials.")
                    except Exception as e:
                        st.sidebar.error(f"Login error: {str(e)}")
        else:
            if st.sidebar.button("Sign Up"):
                if not email or not password:
                    st.sidebar.error("Please enter both email and password.")
                elif len(password) < 6:
                    st.sidebar.error("Password must be at least 6 characters long.")
                else:
                    try:
                        res = supabase.auth.sign_up({"email": email, "password": password})
                        if res.user:
                            st.session_state['user'] = res.user
                            st.sidebar.success("Sign up successful! You are now logged in.")
                            st.rerun()
                        else:
                            st.sidebar.error("Sign up failed. Try a different email.")
                    except Exception as e:
                        st.sidebar.error(f"Sign up error: {str(e)}")
    if not st.session_state['user']:
        st.warning("Please log in or sign up to use the app.")
        st.stop()
    
    # Main content
    uploaded_file = st.file_uploader(
        "📤 Upload your data file",
        type=['csv', 'xlsx', 'json', 'parquet'],
        help="Supported formats: CSV, Excel, JSON, Parquet. Upload your main dataset here."
    )

    # --- Dataset Storage & Retrieval ---
    if uploaded_file is not None:
        try:
            # Save file to Supabase Storage
            user_id = st.session_state['user'].id
            file_path = f"{user_id}/{uploaded_file.name}"
            file_bytes = uploaded_file.getvalue()
            
            # Upload to storage
            storage_response = supabase.storage.from_('datasets').upload(
                file_path,
                file_bytes
            )
            
            if storage_response:
                # Save metadata to dataset_files table
                data = {
                    'user_id': user_id,
                    'filename': uploaded_file.name,
                    'upload_date': datetime.now().isoformat(),
                    'storage_path': file_path
                }
                
                response = supabase.table('dataset_files').insert(data).execute()
                
                if response:
                    st.success(f"File '{uploaded_file.name}' uploaded and saved!")
                    st.session_state.uploaded_file = uploaded_file
                else:
                    st.error("Error saving file metadata. Please try again.")
            else:
                st.error("Error uploading file to storage. Please try again.")
            
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            st.info("Please make sure you're logged in and have the correct permissions.")

    # List previous files for this user
    try:
        user_id = st.session_state['user'].id
        response = supabase.table('dataset_files').select('*').eq('user_id', user_id).execute()
        if response.data:
            st.subheader("📂 Your Uploaded Datasets")
            files = [file['filename'] for file in response.data]
            selected_file = st.selectbox("Select a file to load", files)
            
            if selected_file:
                file_data = next((f for f in response.data if f['filename'] == selected_file), None)
                if file_data:
                    try:
                        file_bytes = supabase.storage.from_('datasets').download(file_data['storage_path'])
                        if file_bytes:
                            # Create a temporary file to read with pandas
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{selected_file.split('.')[-1]}") as tmp_file:
                                tmp_file.write(file_bytes)
                                tmp_file.flush()
                                
                                # Read the file based on its extension
                                file_extension = selected_file.split('.')[-1].lower()
                                if file_extension == 'csv':
                                    df = pd.read_csv(tmp_file.name, low_memory=False)
                                elif file_extension == 'xlsx':
                                    df = pd.read_excel(tmp_file.name)
                                elif file_extension == 'json':
                                    df = pd.read_json(tmp_file.name)
                                elif file_extension == 'parquet':
                                    df = pd.read_parquet(tmp_file.name)
                                else:
                                    st.error(f"Unsupported file type: {file_extension}")
                                    return None
                                
                                st.success(f"Loaded '{selected_file}' from your storage!")
                                return df
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
                        return None
    except Exception as e:
        st.error(f"Error retrieving files: {str(e)}")
        return None

    if uploaded_file is None:
        st.info("Upload a new file or select one of your previous files above to get started.")
        st.stop()

    if uploaded_file is not None:
        try:
            with st.spinner('🔄 Reading and analyzing your data...'):
                # Load and validate data
                df = load_data(uploaded_file)
                if df is None:
                    return
                
                # Check for data issues
                issues = validate_data(df)
                if issues:
                    st.warning("⚠️ <b>Data Quality Issues Detected:</b>", icon="⚠️")
                    for issue in issues:
                        st.write(f"- {issue}")
                
                # Preprocess data
                df_processed = preprocess_data(df)
                
                # Display download link for processed data
                st.markdown(get_csv_download_link(df_processed, "processed_data.csv"), unsafe_allow_html=True)
                
                # Display raw data
                with st.expander("🔍 View Raw Data", expanded=False):
                    st.dataframe(df)
                
                # Create tabs
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
                    "📊 Overview",
                    "🔍 Insights",
                    "📈 Visualizations",
                    "🎯 Machine Learning",
                    "🔬 Advanced Analysis",
                    "🤖 Ask the AI",
                    "🔗 Joining",
                    "🔄 Structuring",
                    "🧹 Cleaning & Wrangling",
                    "💡 Insights & Recommendations"
                ])
                
                with tab1:
                    st.header("Dataset Overview")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Records", f"{len(df):,}")
                    with col2:
                        st.metric("Features", f"{df.shape[1]:,}")
                    with col3:
                        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.2f} MB")
                    
                    st.subheader("Data Types")
                    st.dataframe(df.dtypes)
                
                with tab2:
                    st.markdown(generate_advanced_insights(df_processed))
                
                with tab3:
                    create_advanced_visualizations(df_processed)
                
                with tab4:
                    st.header("Machine Learning")
                    
                    # Model selection
                    problem_type = st.selectbox(
                        "Select Problem Type",
                        ['classification', 'regression'],
                        help="Choose the type of machine learning problem"
                    )
                    
                    target_col = st.selectbox("Select Target Variable", df.columns)
                    
                    # Check if target type matches problem type
                    target_is_numeric = np.issubdtype(df[target_col].dtype, np.number)
                    if problem_type == 'classification' and target_is_numeric:
                        st.warning("You selected classification, but your target variable is continuous. Please select regression or choose a categorical target.")
                        if st.button("Switch to Regression"):
                            problem_type = 'regression'
                    elif problem_type == 'regression' and not target_is_numeric:
                        st.warning("You selected regression, but your target variable is categorical. Please select classification or choose a numeric target.")
                        if st.button("Switch to Classification"):
                            problem_type = 'classification'
                    
                    # Suggest relevant features based on correlation with the target
                    if problem_type == 'regression':
                        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
                        suggested_features = correlations.index[1:6].tolist()  # Top 5 correlated features
                    else:
                        # For classification, use mutual information
                        try:
                            mi_scores = mutual_info_classif(df.drop(columns=[target_col]), df[target_col])
                            mi_scores = pd.Series(mi_scores, index=df.drop(columns=[target_col]).columns)
                            suggested_features = mi_scores.sort_values(ascending=False).index[:5].tolist()
                        except Exception as e:
                            st.warning(f"Could not compute feature relevance for classification: {e}")
                            suggested_features = [col for col in df.columns if col != target_col][:5]
                    
                    st.write("Suggested features based on relevance:")
                    st.write(suggested_features)
                    
                    feature_cols = st.multiselect(
                        "Select Features",
                        [col for col in df.columns if col != target_col],
                        default=suggested_features
                    )
                    
                    if st.button("Train Model"):
                        with st.spinner('Training model...'):
                            try:
                                results = train_model(df_processed[feature_cols + [target_col]], 
                                                   target_col,
                                                   problem_type)
                                if 'error' in results:
                                    st.error(results['error'])
                                    # If error is about label type, prompt user
                                    if 'Unknown label type' in results['error']:
                                        st.warning("This error usually means the target variable type does not match the selected problem type. Please check your selections above.")
                                else:
                                    st.success("Model training complete!")
                                    
                                    if problem_type == 'classification':
                                        st.metric("Model Accuracy", f"{results['accuracy']:.2%}")
                                        
                                        # Confusion Matrix
                                        st.subheader("Confusion Matrix")
                                        fig = px.imshow(
                                            results['confusion_matrix'],
                                            labels=dict(x="Predicted", y="Actual"),
                                            title="Confusion Matrix"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.metric("R² Score", f"{results['r2']:.2%}")
                                        st.metric("Mean Squared Error", f"{results['mse']:.4f}")
                                    
                                    # Feature Importance
                                    st.subheader("Feature Importance")
                                    fig = px.bar(
                                        results['feature_importance'],
                                        x='importance', y='feature',
                                        orientation='h',
                                        title="Feature Importance"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Prediction Section
                                    st.subheader("Make Predictions")
                                    input_data = {}
                                    for feature in feature_cols:
                                        input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
                                    
                                    if st.button("Predict"):
                                        input_df = pd.DataFrame([input_data])
                                        prediction = results['model'].predict(input_df)
                                        # Clear, styled output for prediction
                                        if problem_type == 'classification':
                                            st.success(f"<span style='font-size:1.5em;'><b>Predicted Class:</b> {prediction[0]}</span>", icon="🔮", unsafe_allow_html=True)
                                            st.markdown("<span style='color: #555;'>This is the predicted class for your input. For more details, check the confusion matrix and feature importance above.</span>", unsafe_allow_html=True)
                                        else:
                                            st.info(f"<span style='font-size:1.5em;'><b>Predicted Value:</b> {prediction[0]}</span>", icon="📈", unsafe_allow_html=True)
                                            st.markdown("<span style='color: #555;'>This is the predicted value for your input. For more details, check the R² score and feature importance above.</span>", unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"An error occurred: {str(e)}")
                                if 'Unknown label type' in str(e):
                                    st.warning("This error usually means the target variable type does not match the selected problem type. Please check your selections above.")
                
                with tab5:
                    st.header("Advanced Analysis")
                    
                    # Clustering
                    st.subheader("Data Clustering")
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                    
                    if st.button("Perform Clustering"):
                        with st.spinner('Performing clustering analysis...'):
                            clustering_results = perform_clustering(df_processed)
                            if clustering_results is not None:
                                fig = px.scatter(
                                    x=clustering_results['pca_components'][:, 0],
                                    y=clustering_results['pca_components'][:, 1],
                                    color=clustering_results['clusters'],
                                    title="Data Clusters (PCA)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.info(f"Explained variance: {sum(clustering_results['explained_variance']):.1%}")
                            else:
                                st.warning("Insufficient numerical columns for clustering.")
                    
                    # Statistical Tests
                    if len(df.select_dtypes(include=['int64', 'float64']).columns) >= 2:
                        st.subheader("Statistical Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            var1 = st.selectbox("Select first variable", 
                                              df.select_dtypes(include=['int64', 'float64']).columns)
                        with col2:
                            var2 = st.selectbox("Select second variable",
                                              [col for col in df.select_dtypes(include=['int64', 'float64']).columns
                                               if col != var1])
                        
                        correlation = df[var1].corr(df[var2])
                        st.metric("Correlation Coefficient", f"{correlation:.3f}")

                with tab6:  # AI Chat Tab
                    user_question = st.text_input("What would you like to know about your data?")
                    if st.button("Get Answer"):
                        if user_question:
                            with st.spinner('Getting response from AI...'):
                                # Use only a small sample for context
                                context = df_processed.head(10).to_string()
                                try:
                                    ai_response = get_answer(user_question, context)
                                    st.markdown("### AI Response:")
                                    st.write(ai_response)
                                except Exception as e:
                                    st.error(f"An error occurred: {str(e)}")
                        else:
                            st.warning("Please enter a question.")

                with tab7:  # Joining tab
                    st.header("Join Datasets")
                    st.write("Upload a second dataset to join with your main dataset.")
                    join_file = st.file_uploader("Upload second data file", type=['csv', 'xlsx', 'json', 'parquet'], key='join_file')
                    if join_file is not None:
                        join_df = load_data(join_file)
                        st.write("Second dataset preview:")
                        st.dataframe(join_df.head())
                        # Select join keys
                        col1, col2 = st.columns(2)
                        with col1:
                            left_key = st.selectbox("Select join key from main dataset", df.columns)
                        with col2:
                            right_key = st.selectbox("Select join key from second dataset", join_df.columns)
                        join_type = st.selectbox("Select join type", ['inner', 'left', 'right', 'outer'])
                        if st.button("Join Datasets"):
                            try:
                                merged_df = pd.merge(df, join_df, left_on=left_key, right_on=right_key, how=join_type)
                                st.success(f"Datasets joined successfully! Shape: {merged_df.shape}")
                                st.dataframe(merged_df.head())
                                st.markdown(get_csv_download_link(merged_df, "joined_data.csv"), unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error joining datasets: {str(e)}")

                with tab8:  # Structuring tab
                    st.header("Reshape & Structure Data")
                    st.write("Perform pivot, melt (unpivot), and reorder columns.")
                    struct_op = st.selectbox("Select structuring operation", ["Pivot", "Melt", "Reorder Columns"])
                    if struct_op == "Pivot":
                        index_col = st.selectbox("Index column", df.columns)
                        columns_col = st.selectbox("Columns to pivot", [col for col in df.columns if col != index_col])
                        values_col = st.selectbox("Values column", [col for col in df.columns if col not in [index_col, columns_col]])
                        if st.button("Pivot Data"):
                            try:
                                pivoted = df.pivot(index=index_col, columns=columns_col, values=values_col)
                                st.dataframe(pivoted.head())
                                st.markdown(get_csv_download_link(pivoted.reset_index(), "pivoted_data.csv"), unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error in pivot: {str(e)}")
                    elif struct_op == "Melt":
                        id_vars = st.multiselect("ID variables (keep these columns)", df.columns)
                        value_vars = st.multiselect("Value variables (unpivot these columns)", [col for col in df.columns if col not in id_vars])
                        if st.button("Melt Data"):
                            try:
                                melted = pd.melt(df, id_vars=id_vars, value_vars=value_vars)
                                st.dataframe(melted.head())
                                st.markdown(get_csv_download_link(melted, "melted_data.csv"), unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error in melt: {str(e)}")
                    elif struct_op == "Reorder Columns":
                        new_order = st.multiselect("Select new column order", df.columns, default=list(df.columns))
                        if st.button("Reorder Columns"):
                            try:
                                reordered = df[new_order]
                                st.dataframe(reordered.head())
                                st.markdown(get_csv_download_link(reordered, "reordered_data.csv"), unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error in reordering: {str(e)}")

                with tab9:
                    st.header("Advanced Data Cleaning & Wrangling")
                    wrangle_df = df.copy()

                    st.subheader("1. Missing Value Handling")
                    missing_summary = wrangle_df.isnull().sum()
                    st.write("Missing values per column:")
                    st.dataframe(missing_summary[missing_summary > 0])
                    for col in wrangle_df.columns:
                        if wrangle_df[col].isnull().any():
                            st.write(f"Column: {col}")
                            method = st.selectbox(f"How to handle missing values in {col}?", ["Do nothing", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value", "Forward fill", "Backward fill"], key=f"na_{col}")
                            if method == "Drop rows":
                                wrangle_df = wrangle_df.dropna(subset=[col])
                            elif method == "Fill with mean" and wrangle_df[col].dtype in [np.float64, np.int64]:
                                wrangle_df[col] = wrangle_df[col].fillna(wrangle_df[col].mean())
                            elif method == "Fill with median" and wrangle_df[col].dtype in [np.float64, np.int64]:
                                wrangle_df[col] = wrangle_df[col].fillna(wrangle_df[col].median())
                            elif method == "Fill with mode":
                                wrangle_df[col] = wrangle_df[col].fillna(wrangle_df[col].mode()[0])
                            elif method == "Fill with custom value":
                                custom_val = st.text_input(f"Custom value for {col}", key=f"custom_{col}")
                                if custom_val:
                                    wrangle_df[col] = wrangle_df[col].fillna(custom_val)
                            elif method == "Forward fill":
                                wrangle_df[col] = wrangle_df[col].fillna(method='ffill')
                            elif method == "Backward fill":
                                wrangle_df[col] = wrangle_df[col].fillna(method='bfill')

                    st.subheader("2. Duplicate Handling")
                    dup_count = wrangle_df.duplicated().sum()
                    st.write(f"Duplicate rows: {dup_count}")
                    if dup_count > 0:
                        if st.button("Drop Duplicates"):
                            wrangle_df = wrangle_df.drop_duplicates()
                            st.success("Duplicates dropped.")

                    st.subheader("3. Outlier Detection & Handling")
                    num_cols = wrangle_df.select_dtypes(include=[np.number]).columns
                    outlier_method = st.selectbox("Outlier detection method", ["None", "IQR", "Z-score"])
                    if outlier_method != "None":
                        for col in num_cols:
                            st.write(f"Column: {col}")
                            if outlier_method == "IQR":
                                Q1 = wrangle_df[col].quantile(0.25)
                                Q3 = wrangle_df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower = Q1 - 1.5 * IQR
                                upper = Q3 + 1.5 * IQR
                                outliers = (wrangle_df[col] < lower) | (wrangle_df[col] > upper)
                            elif outlier_method == "Z-score":
                                z = (wrangle_df[col] - wrangle_df[col].mean()) / wrangle_df[col].std()
                                outliers = abs(z) > 3
                            st.write(f"Outliers detected: {outliers.sum()}")
                            outlier_action = st.selectbox(f"How to handle outliers in {col}?", ["Do nothing", "Remove", "Cap to bounds"], key=f"out_{col}")
                            if outlier_action == "Remove":
                                wrangle_df = wrangle_df[~outliers]
                            elif outlier_action == "Cap to bounds":
                                if outlier_method == "IQR":
                                    wrangle_df[col] = np.where(wrangle_df[col] < lower, lower, wrangle_df[col])
                                    wrangle_df[col] = np.where(wrangle_df[col] > upper, upper, wrangle_df[col])
                                elif outlier_method == "Z-score":
                                    cap = 3 * wrangle_df[col].std()
                                    wrangle_df[col] = np.where(wrangle_df[col] < -cap, -cap, wrangle_df[col])
                                    wrangle_df[col] = np.where(wrangle_df[col] > cap, cap, wrangle_df[col])

                    st.subheader("4. Data Type Conversion")
                    for col in wrangle_df.columns:
                        dtype = str(wrangle_df[col].dtype)
                        st.write(f"{col}: {dtype}")
                        new_type = st.selectbox(f"Convert {col} to:", [dtype, "int", "float", "str", "datetime"], key=f"dtype_{col}")
                        if new_type != dtype:
                            try:
                                if new_type == "int":
                                    wrangle_df[col] = wrangle_df[col].astype(int)
                                elif new_type == "float":
                                    wrangle_df[col] = wrangle_df[col].astype(float)
                                elif new_type == "str":
                                    wrangle_df[col] = wrangle_df[col].astype(str)
                                elif new_type == "datetime":
                                    wrangle_df[col] = pd.to_datetime(wrangle_df[col], errors='coerce')
                            except Exception as e:
                                st.error(f"Error converting {col}: {str(e)}")

                    st.subheader("5. String Operations")
                    str_cols = wrangle_df.select_dtypes(include=[object]).columns
                    for col in str_cols:
                        st.write(f"Column: {col}")
                        str_op = st.selectbox(f"String operation for {col}", ["None", "Trim whitespace", "To lowercase", "To uppercase", "Remove special characters", "Extract substring"], key=f"str_{col}")
                        if str_op == "Trim whitespace":
                            wrangle_df[col] = wrangle_df[col].str.strip()
                        elif str_op == "To lowercase":
                            wrangle_df[col] = wrangle_df[col].str.lower()
                        elif str_op == "To uppercase":
                            wrangle_df[col] = wrangle_df[col].str.upper()
                        elif str_op == "Remove special characters":
                            wrangle_df[col] = wrangle_df[col].str.replace(r'[^\w\s]', '', regex=True)
                        elif str_op == "Extract substring":
                            start = st.number_input(f"Start index for {col}", min_value=0, value=0, key=f"start_{col}")
                            end = st.number_input(f"End index for {col}", min_value=0, value=5, key=f"end_{col}")
                            wrangle_df[col] = wrangle_df[col].str[start:end]

                    st.subheader("6. Custom Column Creation")
                    new_col_name = st.text_input("New column name")
                    formula = st.text_input("Formula (e.g., col1 + col2 * 2)")
                    if st.button("Create Column") and new_col_name and formula:
                        try:
                            wrangle_df[new_col_name] = eval(formula, {}, wrangle_df)
                            st.success(f"Column '{new_col_name}' created.")
                        except Exception as e:
                            st.error(f"Error creating column: {str(e)}")

                    st.subheader("Preview & Download Cleaned Data")
                    st.dataframe(wrangle_df.head())
                    st.markdown(get_csv_download_link(wrangle_df, "cleaned_data.csv"), unsafe_allow_html=True)

                with tab10:
                    st.header("Automated Insights & Recommendations")
                    # Use cleaned/wrangled data if available, else use df
                    try:
                        wrangle_df
                    except NameError:
                        wrangle_df = df
                    insights = []
                    recommendations = []

                    # 1. High missing values
                    missing = wrangle_df.isnull().mean()
                    high_missing = missing[missing > 0.2]
                    if not high_missing.empty:
                        for col, pct in high_missing.items():
                            insights.append(f"Column '{col}' has {pct:.0%} missing values.")
                            recommendations.append(f"Consider dropping or imputing '{col}'.")

                    # 2. High cardinality categorical columns
                    cat_cols = wrangle_df.select_dtypes(include='object').columns
                    for col in cat_cols:
                        n_unique = wrangle_df[col].nunique()
                        if n_unique > 50:
                            insights.append(f"Column '{col}' has high cardinality ({n_unique} unique values).")
                            recommendations.append(f"Consider encoding or reducing categories in '{col}'.")

                    # 3. Outliers in numeric columns
                    num_cols = wrangle_df.select_dtypes(include=[np.number]).columns
                    for col in num_cols:
                        Q1 = wrangle_df[col].quantile(0.25)
                        Q3 = wrangle_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        outliers = ((wrangle_df[col] < lower) | (wrangle_df[col] > upper)).sum()
                        if outliers > 0:
                            insights.append(f"Column '{col}' has {outliers} outliers (IQR method).")
                            recommendations.append(f"Consider capping or removing outliers in '{col}'.")

                    # 4. Top correlated features (if target exists)
                    if 'target_col' in locals() and target_col in wrangle_df.columns:
                        corr = wrangle_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
                        top_corr = corr.head(3)
                        for col, val in top_corr.items():
                            insights.append(f"'{col}' is strongly correlated with target ({val:.2f}).")
                            recommendations.append(f"Consider using '{col}' as a key feature for prediction.")

                    # 5. Notable trends (time or group based)
                    date_cols = wrangle_df.select_dtypes(include='datetime').columns
                    if len(date_cols) > 0:
                        for col in date_cols:
                            if wrangle_df[col].nunique() > 10:
                                trend = wrangle_df.groupby(wrangle_df[col].dt.year).size()
                                if trend.max() > 2 * trend.min():
                                    insights.append(f"Significant change in data volume over years in '{col}'.")
                                    recommendations.append(f"Investigate reasons for change in '{col}'.")

                    # 6. Clusters or segments (if clustering done)
                    if 'clustering_results' in locals() and 'clusters' in clustering_results:
                        n_clusters = len(set(clustering_results['clusters']))
                        insights.append(f"Data forms {n_clusters} clusters.")
                        recommendations.append(f"Analyze clusters for targeted actions.")

                    # Display insight cards
                    st.subheader("Key Insights")
                    if insights:
                        for i, insight in enumerate(insights):
                            st.info(f"{i+1}. {insight}")
                    else:
                        st.success("No major issues or trends detected.")

                    st.subheader("Actionable Recommendations")
                    if recommendations:
                        for i, rec in enumerate(recommendations):
                            st.warning(f"{i+1}. {rec}")
                    else:
                        st.success("No immediate actions recommended.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.markdown("""
            ### Troubleshooting Tips:
            1. Check if your CSV file is properly formatted
            2. Ensure the file isn't corrupted
            3. Verify the file size is under 200MB
            4. Make sure the data types are consistent
            """)

if __name__ == "__main__":
    openai.api_key = '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'
    main() 

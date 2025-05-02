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
        df = pd.read_csv(file, low_memory=False)
        
        # Basic cleaning
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Convert date columns to datetime
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                    df[col] = pd.to_datetime(df[col])
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

def main():
    st.title("🤖 Advanced Data Explorer")
    st.markdown("""
    Welcome to the Advanced Data Explorer! Upload your CSV file and let's discover insights together.
    This tool provides comprehensive analysis, visualization, and machine learning capabilities.
    """)
    
    # Sidebar for settings and options
    with st.sidebar:
        st.header("⚙️ Settings")
        show_advanced = st.checkbox("Show Advanced Options", value=False)
        if show_advanced:
            st.subheader("Data Processing")
            handle_missing = st.selectbox(
                "Handle Missing Values",
                ["median/mode", "mean", "drop", "none"],
                help="Choose how to handle missing values in the dataset"
            )
            remove_outliers = st.checkbox(
                "Remove Outliers",
                help="Automatically remove statistical outliers from numerical columns"
            )
    
    # Main content
    uploaded_file = st.file_uploader("📤 Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            with st.spinner('Reading and analyzing your data...'):
                # Load and validate data
                df = load_data(uploaded_file)
                if df is None:
                    return
                
                # Check for data issues
                issues = validate_data(df)
                if issues:
                    st.warning("⚠️ Data Quality Issues Detected:")
                    for issue in issues:
                        st.write(f"- {issue}")
                
                # Preprocess data
                df_processed = preprocess_data(df)
                
                # Display download link for processed data
                st.markdown(get_csv_download_link(df_processed, "processed_data.csv"), unsafe_allow_html=True)
                
                # Display raw data
                with st.expander("🔍 View Raw Data"):
                    st.dataframe(df)
                
                # Create tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Overview",
                    "🔍 Insights",
                    "📈 Visualizations",
                    "🎯 Machine Learning",
                    "🔬 Advanced Analysis"
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
                    feature_cols = st.multiselect(
                        "Select Features",
                        [col for col in df.columns if col != target_col],
                        default=[col for col in df.columns if col != target_col]
                    )
                    
                    if st.button("Train Model"):
                        with st.spinner('Training model...'):
                            results = train_model(df_processed[feature_cols + [target_col]], 
                                               target_col,
                                               problem_type)
                            
                            if 'error' in results:
                                st.error(results['error'])
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
    main() 
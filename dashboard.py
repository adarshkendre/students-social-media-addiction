"""
Social Media Usage Analysis Dashboard - Enhanced Version
Hey! This is the beefed-up version with way cooler graphs and more interactivity.
I added 3D plots, animated charts, and some machine learning stuff too!

To get started:
pip install streamlit pandas matplotlib seaborn plotly numpy scipy scikit-learn

Then run it:
streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Alright, let's set up the page with a nice wide layout
st.set_page_config(
    page_title="Social Media Analytics Pro",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make everything look slick
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: white !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
    }
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Big flashy title
st.title("📱 Advanced Social Media Analytics Dashboard")
st.markdown("### Uncovering insights from student social media behavior with data science 🎯")
st.markdown("---")

# Sidebar setup - this is where all the magic controls live
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/instagram-new.png", width=80)
    st.header("🎛️ Control Panel")
    st.markdown("Tweak these to explore different slices of data!")

# Function to whip up some realistic sample data
@st.cache_data
def generate_sample_data():
    """Making up some fake but realistic data for demo"""
    np.random.seed(42)
    n = 500
    
    platforms = ['Instagram', 'TikTok', 'Twitter', 'Facebook', 'YouTube', 'Snapchat']
    genders = ['Male', 'Female', 'Other']
    levels = ['High School', 'Undergraduate', 'Graduate', 'PhD']
    countries = ['USA', 'India', 'UK', 'Canada', 'Australia', 'Germany', 'Japan']
    relationships = ['Single', 'In a relationship', 'Married']
    
    # Making the data somewhat correlated to be realistic
    ages = np.random.randint(16, 31, n)
    usage_hours = np.clip(np.random.normal(5, 2.5, n), 0.5, 12)
    
    # Higher usage = worse mental health (sad but realistic)
    mental_health = np.clip(100 - usage_hours * 5 + np.random.normal(0, 10, n), 30, 100)
    
    # Higher usage = higher addiction score (obviously)
    addiction = np.clip(usage_hours * 8 + np.random.normal(0, 10, n), 20, 100)
    
    # More usage = less sleep (RIP sleep schedule)
    sleep = np.clip(9 - usage_hours * 0.3 + np.random.normal(0, 1, n), 4, 9)
    
    data = {
        'Student_ID': [f'STD{i:04d}' for i in range(1, n+1)],
        'Age': ages,
        'Gender': np.random.choice(genders, n),
        'Academic_Level': np.random.choice(levels, n),
        'Country': np.random.choice(countries, n),
        'Avg_Daily_Usage_Hours': np.round(usage_hours, 1),
        'Most_Used_Platform': np.random.choice(platforms, n),
        'Affects_Academic_Performance': np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4]),
        'Sleep_Hours_Per_Night': np.round(sleep, 1),
        'Mental_Health_Score': mental_health.astype(int),
        'Relationship_Status': np.random.choice(relationships, n),
        'Conflicts_Over_Social_Media': np.random.choice(['Yes', 'No'], n, p=[0.45, 0.55]),
        'Addicted_Score': addiction.astype(int)
    }
    
    return pd.DataFrame(data)

# Loading the actual data
@st.cache_data
def load_data(file):
    """Just loading CSV with caching to make it fast"""
    return pd.read_csv(file)

# Try loading from file first, otherwise create sample data
try:
    df = pd.read_csv('Students Social Media Addiction.csv')
    st.success("✅ Loaded your dataset! Ready to dive in.")
except FileNotFoundError:
    st.warning("⚠️ Dataset not found. Using sample data instead.")
    df = generate_sample_data()

# Quick data peek for the curious
with st.expander("🔍 Peek at the Raw Data", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Features", len(df.columns))

# Now the fun part - FILTERS!
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filter Your Data")

# Gender filter, you can pick all or specific ones
gender_options = ['All'] + df['Gender'].unique().tolist()
gender_selection = st.sidebar.selectbox("Gender", gender_options)
if gender_selection == 'All':
    gender_filter = df['Gender'].unique().tolist()
else:
    gender_filter = [gender_selection]

# Academic level selector
academic_filter = st.sidebar.multiselect(
    "Academic Level",
    options=df['Academic_Level'].unique().tolist(),
    default=df['Academic_Level'].unique().tolist()
)

# Platform picker
platform_filter = st.sidebar.multiselect(
    "Favorite Platform",
    options=df['Most_Used_Platform'].unique().tolist(),
    default=df['Most_Used_Platform'].unique().tolist()
)

# Age range slider
age_range = st.sidebar.slider(
    "Age Range",
    int(df['Age'].min()),
    int(df['Age'].max()),
    (int(df['Age'].min()), int(df['Age'].max()))
)

# Usage hours slider
usage_range = st.sidebar.slider(
    "Daily Usage (hours)",
    float(df['Avg_Daily_Usage_Hours'].min()),
    float(df['Avg_Daily_Usage_Hours'].max()),
    (float(df['Avg_Daily_Usage_Hours'].min()), float(df['Avg_Daily_Usage_Hours'].max()))
)

# Applying all those filters now
filtered_df = df[
    (df['Gender'].isin(gender_filter)) &
    (df['Academic_Level'].isin(academic_filter)) &
    (df['Most_Used_Platform'].isin(platform_filter)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1]) &
    (df['Avg_Daily_Usage_Hours'] >= usage_range[0]) &
    (df['Avg_Daily_Usage_Hours'] <= usage_range[1])
]

# Showing the count of filtered students
st.sidebar.markdown(f"**📊 Showing:** {len(filtered_df)} / {len(df)} students")
if len(filtered_df) < 10:
    st.sidebar.warning("⚠️ Very few records! Try loosening filters.")

# Setting up the tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Overview",
    "👥 Demographics", 
    "⏱️ Usage Patterns",
    "🧠 Mental Health",
    "📚 Academic Impact",
    "🔗 Correlations",
    "🎯 ML Insights",
    "🌍 Geographic View"
])

# Tab 1: Overview of everything
with tab1:
    st.header("📊 Executive Dashboard")
    st.markdown("Your one-stop view of the key metrics")
    
    # Top metrics with cool styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_usage = filtered_df['Avg_Daily_Usage_Hours'].mean()
        st.metric(
            "⏰ Avg Usage", 
            f"{avg_usage:.1f}h",
            delta=f"{avg_usage - df['Avg_Daily_Usage_Hours'].mean():.1f}h",
            help="Average daily social media usage"
        )
    
    with col2:
        avg_sleep = filtered_df['Sleep_Hours_Per_Night'].mean()
        sleep_delta = avg_sleep - df['Sleep_Hours_Per_Night'].mean()
        st.metric(
            "😴 Avg Sleep", 
            f"{avg_sleep:.1f}h",
            delta=f"{sleep_delta:.1f}h",
            delta_color="normal" if sleep_delta > 0 else "inverse",
            help="Average sleep per night"
        )
    
    with col3:
        avg_mental = filtered_df['Mental_Health_Score'].mean()
        st.metric(
            "🧠 Mental Health", 
            f"{avg_mental:.0f}/100",
            delta=f"{avg_mental - df['Mental_Health_Score'].mean():.0f}",
            help="Average mental health score (higher is better)"
        )
    
    with col4:
        avg_addiction = filtered_df['Addicted_Score'].mean()
        addiction_delta = avg_addiction - df['Addicted_Score'].mean()
        st.metric(
            "⚠️ Addiction", 
            f"{avg_addiction:.0f}/100",
            delta=f"{addiction_delta:.0f}",
            delta_color="inverse",
            help="Average addiction score (lower is better)"
        )
    
    with col5:
        affected_pct = (filtered_df['Affects_Academic_Performance'] == 'Yes').sum() / len(filtered_df) * 100
        st.metric(
            "📉 Academic Hit", 
            f"{affected_pct:.1f}%",
            help="% reporting academic impact"
        )
    
    st.markdown("---")
    
    # Animated gauges for the scores
    col1, col2 = st.columns(2)
    
    with col1:
        # Mental health gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_mental,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Mental Health Score", 'font': {'size': 24}},
            delta={'reference': df['Mental_Health_Score'].mean(), 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ffcccc'},
                    {'range': [50, 75], 'color': '#ffffcc'},
                    {'range': [75, 100], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Addiction gauge, colors are flipped since lower is better
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_addiction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Addiction Level", 'font': {'size': 24}},
            delta={'reference': df['Addicted_Score'].mean(), 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkred"},
                'bar': {'color': "darkred"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ccffcc'},
                    {'range': [40, 70], 'color': '#ffffcc'},
                    {'range': [70, 100], 'color': '#ffcccc'}
                ],
                'threshold': {
                    'line': {'color': "darkred", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution charts next to each other
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Usage Distribution")
        # Violin plot to show distributions better
        fig = go.Figure()
        
        for gender in filtered_df['Gender'].unique():
            gender_data = filtered_df[filtered_df['Gender'] == gender]['Avg_Daily_Usage_Hours']
            fig.add_trace(go.Violin(
                y=gender_data,
                name=gender,
                box_visible=True,
                meanline_visible=True,
                fillcolor='lightseagreen' if gender == 'Male' else 'pink',
                opacity=0.6
            ))
        
        fig.update_layout(
            title="Daily Usage Hours by Gender (Violin Plot)",
            yaxis_title="Hours",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Addiction Distribution")
        # 3D histogram for addiction scores
        fig = px.histogram(
            filtered_df,
            x='Addicted_Score',
            color='Gender',
            marginal='box',
            hover_data=filtered_df.columns,
            title="Addiction Score Distribution with Box Plot"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Animated sunburst chart for platform usage
    st.subheader("🌐 Platform Ecosystem")
    
    # Create hierarchical data for sunburst
    sunburst_data = filtered_df.groupby(['Most_Used_Platform', 'Gender']).size().reset_index(name='count')
    
    fig = px.sunburst(
        sunburst_data,
        path=['Most_Used_Platform', 'Gender'],
        values='count',
        title="Platform Usage by Gender (Click to expand!)",
        color='count',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Time-based pattern simulation (if we had timestamps)
    st.subheader("📈 Trend Analysis Simulation")
    # Create fake hourly data for demonstration
    hours = list(range(24))
    usage_pattern = [0.5, 0.3, 0.2, 0.1, 0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 
                     3.0, 2.8, 2.5, 3.5, 4.0, 4.5, 5.0, 5.5, 4.0, 3.0, 2.0, 1.0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=usage_pattern,
        mode='lines+markers',
        name='Usage Pattern',
        line=dict(color='rgb(67, 147, 195)', width=3),
        fill='tozeroy',
        fillcolor='rgba(67, 147, 195, 0.3)'
    ))
    
    fig.update_layout(
        title="Typical Daily Usage Pattern (Simulated)",
        xaxis_title="Hour of Day",
        yaxis_title="Relative Usage",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Demographics, who is using what
with tab2:
    st.header("👥 Deep Dive into Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Pyramid")
        # Create age groups for better visualization
        filtered_df['Age_Group'] = pd.cut(
            filtered_df['Age'],
            bins=[15, 18, 22, 26, 35],
            labels=['16-18', '19-22', '23-26', '27+']
        )
        
        age_gender = filtered_df.groupby(['Age_Group', 'Gender']).size().reset_index(name='count')
        
        fig = px.bar(
            age_gender,
            x='count',
            y='Age_Group',
            color='Gender',
            orientation='h',
            title="Population Pyramid by Age Group",
            barmode='group',
            color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c', 'Other': '#95a5a6'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Country treemap
        st.subheader("🌍 Geographic Distribution")
        country_counts = filtered_df['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        
        fig = px.treemap(
            country_counts,
            path=['Country'],
            values='Count',
            title="Students by Country (Treemap)",
            color='Count',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Academic Breakdown")
        # Animated donut chart
        academic_counts = filtered_df['Academic_Level'].value_counts().reset_index()
        academic_counts.columns = ['Level', 'Count']
        
        fig = go.Figure(data=[go.Pie(
            labels=academic_counts['Level'],
            values=academic_counts['Count'],
            hole=0.5,
            marker=dict(colors=px.colors.qualitative.Pastel),
            textposition='inside',
            textinfo='percent+label'
        )])
        fig.update_layout(
            title="Students by Academic Level",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Relationship status funnel
        st.subheader("❤️ Relationship Status")
        relationship_counts = filtered_df['Relationship_Status'].value_counts().reset_index()
        relationship_counts.columns = ['Status', 'Count']
        
        fig = go.Figure(go.Funnel(
            y=relationship_counts['Status'],
            x=relationship_counts['Count'],
            textinfo="value+percent initial",
            marker=dict(color=["deepskyblue", "lightsalmon", "lightgreen"])
        ))
        fig.update_layout(title="Relationship Status Funnel", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Platform preference matrix
    st.subheader("📱 Platform Preferences Heatmap")
    platform_demo = pd.crosstab(
        [filtered_df['Gender'], filtered_df['Academic_Level']], 
        filtered_df['Most_Used_Platform']
    )
    
    fig = px.imshow(
        platform_demo,
        labels=dict(x="Platform", y="Gender + Level", color="Count"),
        title="Platform Usage Patterns (Gender x Academic Level)",
        color_continuous_scale='Viridis',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D scatter of demographics
    st.subheader("🎨 3D Demographics Explorer")
    fig = px.scatter_3d(
        filtered_df.sample(min(200, len(filtered_df))),  # Sample for performance
        x='Age',
        y='Avg_Daily_Usage_Hours',
        z='Mental_Health_Score',
        color='Gender',
        size='Addicted_Score',
        hover_data=['Academic_Level', 'Most_Used_Platform'],
        title="3D View: Age x Usage x Mental Health (hover for details!)",
        opacity=0.7
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Usage patterns, when where how much
with tab3:
    st.header("⏱️ Usage Pattern Deep Dive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Usage by Academic Level")
        # Ridge plot style visualization
        fig = go.Figure()
        
        for level in filtered_df['Academic_Level'].unique():
            level_data = filtered_df[filtered_df['Academic_Level'] == level]['Avg_Daily_Usage_Hours']
            fig.add_trace(go.Violin(
                x=level_data,
                name=level,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            title="Usage Distribution by Academic Level",
            xaxis_title="Daily Hours",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Platform comparison
        st.subheader("Platform Usage Leaders")
        platform_stats = filtered_df.groupby('Most_Used_Platform').agg({
            'Avg_Daily_Usage_Hours': ['mean', 'median', 'max']
        }).round(2)
        platform_stats.columns = ['Mean', 'Median', 'Max']
        platform_stats = platform_stats.sort_values('Mean', ascending=False).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Mean',
            x=platform_stats['Most_Used_Platform'],
            y=platform_stats['Mean'],
            marker_color='indianred'
        ))
        fig.add_trace(go.Bar(
            name='Median',
            x=platform_stats['Most_Used_Platform'],
            y=platform_stats['Median'],
            marker_color='lightsalmon'
        ))
        fig.update_layout(
            barmode='group',
            title="Average Usage by Platform",
            yaxis_title="Hours",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Usage vs Sleep Relationship")
        # Advanced scatter with density contours
        fig = px.density_contour(
            filtered_df,
            x='Avg_Daily_Usage_Hours',
            y='Sleep_Hours_Per_Night',
            color='Gender',
            title="Usage vs Sleep (with density contours)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Age trends
        st.subheader("Age Trends")
        age_usage = filtered_df.groupby('Age')['Avg_Daily_Usage_Hours'].agg(['mean', 'std']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=age_usage['Age'],
            y=age_usage['mean'],
            mode='lines+markers',
            name='Mean Usage',
            line=dict(color='rgb(31, 119, 180)'),
            error_y=dict(
                type='data',
                array=age_usage['std'],
                visible=True
            )
        ))
        fig.update_layout(
            title="Usage Trends by Age (with std dev)",
            xaxis_title="Age",
            yaxis_title="Hours",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Usage intensity heatmap by multiple factors
    st.subheader("🔥 Usage Intensity Matrix")
    
    # Create usage categories
    filtered_df['Usage_Level'] = pd.cut(
        filtered_df['Avg_Daily_Usage_Hours'],
        bins=[0, 2, 4, 6, 12],
        labels=['Light', 'Moderate', 'Heavy', 'Extreme']
    )
    
    usage_matrix = pd.crosstab(
        filtered_df['Academic_Level'],
        filtered_df['Usage_Level'],
        normalize='index'
    ) * 100
    
    fig = px.imshow(
        usage_matrix,
        labels=dict(x="Usage Level", y="Academic Level", color="Percentage"),
        title="Usage Intensity Distribution (% within each academic level)",
        color_continuous_scale='RdYlGn_r',
        text_auto='.1f'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Parallel coordinates plot
    st.subheader("🎯 Multi-Dimensional Usage Patterns")
    fig = px.parallel_coordinates(
        filtered_df.sample(min(100, len(filtered_df))),
        dimensions=['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                   'Mental_Health_Score', 'Addicted_Score'],
        color='Avg_Daily_Usage_Hours',
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Parallel Coordinates: See how variables interact! (drag to filter)"
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Mental health, the serious stuff
with tab4:
    st.header("🧠 Mental Health Analysis")
    st.markdown("Understanding the psychological impact of social media")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mental Health Score Distribution")
        # Distribution with multiple visualizations
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Histogram', 'Box Plot by Gender'),
            row_heights=[0.6, 0.4]
        )
        
        for gender in filtered_df['Gender'].unique():
            gender_data = filtered_df[filtered_df['Gender'] == gender]['Mental_Health_Score']
            fig.add_trace(
                go.Histogram(x=gender_data, name=gender, opacity=0.7),
                row=1, col=1
            )
            fig.add_trace(
                go.Box(x=gender_data, name=gender),
                row=2, col=1
            )
        
        fig.update_layout(height=600, showlegend=True, barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
        
        # Platform impact on mental health
        st.subheader("Platform Impact")
        platform_mental = filtered_df.groupby('Most_Used_Platform').agg({
            'Mental_Health_Score': ['mean', 'std'],
            'Student_ID': 'count'
        }).reset_index()
        platform_mental.columns = ['Platform', 'Mean', 'Std', 'Count']
        platform_mental = platform_mental.sort_values('Mean', ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=platform_mental['Platform'],
            y=platform_mental['Mean'],
            error_y=dict(type='data', array=platform_mental['Std']),
            marker_color=platform_mental['Mean'],
            marker_colorscale='RdYlGn',
            text=platform_mental['Mean'].round(1),
            textposition='outside'
        ))
        fig.update_layout(
            title="Average Mental Health Score by Platform (with std dev)",
            yaxis_title="Mental Health Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Usage vs Mental Health")
        # Hexbin-style density plot
        fig = px.density_heatmap(
            filtered_df,
            x='Avg_Daily_Usage_Hours',
            y='Mental_Health_Score',
            title="Usage vs Mental Health Density"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Addiction correlation
        st.subheader("Addiction Impact")
        fig = px.scatter(
            filtered_df,
            x='Addicted_Score',
            y='Mental_Health_Score',
            color='Most_Used_Platform',
            size='Avg_Daily_Usage_Hours',
            trendline='ols',
            title="Addiction Score vs Mental Health (sized by usage)",
            hover_data=['Age', 'Gender'],
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Conflict analysis - this is where drama happens
    st.subheader("💔 Social Media Conflicts Impact")
    
    conflict_comparison = filtered_df.groupby('Conflicts_Over_Social_Media').agg({
        'Mental_Health_Score': 'mean',
        'Addicted_Score': 'mean',
        'Avg_Daily_Usage_Hours': 'mean',
        'Sleep_Hours_Per_Night': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    metrics = ['Mental_Health_Score', 'Addicted_Score', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' '),
            x=conflict_comparison['Conflicts_Over_Social_Media'],
            y=conflict_comparison[metric],
            text=conflict_comparison[metric].round(1),
            textposition='auto',
        ))
    
    fig.update_layout(
        barmode='group',
        title="How Conflicts Affect Everything (grouped comparison)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Mental health risk categories
    st.subheader("⚠️ Mental Health Risk Zones")
    
    # Categorize mental health
    filtered_df['Mental_Health_Category'] = pd.cut(
        filtered_df['Mental_Health_Score'],
        bins=[0, 50, 70, 85, 100],
        labels=['Critical', 'At Risk', 'Moderate', 'Healthy']
    )
    
    risk_dist = filtered_df['Mental_Health_Category'].value_counts().reset_index()
    risk_dist.columns = ['Category', 'Count']
    
    # Fancy waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Students",
        orientation="v",
        measure=["relative"] * len(risk_dist),
        x=risk_dist['Category'],
        y=risk_dist['Count'],
        text=risk_dist['Count'],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ff6b6b"}},
        increasing={"marker": {"color": "#51cf66"}},
        totals={"marker": {"color": "#4dabf7"}}
    ))
    
    fig.update_layout(
        title="Mental Health Distribution Across Risk Categories",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart for platform comparison
    st.subheader("🎯 Platform Comparison Radar")
    
    platform_comparison = filtered_df.groupby('Most_Used_Platform').agg({
        'Mental_Health_Score': 'mean',
        'Addicted_Score': lambda x: 100 - x.mean(),  # Invert so higher is better
        'Sleep_Hours_Per_Night': lambda x: (x.mean() / 9) * 100,  # Normalize to 100
        'Avg_Daily_Usage_Hours': lambda x: 100 - (x.mean() / 12) * 100  # Invert and normalize
    }).reset_index()
    
    # Create radar chart for top 5 platforms
    top_platforms = platform_comparison.nlargest(5, 'Mental_Health_Score')
    
    fig = go.Figure()
    
    categories = ['Mental Health', 'Low Addiction', 'Good Sleep', 'Moderate Usage']
    
    for _, row in top_platforms.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Mental_Health_Score'], row['Addicted_Score'], 
               row['Sleep_Hours_Per_Night'], row['Avg_Daily_Usage_Hours']],
            theta=categories,
            fill='toself',
            name=row['Most_Used_Platform']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Platform Health Scores Comparison (all normalized to 100)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Academic impact, school's getting affected
with tab5:
    st.header("📚 Academic Performance Analysis")
    st.markdown("How social media is messing with grades")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Academic Impact Overview")
        
        impact_counts = filtered_df['Affects_Academic_Performance'].value_counts().reset_index()
        impact_counts.columns = ['Impact', 'Count']
        
        # Animated pie with pull effect
        fig = go.Figure(data=[go.Pie(
            labels=impact_counts['Impact'],
            values=impact_counts['Count'],
            pull=[0.2 if x == 'Yes' else 0 for x in impact_counts['Impact']],
            marker=dict(colors=['#ff6b6b', '#51cf66']),
            textinfo='label+percent+value',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Does Social Media Affect Academic Performance?",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Academic level breakdown
        st.subheader("Impact by Academic Level")
        
        impact_by_level = pd.crosstab(
            filtered_df['Academic_Level'],
            filtered_df['Affects_Academic_Performance'],
            normalize='index'
        ) * 100
        
        fig = go.Figure()
        
        for col in impact_by_level.columns:
            fig.add_trace(go.Bar(
                name=col,
                x=impact_by_level.index,
                y=impact_by_level[col],
                text=impact_by_level[col].round(1),
                texttemplate='%{text}%',
                textposition='inside'
            ))
        
        fig.update_layout(
            barmode='stack',
            title="Academic Impact Distribution by Level (%)",
            yaxis_title="Percentage",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Usage Comparison")
        
        # Box plot comparison
        fig = go.Figure()
        
        for impact in ['Yes', 'No']:
            data = filtered_df[filtered_df['Affects_Academic_Performance'] == impact]['Avg_Daily_Usage_Hours']
            fig.add_trace(go.Box(
                y=data,
                name=f'Affected: {impact}',
                boxmean='sd',  # Show mean and std dev
                marker_color='#ff6b6b' if impact == 'Yes' else '#51cf66'
            ))
        
        fig.update_layout(
            title="Usage Hours: Affected vs Not Affected",
            yaxis_title="Hours per Day",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sleep comparison
        st.subheader("Sleep Pattern Comparison")
        
        fig = go.Figure()
        
        for impact in ['Yes', 'No']:
            data = filtered_df[filtered_df['Affects_Academic_Performance'] == impact]['Sleep_Hours_Per_Night']
            fig.add_trace(go.Violin(
                y=data,
                name=f'Affected: {impact}',
                box_visible=True,
                meanline_visible=True,
                fillcolor='#ff6b6b' if impact == 'Yes' else '#51cf66',
                opacity=0.6
            ))
        
        fig.update_layout(
            title="Sleep Hours: Affected vs Not Affected",
            yaxis_title="Hours per Night",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Comprehensive comparison table with styling
    st.subheader("📊 Side-by-Side Comparison")
    
    comparison_df = filtered_df.groupby('Affects_Academic_Performance').agg({
        'Avg_Daily_Usage_Hours': ['mean', 'median'],
        'Sleep_Hours_Per_Night': ['mean', 'median'],
        'Mental_Health_Score': ['mean', 'median'],
        'Addicted_Score': ['mean', 'median']
    }).round(2)
    
    comparison_df.columns = ['_'.join(col).strip() for col in comparison_df.columns.values]
    comparison_df = comparison_df.reset_index()
    
    st.dataframe(
        comparison_df.style.background_gradient(cmap='RdYlGn', subset=[col for col in comparison_df.columns if 'Mental_Health' in col])
        .background_gradient(cmap='RdYlGn_r', subset=[col for col in comparison_df.columns if 'Addicted' in col]),
        use_container_width=True
    )
    
    # Sankey diagram showing flow
    st.subheader("🔄 Student Flow Analysis")
    
    # Create categories for better flow visualization
    filtered_df['Usage_Cat'] = pd.cut(filtered_df['Avg_Daily_Usage_Hours'], bins=[0, 3, 6, 12], labels=['Low', 'Medium', 'High'])
    
    # Build sankey data
    sankey_data = filtered_df.groupby(['Usage_Cat', 'Affects_Academic_Performance']).size().reset_index(name='count')
    
    # Create unique labels
    source_labels = sankey_data['Usage_Cat'].astype(str).tolist()
    target_labels = ['Academic: ' + str(x) for x in sankey_data['Affects_Academic_Performance'].tolist()]
    
    # Map to indices
    all_labels = list(set(source_labels + target_labels))
    source_indices = [all_labels.index(x) for x in source_labels]
    target_indices = [all_labels.index(x) for x in target_labels]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=['#51cf66', '#ffd93d', '#ff6b6b', '#ff6b6b', '#51cf66']
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=sankey_data['count'].tolist(),
            color='rgba(0,0,0,0.2)'
        )
    )])
    
    fig.update_layout(
        title="Flow from Usage Levels to Academic Impact",
        height=400,
        font_size=12
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 6: Correlations, the math geek section
with tab6:
    st.header("🔗 Correlation & Statistical Analysis")
    st.markdown("Finding hidden relationships in the data")
    
    # Enhanced correlation matrix
    numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                    'Mental_Health_Score', 'Addicted_Score']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Correlation Heatmap")
        
        # Create annotated heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Matrix (click cells for details)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Findings")
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Var1': corr_matrix.columns[i],
                    'Var2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
        
        st.write("**Strongest Correlations:**")
        for _, row in corr_df.head(5).iterrows():
            correlation_strength = "Strong" if abs(row['Correlation']) > 0.5 else "Moderate" if abs(row['Correlation']) > 0.3 else "Weak"
            color = "🔴" if row['Correlation'] < 0 else "🟢"
            st.write(f"{color} **{row['Var1'][:15]}** ↔ **{row['Var2'][:15]}**")
            st.write(f"   {row['Correlation']:.3f} ({correlation_strength})")
            st.write("")
    
    # Scatter matrix with enhanced styling
    st.subheader("🎨 Scatter Plot Matrix")
    
    sample_size = min(200, len(filtered_df))
    sample_df = filtered_df.sample(sample_size)[numeric_cols + ['Gender', 'Most_Used_Platform']]
    
    fig = px.scatter_matrix(
        sample_df,
        dimensions=numeric_cols,
        color='Gender',
        title=f"Pairwise Relationships (sample of {sample_size} students - drag to zoom!)",
        opacity=0.6,
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical tests section
    st.subheader("📈 Statistical Significance Tests")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**T-Test: Gender & Addiction**")
        male_addiction = filtered_df[filtered_df['Gender'] == 'Male']['Addicted_Score']
        female_addiction = filtered_df[filtered_df['Gender'] == 'Female']['Addicted_Score']
        
        if len(male_addiction) > 0 and len(female_addiction) > 0:
            t_stat, p_value = stats.ttest_ind(male_addiction, female_addiction)
            
            st.metric("T-statistic", f"{t_stat:.4f}")
            st.metric("P-value", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ Significant difference!")
            else:
                st.info("ℹ️ No significant difference")
    
    with col2:
        st.write("**Correlation: Usage vs Mental Health**")
        if len(filtered_df) > 2:
            corr_coef, p_val = stats.pearsonr(
                filtered_df['Avg_Daily_Usage_Hours'],
                filtered_df['Mental_Health_Score']
            )
            
            st.metric("Correlation", f"{corr_coef:.4f}")
            st.metric("P-value", f"{p_val:.4f}")
            
            if abs(corr_coef) > 0.5:
                st.warning("⚠️ Strong correlation")
            elif abs(corr_coef) > 0.3:
                st.info("ℹ️ Moderate correlation")
            else:
                st.success("✅ Weak correlation")
    
    with col3:
        st.write("**ANOVA: Platforms & Mental Health**")
        platform_groups = [group['Mental_Health_Score'].values for name, group in filtered_df.groupby('Most_Used_Platform')]
        
        if len(platform_groups) > 2:
            f_stat, p_value = stats.f_oneway(*platform_groups)
            
            st.metric("F-statistic", f"{f_stat:.4f}")
            st.metric("P-value", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ Platforms differ significantly!")
            else:
                st.info("ℹ️ No significant difference")
    
    # Regression analysis visualization
    st.subheader("📉 Regression Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Usage predicting mental health
        fig = px.scatter(
            filtered_df,
            x='Avg_Daily_Usage_Hours',
            y='Mental_Health_Score',
            trendline='ols',
            trendline_color_override='red',
            title="Regression: Usage → Mental Health",
            opacity=0.5
        )
        
        # Add regression equation
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            filtered_df['Avg_Daily_Usage_Hours'],
            filtered_df['Mental_Health_Score']
        )
        
        fig.add_annotation(
            text=f'y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_value**2:.3f}',
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Addiction predicting sleep
        fig = px.scatter(
            filtered_df,
            x='Addicted_Score',
            y='Sleep_Hours_Per_Night',
            trendline='ols',
            trendline_color_override='blue',
            title="Regression: Addiction → Sleep",
            opacity=0.5
        )
        
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
            filtered_df['Addicted_Score'],
            filtered_df['Sleep_Hours_Per_Night']
        )
        
        fig.add_annotation(
            text=f'y = {slope2:.2f}x + {intercept2:.2f}<br>R² = {r_value2**2:.3f}',
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 7: ML insights, the cool AI stuff
with tab7:
    st.header("🎯 Machine Learning Insights")
    st.markdown("Let's use AI to find patterns!")
    
    # PCA Analysis
    st.subheader("🔬 Principal Component Analysis (PCA)")
    st.markdown("Reducing dimensions to see the big picture")
    
    # Prepare data for PCA
    pca_features = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                    'Mental_Health_Score', 'Addicted_Score']
    
    X = filtered_df[pca_features].dropna()
    
    if len(X) > 10:
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        
        # Create PCA dataframe
        pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        pca_df['Gender'] = filtered_df.loc[X.index, 'Gender'].values
        pca_df['Platform'] = filtered_df.loc[X.index, 'Most_Used_Platform'].values
        pca_df['Addiction'] = filtered_df.loc[X.index, 'Addicted_Score'].values
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # PCA scatter plot
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Gender',
                size='Addiction',
                hover_data=['Platform'],
                title=f"PCA Projection (explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance)",
                opacity=0.7
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Variance Explained:**")
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                'Variance': pca.explained_variance_ratio_ * 100
            })
            
            fig = go.Figure(data=[go.Bar(
                x=variance_df['Component'],
                y=variance_df['Variance'],
                text=variance_df['Variance'].round(1),
                texttemplate='%{text}%',
                textposition='auto',
                marker_color='indianred'
            )])
            fig.update_layout(
                title="Variance by Component",
                yaxis_title="Percentage",
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Feature Loadings:**")
            loadings_df = pd.DataFrame(
                pca.components_.T,
                columns=['PC1', 'PC2'],
                index=pca_features
            ).round(3)
            st.dataframe(loadings_df.style.background_gradient(cmap='coolwarm', axis=None))
    
    # K-Means Clustering
    st.subheader("🎯 Student Clustering (K-Means)")
    st.markdown("Grouping students with similar behaviors")
    
    # Let user choose number of clusters
    n_clusters = st.slider("Number of clusters", 2, 6, 3)
    
    if len(X) > n_clusters:
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        cluster_df = filtered_df.loc[X.index].copy()
        cluster_df['Cluster'] = clusters
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 3D cluster visualization
            fig = px.scatter_3d(
                cluster_df,
                x='Avg_Daily_Usage_Hours',
                y='Mental_Health_Score',
                z='Addicted_Score',
                color='Cluster',
                symbol='Gender',
                title="Student Clusters in 3D Space",
                opacity=0.7,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cluster characteristics
            st.write("**Cluster Characteristics:**")
            
            cluster_stats = cluster_df.groupby('Cluster').agg({
                'Avg_Daily_Usage_Hours': 'mean',
                'Mental_Health_Score': 'mean',
                'Addicted_Score': 'mean',
                'Sleep_Hours_Per_Night': 'mean',
                'Age': 'mean'
            }).round(2)
            
            cluster_stats['Size'] = cluster_df['Cluster'].value_counts().sort_index()
            
            st.dataframe(cluster_stats.style.background_gradient(cmap='RdYlGn', subset=['Mental_Health_Score'])
                        .background_gradient(cmap='RdYlGn_r', subset=['Addicted_Score']))
            
            # Cluster distribution
            cluster_counts = cluster_df['Cluster'].value_counts().sort_index()
            fig = go.Figure(data=[go.Pie(
                labels=[f'Cluster {i}' for i in cluster_counts.index],
                values=cluster_counts.values,
                hole=0.4
            )])
            fig.update_layout(title="Cluster Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster profiles
        st.subheader("📋 Cluster Profiles")
        
        # Create interpretable names based on characteristics
        cluster_names = []
        for i in range(n_clusters):
            stats = cluster_stats.loc[i]
            if stats['Addicted_Score'] > 70:
                name = "🔴 High Risk"
            elif stats['Mental_Health_Score'] > 75:
                name = "🟢 Healthy Users"
            elif stats['Avg_Daily_Usage_Hours'] > 6:
                name = "🟡 Heavy Users"
            else:
                name = "🔵 Moderate Users"
            cluster_names.append(name)
        
        for i in range(n_clusters):
            with st.expander(f"Cluster {i}: {cluster_names[i]} ({cluster_stats.loc[i, 'Size']} students)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Usage", f"{cluster_stats.loc[i, 'Avg_Daily_Usage_Hours']:.1f}h")
                    st.metric("Avg Age", f"{cluster_stats.loc[i, 'Age']:.1f} yrs")
                
                with col2:
                    st.metric("Mental Health", f"{cluster_stats.loc[i, 'Mental_Health_Score']:.0f}/100")
                    st.metric("Sleep", f"{cluster_stats.loc[i, 'Sleep_Hours_Per_Night']:.1f}h")
                
                with col3:
                    st.metric("Addiction", f"{cluster_stats.loc[i, 'Addicted_Score']:.0f}/100")
                    
                    # Most common platform
                    common_platform = cluster_df[cluster_df['Cluster'] == i]['Most_Used_Platform'].mode()
                    if len(common_platform) > 0:
                        st.metric("Top Platform", common_platform.values[0])

# Tab 8: Geographic view, where in the world
with tab8:
    st.header("🌍 Geographic Analysis")
    st.markdown("Social media usage patterns around the globe")
    
    # Country-wise statistics
    country_stats = filtered_df.groupby('Country').agg({
        'Student_ID': 'count',
        'Avg_Daily_Usage_Hours': 'mean',
        'Mental_Health_Score': 'mean',
        'Addicted_Score': 'mean',
        'Sleep_Hours_Per_Night': 'mean'
    }).round(2)
    country_stats.columns = ['Students', 'Avg Usage', 'Mental Health', 'Addiction', 'Sleep']
    country_stats = country_stats.sort_values('Students', ascending=False).reset_index()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Choropleth map (simulated with available data)
        st.subheader("World Map View")
        
        # Note: This would work better with actual geographic coordinates
        fig = px.choropleth(
            country_stats,
            locations='Country',
            locationmode='country names',
            color='Avg Usage',
            hover_name='Country',
            hover_data=['Students', 'Mental Health', 'Addiction'],
            title="Average Daily Usage by Country",
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Countries")
        
        # Show top countries table
        st.dataframe(
            country_stats.head(10).style.background_gradient(cmap='Blues', subset=['Students'])
            .background_gradient(cmap='Reds', subset=['Avg Usage'])
            .background_gradient(cmap='RdYlGn', subset=['Mental Health']),
            use_container_width=True
        )
    
    # Country comparison
    st.subheader("📊 Cross-Country Comparison")
    
    # Parallel categories for country analysis
    fig = px.parallel_categories(
        filtered_df.sample(min(200, len(filtered_df))),
        dimensions=['Country', 'Gender', 'Academic_Level', 'Most_Used_Platform'],
        color='Addicted_Score',
        color_continuous_scale='Turbo',
        title="Student Flow: Country → Gender → Level → Platform (color = Addiction)"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Bubble chart - countries
    st.subheader("🎈 Country Bubble Chart")
    
    fig = px.scatter(
        country_stats,
        x='Avg Usage',
        y='Mental Health',
        size='Students',
        color='Addiction',
        hover_name='Country',
        title="Usage vs Mental Health by Country (bubble size = # of students)",
        color_continuous_scale='RdYlGn_r',
        size_max=60
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Download section - now with multiple format options
st.markdown("---")
st.header("📥 Export Your Data")

col1, col2, col3 = st.columns(3)

with col1:
    # CSV download
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        label="📄 Download as CSV",
        data=csv,
        file_name=f'social_media_data_{len(filtered_df)}_records.csv',
        mime='text/csv',
        help="Download filtered data in CSV format"
    )

with col2:
    # Summary statistics download
    @st.cache_data
    def create_summary_report(df):
        summary = df.describe().T
        summary['missing'] = df.isnull().sum()
        return summary.to_csv().encode('utf-8')
    
    summary_csv = create_summary_report(filtered_df)
    st.download_button(
        label="📊 Download Statistics",
        data=summary_csv,
        file_name='summary_statistics.csv',
        mime='text/csv',
        help="Download summary statistics"
    )

with col3:
    # Correlation matrix download
    @st.cache_data
    def create_correlation_report(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].corr().to_csv().encode('utf-8')
    
    if len(filtered_df.select_dtypes(include=[np.number]).columns) > 0:
        corr_csv = create_correlation_report(filtered_df)
        st.download_button(
            label="🔗 Download Correlations",
            data=corr_csv,
            file_name='correlation_matrix.csv',
            mime='text/csv',
            help="Download correlation matrix"
        )

# Insights and Recommendations section
st.markdown("---")
st.header("💡 Key Insights & Recommendations")

# Calculate key metrics for insights
avg_usage = filtered_df['Avg_Daily_Usage_Hours'].mean()
avg_mental = filtered_df['Mental_Health_Score'].mean()
avg_addiction = filtered_df['Addicted_Score'].mean()
affected_pct = (filtered_df['Affects_Academic_Performance'] == 'Yes').sum() / len(filtered_df) * 100

# Create insights based on data
insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.subheader("🔍 What We Found")
    
    # Dynamic insights based on actual data
    if avg_usage > 6:
        st.warning(f"⚠️ **High Usage Alert:** Students average {avg_usage:.1f} hours daily - that's more than 25% of waking hours!")
    else:
        st.info(f"ℹ️ **Moderate Usage:** Average usage is {avg_usage:.1f} hours per day")
    
    if avg_mental < 60:
        st.error(f"🚨 **Mental Health Concern:** Average score is only {avg_mental:.0f}/100 - below healthy threshold")
    elif avg_mental < 75:
        st.warning(f"⚠️ **Mental Health Watch:** Average score is {avg_mental:.0f}/100 - room for improvement")
    else:
        st.success(f"✅ **Good Mental Health:** Average score is {avg_mental:.0f}/100")
    
    if affected_pct > 50:
        st.error(f"📉 **Academic Crisis:** {affected_pct:.1f}% report negative academic impact")
    else:
        st.info(f"📚 **Academic Impact:** {affected_pct:.1f}% report being affected")
    
    # Correlation insights
    usage_mental_corr = filtered_df['Avg_Daily_Usage_Hours'].corr(filtered_df['Mental_Health_Score'])
    if usage_mental_corr < -0.3:
        st.warning(f"📊 **Strong Negative Correlation:** More usage = worse mental health (r={usage_mental_corr:.2f})")
    
    usage_sleep_corr = filtered_df['Avg_Daily_Usage_Hours'].corr(filtered_df['Sleep_Hours_Per_Night'])
    if usage_sleep_corr < -0.2:
        st.warning(f"😴 **Sleep Disruption:** High usage linked to poor sleep (r={usage_sleep_corr:.2f})")

with insights_col2:
    st.subheader("✅ Recommendations")
    
    st.markdown("""
    **For Students:**
    - 🎯 Set daily usage limits (aim for under 3 hours)
    - 📱 Use app timers and screen time trackers
    - 🌙 No screens 1 hour before bed
    - 🧘 Take regular digital detox breaks
    - 📚 Prioritize study time over scrolling
    
    **For Educators:**
    - 📊 Monitor students showing signs of high usage
    - 🗣️ Open dialogue about healthy digital habits
    - 📖 Integrate digital wellness into curriculum
    - 🤝 Provide support resources for at-risk students
    
    **For Parents:**
    - 👨‍👩‍👧 Model healthy social media behavior
    - ⏰ Establish family screen-free times
    - 💬 Talk openly about online experiences
    - 🎮 Encourage offline activities and hobbies
    """)

# Interactive recommendations based on user selections
st.markdown("---")
st.subheader("🎯 Personalized Insights")

# Allow user to input their own stats
with st.expander("📝 Get Personalized Recommendations (Optional)"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_usage = st.number_input("Your daily usage (hours)", 0.0, 24.0, 5.0, 0.5)
        user_sleep = st.number_input("Your sleep (hours)", 0.0, 12.0, 7.0, 0.5)
    
    with col2:
        user_mental = st.slider("Your mental health (1-100)", 0, 100, 70)
        user_addiction = st.slider("Your addiction level (1-100)", 0, 100, 50)
    
    with col3:
        user_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        user_academic = st.selectbox("Academic Level", df['Academic_Level'].unique())
    
    if st.button("🔍 Analyze My Profile"):
        st.markdown("---")
        st.subheader("Your Personal Analysis")
        
        # Compare with averages
        usage_diff = user_usage - avg_usage
        mental_diff = user_mental - avg_mental
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if usage_diff > 2:
                st.error(f"⚠️ You use {usage_diff:.1f} hours MORE than average")
                st.write("**Suggestion:** Try reducing by 30 mins daily")
            elif usage_diff < -2:
                st.success(f"✅ You use {abs(usage_diff):.1f} hours LESS than average")
            else:
                st.info("ℹ️ Your usage is around average")
        
        with col2:
            if user_mental < 60:
                st.error("🚨 Your mental health score is concerning")
                st.write("**Action:** Consider talking to a counselor")
            elif user_mental < 75:
                st.warning("⚠️ Your mental health could improve")
                st.write("**Action:** Try mindfulness apps")
            else:
                st.success("✅ Your mental health is good!")
        
        with col3:
            if user_addiction > 70:
                st.error("🚨 High addiction risk detected")
                st.write("**Action:** Seek digital wellness support")
            elif user_addiction > 50:
                st.warning("⚠️ Moderate addiction risk")
                st.write("**Action:** Set strict daily limits")
            else:
                st.success("✅ Low addiction risk")
        
        # Find similar students
        similar_students = filtered_df[
            (filtered_df['Gender'] == user_gender) &
            (filtered_df['Academic_Level'] == user_academic) &
            (abs(filtered_df['Avg_Daily_Usage_Hours'] - user_usage) < 2)
        ]
        
        if len(similar_students) > 0:
            st.write(f"**📊 Compared to {len(similar_students)} similar students:**")
            st.write(f"- Average Mental Health: {similar_students['Mental_Health_Score'].mean():.0f}")
            st.write(f"- Average Sleep: {similar_students['Sleep_Hours_Per_Night'].mean():.1f} hours")
            st.write(f"- {(similar_students['Affects_Academic_Performance']=='Yes').sum()/len(similar_students)*100:.0f}% report academic impact")

# Footer with additional info
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
    <h3>📱 Social Media Analytics Dashboard</h3>
    <p>Built with ❤️ using Streamlit, Plotly, and Machine Learning</p>
    <p><b>Data Sources:</b> Student Survey Data | <b>Analysis Date:</b> {}</p>
    <p style='font-size: 12px; margin-top: 10px;'>
        This dashboard uses advanced statistical methods and ML to uncover insights.<br>
        Remember: Balance is key! Use social media wisely 🧠✨
    </p>
</div>
""".format(pd.Timestamp.now().strftime('%Y-%m-%d')), unsafe_allow_html=True)

# Hidden easter egg for power users
if st.sidebar.checkbox("🔓 Show Advanced Options", value=False):
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Advanced Settings")
    
    show_raw_stats = st.sidebar.checkbox("Show Raw Statistics")
    if show_raw_stats:
        st.sidebar.write("**Dataset Info:**")
        st.sidebar.write(f"- Shape: {filtered_df.shape}")
        st.sidebar.write(f"- Memory: {filtered_df.memory_usage().sum() / 1024:.2f} KB")
        st.sidebar.write(f"- Duplicates: {filtered_df.duplicated().sum()}")
        st.sidebar.write(f"- Missing: {filtered_df.isnull().sum().sum()}")
    
    if st.sidebar.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")
    
    if st.sidebar.button("💾 Save Current View"):
        st.sidebar.success("View saved! (simulated)")

# Performance metrics (hidden)
if st.sidebar.checkbox("📊 Show Performance Metrics", value=False):
    import time
    st.sidebar.write(f"**App Performance:**")
    st.sidebar.write(f"- Records loaded: {len(df)}")
    st.sidebar.write(f"- Records filtered: {len(filtered_df)}")
    st.sidebar.write(f"- Filter ratio: {len(filtered_df)/len(df)*100:.1f}%")
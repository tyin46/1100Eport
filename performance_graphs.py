import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Set random seed for reproducible "demo" data
np.random.seed(42)
random.seed(42)

def generate_retrieval_accuracy_plot():
    """Generate a plot showing retrieval accuracy improvement over time"""
    
    # Simulate accuracy improvement over development iterations
    dates = pd.date_range(start='2024-01-01', end='2024-11-12', freq='W')
    
    # Base accuracy with realistic improvement curve
    base_accuracy = 78.3
    iterations = len(dates)
    
    # Create realistic improvement curve with some volatility
    improvement_curve = []
    current_accuracy = base_accuracy
    
    for i in range(iterations):
        # Major improvements at certain milestones
        if i == 10:  # Embedding model upgrade
            current_accuracy += 4.5
        elif i == 20:  # Hybrid retrieval implementation
            current_accuracy += 3.2
        elif i == 30:  # Fine-tuning optimization
            current_accuracy += 2.8
        elif i == 35:  # Caching and optimization
            current_accuracy += 1.6
        
        # Small random improvements and occasional setbacks
        random_change = np.random.normal(0.1, 0.3)
        current_accuracy += random_change
        
        # Ensure accuracy doesn't exceed realistic bounds
        current_accuracy = min(max(current_accuracy, 75.0), 95.0)
        improvement_curve.append(current_accuracy)
    
    # Create the plot
    fig = go.Figure()
    
    # Main accuracy line
    fig.add_trace(go.Scatter(
        x=dates,
        y=improvement_curve,
        mode='lines+markers',
        name='Retrieval Accuracy',
        line=dict(color='#4a90e2', width=3),
        marker=dict(size=6, color='#4a90e2'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Accuracy:</b> %{y:.1f}%<extra></extra>'
    ))
    
    # Add milestone annotations
    milestones = [
        (dates[10], improvement_curve[10], "Embedding Model<br>Upgrade"),
        (dates[20], improvement_curve[20], "Hybrid Retrieval<br>Implementation"),
        (dates[30], improvement_curve[30], "Fine-tuning<br>Optimization"),
        (dates[35], improvement_curve[35], "Caching &<br>Performance Boost")
    ]
    
    for date, accuracy, label in milestones:
        fig.add_annotation(
            x=date,
            y=accuracy,
            text=label,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#666",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#666",
            borderwidth=1
        )
    
    # Add target line
    fig.add_hline(
        y=90.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Target: 90% Accuracy",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title={
            'text': 'Document Retrieval Accuracy Improvement Over Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Development Timeline',
        yaxis_title='Retrieval Accuracy (%)',
        yaxis=dict(range=[75, 95]),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def generate_response_time_analysis():
    """Generate response time distribution and trend analysis"""
    
    # Simulate response times for different query types
    query_types = ['Simple Fact', 'Complex Analysis', 'Multi-document', 'Mathematical', 'Technical Spec']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    fig = go.Figure()
    
    # Generate realistic response time distributions
    for i, (query_type, color) in enumerate(zip(query_types, colors)):
        if query_type == 'Simple Fact':
            times = np.random.gamma(2, 0.4, 1000) + 0.5  # Faster for simple queries
        elif query_type == 'Complex Analysis':
            times = np.random.gamma(3, 0.8, 1000) + 1.2  # Slower for complex
        elif query_type == 'Multi-document':
            times = np.random.gamma(4, 0.6, 1000) + 1.0  # Variable timing
        elif query_type == 'Mathematical':
            times = np.random.gamma(2.5, 0.7, 1000) + 0.8
        else:  # Technical Spec
            times = np.random.gamma(3.2, 0.5, 1000) + 1.1
        
        # Cap at reasonable maximum
        times = np.clip(times, 0.3, 6.0)
        
        fig.add_trace(go.Box(
            y=times,
            name=query_type,
            boxpoints='outliers',
            marker_color=color,
            line_color=color
        ))
    
    fig.update_layout(
        title={
            'text': 'Response Time Distribution by Query Type',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        yaxis_title='Response Time (seconds)',
        xaxis_title='Query Type',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def generate_user_satisfaction_metrics():
    """Generate user satisfaction analysis over time"""
    
    # Simulate user feedback data
    dates = pd.date_range(start='2024-01-01', end='2024-11-12', freq='D')
    
    # Generate satisfaction scores (1-5 scale)
    base_satisfaction = 3.2
    satisfaction_scores = []
    
    for i, date in enumerate(dates):
        # Gradual improvement with some volatility
        trend_improvement = (i / len(dates)) * 1.5  # Improve by 1.5 points over time
        daily_variation = np.random.normal(0, 0.15)
        
        score = base_satisfaction + trend_improvement + daily_variation
        score = np.clip(score, 1.0, 5.0)  # Keep within valid range
        satisfaction_scores.append(score)
    
    # Create subplot with satisfaction trend and distribution
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Satisfaction Trend Over Time', 'Current Rating Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Satisfaction trend
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=satisfaction_scores,
            mode='lines',
            name='Daily Average',
            line=dict(color='#4a90e2', width=2),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add moving average
    window_size = 30
    moving_avg = pd.Series(satisfaction_scores).rolling(window=window_size).mean()
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=moving_avg,
            mode='lines',
            name='30-Day Moving Average',
            line=dict(color='#e74c3c', width=3)
        ),
        row=1, col=1
    )
    
    # Current rating distribution (simulate recent ratings)
    recent_ratings = np.random.choice([1, 2, 3, 4, 5], size=500, 
                                    p=[0.02, 0.08, 0.15, 0.35, 0.40])  # Skewed toward positive
    
    rating_counts = pd.Series(recent_ratings).value_counts().sort_index()
    
    fig.add_trace(
        go.Bar(
            x=[f"{i} Star{'s' if i != 1 else ''}" for i in rating_counts.index],
            y=rating_counts.values,
            name='Rating Distribution',
            marker_color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60'],
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title={
            'text': 'User Satisfaction Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        template='plotly_white',
        height=400
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Satisfaction Score (1-5)", row=1, col=1)
    fig.update_yaxes(title_text="Number of Ratings", row=1, col=2)
    
    return fig

def generate_system_load_analysis():
    """Generate system resource utilization analysis"""
    
    # Simulate 24-hour system load data
    hours = list(range(24))
    
    # CPU Usage (%)
    cpu_usage = []
    for hour in hours:
        if 9 <= hour <= 17:  # Business hours
            base_cpu = 45 + np.random.normal(0, 8)
        elif 18 <= hour <= 22:  # Evening peak
            base_cpu = 35 + np.random.normal(0, 6)
        else:  # Night/early morning
            base_cpu = 15 + np.random.normal(0, 4)
        
        cpu_usage.append(max(5, min(85, base_cpu)))
    
    # Memory Usage (%)
    memory_usage = []
    for hour in hours:
        if 9 <= hour <= 17:
            base_memory = 60 + np.random.normal(0, 10)
        elif 18 <= hour <= 22:
            base_memory = 50 + np.random.normal(0, 8)
        else:
            base_memory = 25 + np.random.normal(0, 5)
        
        memory_usage.append(max(10, min(90, base_memory)))
    
    # Query Volume
    query_volume = []
    for hour in hours:
        if 9 <= hour <= 17:
            base_queries = 120 + np.random.normal(0, 25)
        elif 18 <= hour <= 22:
            base_queries = 80 + np.random.normal(0, 15)
        else:
            base_queries = 20 + np.random.normal(0, 8)
        
        query_volume.append(max(0, base_queries))
    
    # Create subplot
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU Usage', 'Memory Usage', 'Query Volume', 'Response Time vs Load'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # CPU Usage
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=cpu_usage,
            mode='lines+markers',
            name='CPU Usage',
            line=dict(color='#e74c3c', width=2),
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.1)'
        ),
        row=1, col=1
    )
    
    # Memory Usage
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=memory_usage,
            mode='lines+markers',
            name='Memory Usage',
            line=dict(color='#3498db', width=2),
            fill='tonexty',
            fillcolor='rgba(52, 152, 219, 0.1)'
        ),
        row=1, col=2
    )
    
    # Query Volume
    fig.add_trace(
        go.Bar(
            x=hours,
            y=query_volume,
            name='Queries/Hour',
            marker_color='#2ecc71',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Response Time vs Load correlation
    load_levels = np.linspace(10, 80, 50)
    response_times = 1.2 + (load_levels / 100) * 2.5 + np.random.normal(0, 0.2, 50)
    response_times = np.clip(response_times, 0.5, 5.0)
    
    fig.add_trace(
        go.Scatter(
            x=load_levels,
            y=response_times,
            mode='markers',
            name='Response Time vs CPU Load',
            marker=dict(
                color=load_levels,
                colorscale='Viridis',
                size=8,
                showscale=True,
                colorbar=dict(title="CPU Load (%)")
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title={
            'text': 'System Resource Utilization - 24 Hour Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
    fig.update_xaxes(title_text="CPU Load (%)", row=2, col=2)
    
    fig.update_yaxes(title_text="CPU Usage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Memory Usage (%)", row=1, col=2)
    fig.update_yaxes(title_text="Queries per Hour", row=2, col=1)
    fig.update_yaxes(title_text="Response Time (s)", row=2, col=2)
    
    return fig

def generate_accuracy_by_document_type():
    """Generate accuracy analysis by document type"""
    
    doc_types = ['Technical Manual', 'Research Paper', 'Legal Document', 'Financial Report', 'User Guide']
    accuracy_scores = [94.2, 91.8, 88.5, 92.7, 96.1]
    sample_sizes = [450, 320, 180, 290, 380]
    
    fig = go.Figure()
    
    # Create bubble chart
    fig.add_trace(go.Scatter(
        x=doc_types,
        y=accuracy_scores,
        mode='markers',
        marker=dict(
            size=[s/10 for s in sample_sizes],  # Scale bubble size
            color=accuracy_scores,
            colorscale='RdYlGn',
            showscale=True,
            sizemode='diameter',
            sizeref=2,
            colorbar=dict(title="Accuracy (%)")
        ),
        text=[f'Accuracy: {acc}%<br>Samples: {size}' for acc, size in zip(accuracy_scores, sample_sizes)],
        hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Retrieval Accuracy by Document Type',
        xaxis_title='Document Type',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[85, 100]),
        template='plotly_white'
    )
    
    return fig

if __name__ == "__main__":
    # Test the functions
    print("Testing performance visualization functions...")
    
    fig1 = generate_retrieval_accuracy_plot()
    print("✓ Retrieval accuracy plot generated")
    
    fig2 = generate_response_time_analysis()
    print("✓ Response time analysis generated")
    
    fig3 = generate_user_satisfaction_metrics()
    print("✓ User satisfaction metrics generated")
    
    fig4 = generate_system_load_analysis()
    print("✓ System load analysis generated")
    
    fig5 = generate_accuracy_by_document_type()
    print("✓ Accuracy by document type generated")
    
    print("All performance visualization functions working correctly!")
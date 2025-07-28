import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

class SowForecastingVisualizer:
    """
    Comprehensive visualization toolkit for sow feed intake forecasting system.
    Generates publication-quality plots for trajectory analysis, clustering, and forecasting.
    """
    
    def __init__(self, forecaster):
        """
        Initialize visualizer with a trained forecasting system.
        
        Args:
            forecaster: Trained SowFeedIntakeForecaster instance
        """
        self.forecaster = forecaster
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    # ... other methods unchanged ...

    def create_interactive_dashboard(self, df, trajectory_matrix, cluster_assignments):
        """
        Create an interactive Plotly dashboard for comprehensive analysis.
        """
        print("🚀 Creating interactive dashboard...")
        
        # Merge data
        df_with_clusters = df.merge(cluster_assignments, on='sow_id', how='left')
        
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Trajectory Clusters', 'Environmental Impact', 
                            'Parity Distribution', 'Daily Patterns',
                            'Performance Metrics', 'Prediction Confidence'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"type": "pie"}, {"secondary_y": False}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Interactive trajectory clusters
        for cluster in np.unique(cluster_assignments['cluster']):
            cluster_sows = cluster_assignments[cluster_assignments['cluster'] == cluster]['sow_id']
            cluster_trajectories = trajectory_matrix.loc[cluster_sows]
            mean_trajectory = cluster_trajectories.mean()
            
            fig.add_trace(
                go.Scatter(x=mean_trajectory.index, y=mean_trajectory.values,
                          mode='lines+markers', name=f'Cluster {cluster} Mean',
                          line=dict(width=3), opacity=0.8),
                row=1, col=1
            )
        
        # 2. Environmental impact over time
        daily_env = df_with_clusters.groupby('lactation_day').agg({
            'temperature': 'mean',
            'daily_feed_intake': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(x=daily_env['lactation_day'], y=daily_env['daily_feed_intake'],
                      mode='lines+markers', name='Average Intake',
                      line=dict(width=2), marker=dict(color='blue')),
            row=1, col=2, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=daily_env['lactation_day'], y=daily_env['temperature'],
                      mode='lines', name='Temperature',
                      line=dict(width=2, dash='dash'), marker=dict(color='red')),
            row=1, col=2, secondary_y=True
        )
        
        # 3. Parity distribution pie chart
        parity_dist = df_with_clusters['Parity'].value_counts()
        fig.add_trace(
            go.Pie(labels=parity_dist.index, values=parity_dist.values,
                   name="Parity Distribution", hole=0.4),
            row=2, col=1
        )
        
        # 4. Daily intake patterns by cluster
        daily_patterns = df_with_clusters.groupby(['lactation_day', 'cluster'])['daily_feed_intake'].mean().reset_index()
        for cluster in np.unique(daily_patterns['cluster']):
            cluster_data = daily_patterns[daily_patterns['cluster'] == cluster]
            fig.add_trace(
                go.Scatter(x=cluster_data['lactation_day'], y=cluster_data['daily_feed_intake'],
                           mode='lines', name=f'Cluster {cluster} Pattern',
                           opacity=0.7),
                row=2, col=2
            )
        
        # 5. Performance metrics bar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [0.85, 0.83, 0.87, 0.85]  # Example values
        fig.add_trace(
            go.Bar(x=metrics, y=values, marker_color=self.colors[:len(metrics)], name='Metrics'),
            row=3, col=1
        )
        
        # 6. Prediction confidence distribution
        # Simulate or use real probabilities if available
        if hasattr(self.forecaster, 'early_classifier') and hasattr(self.forecaster.early_classifier, 'predict_proba'):
            y_proba = self.forecaster.early_classifier.predict_proba(self.forecaster.X_test)
            max_proba = np.max(y_proba, axis=1)
        else:
            # simulated
            max_proba = np.random.beta(5, 2, size=100)
        
        fig.add_trace(
            go.Histogram(x=max_proba, nbinsx=20, name='Prediction Confidence'),
            row=3, col=2
        )
        
        # Layout adjustments
        fig.update_layout(
            title_text="Interactive Sow Feed Intake Forecasting Dashboard",
            height=900, width=1200,
            legend=dict(orientation='h', y=-0.1)
        )
        fig.update_xaxes(title_text="Lactation Day", row=1, col=1)
        fig.update_yaxes(title_text="Intake (kg)", row=1, col=1)
        
        fig.show()

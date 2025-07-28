# Enhanced Visualization Module for Sow Feed Intake Forecasting
# This module extends the main forecasting system with comprehensive visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent, professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
        
    def plot_data_overview(self, df):
        """
        Create comprehensive data overview visualizations.
        """
        print("📊 Generating data overview visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sow Feed Intake Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Distribution of feed intake
        axes[0, 0].hist(df['daily_feed_intake'], bins=50, alpha=0.7, color=self.colors[0])
        axes[0, 0].set_title('Distribution of Daily Feed Intake')
        axes[0, 0].set_xlabel('Daily Feed Intake (kg)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feed intake by parity
        parity_order = ['P1', 'P2', 'P3+']
        df_parity = df[df['Parity'].isin(parity_order)]
        sns.boxplot(data=df_parity, x='Parity', y='daily_feed_intake', ax=axes[0, 1])
        axes[0, 1].set_title('Feed Intake Distribution by Parity')
        axes[0, 1].set_ylabel('Daily Feed Intake (kg)')
        
        # 3. Feed intake over lactation days
        daily_avg = df.groupby('lactation_day')['daily_feed_intake'].agg(['mean', 'std']).reset_index()
        axes[0, 2].plot(daily_avg['lactation_day'], daily_avg['mean'], 'o-', color=self.colors[0])
        axes[0, 2].fill_between(daily_avg['lactation_day'], 
                               daily_avg['mean'] - daily_avg['std'],
                               daily_avg['mean'] + daily_avg['std'], 
                               alpha=0.3, color=self.colors[0])
        axes[0, 2].set_title('Average Feed Intake Trajectory')
        axes[0, 2].set_xlabel('Lactation Day')
        axes[0, 2].set_ylabel('Daily Feed Intake (kg)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Environmental conditions distribution
        axes[1, 0].hist(df['temperature'], bins=30, alpha=0.7, color=self.colors[1], label='Temperature')
        ax_twin = axes[1, 0].twinx()
        ax_twin.hist(df['dew_point'], bins=30, alpha=0.5, color=self.colors[2], label='Dew Point')
        axes[1, 0].set_title('Environmental Conditions Distribution')
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Frequency - Temperature', color=self.colors[1])
        ax_twin.set_ylabel('Frequency - Dew Point', color=self.colors[2])
        
        # 5. Heat stress hours distribution
        axes[1, 1].hist(df['Hours_T2M_GT_24'], bins=range(0, 25), alpha=0.7, color=self.colors[3])
        axes[1, 1].set_title('Heat Stress Hours Distribution')
        axes[1, 1].set_xlabel('Hours Above 24°C')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Sample size by parity
        parity_counts = df['Parity'].value_counts()
        axes[1, 2].pie(parity_counts.values, labels=parity_counts.index, autopct='%1.1f%%',
                      colors=self.colors[:len(parity_counts)])
        axes[1, 2].set_title('Sample Distribution by Parity')
        
        plt.tight_layout()
        plt.show()
        
        # Additional summary statistics
        print("\n📈 Dataset Summary Statistics:")
        print(f"  • Total records: {len(df):,}")
        print(f"  • Unique sows: {df['sow_id'].nunique():,}")
        print(f"  • Lactation days range: {df['lactation_day'].min()}-{df['lactation_day'].max()}")
        print(f"  • Average intake: {df['daily_feed_intake'].mean():.2f} kg")
        print(f"  • Intake range: {df['daily_feed_intake'].min():.2f}-{df['daily_feed_intake'].max():.2f} kg")
        
    def plot_trajectory_clusters_advanced(self, trajectory_matrix, cluster_assignments):
        """
        Create advanced trajectory cluster visualizations with statistical analysis.
        """
        print("🎯 Generating advanced trajectory cluster analysis...")
        
        n_clusters = len(np.unique(cluster_assignments['cluster']))
        
        # Create comprehensive cluster visualization
        fig = plt.figure(figsize=(20, 15))
        
        # Main trajectory plots
        for i, cluster in enumerate(np.unique(cluster_assignments['cluster'])):
            ax = plt.subplot(3, n_clusters, i + 1)
            
            cluster_sows = cluster_assignments[cluster_assignments['cluster'] == cluster]['sow_id']
            cluster_trajectories = trajectory_matrix.loc[cluster_sows]
            
            # Plot individual trajectories with transparency
            for sow_id in cluster_sows:
                trajectory = trajectory_matrix.loc[sow_id]
                days = trajectory.index
                ax.plot(days, trajectory.values, alpha=0.1, color=self.colors[i], linewidth=0.5)
            
            # Calculate and plot percentiles
            percentiles = cluster_trajectories.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            days = percentiles.columns
            
            ax.fill_between(days, percentiles.loc[0.1], percentiles.loc[0.9], 
                           alpha=0.2, color=self.colors[i], label='10th-90th percentile')
            ax.fill_between(days, percentiles.loc[0.25], percentiles.loc[0.75], 
                           alpha=0.3, color=self.colors[i], label='25th-75th percentile')
            ax.plot(days, percentiles.loc[0.5], color=self.colors[i], linewidth=3, 
                   label=f'Median (n={len(cluster_sows)})')
            
            ax.set_title(f'Cluster {cluster} - Feeding Trajectories', fontweight='bold')
            ax.set_xlabel('Lactation Day')
            ax.set_ylabel('Daily Feed Intake (kg)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Cluster comparison plots
        ax_compare = plt.subplot(3, 1, 2)
        for i, cluster in enumerate(np.unique(cluster_assignments['cluster'])):
            cluster_sows = cluster_assignments[cluster_assignments['cluster'] == cluster]['sow_id']
            cluster_trajectories = trajectory_matrix.loc[cluster_sows]
            mean_trajectory = cluster_trajectories.mean()
            std_trajectory = cluster_trajectories.std()
            
            days = mean_trajectory.index
            ax_compare.plot(days, mean_trajectory.values, color=self.colors[i], 
                           linewidth=3, label=f'Cluster {cluster} Mean', marker='o')
            ax_compare.fill_between(days, 
                                   mean_trajectory - std_trajectory,
                                   mean_trajectory + std_trajectory,
                                   alpha=0.2, color=self.colors[i])
        
        ax_compare.set_title('Cluster Comparison - Mean Trajectories ± 1 SD', fontweight='bold')
        ax_compare.set_xlabel('Lactation Day')
        ax_compare.set_ylabel('Daily Feed Intake (kg)')
        ax_compare.legend()
        ax_compare.grid(True, alpha=0.3)
        
        # Cluster characteristics heatmap
        ax_heatmap = plt.subplot(3, 1, 3)
        
        # Calculate cluster characteristics
        characteristics = []
        for cluster in np.unique(cluster_assignments['cluster']):
            cluster_sows = cluster_assignments[cluster_assignments['cluster'] == cluster]['sow_id']
            cluster_trajectories = trajectory_matrix.loc[cluster_sows]
            
            char = {
                'Initial Intake (Day 2)': cluster_trajectories.iloc[:, 0].mean(),
                'Peak Intake': cluster_trajectories.max(axis=1).mean(),
                'Final Intake': cluster_trajectories.iloc[:, -1].mean(),
                'Days to Peak': cluster_trajectories.idxmax(axis=1).mean(),
                'Intake Variability': cluster_trajectories.std(axis=1).mean(),
                'Total Intake': cluster_trajectories.sum(axis=1).mean()
            }
            characteristics.append(char)
        
        char_df = pd.DataFrame(characteristics, index=[f'Cluster {i}' for i in range(n_clusters)])
        
        # Normalize for heatmap
        char_normalized = (char_df - char_df.min()) / (char_df.max() - char_df.min())
        
        sns.heatmap(char_normalized.T, annot=char_df.T, fmt='.2f', cmap='RdYlBu_r', 
                   ax=ax_heatmap, cbar_kws={'label': 'Normalized Value'})
        ax_heatmap.set_title('Cluster Characteristics Comparison', fontweight='bold')
        ax_heatmap.set_xlabel('Cluster')
        
        plt.tight_layout()
        plt.show()
        
    def plot_clustering_validation(self, cluster_scores):
        """
        Visualize clustering validation metrics.
        """
        print("✅ Generating clustering validation plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        n_clusters = list(cluster_scores.keys())
        silhouette_scores = [cluster_scores[k]['silhouette'] for k in n_clusters]
        calinski_scores = [cluster_scores[k]['calinski'] for k in n_clusters]
        
        # Silhouette scores
        axes[0].plot(n_clusters, silhouette_scores, 'o-', color=self.colors[0], linewidth=

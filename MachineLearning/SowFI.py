# Sow Feed Intake Trajectory Clustering and Forecasting System
# Implementation of offline learning and online forecasting for precision feeding

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import dtw
import warnings
warnings.filterwarnings('ignore')

class SowFeedIntakeForecaster:
    """
    A comprehensive system for clustering sow feed intake trajectories and forecasting future intake.
    Implements both offline learning (trajectory clustering) and online forecasting procedures.
    """
    
    def __init__(self):
        self.trajectory_clusters = None
        self.cluster_model = None
        self.early_classifier = None
        self.trajectory_scaler = None
        self.cluster_centers = None
        self.cluster_labels = None
        self.sow_metadata = None
        
    def load_and_prepare_data(self, df):
        """
        Load and prepare the sow feed intake data for analysis.
        
        Args:
            df: DataFrame with columns: sow_id, lactation_day, daily_feed_intake, Parity, 
                temperature, dew_point, Hours_T2M_GT_24
        
        Returns:
            Prepared data structures for trajectory analysis
        """
        print("📊 Loading and preparing data...")
        
        # Create pivot table: rows = sows, columns = lactation days, values = feed intake
        trajectory_matrix = df.pivot(index='sow_id', columns='lactation_day', values='daily_feed_intake')
        
        # Get metadata for each sow (parity, environmental averages)
        sow_metadata = df.groupby('sow_id').agg({
            'Parity': 'first',
            'temperature': 'mean',
            'dew_point': 'mean', 
            'Hours_T2M_GT_24': 'mean',
            'lactation_day': 'max'  # Total lactation length
        }).reset_index()
        sow_metadata.columns = ['sow_id', 'parity', 'avg_temp', 'avg_dew_point', 'avg_hours_gt24', 'lactation_length']
        
        # Fill missing values with forward fill, then backward fill
        trajectory_matrix = trajectory_matrix.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
        
        # Only keep sows with at least 15 days of data for reliable clustering
        min_days = 15
        valid_sows = trajectory_matrix.dropna(thresh=min_days, axis=0)
        
        print(f"✅ Data prepared: {len(valid_sows)} sows with sufficient data")
        print(f"📈 Lactation days range: {trajectory_matrix.columns.min()}-{trajectory_matrix.columns.max()}")
        
        self.sow_metadata = sow_metadata
        return trajectory_matrix, sow_metadata
    
    def offline_trajectory_clustering(self, trajectory_matrix, n_clusters_range=[2, 3, 4]):
        """
        OFFLINE LEARNING PROCEDURE
        Perform time-series clustering to identify distinct feed intake trajectory patterns.
        Uses DTW-based clustering similar to kShape for scale/shift invariance.
        
        Args:
            trajectory_matrix: Pivot table of feed intake trajectories
            n_clusters_range: Range of cluster numbers to evaluate
        
        Returns:
            Best clustering model and cluster assignments
        """
        print("\n🔍 OFFLINE LEARNING: Performing trajectory clustering...")
        
        # Prepare time series data for clustering
        # Convert to 3D array format required by tslearn
        time_series_data = []
        valid_sow_ids = []
        
        for sow_id in trajectory_matrix.index:
            trajectory = trajectory_matrix.loc[sow_id].values
            if not np.isnan(trajectory).all():  # Skip completely empty trajectories
                time_series_data.append(trajectory)
                valid_sow_ids.append(sow_id)
        
        time_series_data = np.array(time_series_data)
        print(f"📊 Clustering {len(time_series_data)} valid trajectories")
        
        # Scale the time series data to make clustering more robust
        self.trajectory_scaler = TimeSeriesScalerMeanVariance()
        scaled_time_series = self.trajectory_scaler.fit_transform(time_series_data)
        
        # Find optimal number of clusters
        best_score = -1
        best_n_clusters = 2
        best_model = None
        cluster_scores = {}
        
        for n_clusters in n_clusters_range:
            print(f"  Testing {n_clusters} clusters...")
            
            # Use DTW-based TimeSeriesKMeans (similar to kShape)
            model = TimeSeriesKMeans(n_clusters=n_clusters, 
                                   metric="dtw",  # Dynamic Time Warping for trajectory comparison
                                   max_iter=50,
                                   random_state=42)
            
            cluster_labels = model.fit_predict(scaled_time_series)
            
            # Calculate clustering quality metrics
            silhouette = silhouette_score(scaled_time_series.reshape(len(scaled_time_series), -1), cluster_labels)
            calinski = calinski_harabasz_score(scaled_time_series.reshape(len(scaled_time_series), -1), cluster_labels)
            
            cluster_scores[n_clusters] = {
                'silhouette': silhouette,
                'calinski': calinski,
                'model': model,
                'labels': cluster_labels
            }
            
            print(f"    Silhouette Score: {silhouette:.3f}")
            print(f"    Calinski-Harabasz Score: {calinski:.1f}")
            
            # Simple scoring: prioritize silhouette score
            if silhouette > best_score:
                best_score = silhouette
                best_n_clusters = n_clusters
                best_model = model
        
        print(f"\n🎯 Best clustering: {best_n_clusters} clusters (Silhouette: {best_score:.3f})")
        
        # Store results
        self.cluster_model = best_model
        self.cluster_labels = cluster_scores[best_n_clusters]['labels']
        self.cluster_centers = best_model.cluster_centers_
        
        # Create cluster assignment DataFrame
        cluster_assignments = pd.DataFrame({
            'sow_id': valid_sow_ids,
            'cluster': self.cluster_labels
        })
        
        return cluster_assignments, cluster_scores
    
    def analyze_cluster_characteristics(self, trajectory_matrix, cluster_assignments):
        """
        Analyze and visualize the characteristics of identified trajectory clusters.
        """
        print("\n📈 Analyzing cluster characteristics...")
        
        # Merge cluster assignments with metadata
        cluster_data = cluster_assignments.merge(self.sow_metadata, on='sow_id', how='left')
        
        # Plot cluster trajectories
        plt.figure(figsize=(15, 10))
        
        n_clusters = len(np.unique(cluster_assignments['cluster']))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for i, cluster in enumerate(np.unique(cluster_assignments['cluster'])):
            plt.subplot(2, 2, i+1)
            
            cluster_sows = cluster_assignments[cluster_assignments['cluster'] == cluster]['sow_id']
            cluster_trajectories = trajectory_matrix.loc[cluster_sows]
            
            # Plot individual trajectories (light lines)
            for sow_id in cluster_sows[:20]:  # Show first 20 for clarity
                trajectory = trajectory_matrix.loc[sow_id]
                days = trajectory.index
                plt.plot(days, trajectory.values, alpha=0.3, color=colors[i], linewidth=0.5)
            
            # Plot mean trajectory (bold line)
            mean_trajectory = cluster_trajectories.mean()
            plt.plot(mean_trajectory.index, mean_trajectory.values, 
                    color=colors[i], linewidth=3, label=f'Cluster {cluster} Mean')
            
            plt.title(f'Cluster {cluster} Trajectories (n={len(cluster_sows)})')
            plt.xlabel('Lactation Day')
            plt.ylabel('Daily Feed Intake (kg)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print cluster statistics
        print("\n📊 Cluster Statistics:")
        for cluster in np.unique(cluster_assignments['cluster']):
            cluster_sows = cluster_data[cluster_data['cluster'] == cluster]
            print(f"\nCluster {cluster} (n={len(cluster_sows)}):")
            print(f"  Parity distribution: {cluster_sows['parity'].value_counts().to_dict()}")
            print(f"  Avg temperature: {cluster_sows['avg_temp'].mean():.1f}°C")
            print(f"  Avg lactation length: {cluster_sows['lactation_length'].mean():.1f} days")
            
            # Calculate trajectory characteristics
            cluster_trajectories = trajectory_matrix.loc[cluster_sows['sow_id']]
            mean_traj = cluster_trajectories.mean()
            
            print(f"  Initial intake (day 2): {mean_traj.iloc[0]:.2f} kg")
            print(f"  Peak intake: {mean_traj.max():.2f} kg (day {mean_traj.idxmax()})")
            print(f"  Final intake: {mean_traj.iloc[-1]:.2f} kg")
    
    def train_early_classifier(self, df, cluster_assignments, early_days=[2, 3, 4, 5]):
        """
        ONLINE FORECASTING PROCEDURE - Part 1
        Train a classifier to predict cluster membership from early lactation data (days 1-5).
        This enables real-time cluster assignment for new sows.
        
        Args:
            df: Original DataFrame
            cluster_assignments: Results from offline clustering
            early_days: Which early lactation days to use for prediction
        
        Returns:
            Trained classifier for early cluster prediction
        """
        print(f"\n🤖 ONLINE FORECASTING: Training early classifier using days {early_days}...")
        
        # Prepare features from early lactation days
        early_features = []
        target_clusters = []
        
        for sow_id in cluster_assignments['sow_id']:
            sow_data = df[df['sow_id'] == sow_id]
            cluster = cluster_assignments[cluster_assignments['sow_id'] == sow_id]['cluster'].iloc[0]
            
            # Get early lactation data
            early_data = sow_data[sow_data['lactation_day'].isin(early_days)]
            
            if len(early_data) >= len(early_days) - 1:  # Allow missing 1 day
                features = []
                
                # Feed intake features
                feed_intakes = early_data['daily_feed_intake'].values
                features.extend([
                    np.mean(feed_intakes),  # Average early intake
                    np.std(feed_intakes),   # Variability in early intake
                    np.max(feed_intakes),   # Peak early intake
                    feed_intakes[-1] - feed_intakes[0] if len(feed_intakes) > 1 else 0,  # Trend
                ])
                
                # Environmental features
                features.extend([
                    early_data['temperature'].mean(),
                    early_data['dew_point'].mean(),
                    early_data['Hours_T2M_GT_24'].mean(),
                ])
                
                # Sow characteristics
                parity = early_data['Parity'].iloc[0]
                parity_encoding = {'P1': 1, 'P2': 2, 'P3+': 3}
                features.append(parity_encoding.get(parity, 1))
                
                early_features.append(features)
                target_clusters.append(cluster)
        
        # Convert to arrays
        X = np.array(early_features)
        y = np.array(target_clusters)
        
        print(f"📊 Training data: {len(X)} sows with early lactation features")
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest classifier (good for this type of tabular data)
        self.early_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.early_classifier.fit(X_train, y_train)
        
        # Evaluate classifier
        y_pred = self.early_classifier.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"✅ Early classifier accuracy: {accuracy:.3f}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_names = ['Avg_Early_Intake', 'Std_Early_Intake', 'Max_Early_Intake', 'Intake_Trend',
                        'Avg_Temperature', 'Avg_DewPoint', 'Avg_Hours_GT24', 'Parity']
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.early_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Feature Importance:")
        for _, row in importance_df.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return self.early_classifier
    
    def forecast_remaining_intake(self, df, cluster_assignments, forecast_day=10):
        """
        ONLINE FORECASTING PROCEDURE - Part 2
        Develop predictive functions to forecast remaining lactation feed intake
        based on cluster membership and environmental variables.
        
        Args:
            df: Original DataFrame
            cluster_assignments: Results from offline clustering
            forecast_day: Day from which to forecast remaining intake
        
        Returns:
            Forecasting results and evaluation metrics
        """
        print(f"\n📊 ONLINE FORECASTING: Training intake forecasting from day {forecast_day}...")
        
        forecast_results = []
        
        # For each cluster, build a forecasting model
        cluster_models = {}
        
        for cluster in np.unique(cluster_assignments['cluster']):
            print(f"\n  Training forecasting model for Cluster {cluster}...")
            
            cluster_sows = cluster_assignments[cluster_assignments['cluster'] == cluster]['sow_id']
            cluster_data = []
            
            for sow_id in cluster_sows:
                sow_data = df[df['sow_id'] == sow_id].sort_values('lactation_day')
                
                if len(sow_data) > forecast_day + 3:  # Need some days after forecast_day
                    # Features: intake up to forecast_day + environmental variables
                    intake_history = sow_data[sow_data['lactation_day'] <= forecast_day]['daily_feed_intake'].values
                    
                    if len(intake_history) >= forecast_day - 1:  # Sufficient history
                        features = []
                        
                        # Historical intake features
                        features.extend([
                            np.mean(intake_history),
                            np.std(intake_history),
                            np.max(intake_history),
                            intake_history[-1],  # Most recent intake
                            intake_history[-1] - intake_history[0] if len(intake_history) > 1 else 0,  # Trend
                        ])
                        
                        # Environmental features (average up to forecast day)
                        env_data = sow_data[sow_data['lactation_day'] <= forecast_day]
                        features.extend([
                            env_data['temperature'].mean(),
                            env_data['dew_point'].mean(),
                            env_data['Hours_T2M_GT_24'].mean(),
                        ])
                        
                        # Target: remaining total intake
                        remaining_data = sow_data[sow_data['lactation_day'] > forecast_day]
                        total_remaining = remaining_data['daily_feed_intake'].sum()
                        
                        cluster_data.append({
                            'features': features,
                            'target': total_remaining,
                            'days_remaining': len(remaining_data),
                            'sow_id': sow_id
                        })
            
            if len(cluster_data) > 10:  # Need enough data to train
                # Prepare training data
                X_cluster = np.array([item['features'] for item in cluster_data])
                y_cluster = np.array([item['target'] for item in cluster_data])
                
                # Train Random Forest regressor for this cluster
                from sklearn.ensemble import RandomForestRegressor
                cluster_model = RandomForestRegressor(n_estimators=100, random_state=42)
                cluster_model.fit(X_cluster, y_cluster)
                
                cluster_models[cluster] = {
                    'model': cluster_model,
                    'training_data': cluster_data
                }
                
                # Evaluate with cross-validation
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(cluster_model, X_cluster, y_cluster, cv=5, scoring='neg_mean_squared_error')
                rmse = np.sqrt(-cv_scores.mean())
                
                print(f"    Cluster {cluster}: {len(cluster_data)} training samples, RMSE: {rmse:.2f} kg")
        
        self.cluster_forecasting_models = cluster_models
        return cluster_models
    
    def predict_new_sow(self, early_data_dict, forecast_day=10):
        """
        Complete online prediction pipeline for a new sow.
        1. Classify into trajectory cluster using early data
        2. Forecast remaining intake based on cluster membership
        
        Args:
            early_data_dict: Dictionary with early lactation data
                           {'feed_intake': [day2, day3, day4, day5], 
                            'temperature': [temp2, temp3, temp4, temp5],
                            'dew_point': [...], 'hours_gt24': [...], 'parity': 'P1'}
            forecast_day: Day from which to forecast
        
        Returns:
            Prediction results
        """
        print(f"\n🔮 ONLINE PREDICTION for new sow...")
        
        # Step 1: Predict cluster membership
        feed_intakes = early_data_dict['feed_intake']
        features = [
            np.mean(feed_intakes),
            np.std(feed_intakes),
            np.max(feed_intakes),
            feed_intakes[-1] - feed_intakes[0] if len(feed_intakes) > 1 else 0,
            np.mean(early_data_dict['temperature']),
            np.mean(early_data_dict['dew_point']),
            np.mean(early_data_dict['hours_gt24']),
            {'P1': 1, 'P2': 2, 'P3+': 3}.get(early_data_dict['parity'], 1)
        ]
        
        predicted_cluster = self.early_classifier.predict([features])[0]
        cluster_proba = self.early_classifier.predict_proba([features])[0]
        
        print(f"  🎯 Predicted cluster: {predicted_cluster}")
        print(f"  📊 Cluster probabilities: {dict(zip(self.early_classifier.classes_, cluster_proba))}")
        
        # Step 2: Forecast remaining intake (if we have the forecasting model)
        if hasattr(self, 'cluster_forecasting_models') and predicted_cluster in self.cluster_forecasting_models:
            # This would require intake history up to forecast_day
            # For demo purposes, we'll show how it would work
            print(f"  📈 Forecasting model available for cluster {predicted_cluster}")
            print(f"  💡 To forecast remaining intake, provide intake data up to day {forecast_day}")
        
        return {
            'predicted_cluster': predicted_cluster,
            'cluster_probabilities': dict(zip(self.early_classifier.classes_, cluster_proba))
        }

# Example usage and main execution
def main():
    """
    Main execution function demonstrating the complete workflow.
    """
    print("🚀 Sow Feed Intake Trajectory Analysis and Forecasting System")
    print("=" * 70)
    
    # Initialize the forecasting system
    forecaster = SowFeedIntakeForecaster()
    
    # Note: Replace this with your actual data loading
    print("\n📁 Load your data using:")
    print("df = pd.read_csv('your_sow_data.csv')")
    print("\nRequired columns: sow_id, lactation_day, daily_feed_intake, Parity, temperature, dew_point, Hours_T2M_GT_24")
    
    # Demonstration with sample workflow (uncomment when you have data):
    """
    # 1. Load and prepare data
    trajectory_matrix, sow_metadata = forecaster.load_and_prepare_data(df)
    
    # 2. OFFLINE LEARNING: Perform trajectory clustering
    cluster_assignments, cluster_scores = forecaster.offline_trajectory_clustering(trajectory_matrix)
    
    # 3. Analyze cluster characteristics
    forecaster.analyze_cluster_characteristics(trajectory_matrix, cluster_assignments)
    
    # 4. ONLINE FORECASTING: Train early classifier
    early_classifier = forecaster.train_early_classifier(df, cluster_assignments)
    
    # 5. Train intake forecasting models
    forecasting_models = forecaster.forecast_remaining_intake(df, cluster_assignments)
    
    # 6. Example prediction for a new sow
    new_sow_data = {
        'feed_intake': [3.2, 3.8, 4.1, 4.5],  # Days 2-5
        'temperature': [25.1, 26.3, 24.8, 27.2],
        'dew_point': [18.5, 19.1, 17.8, 20.1],
        'hours_gt24': [6, 8, 5, 9],
        'parity': 'P1'
    }
    
    prediction = forecaster.predict_new_sow(new_sow_data)
    print(f"Prediction results: {prediction}")
    """
    
    print("\n✅ System ready! Load your data and uncomment the execution code above.")

if __name__ == "__main__":
    main()

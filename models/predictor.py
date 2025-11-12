#!/usr/bin/env python3
"""
Pass Probability Predictor
Logistic regression model for predicting certification exam success
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import pickle


class PassPredictor:
    """Predicts certification exam pass probability using logistic regression"""
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_names = [
            'practice_score',
            'study_hours',
            'compute_score',
            'storage_score',
            'networking_score',
            'security_score',
            'previous_attempts',
            'domain_balance'
        ]
        self.is_trained = False
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for modeling
        
        Args:
            data: DataFrame with learner information
        
        Returns:
            Feature matrix
        """
        features = []
        
        for _, row in data.iterrows():
            # Calculate domain balance (standard deviation of domain scores)
            domain_scores = [
                row.get('compute_score', 0),
                row.get('storage_score', 0),
                row.get('networking_score', 0),
                row.get('security_score', 0)
            ]
            domain_balance = 100 - np.std(domain_scores)  # Higher = more balanced
            
            feature_row = [
                row.get('practice_score', 0),
                row.get('study_hours', 0),
                row.get('compute_score', 0),
                row.get('storage_score', 0),
                row.get('networking_score', 0),
                row.get('security_score', 0),
                row.get('previous_attempts', 0),
                domain_balance
            ]
            features.append(feature_row)
        
        return np.array(features)
    
    def train(self, data: pd.DataFrame, target_column: str = 'passed') -> Dict:
        """
        Train the prediction model
        
        Args:
            data: Training data with features and target
            target_column: Name of the target column
        
        Returns:
            Training metrics
        """
        # Prepare features and target
        X = self.prepare_features(data)
        y = data[target_column].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall': round(recall_score(y_test, y_pred), 4),
            'f1_score': round(2 * (precision_score(y_test, y_pred) * recall_score(y_test, y_pred)) / 
                            (precision_score(y_test, y_pred) + recall_score(y_test, y_pred)), 4),
            'auc_roc': round(roc_auc_score(y_test, y_pred_proba), 4),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        metrics['cv_score_mean'] = round(cv_scores.mean(), 4)
        metrics['cv_score_std'] = round(cv_scores.std(), 4)
        
        # Feature importance (coefficients)
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_importance[name] = round(self.model.coef_[0][i], 4)
        
        metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def predict(self, learner_data: Dict) -> Dict:
        """
        Predict pass probability for a learner
        
        Args:
            learner_data: Dictionary with learner features
        
        Returns:
            Prediction results with probability and recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        domain_scores = [
            learner_data.get('compute_score', 0),
            learner_data.get('storage_score', 0),
            learner_data.get('networking_score', 0),
            learner_data.get('security_score', 0)
        ]
        domain_balance = 100 - np.std(domain_scores)
        
        features = np.array([[
            learner_data.get('practice_score', 0),
            learner_data.get('study_hours', 0),
            learner_data.get('compute_score', 0),
            learner_data.get('storage_score', 0),
            learner_data.get('networking_score', 0),
            learner_data.get('security_score', 0),
            learner_data.get('previous_attempts', 0),
            domain_balance
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        probability = self.model.predict_proba(features_scaled)[0][1]
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence interval (approximate)
        ci_margin = 1.96 * 0.05  # 95% CI with assumed SE
        ci_lower = max(0, probability - ci_margin)
        ci_upper = min(1, probability + ci_margin)
        
        # Determine risk level
        if probability >= 0.80:
            risk_level = "low"
        elif probability >= 0.60:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(learner_data, probability)
        
        # Key factors (top 3 by importance)
        feature_importance = []
        for name, value in zip(self.feature_names, features[0]):
            coef = self.model.coef_[0][self.feature_names.index(name)]
            feature_importance.append({
                'factor': name,
                'importance': round(abs(coef), 3),
                'current_value': round(value, 1)
            })
        
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            'pass_probability': round(probability, 3),
            'confidence_interval': [round(ci_lower, 3), round(ci_upper, 3)],
            'prediction': 'pass' if prediction == 1 else 'fail',
            'risk_level': risk_level,
            'key_factors': feature_importance[:3],
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, learner_data: Dict, probability: float) -> List[str]:
        """Generate personalized recommendations based on learner data"""
        recommendations = []
        
        practice_score = learner_data.get('practice_score', 0)
        study_hours = learner_data.get('study_hours', 0)
        
        domain_scores = {
            'compute': learner_data.get('compute_score', 0),
            'storage': learner_data.get('storage_score', 0),
            'networking': learner_data.get('networking_score', 0),
            'security': learner_data.get('security_score', 0)
        }
        
        # Overall readiness
        if probability >= 0.80:
            recommendations.append("Strong performance predicted - maintain current study approach")
        elif probability >= 0.60:
            recommendations.append("Moderate readiness - additional preparation recommended")
        else:
            recommendations.append("High risk - significant additional study needed before attempting exam")
        
        # Practice score
        if practice_score < 70:
            recommendations.append(f"Practice score is low ({practice_score}%) - aim for 75%+ before exam")
        
        # Study hours
        if study_hours < 60:
            recommendations.append(f"Study time below optimal range - consider {60 - study_hours} more hours")
        elif study_hours > 100:
            recommendations.append("Very high study hours - ensure quality over quantity")
        
        # Weak domains
        weak_domains = [domain for domain, score in domain_scores.items() if score < 65]
        if weak_domains:
            recommendations.append(f"Focus on weak domains: {', '.join(weak_domains)}")
        
        # Domain balance
        domain_std = np.std(list(domain_scores.values()))
        if domain_std > 15:
            recommendations.append("Unbalanced domain knowledge - work on weaker areas for better overall performance")
        
        return recommendations
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True


# Example usage
if __name__ == '__main__':
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'practice_score': np.random.normal(70, 15, n_samples),
        'study_hours': np.random.normal(65, 20, n_samples),
        'compute_score': np.random.normal(75, 12, n_samples),
        'storage_score': np.random.normal(72, 13, n_samples),
        'networking_score': np.random.normal(68, 14, n_samples),
        'security_score': np.random.normal(73, 12, n_samples),
        'previous_attempts': np.random.poisson(0.3, n_samples)
    })
    
    # Clip scores to 0-100 range
    for col in ['practice_score', 'compute_score', 'storage_score', 'networking_score', 'security_score']:
        sample_data[col] = sample_data[col].clip(0, 100)
    
    # Generate target (passed/failed) based on features
    sample_data['passed'] = (
        (sample_data['practice_score'] > 65) & 
        (sample_data['study_hours'] > 50) & 
        (sample_data[['compute_score', 'storage_score', 'networking_score', 'security_score']].mean(axis=1) > 65)
    ).astype(int)
    
    # Train model
    predictor = PassPredictor()
    print("\n" + "="*70)
    print("Training Pass Probability Predictor")
    print("="*70 + "\n")
    
    metrics = predictor.train(sample_data)
    
    print("Model Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall: {metrics['recall']:.1%}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    print(f"  CV Score: {metrics['cv_score_mean']:.1%} (±{metrics['cv_score_std']:.3f})")
    
    print("\nFeature Importance (Coefficients):")
    for feature, importance in sorted(metrics['feature_importance'].items(), 
                                     key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feature}: {importance:+.3f}")
    
    # Test prediction
    print("\n" + "="*70)
    print("Sample Prediction")
    print("="*70 + "\n")
    
    test_learner = {
        'practice_score': 75,
        'study_hours': 60,
        'compute_score': 80,
        'storage_score': 70,
        'networking_score': 65,
        'security_score': 75,
        'previous_attempts': 0
    }
    
    result = predictor.predict(test_learner)
    
    print(f"Pass Probability: {result['pass_probability']:.1%}")
    print(f"95% CI: [{result['confidence_interval'][0]:.1%}, {result['confidence_interval'][1]:.1%}]")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Risk Level: {result['risk_level'].upper()}")
    
    print("\nKey Factors:")
    for factor in result['key_factors']:
        print(f"  {factor['factor']}: importance={factor['importance']:.3f}, value={factor['current_value']:.1f}")
    
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")

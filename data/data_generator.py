#!/usr/bin/env python3
"""
Synthetic Data Generator
Generates realistic exam performance data for testing and demonstration
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta


class ExamDataGenerator:
    """Generates synthetic exam performance data"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        self.domains = ['compute', 'storage', 'networking', 'security']
        self.learner_segments = ['beginner', 'intermediate', 'advanced']
    
    def generate_learners(self, n_learners: int = 1000) -> pd.DataFrame:
        """Generate learner profiles with exam results"""
        
        learners = []
        
        for i in range(n_learners):
            # Determine learner segment
            segment = np.random.choice(self.learner_segments, p=[0.3, 0.5, 0.2])
            
            # Generate features based on segment
            if segment == 'beginner':
                practice_score = np.random.normal(55, 12, 1)[0]
                study_hours = np.random.normal(45, 15, 1)[0]
                domain_base = 55
            elif segment == 'intermediate':
                practice_score = np.random.normal(70, 10, 1)[0]
                study_hours = np.random.normal(65, 12, 1)[0]
                domain_base = 70
            else:  # advanced
                practice_score = np.random.normal(82, 8, 1)[0]
                study_hours = np.random.normal(75, 10, 1)[0]
                domain_base = 80
            
            # Generate domain scores with some variation
            compute_score = domain_base + np.random.normal(0, 8, 1)[0]
            storage_score = domain_base + np.random.normal(-3, 8, 1)[0]
            networking_score = domain_base + np.random.normal(-5, 10, 1)[0]  # Typically harder
            security_score = domain_base + np.random.normal(2, 8, 1)[0]
            
            # Clip scores to 0-100
            practice_score = np.clip(practice_score, 0, 100)
            domain_scores = [
                np.clip(compute_score, 0, 100),
                np.clip(storage_score, 0, 100),
                np.clip(networking_score, 0, 100),
                np.clip(security_score, 0, 100)
            ]
            
            # Previous attempts (mostly 0, some 1-2)
            previous_attempts = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            
            # Calculate pass probability (logistic function)
            score_factor = (practice_score - 50) / 50
            hours_factor = (study_hours - 40) / 40
            domain_factor = (np.mean(domain_scores) - 50) / 50
            attempts_penalty = -0.3 * previous_attempts
            
            logit = 0.5 + 1.5*score_factor + 0.8*hours_factor + 1.0*domain_factor + attempts_penalty
            pass_prob = 1 / (1 + np.exp(-logit))
            
            # Determine if passed
            passed = 1 if np.random.random() < pass_prob else 0
            
            # Generate final score (if passed, higher score)
            if passed:
                final_score = np.random.normal(75, 8, 1)[0]
            else:
                final_score = np.random.normal(58, 10, 1)[0]
            final_score = np.clip(final_score, 0, 100)
            
            learner = {
                'learner_id': f'learner_{i+1:04d}',
                'segment': segment,
                'practice_score': round(practice_score, 1),
                'study_hours': round(study_hours, 1),
                'compute_score': round(domain_scores[0], 1),
                'storage_score': round(domain_scores[1], 1),
                'networking_score': round(domain_scores[2], 1),
                'security_score': round(domain_scores[3], 1),
                'previous_attempts': int(previous_attempts),
                'passed': int(passed),
                'final_score': round(final_score, 1),
                'exam_date': (datetime.now() - timedelta(days=np.random.randint(1, 180))).strftime('%Y-%m-%d')
            }
            
            learners.append(learner)
        
        return pd.DataFrame(learners)
    
    def generate_question_responses(self, n_learners: int = 100, 
                                    n_questions: int = 65) -> pd.DataFrame:
        """Generate item-level responses for IRT analysis"""
        
        # Generate learner abilities
        abilities = np.random.normal(0, 1, n_learners)
        
        # Generate question parameters
        difficulties = np.random.uniform(-2, 2, n_questions)
        discriminations = np.random.uniform(0.5, 2.5, n_questions)
        
        # Add some problematic questions
        difficulties[0] = -2.5  # Too easy
        difficulties[1] = 2.8   # Too hard
        discriminations[2] = 0.1  # Low discrimination
        
        # Generate responses using 2PL IRT
        responses = np.zeros((n_learners, n_questions))
        for i in range(n_learners):
            for j in range(n_questions):
                prob = 1 / (1 + np.exp(-discriminations[j] * (abilities[i] - difficulties[j])))
                responses[i, j] = 1 if np.random.random() < prob else 0
        
        # Create DataFrame
        columns = [f'Q_{i+1:03d}' for i in range(n_questions)]
        return pd.DataFrame(responses, columns=columns)
    
    def save_data(self, learners_df: pd.DataFrame, filepath: str = 'data/sample_data.json'):
        """Save generated data to JSON file"""
        data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'n_learners': len(learners_df),
                'description': 'Synthetic certification exam performance data'
            },
            'learners': learners_df.to_dict('records')
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data saved to {filepath}")


# Example usage
if __name__ == '__main__':
    generator = ExamDataGenerator()
    
    print("\n" + "="*70)
    print("Generating Synthetic Exam Data")
    print("="*70 + "\n")
    
    # Generate learner data
    print("Generating learner profiles...")
    learners = generator.generate_learners(n_learners=1000)
    
    print(f"Generated {len(learners)} learner profiles")
    print(f"\nPass Rate: {learners['passed'].mean():.1%}")
    print(f"Average Final Score: {learners['final_score'].mean():.1f}")
    
    print("\nSegment Distribution:")
    print(learners['segment'].value_counts())
    
    print("\nSample Learner:")
    print(learners.iloc[0].to_dict())
    
    # Generate question responses
    print("\n" + "="*70)
    print("Generating Question Responses")
    print("="*70 + "\n")
    
    responses = generator.generate_question_responses(n_learners=200, n_questions=65)
    print(f"Generated {len(responses)} x {len(responses.columns)} response matrix")
    print(f"Average Score: {responses.sum(axis=1).mean():.1f}/{len(responses.columns)}")
    
    # Save data
    print("\n" + "="*70)
    print("Saving Data")
    print("="*70 + "\n")
    
    generator.save_data(learners, 'sample_data.json')
    
    # Save as CSV too
    learners.to_csv('learners_data.csv', index=False)
    print("Data also saved to learners_data.csv")
    
    # Statistics
    print("\n" + "="*70)
    print("Data Statistics")
    print("="*70 + "\n")
    
    print("Practice Score Statistics:")
    print(f"  Mean: {learners['practice_score'].mean():.1f}")
    print(f"  Std: {learners['practice_score'].std():.1f}")
    print(f"  Range: [{learners['practice_score'].min():.1f}, {learners['practice_score'].max():.1f}]")
    
    print("\nStudy Hours Statistics:")
    print(f"  Mean: {learners['study_hours'].mean():.1f}")
    print(f"  Std: {learners['study_hours'].std():.1f}")
    
    print("\nDomain Scores (Average):")
    for domain in ['compute', 'storage', 'networking', 'security']:
        col = f'{domain}_score'
        print(f"  {domain.capitalize()}: {learners[col].mean():.1f}")

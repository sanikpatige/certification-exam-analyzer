#!/usr/bin/env python3
"""
Question Quality Analysis
Analyzes exam questions using classical test theory and IRT
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List


class QuestionAnalyzer:
    """Analyzes exam question quality and psychometric properties"""
    
    def __init__(self):
        self.questions = {}
    
    def analyze_question(self, question_id: str, responses: np.ndarray, 
                        total_scores: np.ndarray) -> Dict:
        """
        Analyze a single question
        
        Args:
            question_id: Question identifier
            responses: Binary array (1=correct, 0=incorrect)
            total_scores: Array of total exam scores for each learner
        
        Returns:
            Question statistics
        """
        n_responses = len(responses)
        n_correct = np.sum(responses)
        
        # Difficulty (proportion correct)
        difficulty = n_correct / n_responses
        
        # Discrimination (point-biserial correlation)
        if np.std(responses) > 0 and np.std(total_scores) > 0:
            discrimination = stats.pointbiserialr(responses, total_scores)[0]
        else:
            discrimination = 0
        
        # Classify difficulty
        if difficulty >= 0.85:
            difficulty_category = "easy"
        elif difficulty >= 0.50:
            difficulty_category = "medium"
        else:
            difficulty_category = "hard"
        
        # Flag issues
        issues = []
        if discrimination < 0.2:
            issues.append("low_discrimination")
        if difficulty > 0.90:
            issues.append("too_easy")
        if difficulty < 0.30:
            issues.append("too_hard")
        if discrimination < 0:
            issues.append("negative_discrimination")
        
        # Recommendation
        if "negative_discrimination" in issues:
            recommendation = "Critical: Question may have wrong answer key or is misleading"
        elif "low_discrimination" in issues:
            recommendation = "Review question wording - may be confusing or poorly written"
        elif "too_easy" in issues:
            recommendation = "Consider increasing difficulty or removing question"
        elif "too_hard" in issues:
            recommendation = "Review for clarity or consider as bonus/advanced question"
        else:
            recommendation = "Question performs well - no changes needed"
        
        analysis = {
            'question_id': question_id,
            'n_responses': n_responses,
            'difficulty': round(difficulty, 3),
            'discrimination': round(discrimination, 3),
            'difficulty_category': difficulty_category,
            'correct_rate': round(difficulty, 3),
            'item_total_correlation': round(discrimination, 3),
            'issues': issues,
            'recommendation': recommendation
        }
        
        self.questions[question_id] = analysis
        return analysis
    
    def calculate_reliability(self, item_responses: pd.DataFrame) -> Dict:
        """
        Calculate exam reliability using Cronbach's alpha
        
        Args:
            item_responses: DataFrame where rows=learners, columns=questions (0/1)
        
        Returns:
            Reliability metrics
        """
        n_items = item_responses.shape[1]
        
        # Calculate variance of each item
        item_variances = item_responses.var(axis=0)
        
        # Calculate total score variance
        total_scores = item_responses.sum(axis=1)
        total_variance = total_scores.var()
        
        # Cronbach's alpha
        if total_variance > 0:
            alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
        else:
            alpha = 0
        
        # Standard error of measurement
        total_std = total_scores.std()
        sem = total_std * np.sqrt(1 - alpha)
        
        # Interpretation
        if alpha >= 0.90:
            interpretation = "Excellent"
        elif alpha >= 0.80:
            interpretation = "Good"
        elif alpha >= 0.70:
            interpretation = "Acceptable"
        else:
            interpretation = "Poor"
        
        return {
            'cronbach_alpha': round(alpha, 3),
            'standard_error': round(sem, 2),
            'interpretation': interpretation,
            'n_items': n_items,
            'n_learners': len(item_responses)
        }
    
    def analyze_exam(self, item_responses: pd.DataFrame, 
                    total_scores: np.ndarray = None) -> Dict:
        """
        Comprehensive exam analysis
        
        Args:
            item_responses: DataFrame where rows=learners, columns=questions (0/1)
            total_scores: Optional total scores (if None, calculated from items)
        
        Returns:
            Complete exam analysis
        """
        if total_scores is None:
            total_scores = item_responses.sum(axis=1).values
        
        # Analyze each question
        question_analyses = []
        for question_id in item_responses.columns:
            responses = item_responses[question_id].values
            analysis = self.analyze_question(question_id, responses, total_scores)
            question_analyses.append(analysis)
        
        # Calculate reliability
        reliability = self.calculate_reliability(item_responses)
        
        # Summary statistics
        difficulties = [q['difficulty'] for q in question_analyses]
        discriminations = [q['discrimination'] for q in question_analyses]
        
        # Count by difficulty
        difficulty_dist = {
            'easy': sum(1 for q in question_analyses if q['difficulty_category'] == 'easy'),
            'medium': sum(1 for q in question_analyses if q['difficulty_category'] == 'medium'),
            'hard': sum(1 for q in question_analyses if q['difficulty_category'] == 'hard')
        }
        
        # Flagged questions
        flagged = [q for q in question_analyses if q['issues']]
        
        return {
            'total_questions': len(question_analyses),
            'difficulty_distribution': difficulty_dist,
            'average_difficulty': round(np.mean(difficulties), 3),
            'average_discrimination': round(np.mean(discriminations), 3),
            'reliability': reliability,
            'flagged_questions': flagged,
            'question_details': question_analyses
        }
    
    def get_summary(self) -> Dict:
        """Get summary of analyzed questions"""
        if not self.questions:
            return {'total_questions': 0}
        
        flagged = [q for q in self.questions.values() if q['issues']]
        
        return {
            'total_questions': len(self.questions),
            'flagged_count': len(flagged),
            'average_difficulty': round(np.mean([q['difficulty'] for q in self.questions.values()]), 3),
            'average_discrimination': round(np.mean([q['discrimination'] for q in self.questions.values()]), 3)
        }


# Example usage
if __name__ == '__main__':
    np.random.seed(42)
    
    # Simulate exam responses (100 learners, 20 questions)
    n_learners = 100
    n_questions = 20
    
    # Generate ability levels for learners
    abilities = np.random.normal(0, 1, n_learners)
    
    # Generate question parameters
    difficulties = np.random.uniform(-2, 2, n_questions)
    discriminations = np.random.uniform(0.5, 2, n_questions)
    
    # Simulate responses using 2PL IRT model
    responses = np.zeros((n_learners, n_questions))
    for i in range(n_learners):
        for j in range(n_questions):
            prob = 1 / (1 + np.exp(-discriminations[j] * (abilities[i] - difficulties[j])))
            responses[i, j] = 1 if np.random.random() < prob else 0
    
    # Create DataFrame
    question_ids = [f"Q_{i+1:03d}" for i in range(n_questions)]
    item_responses = pd.DataFrame(responses, columns=question_ids)
    
    # Analyze exam
    analyzer = QuestionAnalyzer()
    
    print("\n" + "="*70)
    print("Exam Question Analysis")
    print("="*70 + "\n")
    
    exam_analysis = analyzer.analyze_exam(item_responses)
    
    print("Overall Statistics:")
    print(f"  Total Questions: {exam_analysis['total_questions']}")
    print(f"  Average Difficulty: {exam_analysis['average_difficulty']:.3f}")
    print(f"  Average Discrimination: {exam_analysis['average_discrimination']:.3f}")
    
    print("\nDifficulty Distribution:")
    for category, count in exam_analysis['difficulty_distribution'].items():
        print(f"  {category.capitalize()}: {count}")
    
    print("\nReliability:")
    rel = exam_analysis['reliability']
    print(f"  Cronbach's Alpha: {rel['cronbach_alpha']:.3f} ({rel['interpretation']})")
    print(f"  Standard Error: {rel['standard_error']:.2f}")
    
    print(f"\nFlagged Questions: {len(exam_analysis['flagged_questions'])}")
    for q in exam_analysis['flagged_questions'][:3]:  # Show first 3
        print(f"\n  {q['question_id']}:")
        print(f"    Difficulty: {q['difficulty']:.3f}")
        print(f"    Discrimination: {q['discrimination']:.3f}")
        print(f"    Issues: {', '.join(q['issues'])}")
        print(f"    Recommendation: {q['recommendation']}")

#!/usr/bin/env python3
"""
Test script for Certification Exam Analyzer
Tests all major functionality and displays results
"""

import requests
import time

API_URL = "http://localhost:8000"


def test_prediction():
    """Test pass probability prediction"""
    print("\n" + "="*70)
    print("🎯 Testing Pass Probability Prediction")
    print("="*70 + "\n")
    
    test_learners = [
        {
            "name": "High Performer",
            "data": {
                "learner_id": "test_001",
                "practice_score": 85,
                "study_hours": 75,
                "domain_scores": {
                    "compute": 88,
                    "storage": 82,
                    "networking": 80,
                    "security": 86
                },
                "previous_attempts": 0
            }
        },
        {
            "name": "At-Risk Learner",
            "data": {
                "learner_id": "test_002",
                "practice_score": 55,
                "study_hours": 35,
                "domain_scores": {
                    "compute": 60,
                    "storage": 58,
                    "networking": 50,
                    "security": 62
                },
                "previous_attempts": 1
            }
        },
        {
            "name": "Average Performer",
            "data": {
                "learner_id": "test_003",
                "practice_score": 70,
                "study_hours": 60,
                "domain_scores": {
                    "compute": 72,
                    "storage": 68,
                    "networking": 65,
                    "security": 71
                },
                "previous_attempts": 0
            }
        }
    ]
    
    for learner in test_learners:
        print(f"Testing: {learner['name']}")
        print("-" * 70)
        
        response = requests.post(f"{API_URL}/predict", json=learner['data'])
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Pass Probability: {result['pass_probability']:.1%}")
            print(f"95% CI: [{result['confidence_interval'][0]:.1%}, {result['confidence_interval'][1]:.1%}]")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Risk Level: {result['risk_level'].upper()}")
            
            print("\nKey Factors:")
            for factor in result['key_factors']:
                print(f"  • {factor['factor']}: {factor['current_value']:.1f} (importance: {factor['importance']:.3f})")
            
            print("\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  • {rec}")
            
            print()
        else:
            print(f"✗ Error: {response.status_code}")
        
        time.sleep(0.5)


def test_analytics():
    """Test analytics endpoints"""
    print("\n" + "="*70)
    print("📊 Testing Analytics")
    print("="*70 + "\n")
    
    # Summary
    print("Overall Summary:")
    print("-" * 70)
    response = requests.get(f"{API_URL}/analytics/summary")
    if response.status_code == 200:
        data = response.json()
        print(f"Total Exams: {data['total_exams']}")
        print(f"Pass Rate: {data['pass_rate']}%")
        print(f"Average Score: {data['average_score']}")
        print(f"\nDomain Performance:")
        for domain, score in data['domain_performance'].items():
            print(f"  {domain.capitalize()}: {score}")
        print(f"\nKey Correlations:")
        print(f"  Practice Score: r={data['practice_score_correlation']}")
        print(f"  Study Hours: r={data['study_hours_correlation']}")
    
    # Question Analysis
    print("\n" + "="*70)
    print("Question Quality Analysis:")
    print("-" * 70)
    response = requests.get(f"{API_URL}/analytics/questions")
    if response.status_code == 200:
        data = response.json()
        print(f"Total Questions: {data['total_questions']}")
        print(f"Average Difficulty: {data['average_difficulty']:.3f}")
        print(f"Average Discrimination: {data['average_discrimination']:.3f}")
        print(f"Cronbach's Alpha: {data['reliability']['cronbach_alpha']} ({data['reliability']['interpretation']})")
        print(f"\nFlagged Questions: {len(data['flagged_questions'])}")
        
        for q in data['flagged_questions'][:3]:
            print(f"\n  {q['question_id']}:")
            print(f"    Issues: {', '.join(q['issues'])}")
            print(f"    Recommendation: {q['recommendation']}")
    
    # Domain Analysis
    print("\n" + "="*70)
    print("Domain Analysis:")
    print("-" * 70)
    response = requests.get(f"{API_URL}/analytics/domains")
    if response.status_code == 200:
        data = response.json()
        print(f"Weakest Domain: {data['weakest_domain']['name']} ({data['weakest_domain']['average_score']})")
        print("\nDomain Statistics:")
        for domain, stats in data['domain_statistics'].items():
            print(f"\n  {domain.capitalize()}:")
            print(f"    Overall Average: {stats['overall_average']}")
            print(f"    Pass/Fail Gap: {stats['performance_gap']}")


def test_research_report():
    """Test research report generation"""
    print("\n" + "="*70)
    print("📄 Generating Research Report")
    print("="*70 + "\n")
    
    response = requests.get(f"{API_URL}/research/report")
    
    if response.status_code == 200:
        report = response.json()
        
        print("Executive Summary:")
        print("-" * 70)
        summary = report['executive_summary']
        print(f"Total Exams Analyzed: {summary['total_exams_analyzed']}")
        print(f"Overall Pass Rate: {summary['overall_pass_rate']}%")
        print(f"Model Accuracy: {summary['model_accuracy']}%")
        print(f"Key Finding: {summary['key_finding']}")
        
        print("\n" + "="*70)
        print("Recommendations:")
        print("-" * 70)
        
        print("\nFor Learners:")
        for rec in report['recommendations']['for_learners']:
            print(f"  • {rec}")
        
        print("\nFor Exam Designers:")
        for rec in report['recommendations']['for_exam_designers']:
            print(f"  • {rec}")
        
        print("\nFor Training Programs:")
        for rec in report['recommendations']['for_training_programs']:
            print(f"  • {rec}")


def main():
    print("\n" + "="*70)
    print("🎓 Certification Exam Analyzer - Test Suite")
    print("="*70)
    print("\nMake sure the API server is running: python app.py")
    print("\nPress Enter to start testing or Ctrl+C to cancel...")
    input()
    
    # Run tests
    test_prediction()
    test_analytics()
    test_research_report()
    
    print("\n" + "="*70)
    print("✅ All Tests Completed!")
    print("="*70)
    print("\nNext steps:")
    print("  1. View interactive API docs: http://localhost:8000/docs")
    print("  2. Generate full report: curl http://localhost:8000/research/report")
    print("  3. Test predictions with your own data")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")

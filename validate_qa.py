"""
QA Dataset Validation Script
Run this to check if your QA pairs are properly formatted

Usage: python validate_qa.py data/qa_pairs/qa_dataset.json
"""

import json
import os
from pathlib import Path

def validate_qa_dataset(qa_file_path):
    """
    Validates the QA dataset JSON file
    Returns: (is_valid, error_messages, warnings, statistics)
    """
    errors = []
    warnings = []
    stats = {
        'total_questions': 0,
        'by_type': {},
        'by_difficulty': {},
        'by_paper': {},
        'missing_pages': 0,
        'short_answers': 0,
        'short_questions': 0
    }
    
    # Check file exists
    if not os.path.exists(qa_file_path):
        return False, [f"File not found: {qa_file_path}"], warnings, stats
    
    # Load JSON
    try:
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON format: {str(e)}"], warnings, stats
    
    # Check if list
    if not isinstance(qa_data, list):
        return False, ["QA data must be a list of objects"], warnings, stats
    
    stats['total_questions'] = len(qa_data)
    
    # Validate each QA pair
    required_fields = ['id', 'paper_title', 'paper_file', 'page_numbers', 
                       'question', 'answer', 'question_type', 'difficulty']
    
    valid_types = ['factual', 'conceptual', 'comparative', 'methodological']
    valid_difficulties = ['easy', 'medium', 'hard']
    
    for idx, qa in enumerate(qa_data):
        qa_id = qa.get('id', idx + 1)
        
        # Check required fields
        for field in required_fields:
            if field not in qa:
                errors.append(f"QA #{qa_id}: Missing required field '{field}'")
        
        # Validate paper_title
        if 'paper_title' in qa:
            if not qa['paper_title'] or qa['paper_title'].strip() == '':
                errors.append(f"QA #{qa_id}: paper_title is empty")
            else:
                paper = qa['paper_title']
                stats['by_paper'][paper] = stats['by_paper'].get(paper, 0) + 1
        
        # Validate paper_file
        if 'paper_file' in qa:
            if not qa['paper_file'] or qa['paper_file'].strip() == '':
                errors.append(f"QA #{qa_id}: paper_file is empty")
        
        # Validate page_numbers
        if 'page_numbers' in qa:
            if not isinstance(qa['page_numbers'], list):
                errors.append(f"QA #{qa_id}: page_numbers must be a list")
            elif len(qa['page_numbers']) == 0:
                stats['missing_pages'] += 1
                warnings.append(f"QA #{qa_id}: No page numbers provided")
            else:
                # Check if all page numbers are integers
                for page in qa['page_numbers']:
                    if not isinstance(page, int):
                        errors.append(f"QA #{qa_id}: page_numbers must contain only integers, found {type(page)}")
        
        # Validate question
        if 'question' in qa:
            question = qa['question'].strip()
            if len(question) == 0:
                errors.append(f"QA #{qa_id}: Question is empty")
            elif len(question) < 10:
                stats['short_questions'] += 1
                warnings.append(f"QA #{qa_id}: Question is very short ({len(question)} chars)")
            if question and not question.endswith('?'):
                warnings.append(f"QA #{qa_id}: Question doesn't end with '?'")
        
        # Validate answer
        if 'answer' in qa:
            answer = qa['answer'].strip()
            if len(answer) == 0:
                errors.append(f"QA #{qa_id}: Answer is empty")
            elif len(answer) < 20:
                stats['short_answers'] += 1
                warnings.append(f"QA #{qa_id}: Answer is very short ({len(answer)} chars)")
        
        # Validate question_type
        if 'question_type' in qa:
            qtype = qa['question_type']
            if qtype not in valid_types:
                errors.append(f"QA #{qa_id}: Invalid question_type '{qtype}'. Must be one of {valid_types}")
            else:
                stats['by_type'][qtype] = stats['by_type'].get(qtype, 0) + 1
        
        # Validate difficulty
        if 'difficulty' in qa:
            diff = qa['difficulty']
            if diff not in valid_difficulties:
                errors.append(f"QA #{qa_id}: Invalid difficulty '{diff}'. Must be one of {valid_difficulties}")
            else:
                stats['by_difficulty'][diff] = stats['by_difficulty'].get(diff, 0) + 1
    
    # Check for duplicate IDs
    ids = [qa.get('id') for qa in qa_data if 'id' in qa]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate IDs found in dataset")
    
    # Check question count target
    if stats['total_questions'] < 100:
        warnings.append(f"Dataset has {stats['total_questions']} questions. Target is 100.")
    
    # Final validation
    is_valid = len(errors) == 0
    
    return is_valid, errors, warnings, stats


def print_validation_report(qa_file_path):
    """Print a detailed validation report"""
    print("="*70)
    print("QA DATASET VALIDATION REPORT")
    print("="*70)
    print(f"\nFile: {qa_file_path}\n")
    
    is_valid, errors, warnings, stats = validate_qa_dataset(qa_file_path)
    
    # Statistics
    print("STATISTICS:")
    print(f"  Total Questions: {stats['total_questions']} / 100 target")
    
    if stats['by_type']:
        print(f"\n  By Question Type:")
        for qtype, count in sorted(stats['by_type'].items()):
            print(f"    {qtype}: {count}")
    
    if stats['by_difficulty']:
        print(f"\n  By Difficulty:")
        for diff, count in sorted(stats['by_difficulty'].items()):
            print(f"    {diff}: {count}")
    
    if stats['by_paper']:
        print(f"\n  By Paper ({len(stats['by_paper'])} papers):")
        for paper, count in sorted(stats['by_paper'].items()):
            # Truncate long paper titles
            paper_display = paper if len(paper) <= 50 else paper[:47] + "..."
            print(f"    {paper_display}: {count}")
    
    print(f"\n  Potential Issues:")
    print(f"    Missing page numbers: {stats['missing_pages']}")
    print(f"    Short questions (<10 chars): {stats['short_questions']}")
    print(f"    Short answers (<20 chars): {stats['short_answers']}")
    
    # Errors
    if errors:
        print("\n" + "="*70)
        print("❌ ERRORS (MUST FIX BEFORE SUBMISSION):")
        print("="*70)
        for error in errors:
            print(f"  {error}")
    
    # Warnings
    if warnings:
        print("\n" + "="*70)
        print("⚠️  WARNINGS (SHOULD FIX FOR BETTER QUALITY):")
        print("="*70)
        for warning in warnings[:25]:  # Show first 25 warnings
            print(f"  {warning}")
        if len(warnings) > 25:
            print(f"  ... and {len(warnings) - 25} more warnings")
    
    # Final result
    print("\n" + "="*70)
    if is_valid:
        if stats['total_questions'] >= 100:
            print("✅ VALIDATION PASSED - Dataset is complete and ready!")
        else:
            print(f"✅ VALIDATION PASSED - No errors found")
            print(f"⚠️  Need {100 - stats['total_questions']} more questions to reach target")
        if warnings:
            print(f"   ({len(warnings)} warnings - consider fixing for higher quality)")
    else:
        print("❌ VALIDATION FAILED - Fix all errors before proceeding")
        print(f"   Found {len(errors)} error(s) that must be fixed")
    print("="*70)
    
    return is_valid


if __name__ == "__main__":
    import sys
    
    # Default file path
    qa_file = "data/qa_pairs/qa_dataset.json"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        qa_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(qa_file):
        print(f"\n❌ Error: File not found: {qa_file}")
        print("\nUsage: python validate_qa.py [path_to_qa_file.json]")
        print(f"Example: python validate_qa.py data/qa_pairs/qa_dataset.json")
        sys.exit(1)
    
    # Run validation
    is_valid = print_validation_report(qa_file)
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)
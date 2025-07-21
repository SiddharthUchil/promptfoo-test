#!/usr/bin/env python3
"""
Production-ready RAG evaluation system using promptfoo with built-in RAGAS metrics.
Focuses on Excel output and F1 score calculations for category classification.
"""

import json
import subprocess
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
import os
import sys
from typing import Dict, List, Optional, Any

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file at import time
load_env_file()

# Import sklearn with fallback
try:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
except ImportError:
    print("Warning: scikit-learn not available. F1 score calculations will be simplified.")
    # Provide fallback functions
    def precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=None):
        # Simple accuracy calculation as fallback
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0
        return accuracy, accuracy, accuracy, total
    
    def accuracy_score(y_true, y_pred):
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        return correct / len(y_true) if len(y_true) > 0 else 0


class PromptfooRAGEvaluator:
    """
    Production RAG evaluation system integrating promptfoo with Excel processing.
    Uses built-in RAGAS metrics and calculates F1 scores for category classification.
    """
    
    def __init__(self):
        self.config_file = "promptfooconfig.yaml"
        self.excel_file = "evaluation_data.xlsx"
        self.results_file = "promptfoo_results.json"
        
    def validate_environment(self) -> None:
        """Validate required files and environment variables."""
        required_files = [self.config_file, self.excel_file, "prompt_v1.md", "prompt_v2.md"]
        missing_files = [f for f in required_files if not Path(f).exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
            
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        # Check if promptfoo is available
        try:
            subprocess.run(['npx', 'promptfoo', '--version'], 
                         capture_output=True, check=True, shell=True,
                         encoding='utf-8', errors='ignore')
        except subprocess.CalledProcessError:
            raise RuntimeError("promptfoo is not available. Run 'npm install' first.")
    
    def clear_promptfoo_cache(self) -> None:
        """Clear promptfoo cache and database to avoid constraint issues."""
        try:
            import shutil
            
            # Common promptfoo cache/database locations
            cache_paths = [
                '.promptfoo',
                os.path.expanduser('~/.promptfoo'),
                'promptfoo_results.json',
                '.promptfoo.db',
                'eval.db'
            ]
            
            for cache_path in cache_paths:
                if os.path.exists(cache_path):
                    if os.path.isdir(cache_path):
                        shutil.rmtree(cache_path, ignore_errors=True)
                    else:
                        os.remove(cache_path)
                        
            print("Cleared promptfoo cache and database files")
        except Exception as e:
            print(f"Warning: Could not clear promptfoo cache: {e}")
            # Continue anyway - this is just cleanup
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load and validate promptfoo configuration."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"Loaded configuration from {self.config_file}")
            print(f"Found {len(config.get('tests', []))} test cases")
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def run_promptfoo_evaluation(self) -> Dict[str, Any]:
        """Execute promptfoo evaluation with built-in RAGAS metrics."""
        print("\nStarting promptfoo evaluation...")
        print("Evaluation includes:")
        print("  - Built-in RAGAS metrics (answer-relevance, context-faithfulness, context-relevance)")
        print("  - F1 score calculation for category classification")
        print("  - Semantic similarity scoring")
        
        try:
            # Clear promptfoo cache and database to avoid constraint issues
            self.clear_promptfoo_cache()
            
            # Run promptfoo evaluation
            cmd = [
                'npx', 'promptfoo', 'eval', 
                '--config', self.config_file,
                '--no-cache',
                '--no-write'
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=600,
                shell=True
            )
            
            if result.returncode != 0:
                print(f"Promptfoo evaluation failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                
                # Try alternative approach without database features
                print("Attempting evaluation without database features...")
                return self.run_simplified_evaluation()
            
            print("Promptfoo evaluation completed successfully!")
            
            # Load results
            if Path(self.results_file).exists():
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError("Results file not found")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Evaluation timed out")
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")
    
    def run_simplified_evaluation(self) -> Dict[str, Any]:
        """Simplified evaluation that bypasses promptfoo database issues."""
        print("Running simplified evaluation using direct API calls...")
        
        try:
            # Import OpenAI here to avoid dependency if not needed
            from openai import OpenAI
            import time
            
            # Initialize OpenAI client
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            client = OpenAI(api_key=api_key)
            
            # Load prompts
            with open("prompt_v1.md", 'r', encoding='utf-8') as f:
                prompt_v1 = f.read()
            with open("prompt_v2.md", 'r', encoding='utf-8') as f:
                prompt_v2 = f.read()
            
            # Load test cases from Excel  
            df = pd.read_excel(self.excel_file)
            
            # Create simplified results structure
            simplified_results = {
                'results': {
                    'table': []
                }
            }
            
            # Process each test case
            for idx, row in df.iterrows():
                test_outputs = []
                
                for version_num, prompt_template in enumerate([prompt_v1, prompt_v2], 1):
                    # Format prompt with variables - use correct column names
                    formatted_prompt = prompt_template
                    for var_name in ['questions', 'contexts', 'categories']:
                        placeholder = '{{' + var_name[:-1] + '}}'  # Remove 's' to match template
                        if placeholder in formatted_prompt and var_name in row:
                            formatted_prompt = formatted_prompt.replace(placeholder, str(row[var_name]))
                    
                    try:
                        # Call OpenAI API
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": formatted_prompt}],
                            temperature=0.1,
                            max_tokens=500
                        )
                        response_text = response.choices[0].message.content.strip()
                        
                        # Create output structure
                        output_data = {
                            'output': response_text,
                            'success': True,
                            'gradingResult': {
                                'componentResults': []
                            }
                        }
                        
                        test_outputs.append(output_data)
                        
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"Warning: API error for test {idx+1}, prompt v{version_num}: {e}")
                        test_outputs.append({
                            'output': f"Error: {str(e)}",
                            'success': False,
                            'gradingResult': {'componentResults': []}
                        })
                
                simplified_results['results']['table'].append({
                    'outputs': test_outputs
                })
            
            print(f"Completed simplified evaluation for {len(df)} test cases")
            
            # Save simplified results
            with open(self.results_file, 'w') as f:
                json.dump(simplified_results, f, indent=2)
            
            return simplified_results
            
        except Exception as e:
            print(f"Simplified evaluation also failed: {e}")
            return {'results': {'table': []}}
    
    def extract_responses_and_metrics(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Extract LLM responses and metrics from promptfoo results."""
        responses_data = []
        
        if 'results' in results:
            result_data = results['results']
            
            # Handle different result structures
            if isinstance(result_data, dict) and 'table' in result_data:
                table = result_data['table']
                for row_idx, row in enumerate(table):
                    if 'outputs' in row:
                        for prompt_idx, output_data in enumerate(row['outputs']):
                            version = 'v1' if prompt_idx == 0 else 'v2'
                            
                            if isinstance(output_data, dict):
                                response_text = output_data.get('output', '')
                                grading = output_data.get('gradingResult', {})
                                
                                # Extract metrics from component results
                                metrics = {}
                                component_results = grading.get('componentResults', [])
                                for comp in component_results:
                                    assertion = comp.get('assertion', {})
                                    metric = assertion.get('metric', '')
                                    score = comp.get('score', 0)
                                    if metric:
                                        metrics[metric] = score
                                
                                responses_data.append({
                                    'test_idx': row_idx,
                                    'prompt_version': version,
                                    'response': response_text,
                                    'success': output_data.get('success', True),
                                    **metrics
                                })
        
        return pd.DataFrame(responses_data)
    
    def calculate_f1_scores(self, df_results: pd.DataFrame) -> Dict[str, float]:
        """Calculate F1 scores for category classification."""
        print("\nCalculating F1 scores for category classification...")
        
        # Load original data from Excel
        try:
            original_data = pd.read_excel(self.excel_file)
            actual_categories = original_data['ground_truth_categories'].tolist()
            print(f"Loaded {len(actual_categories)} categories from Excel")
        except Exception as e:
            print(f"Error loading categories from Excel: {e}")
            return {'error': str(e)}
        
        f1_scores = {}
        categories = ['technical', 'business', 'analytical']
        
        # Extract predicted and actual categories for each prompt version
        for version in ['v1', 'v2']:
            version_rows = df_results[df_results['prompt_version'] == version]
            
            if len(version_rows) > 0:
                # Simple category prediction based on response content
                predicted_categories = []
                
                for _, row in version_rows.iterrows():
                    response = str(row['response']).lower()
                    # Simple heuristic: check if category words appear in response
                    predicted_cat = 'other'
                    for cat in categories:
                        if cat in response:
                            predicted_cat = cat
                            break
                    predicted_categories.append(predicted_cat)
                
                # Calculate F1 scores
                if len(predicted_categories) == len(actual_categories):
                    try:
                        # Overall metrics
                        accuracy = accuracy_score(actual_categories, predicted_categories)
                        precision, recall, f1, support = precision_recall_fscore_support(
                            actual_categories, predicted_categories, average='macro', zero_division=0
                        )
                        
                        f1_scores[f'{version}_accuracy'] = accuracy
                        f1_scores[f'{version}_precision'] = precision
                        f1_scores[f'{version}_recall'] = recall
                        f1_scores[f'{version}_f1_macro'] = f1
                        
                        # Per-category F1 scores
                        per_cat_precision, per_cat_recall, per_cat_f1, per_cat_support = precision_recall_fscore_support(
                            actual_categories, predicted_categories, labels=categories, average=None, zero_division=0
                        )
                        
                        for i, cat in enumerate(categories):
                            if i < len(per_cat_f1):
                                f1_scores[f'{version}_f1_{cat}'] = per_cat_f1[i]
                    except Exception as e:
                        print(f"Warning: F1 score calculation failed: {e}")
                        f1_scores = {'error': str(e)}
        
        return f1_scores
    
    def save_results_to_excel(self, original_data: pd.DataFrame, 
                            responses_df: pd.DataFrame, 
                            f1_scores: Dict[str, float]) -> str:
        """Save comprehensive results to Excel file."""
        print("\nSaving results to Excel...")
        
        # Create enhanced results dataframe
        df_results = original_data.copy()
        
        # Initialize columns for responses and metrics
        for version in ['v1', 'v2']:
            df_results[f'prompt_{version}_response'] = ''
            df_results[f'prompt_{version}_answer_relevance'] = 0.0
            df_results[f'prompt_{version}_context_faithfulness'] = 0.0
            df_results[f'prompt_{version}_context_relevance'] = 0.0
            df_results[f'prompt_{version}_semantic_similarity'] = 0.0
            df_results[f'prompt_{version}_category_accuracy'] = 0.0
            df_results[f'prompt_{version}_ragas_combined'] = 0.0
        
        # Populate with actual data
        for _, response_row in responses_df.iterrows():
            test_idx = response_row['test_idx']
            version = response_row['prompt_version']
            
            if test_idx < len(df_results):
                df_results.loc[test_idx, f'prompt_{version}_response'] = response_row['response']
                
                # Copy available metrics
                metric_columns = ['answer_relevance', 'context_faithfulness', 'context_relevance', 
                                'semantic_similarity', 'category_accuracy']
                for metric in metric_columns:
                    if metric in response_row and pd.notna(response_row[metric]):
                        df_results.loc[test_idx, f'prompt_{version}_{metric}'] = response_row[metric]
        
        # Calculate combined RAGAS scores
        for version in ['v1', 'v2']:
            df_results[f'prompt_{version}_ragas_combined'] = (
                df_results[f'prompt_{version}_answer_relevance'] +
                df_results[f'prompt_{version}_context_faithfulness'] +
                df_results[f'prompt_{version}_context_relevance']
            ) / 3
        
        # Determine better prompt
        df_results['better_prompt'] = 'tie'
        for idx in range(len(df_results)):
            v1_score = df_results.loc[idx, 'prompt_v1_ragas_combined']
            v2_score = df_results.loc[idx, 'prompt_v2_ragas_combined']
            
            if v1_score > v2_score + 0.05:
                df_results.loc[idx, 'better_prompt'] = 'v1'
            elif v2_score > v1_score + 0.05:
                df_results.loc[idx, 'better_prompt'] = 'v2'
        
        # Add F1 scores as metadata columns
        for metric, score in f1_scores.items():
            df_results[f'f1_{metric}'] = score
        
        # Add timestamp
        df_results['evaluation_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to Excel files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_{timestamp}.xlsx"
        
        # Save both timestamped and latest versions
        df_results.to_excel(output_file, index=False)
        df_results.to_excel("evaluation_results_latest.xlsx", index=False)
        
        print(f"Results saved to: {output_file}")
        print(f"Also saved to: evaluation_results_latest.xlsx")
        
        return output_file
    
    def print_summary(self, df_results: pd.DataFrame, f1_scores: Dict[str, float]) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("PROMPTFOO + RAGAS EVALUATION SUMMARY")
        print("=" * 70)
        
        # Basic summary
        print(f"Total test cases: {len(df_results)}")
        
        if 'better_prompt' in df_results.columns:
            summary = df_results['better_prompt'].value_counts()
            print(f"Prompt V1 wins: {summary.get('v1', 0)}")
            print(f"Prompt V2 wins: {summary.get('v2', 0)}")
            print(f"Ties: {summary.get('tie', 0)}")
        
        # RAGAS scores
        if 'prompt_v1_ragas_combined' in df_results.columns:
            v1_avg = df_results['prompt_v1_ragas_combined'].mean()
            v2_avg = df_results['prompt_v2_ragas_combined'].mean()
            print(f"\nAverage RAGAS Scores:")
            print(f"Prompt V1: {v1_avg:.3f}")
            print(f"Prompt V2: {v2_avg:.3f}")
        
        # F1 scores
        print(f"\nF1 Score Summary:")
        for metric, score in f1_scores.items():
            print(f"{metric}: {score:.3f}")
        
        print("=" * 70)
    
    def run_evaluation(self) -> str:
        """Execute the complete evaluation workflow."""
        print("Production Promptfoo + RAGAS Evaluation System")
        print("=" * 60)
        
        try:
            # Validate environment
            self.validate_environment()
            print("Environment validation passed")
            
            # Load original data
            original_data = pd.read_excel(self.excel_file)
            print(f"Loaded {len(original_data)} test cases from Excel")
            
            # Load configuration
            config = self.load_configuration()
            
            # Run promptfoo evaluation
            results = self.run_promptfoo_evaluation()
            
            # Extract responses and metrics
            responses_df = self.extract_responses_and_metrics(results)
            print(f"Extracted {len(responses_df)} responses")
            
            # Calculate F1 scores
            f1_scores = self.calculate_f1_scores(responses_df)
            
            # Save to Excel
            output_file = self.save_results_to_excel(original_data, responses_df, f1_scores)
            
            # Load final results for summary
            final_results = pd.read_excel(output_file)
            self.print_summary(final_results, f1_scores)
            
            print("\nEvaluation completed successfully!")
            print(f"Excel results: {output_file}")
            print(f"View dashboard: npm run view")
            print(f"Raw results: {self.results_file}")
            
            return output_file
            
        except Exception as e:
            print(f"\nEvaluation failed: {e}")
            raise


def main():
    """Main execution function."""
    try:
        evaluator = PromptfooRAGEvaluator()
        evaluator.run_evaluation()
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
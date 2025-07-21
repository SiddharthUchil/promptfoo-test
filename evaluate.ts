#!/usr/bin/env tsx
/**
 * Production-ready evaluation system using promptfoo library with RAGAS metrics (TypeScript)
 * Alternative to the Python implementation with better type safety
 */

import promptfoo from 'promptfoo';
import fs from 'fs';
import { execSync } from 'child_process';
import yaml from 'js-yaml';

interface EvaluationConfig {
  description: string;
  providers: string[];
  prompts: string[];
  tests: TestCase[];
  defaultTest?: any;
  derivedMetrics?: any[];
  outputPath?: string;
  writeLatestResults?: boolean;
}

interface TestCase {
  description?: string;
  vars: {
    question: string;
    context: string;
    category: string;
    ground_truth_response: string;
    ground_truth_category: string;
  };
  assert?: any[];
}

class PromptfooEvaluatorTS {
  private configFile = 'promptfooconfig.yaml';
  private excelFile = 'evaluation_data.xlsx';
  private resultsFile = 'promptfoo_results.json';

  /**
   * Check if required files exist and environment is set up
   */
  private validateFiles(): void {
    const requiredFiles = [this.configFile, this.excelFile, 'prompt_v1.md', 'prompt_v2.md'];
    const missingFiles = requiredFiles.filter(file => !fs.existsSync(file));
    
    if (missingFiles.length > 0) {
      throw new Error(`Missing required files: ${missingFiles.join(', ')}`);
    }
    
    if (!process.env.OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY environment variable is required');
    }
  }

  /**
   * Load and parse YAML configuration
   */
  private async loadConfig(): Promise<EvaluationConfig> {
    try {
      const configContent = fs.readFileSync(this.configFile, 'utf8');
      const config = yaml.load(configContent) as EvaluationConfig;
      console.log(`Loaded configuration from ${this.configFile}`);
      console.log(`Found ${config.tests?.length || 0} test cases`);
      return config;
    } catch (error) {
      throw new Error(`Failed to load config: ${(error as Error).message}`);
    }
  }

  /**
   * Run promptfoo evaluation using the library directly
   */
  private async runEvaluation(config: EvaluationConfig): Promise<any> {
    console.log('\nStarting promptfoo evaluation...');
    console.log('Evaluation includes:');
    console.log('  - Built-in RAGAS metrics (answer-relevance, context-faithfulness, context-relevance)');
    console.log('  - F1 score calculation for category classification');
    console.log('  - Semantic similarity scoring');
    
    try {
      const options = {
        maxConcurrency: 4,
        writeLatestResults: true,
        cache: true
      };

      const results = await promptfoo.evaluate(config, options);
      
      console.log('Evaluation completed successfully!');
      console.log('Evaluation results generated');
      
      return results;
    } catch (error) {
      throw new Error(`Evaluation failed: ${(error as Error).message}`);
    }
  }

  /**
   * Process results and save to Excel using Python
   */
  private async saveResultsToExcel(): Promise<void> {
    console.log('\nProcessing results and saving to Excel...');
    
    try {
      const pythonScript = `
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def process_typescript_results():
    try:
        # Load promptfoo results
        with open('${this.resultsFile}', 'r') as f:
            results = json.load(f)
        
        # Load original Excel data
        df_original = pd.read_excel('${this.excelFile}')
        df_results = df_original.copy()
        
        # Initialize columns
        for version in ['v1', 'v2']:
            df_results[f'prompt_{version}_response'] = ''
            df_results[f'prompt_{version}_answer_relevance'] = 0.0
            df_results[f'prompt_{version}_context_faithfulness'] = 0.0
            df_results[f'prompt_{version}_context_relevance'] = 0.0
            df_results[f'prompt_{version}_semantic_similarity'] = 0.0
            df_results[f'prompt_{version}_ragas_combined'] = 0.0
        
        # Process results (similar to Python version)
        if 'results' in results and isinstance(results['results'], dict):
            if 'table' in results['results']:
                table = results['results']['table']
                for row_idx, row in enumerate(table):
                    if 'outputs' in row and row_idx < len(df_results):
                        for prompt_idx, output_data in enumerate(row['outputs']):
                            version = 'v1' if prompt_idx == 0 else 'v2'
                            if isinstance(output_data, dict):
                                df_results.loc[row_idx, f'prompt_{version}_response'] = output_data.get('output', '')
                                
                                grading = output_data.get('gradingResult', {})
                                component_results = grading.get('componentResults', [])
                                
                                for comp in component_results:
                                    assertion = comp.get('assertion', {})
                                    metric = assertion.get('metric', '')
                                    score = comp.get('score', 0)
                                    
                                    if metric in ['answer_relevance', 'context_faithfulness', 'context_relevance', 'semantic_similarity']:
                                        df_results.loc[row_idx, f'prompt_{version}_{metric}'] = score
        
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
        
        # Calculate F1 scores for categories
        categories = ['technical', 'business', 'analytical']
        for version in ['v1', 'v2']:
            version_responses = []
            actual_categories = df_results['ground_truth_categories'].tolist()
            
            for idx, row in df_results.iterrows():
                response = str(row[f'prompt_{version}_response']).lower()
                predicted_cat = 'other'
                for cat in categories:
                    if cat in response:
                        predicted_cat = cat
                        break
                version_responses.append(predicted_cat)
            
            if len(version_responses) == len(actual_categories):
                accuracy = accuracy_score(actual_categories, version_responses)
                precision, recall, f1, support = precision_recall_fscore_support(
                    actual_categories, version_responses, average='macro', zero_division=0
                )
                
                df_results[f'f1_{version}_accuracy'] = accuracy
                df_results[f'f1_{version}_precision'] = precision
                df_results[f'f1_{version}_recall'] = recall
                df_results[f'f1_{version}_f1_macro'] = f1
        
        # Add timestamp
        df_results['evaluation_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_{timestamp}.xlsx"
        
        df_results.to_excel(output_file, index=False)
        df_results.to_excel("evaluation_results_latest.xlsx", index=False)
        
        print(f"TypeScript evaluation results saved to: {output_file}")
        print(f"Also saved to: evaluation_results_latest.xlsx")
        
        # Print summary
        if 'better_prompt' in df_results.columns:
            summary = df_results['better_prompt'].value_counts()
            print(f"\\nResults Summary:")
            print(f"Prompt V1 wins: {summary.get('v1', 0)}")
            print(f"Prompt V2 wins: {summary.get('v2', 0)}")
            print(f"Ties: {summary.get('tie', 0)}")
            
            v1_avg = df_results['prompt_v1_ragas_combined'].mean()
            v2_avg = df_results['prompt_v2_ragas_combined'].mean()
            print(f"\\nAverage RAGAS Scores:")
            print(f"Prompt V1: {v1_avg:.3f}")
            print(f"Prompt V2: {v2_avg:.3f}")
        
    except Exception as e:
        print(f"Error processing TypeScript results: {e}")

if __name__ == "__main__":
    process_typescript_results()
`;

      fs.writeFileSync('process_typescript_results.py', pythonScript);
      execSync('python process_typescript_results.py', { stdio: 'inherit' });
      fs.unlinkSync('process_typescript_results.py');
      
      console.log('Excel processing completed!');
      
    } catch (error) {
      console.error(`Excel processing failed: ${(error as Error).message}`);
      throw error;
    }
  }

  /**
   * Main evaluation workflow
   */
  public async run(): Promise<void> {
    try {
      console.log('Production Promptfoo + RAGAS Evaluation System (TypeScript)');
      console.log('=' .repeat(60));
      
      this.validateFiles();
      console.log('Environment validation passed');
      
      const config = await this.loadConfig();
      const results = await this.runEvaluation(config);
      await this.saveResultsToExcel();
      
      console.log('\nEvaluation completed successfully!');
      console.log('View results: evaluation_results_latest.xlsx');
      console.log('View dashboard: npm run view');
      console.log('Raw results: promptfoo_results.json');
      
    } catch (error) {
      console.error(`\nEvaluation failed: ${(error as Error).message}`);
      process.exit(1);
    }
  }
}

// Run the evaluation if this script is executed directly
if (require.main === module) {
  const evaluator = new PromptfooEvaluatorTS();
  evaluator.run();
}

export default PromptfooEvaluatorTS; 
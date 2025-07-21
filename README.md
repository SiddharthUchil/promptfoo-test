# Promptfoo RAG Evaluation System

Production-ready evaluation system that integrates promptfoo with built-in RAGAS metrics and F1 score calculations for category classification. Primary implementation in Python with TypeScript alternative available.

## Features

- Built-in RAGAS Metrics: Uses native promptfoo RAGAS implementations
  - Answer relevance
  - Context faithfulness
  - Context relevance
- F1 Score Calculation: Comprehensive category classification performance metrics using scikit-learn
- Excel-Focused Output: Primary output is Excel files, not JSON
- Python-First Approach: Clean Python implementation with optional TypeScript alternative
- Production Ready: Clean code without development artifacts

## Quick Start

### Prerequisites

- Node.js >= 18.0.0
- Python >= 3.8 (for Excel processing)
- OpenAI API key

### Installation

1. Install Node.js dependencies:

```bash
npm install
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Usage

Run the evaluation (Python - Recommended):

```bash
npm run eval
# or directly: python evaluate.py
```

Run with TypeScript (Alternative):

```bash
# First install TypeScript dependencies
npm install tsx @types/js-yaml
# Then run
npx tsx evaluate.ts
```

View results dashboard:

```bash
npm run view
```

## Architecture

### Core Components

| Component            | Technology                     | Purpose                                          |
| -------------------- | ------------------------------ | ------------------------------------------------ |
| Evaluation Engine    | promptfoo CLI                  | LLM evaluation orchestration                     |
| RAGAS Metrics        | Built-in promptfoo             | Answer relevance, context faithfulness/relevance |
| F1 Score Calculation | Python/scikit-learn            | Category classification performance              |
| Data Processing      | Python/pandas                  | Excel input/output handling                      |
| Alternative Engine   | promptfoo library + TypeScript | Direct library integration option                |

### Metrics

#### RAGAS Metrics

- Answer Relevance: Measures how well the answer addresses the question
- Context Faithfulness: Evaluates if the answer is grounded in the provided context
- Context Relevance: Assesses how relevant the context is to the question

#### Classification Metrics

- F1 Score: Harmonic mean of precision and recall for each category
- Category Accuracy: Simple accuracy for category prediction
- Overall F1: Macro-averaged F1 score across all categories

### Data Flow

1. Load test data from `evaluation_data.xlsx`
2. Execute promptfoo evaluation with built-in RAGAS metrics
3. Calculate F1 scores for category classification
4. Generate comprehensive Excel report with all metrics
5. Provide interactive dashboard for result analysis

## Configuration

The system uses `promptfooconfig.yaml` for configuration:

```yaml
description: Production RAG Evaluation with Built-in RAGAS Metrics and F1 Score

providers:
  - openai:gpt-4o-mini

prompts:
  - file://prompt_v1.md
  - file://prompt_v2.md

defaultTest:
  assert:
    - type: answer-relevance
      threshold: 0.7
    - type: context-faithfulness
      threshold: 0.8
    - type: context-relevance
      threshold: 0.7
    # ... additional metrics
```

## Input Data Format

Excel file (`evaluation_data.xlsx`) should contain:

| Column                  | Description                      |
| ----------------------- | -------------------------------- |
| questions               | Test questions                   |
| context                 | Relevant context for RAG         |
| categories              | Expected category classification |
| ground_truth_response   | Reference answers                |
| ground_truth_categories | Reference categories             |

## Output

The system generates:

1. Excel Report: `evaluation_results_latest.xlsx`

   - Original test data
   - LLM responses for both prompt versions
   - All RAGAS metric scores
   - F1 scores by category
   - Comparative analysis

2. Interactive Dashboard: Accessible via `npm run view`

   - Visual comparison of prompt performance
   - Detailed metric breakdowns
   - Test case analysis

3. Raw Results: `promptfoo_results.json`
   - Complete evaluation data
   - Detailed assertion results
   - Performance metrics

## File Structure

```
promptfoo_architecture/
├── evaluate.py                # Main evaluation script (Python)
├── evaluate.ts                # Alternative evaluation script (TypeScript)
├── promptfooconfig.yaml       # Promptfoo configuration
├── evaluation_data.xlsx       # Input test data
├── prompt_v1.md              # First prompt version
├── prompt_v2.md              # Second prompt version
├── package.json              # Node.js dependencies
├── requirements.txt          # Python dependencies
├── env.example               # Environment variables template
└── README.md                 # This documentation
```

## Development

### Adding New Test Cases

1. Add rows to `evaluation_data.xlsx`
2. Ensure all required columns are populated
3. Run evaluation: `npm run eval`

### Customizing Metrics

Modify `promptfooconfig.yaml`:

- Adjust threshold values for RAGAS metrics
- Add new assertion types
- Configure derived metrics

### Extending Categories

For new categories:

1. Update test data with new category values
2. Modify F1 score calculations in `promptfooconfig.yaml`
3. Update derived metrics section

## Troubleshooting

### Common Issues

**Missing API Key**

```
Error: OPENAI_API_KEY environment variable is required
```

Solution: Set the environment variable or use `.env` file

File Not Found:

```
Error: Missing required files: evaluation_data.xlsx
```

Solution: Ensure all required files exist in the working directory

Python Dependencies:

```
Error: Excel processing failed
```

Solution: Install Python dependencies: `pip install -r requirements.txt`

### Performance Optimization

- Adjust timeout values in Python script for large evaluations
- Use caching for repeated evaluations (enabled by default)
- Consider batch processing for large datasets
- Python approach is generally faster for Excel processing

## License

MIT License - see package.json for details.

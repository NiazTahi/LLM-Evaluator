# LLM Evaluation Framework for Bangla Language Agents

A comprehensive evaluation framework specifically designed for assessing the quality and performance of Bangla (Bengali) language conversational agents across multiple critical dimensions.

## Overview

This framework provides systematic evaluation of LLM-driven conversational agents with a focus on Bangla language interactions. It evaluates seven key dimensions of agent performance and provides detailed scoring, analysis, and recommendations.

## Features

### 🎯 **Seven Evaluation Dimensions**

1. **Conversation Fluency** - Natural flow and coherence in Bangla conversations
2. **Tool Calling Performance** - Contextual relevance and accuracy of tool usage
3. **Guardrails Compliance** - Adherence to ethical, safety, and policy guidelines
4. **Edge Case Handling** - Management of ambiguous, unexpected, or out-of-scope queries
5. **Special Instruction Adherence** - Following specific task or prompt-based instructions
6. **Language Proficiency** - Grammar, tone, style, and register appropriateness in Bangla
7. **Task Execution Accuracy** - Correct understanding and completion of specified tasks

### 🔧 **Framework Architecture**

- **Modular Design**: Each evaluation dimension is implemented as a separate evaluator
- **Extensible**: Easy to add new dimensions or modify existing ones
- **Configurable**: Adjustable scoring weights and criteria
- **Comprehensive**: Handles both positive cases and failure detection

### 📊 **Evaluation Methodology**

- **Hybrid Approach**: Combines heuristic, statistical, and rule-based methods
- **Bangla-Specific**: Tailored for Bangla language patterns and cultural context
- **Multi-faceted Scoring**: 0.0 to 1.0 scale with detailed reasoning
- **Confidence Scoring**: Each evaluation includes confidence metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see requirements.txt)

### Setup

```bash
# Clone the repository
git clone https://github.com/NiazTahi/LLM-Evaluator.git
cd llm-evaluation-framework

# Install dependencies
pip install -r requirements.txt

# Run the framework
python main_evaluation.py --complete-pipeline
```

## Usage

### Quick Start

```python
from llm_eval_framework import LLMEvaluationFramework, ConversationSample

# Create a sample conversation
sample = ConversationSample(
    id="test_001",
    messages=[
        {"role": "user", "content": "আপনি কেমন আছেন?"},
        {"role": "assistant", "content": "আমি ভালো আছি, ধন্যবাদ। আপনি কেমন আছেন?"}
    ]
)

# Initialize framework and evaluate
framework = LLMEvaluationFramework()
result = framework.evaluate_sample(sample)

print(f"Overall Score: {result.overall_score:.2f}")
```

### Command Line Interface

```bash
# Generate dataset and run complete evaluation
python main_evaluation.py --complete-pipeline

# Generate dataset only
python main_evaluation.py --generate-dataset

# Run evaluation on existing dataset
python main_evaluation.py --run-evaluation --dataset-file dataset.json

# Specify output directory
python main_evaluation.py --complete-pipeline --output-dir ./results
```

### Advanced Usage

```python
from dataset_generator import BanglaDatasetGenerator
from main_evaluation_script import EvaluationOrchestrator

# Generate comprehensive test dataset
generator = BanglaDatasetGenerator()
dataset = generator.generate_complete_dataset()

# Run evaluation with custom orchestrator
orchestrator = EvaluationOrchestrator("custom_output")
results = orchestrator.run_evaluation(dataset)

# Generate detailed report
report = orchestrator.generate_comprehensive_report(results)
```

## Project Structure

```
llm-evaluation-framework/
├── llm_eval_framework.py          # Core framework and base evaluators
├── additional_evaluators.py       # Specialized dimension evaluators
├── dataset_generator.py           # Test dataset generation
├── main_evaluation_script.py      # Main orchestration and CLI
├── README.md                      # This file
├── requirements.txt               # Python dependencies

## Dataset

### Included Test Dataset

The framework includes a comprehensive test dataset with:

- **Positive Cases**: High-quality examples demonstrating excellent performance
- **Negative Cases**: Poor examples to test failure detection
- **Edge Cases**: Challenging scenarios to test framework limits

### Dataset Structure

```json
{
  "metadata": {
    "total_samples": 50,
    "positive_samples": 20,
    "negative_samples": 15,
    "edge_cases": 15
  },
  "samples": [
    {
      "id": "sample_001",
      "messages": [...],
      "expected_tools": [...],
      "ground_truth": "...",
      "instruction": "...",
      "metadata": {...},
      "is_failure_case": false
    }
  ]
}
```

### Creating Custom Datasets

```python
from dataset_generator import BanglaDatasetGenerator

generator = BanglaDatasetGenerator()

# Create custom sample
custom_sample = ConversationSample(
    id="custom_001",
    messages=[
        {"role": "user", "content": "Your Bangla query here"},
        {"role": "assistant", "content": "Expected response"}
    ],
    expected_tools=["tool1", "tool2"],  # Optional
    ground_truth="Expected outcome",    # Optional
    instruction="Special instructions", # Optional
    metadata={"custom_field": "value"}  # Optional
)

# Add to dataset and save
dataset = [custom_sample] + generator.generate_complete_dataset()
generator.save_dataset_to_json(dataset, "custom_dataset.json")
```

## Evaluation Metrics

### Scoring Scale

- **0.8 - 1.0**: Excellent performance
- **0.6 - 0.79**: Good performance
- **0.4 - 0.59**: Moderate performance
- **0.2 - 0.39**: Poor performance
- **0.0 - 0.19**: Very poor performance

### Dimension Weights

| Dimension | Default Weight | Description |
|-----------|----------------|-------------|
| Conversation Fluency | 15% | Natural flow in Bangla |
| Tool Calling Performance | 15% | Appropriate tool usage |
| Guardrails Compliance | 15% | Safety and ethics |
| Edge Case Handling | 15% | Handling difficult queries |
| Special Instructions | 10% | Following specific instructions |
| Language Proficiency | 15% | Bangla language quality |
| Task Execution | 15% | Completing specified tasks |

### Sample Evaluation Output

```json
{
  "sample_id": "test_001",
  "overall_score": 0.85,
  "dimension_scores": {
    "conversation_fluency": {
      "score": 0.9,
      "reasoning": "Excellent conversation fluency in Bangla with natural flow",
      "confidence": 0.95
    },
    "language_proficiency": {
      "score": 0.8,
      "reasoning": "Good Bangla proficiency with proper grammar",
      "confidence": 0.9
    }
  },
  "timestamp": "2025-07-03T10:30:00"
}
```

## Report Generation

The framework generates comprehensive reports including:

### Executive Summary
- Total samples evaluated
- Overall performance metrics
- Score distributions
- Pass/fail rates

### Dimension Analysis
- Performance by evaluation dimension
- Strengths and weaknesses identification
- Correlation analysis

### Failure Analysis
- Common failure patterns
- Problematic dimensions
- Specific failure reasons

### Recommendations
- Actionable improvement suggestions
- Framework optimization tips
- Training data recommendations

## Customization

### Adding Custom Evaluators

```python
from llm_eval_framework import BaseEvaluator, EvaluationDimension

class CustomEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(EvaluationDimension.CUSTOM)
    
    def evaluate(self, sample):
        # Your custom evaluation logic
        score = self.calculate_custom_score(sample)
        return EvaluationResult(
            dimension=self.dimension,
            score=score,
            details={"custom_metric": value},
            reasoning="Custom evaluation reasoning"
        )
```
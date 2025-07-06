from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod
import re
import statistics
from additional_evaluators import (
    ToolCallingPerformanceEvaluator,
    GuardrailsComplianceEvaluator,
    EdgeCaseHandlingEvaluator,
    SpecialInstructionAdherenceEvaluator,
    LanguageProficiencyEvaluator,
    TaskExecutionAccuracyEvaluator
)
from evaluation_core import (
    EvaluationDimension,
    ConversationSample,
    EvaluationResult,
    ComprehensiveEvaluation,
    BaseEvaluator,
    BanglaLanguageProcessor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationFluencyEvaluator(BaseEvaluator):
    """Evaluates conversation fluency in Bangla"""
    
    def __init__(self):
        super().__init__(EvaluationDimension.CONVERSATION_FLUENCY)
        self.lang_processor = BanglaLanguageProcessor()
    
    def evaluate(self, sample: ConversationSample) -> EvaluationResult:
        """Evaluate conversation fluency"""
        try:
            scores = []
            details = {}
            
            # Analyze each assistant response
            assistant_responses = [
                msg['content'] for msg in sample.messages 
                if msg['role'] == 'assistant'
            ]
            
            if not assistant_responses:
                return EvaluationResult(
                    dimension=self.dimension,
                    score=0.0,
                    details={"error": "No assistant responses found"},
                    reasoning="No assistant responses to evaluate"
                )
            
            # Evaluate each response
            for i, response in enumerate(assistant_responses):
                response_score = self._evaluate_response_fluency(response)
                scores.append(response_score)
                details[f"response_{i}"] = response_score
            
            # Calculate overall fluency score
            overall_score = statistics.mean(scores) if scores else 0.0
            
            # Generate reasoning
            reasoning = self._generate_fluency_reasoning(details, overall_score)
            
            return EvaluationResult(
                dimension=self.dimension,
                score=overall_score,
                details=details,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in fluency evaluation: {str(e)}")
            return EvaluationResult(
                dimension=self.dimension,
                score=0.0,
                details={"error": str(e)},
                reasoning="Error during evaluation"
            )
    
    def _evaluate_response_fluency(self, response: str) -> Dict[str, Any]:
        """Evaluate fluency of a single response"""
        # Check if response contains Bangla
        has_bangla = self.lang_processor.is_bangla_text(response)
        
        if not has_bangla:
            return {
                "score": 0.0,
                "has_bangla": False,
                "reasoning": "Response does not contain Bangla text"
            }
        
        # Extract Bangla sentences
        bangla_sentences = self.lang_processor.extract_bangla_sentences(response)
        
        # Fluency metrics
        metrics = {
            "has_bangla": has_bangla,
            "sentence_count": len(bangla_sentences),
            "word_count": self.lang_processor.count_bangla_words(response),
            "avg_sentence_length": 0,
            "formality_level": self.lang_processor.detect_formality_level(response)
        }
        
        if bangla_sentences:
            sentence_lengths = [self.lang_processor.count_bangla_words(s) for s in bangla_sentences]
            metrics["avg_sentence_length"] = statistics.mean(sentence_lengths)
        
        # Calculate fluency score based on metrics
        score = self._calculate_fluency_score(metrics)
        
        return {
            "score": score,
            "metrics": metrics,
            "reasoning": f"Fluency score based on {len(bangla_sentences)} sentences"
        }
    
    def _calculate_fluency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate fluency score from metrics"""
        score = 0.0
        
        # Base score for having Bangla content
        if metrics["has_bangla"]:
            score += 0.3
        
        # Score based on sentence structure
        if metrics["sentence_count"] > 0:
            score += 0.3
        
        # Score based on word count (reasonable length)
        if 5 <= metrics["word_count"] <= 100:
            score += 0.2
        elif metrics["word_count"] > 0:
            score += 0.1
        
        # Score based on average sentence length
        if 3 <= metrics["avg_sentence_length"] <= 20:
            score += 0.2
        elif metrics["avg_sentence_length"] > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_fluency_reasoning(self, details: Dict[str, Any], score: float) -> str:
        """Generate reasoning for fluency evaluation"""
        if score >= 0.8:
            return "Excellent conversation fluency in Bangla with natural flow"
        elif score >= 0.6:
            return "Good conversation fluency with minor issues"
        elif score >= 0.4:
            return "Moderate fluency with some coherence issues"
        elif score >= 0.2:
            return "Poor fluency with significant issues"
        else:
            return "Very poor or no Bangla fluency detected"

class LLMEvaluationFramework:
    """Main evaluation framework orchestrating all dimension evaluators"""
    
    def __init__(self):
        self.evaluators = {
            EvaluationDimension.CONVERSATION_FLUENCY: ConversationFluencyEvaluator(),
            EvaluationDimension.TOOL_CALLING_PERFORMANCE: ToolCallingPerformanceEvaluator(),
            EvaluationDimension.GUARDRAILS_COMPLIANCE: GuardrailsComplianceEvaluator(),
            EvaluationDimension.EDGE_CASE_HANDLING: EdgeCaseHandlingEvaluator(),
            EvaluationDimension.SPECIAL_INSTRUCTION_ADHERENCE: SpecialInstructionAdherenceEvaluator(),
            EvaluationDimension.LANGUAGE_PROFICIENCY: LanguageProficiencyEvaluator(),
            EvaluationDimension.TASK_EXECUTION_ACCURACY: TaskExecutionAccuracyEvaluator(),
        }
        
    def evaluate_sample(self, sample: ConversationSample) -> ComprehensiveEvaluation:
        """Evaluate a single conversation sample across all dimensions"""
        from datetime import datetime
        
        dimension_scores = {}
        
        # Evaluate each dimension
        for dimension, evaluator in self.evaluators.items():
            try:
                result = evaluator.evaluate(sample)
                dimension_scores[dimension] = result
                logger.info(f"Evaluated {dimension.value}: {result.score:.2f}")
            except Exception as e:
                logger.error(f"Error evaluating {dimension.value}: {str(e)}")
                dimension_scores[dimension] = EvaluationResult(
                    dimension=dimension,
                    score=0.0,
                    details={"error": str(e)},
                    reasoning="Evaluation failed"
                )
        
        # Calculate overall score
        scores = [result.score for result in dimension_scores.values()]
        overall_score = statistics.mean(scores) if scores else 0.0
        
        return ComprehensiveEvaluation(
            sample_id=sample.id,
            dimension_scores=dimension_scores,
            overall_score=overall_score,
            timestamp=datetime.now().isoformat()
        )
    
    def evaluate_dataset(self, samples: List[ConversationSample]) -> List[ComprehensiveEvaluation]:
        """Evaluate a dataset of conversation samples"""
        results = []
        
        for sample in samples:
            logger.info(f"Evaluating sample: {sample.id}")
            result = self.evaluate_sample(sample)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[ComprehensiveEvaluation]) -> Dict[str, Any]:
        """Generate evaluation report from results"""
        if not results:
            return {"error": "No results to generate report"}
        
        # Calculate aggregate statistics
        overall_scores = [r.overall_score for r in results]
        dimension_stats = {}
        
        for dimension in EvaluationDimension:
            scores = [
                r.dimension_scores[dimension].score 
                for r in results 
                if dimension in r.dimension_scores
            ]
            if scores:
                dimension_stats[dimension.value] = {
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores),
                    "max": max(scores)
                }
        
        return {
            "summary": {
                "total_samples": len(results),
                "overall_mean_score": statistics.mean(overall_scores),
                "overall_median_score": statistics.median(overall_scores),
                "overall_std": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0
            },
            "dimension_statistics": dimension_stats,
            "sample_count": len(results)
        }

# Example usage
if __name__ == "__main__":
    # Create framework
    framework = LLMEvaluationFramework()
    
    # Example conversation sample
    sample = ConversationSample(
        id="test_001",
        messages=[
            {"role": "user", "content": "আপনি কেমন আছেন?"},
            {"role": "assistant", "content": "আমি ভালো আছি, ধন্যবাদ। আপনি কেমন আছেন?"}
        ]
    )
    
    # Evaluate sample
    result = framework.evaluate_sample(sample)
    print(f"Overall Score: {result.overall_score:.2f}")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

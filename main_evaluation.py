import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import logging

# Import our framework components
from llm_eval_framework import (
    LLMEvaluationFramework, ConversationSample, EvaluationDimension,
    ComprehensiveEvaluation
)
from additional_evaluators import (
    ToolCallingPerformanceEvaluator, GuardrailsComplianceEvaluator,
    EdgeCaseHandlingEvaluator, SpecialInstructionAdherenceEvaluator,
    LanguageProficiencyEvaluator, TaskExecutionAccuracyEvaluator
)
from dataset_generator import BanglaDatasetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvaluationOrchestrator:
    """Main orchestrator for the evaluation process"""
    
    def __init__(self, output_dir: str = "output"):
        self.framework = LLMEvaluationFramework()
        self.output_dir = output_dir
        self.generator = BanglaDatasetGenerator()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Evaluation orchestrator initialized with output directory: {output_dir}")
    
    def generate_dataset(self, filename: str = None) -> List[ConversationSample]:
        """Generate a comprehensive test dataset"""
        logger.info("Generating comprehensive test dataset...")
        
        dataset = self.generator.generate_complete_dataset()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bangla_dataset_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        self.generator.save_dataset_to_json(dataset, filepath)
        
        # Also save the rubric
        rubric_path = os.path.join(self.output_dir, "evaluation_rubric.json")
        rubric = self.generator.create_evaluation_rubric()
        with open(rubric_path, 'w', encoding='utf-8') as f:
            json.dump(rubric, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset generated with {len(dataset)} samples")
        logger.info(f"Dataset saved to: {filepath}")
        logger.info(f"Rubric saved to: {rubric_path}")
        
        return dataset
    
    def load_dataset_from_json(self, filepath: str) -> List[ConversationSample]:
        """Load dataset from JSON file"""
        logger.info(f"Loading dataset from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for sample_data in data['samples']:
            sample = ConversationSample(
                id=sample_data['id'],
                messages=sample_data['messages'],
                expected_tools=sample_data.get('expected_tools'),
                ground_truth=sample_data.get('ground_truth'),
                instruction=sample_data.get('instruction'),
                metadata=sample_data.get('metadata'),
                is_failure_case=sample_data.get('is_failure_case', False)
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from dataset")
        return samples
    
    def run_evaluation(self, samples: List[ConversationSample]) -> List[ComprehensiveEvaluation]:
        """Run comprehensive evaluation on the dataset"""
        logger.info(f"Starting evaluation of {len(samples)} samples...")
        
        results = []
        
        for i, sample in enumerate(samples, 1):
            logger.info(f"Evaluating sample {i}/{len(samples)}: {sample.id}")
            
            try:
                result = self.framework.evaluate_sample(sample)
                results.append(result)
                
                # Log summary for each sample
                logger.info(f"Sample {sample.id} overall score: {result.overall_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluating sample {sample.id}: {str(e)}")
                continue
        
        logger.info(f"Evaluation completed. {len(results)} samples successfully evaluated.")
        return results
    
    def save_results(self, results: List[ComprehensiveEvaluation], filename: str = None) -> str:
        """Save evaluation results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        results_data = {
            "metadata": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_samples": len(results),
                "framework_version": "1.0.0"
            },
            "results": [result.to_dict() for result in results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
        return filepath
    
    def generate_comprehensive_report(self, results: List[ComprehensiveEvaluation]) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report"""
        logger.info("Generating comprehensive evaluation report...")
        
        if not results:
            return {"error": "No results to generate report"}
        
        # Basic statistics
        overall_scores = [r.overall_score for r in results]
        total_samples = len(results)
        
        # Dimension-wise statistics
        dimension_stats = {}
        for dimension in EvaluationDimension:
            scores = []
            for result in results:
                if dimension in result.dimension_scores:
                    scores.append(result.dimension_scores[dimension].score)
            
            if scores:
                dimension_stats[dimension.value] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "samples_count": len(scores),
                    "scores_distribution": self._calculate_distribution(scores)
                }
        
        # Performance by sample type
        positive_samples = [r for r in results if not self._is_failure_case(r)]
        negative_samples = [r for r in results if self._is_failure_case(r)]
        
        # Top and bottom performing samples
        sorted_results = sorted(results, key=lambda x: x.overall_score, reverse=True)
        top_samples = sorted_results[:5]
        bottom_samples = sorted_results[-5:]
        
        # Failure analysis
        failure_analysis = self._analyze_failures(results)
        
        report = {
            "executive_summary": {
                "total_samples_evaluated": total_samples,
                "overall_mean_score": sum(overall_scores) / len(overall_scores),
                "overall_min_score": min(overall_scores),
                "overall_max_score": max(overall_scores),
                "positive_samples_count": len(positive_samples),
                "negative_samples_count": len(negative_samples),
                "samples_above_threshold": len([s for s in overall_scores if s >= 0.6]),
                "excellence_samples": len([s for s in overall_scores if s >= 0.8])
            },
            "dimension_performance": dimension_stats,
            "sample_type_analysis": {
                "positive_samples": {
                    "count": len(positive_samples),
                    "mean_score": sum(r.overall_score for r in positive_samples) / len(positive_samples) if positive_samples else 0,
                    "score_range": [min(r.overall_score for r in positive_samples), max(r.overall_score for r in positive_samples)] if positive_samples else [0, 0]
                },
                "negative_samples": {
                    "count": len(negative_samples),
                    "mean_score": sum(r.overall_score for r in negative_samples) / len(negative_samples) if negative_samples else 0,
                    "score_range": [min(r.overall_score for r in negative_samples), max(r.overall_score for r in negative_samples)] if negative_samples else [0, 0]
                }
            },
            "top_performing_samples": [
                {
                    "sample_id": r.sample_id,
                    "overall_score": r.overall_score,
                    "strongest_dimension": max(r.dimension_scores.items(), key=lambda x: x[1].score)[0].value,
                    "strongest_score": max(r.dimension_scores.values(), key=lambda x: x.score).score
                } for r in top_samples
            ],
            "bottom_performing_samples": [
                {
                    "sample_id": r.sample_id,
                    "overall_score": r.overall_score,
                    "weakest_dimension": min(r.dimension_scores.items(), key=lambda x: x[1].score)[0].value,
                    "weakest_score": min(r.dimension_scores.values(), key=lambda x: x.score).score
                } for r in bottom_samples
            ],
            "failure_analysis": failure_analysis,
            "recommendations": self._generate_recommendations(dimension_stats, failure_analysis),
            "framework_effectiveness": self._assess_framework_effectiveness(results)
        }
        
        return report
    
    def _calculate_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate score distribution"""
        distribution = {
            "excellent_0.8+": 0,
            "good_0.6-0.79": 0,
            "moderate_0.4-0.59": 0,
            "poor_0.2-0.39": 0,
            "very_poor_0.0-0.19": 0
        }
        
        for score in scores:
            if score >= 0.8:
                distribution["excellent_0.8+"] += 1
            elif score >= 0.6:
                distribution["good_0.6-0.79"] += 1
            elif score >= 0.4:
                distribution["moderate_0.4-0.59"] += 1
            elif score >= 0.2:
                distribution["poor_0.2-0.39"] += 1
            else:
                distribution["very_poor_0.0-0.19"] += 1
        
        return distribution
    
    def _is_failure_case(self, result: ComprehensiveEvaluation) -> bool:
        """Determine if a result represents a failure case"""
        # This is a simplified heuristic - you might want to enhance this
        return result.overall_score < 0.4
    
    def _analyze_failures(self, results: List[ComprehensiveEvaluation]) -> Dict[str, Any]:
        """Analyze common failure patterns"""
        failure_samples = [r for r in results if r.overall_score < 0.4]
        
        if not failure_samples:
            return {"total_failures": 0, "analysis": "No significant failures detected"}
        
        # Common failure dimensions
        dimension_failures = {}
        for dimension in EvaluationDimension:
            low_scores = [
                r.dimension_scores[dimension].score 
                for r in failure_samples 
                if dimension in r.dimension_scores and r.dimension_scores[dimension].score < 0.3
            ]
            if low_scores:
                dimension_failures[dimension.value] = {
                    "failure_count": len(low_scores),
                    "mean_score": sum(low_scores) / len(low_scores),
                    "percentage_of_failures": len(low_scores) / len(failure_samples) * 100
                }
        
        # Most common failure reasons
        common_failure_reasons = self._extract_failure_reasons(failure_samples)
        
        return {
            "total_failures": len(failure_samples),
            "failure_rate": len(failure_samples) / len(results) * 100,
            "dimension_failures": dimension_failures,
            "common_failure_reasons": common_failure_reasons,
            "most_problematic_dimension": max(dimension_failures.items(), key=lambda x: x[1]["failure_count"])[0] if dimension_failures else None
        }
    
    def _extract_failure_reasons(self, failure_samples: List[ComprehensiveEvaluation]) -> Dict[str, int]:
        """Extract and count common failure reasons"""
        reasons = {}
        
        for result in failure_samples:
            for dimension_result in result.dimension_scores.values():
                if dimension_result.score < 0.3:
                    reasoning = dimension_result.reasoning.lower()
                    # Extract key failure indicators
                    if "no bangla" in reasoning or "language" in reasoning:
                        reasons["language_issues"] = reasons.get("language_issues", 0) + 1
                    if "violation" in reasoning or "compliance" in reasoning:
                        reasons["compliance_issues"] = reasons.get("compliance_issues", 0) + 1
                    if "tool" in reasoning and "poor" in reasoning:
                        reasons["tool_usage_issues"] = reasons.get("tool_usage_issues", 0) + 1
                    if "instruction" in reasoning and "poor" in reasoning:
                        reasons["instruction_following_issues"] = reasons.get("instruction_following_issues", 0) + 1
        
        return dict(sorted(reasons.items(), key=lambda x: x[1], reverse=True))
    
    def _generate_recommendations(self, dimension_stats: Dict[str, Any], 
                                failure_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Check for low-performing dimensions
        for dimension, stats in dimension_stats.items():
            if stats["mean"] < 0.6:
                recommendations.append(
                    f"Improve {dimension.replace('_', ' ')}: Mean score {stats['mean']:.2f} is below threshold"
                )
        
        # Recommendations based on failure analysis
        if failure_analysis.get("failure_rate", 0) > 30:
            recommendations.append("High failure rate detected - consider reviewing training data and model parameters")
        
        common_failures = failure_analysis.get("common_failure_reasons", {})
        if "language_issues" in common_failures:
            recommendations.append("Focus on improving Bangla language proficiency and response quality")
        
        if "compliance_issues" in common_failures:
            recommendations.append("Strengthen guardrails and safety measures")
        
        if "tool_usage_issues" in common_failures:
            recommendations.append("Improve tool calling logic and contextual relevance")
        
        # General recommendations
        recommendations.extend([
            "Regularly update the evaluation dataset with new edge cases",
            "Consider implementing adaptive scoring based on conversation context",
            "Monitor performance trends over time",
            "Validate framework effectiveness with human evaluators"
        ])
        
        return recommendations
    
    def _assess_framework_effectiveness(self, results: List[ComprehensiveEvaluation]) -> Dict[str, Any]:
        """Assess the effectiveness of the evaluation framework itself"""
        # This is a meta-evaluation of how well our framework is working
        
        # Check score distributions
        overall_scores = [r.overall_score for r in results]
        score_variance = sum((s - sum(overall_scores)/len(overall_scores))**2 for s in overall_scores) / len(overall_scores)
        
        # Check if framework can distinguish between good and bad samples
        distinguishability = max(overall_scores) - min(overall_scores)
        
        # Check consistency across dimensions
        dimension_correlations = self._calculate_dimension_correlations(results)
        
        return {
            "score_variance": score_variance,
            "distinguishability": distinguishability,
            "dimension_correlations": dimension_correlations,
            "framework_reliability": "high" if score_variance > 0.1 and distinguishability > 0.5 else "moderate",
            "evaluation_quality": "The framework shows good discriminative power" if distinguishability > 0.6 else "Framework may need calibration"
        }
    
    def _calculate_dimension_correlations(self, results: List[ComprehensiveEvaluation]) -> Dict[str, float]:
        """Calculate correlations between dimensions (simplified)"""
        # This is a simplified correlation calculation
        correlations = {}
        
        dimensions = list(EvaluationDimension)
        for i, dim1 in enumerate(dimensions):
            for dim2 in dimensions[i+1:]:
                scores1 = [r.dimension_scores[dim1].score for r in results if dim1 in r.dimension_scores]
                scores2 = [r.dimension_scores[dim2].score for r in results if dim2 in r.dimension_scores]
                
                if len(scores1) == len(scores2) and len(scores1) > 1:
                    # Simple correlation coefficient
                    mean1, mean2 = sum(scores1)/len(scores1), sum(scores2)/len(scores2)
                    num = sum((s1 - mean1) * (s2 - mean2) for s1, s2 in zip(scores1, scores2))
                    den = (sum((s1 - mean1)**2 for s1 in scores1) * sum((s2 - mean2)**2 for s2 in scores2))**0.5
                    correlation = num / den if den != 0 else 0
                    correlations[f"{dim1.value}_vs_{dim2.value}"] = correlation
        
        return correlations
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save comprehensive report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Comprehensive report saved to: {filepath}")
        return filepath
    
    def run_complete_evaluation_pipeline(self, dataset_file: str = None) -> Dict[str, str]:
        """Run the complete evaluation pipeline"""
        logger.info("Starting complete evaluation pipeline...")
        
        # Step 1: Load or generate dataset
        if dataset_file and os.path.exists(dataset_file):
            samples = self.load_dataset_from_json(dataset_file)
        else:
            logger.info("Generating new dataset...")
            samples = self.generate_dataset()
        
        # Step 2: Run evaluation
        results = self.run_evaluation(samples)
        
        # Step 3: Save results
        results_file = self.save_results(results)
        
        # Step 4: Generate and save report
        report = self.generate_comprehensive_report(results)
        report_file = self.save_report(report)
        
        # Step 5: Generate summary
        self._print_evaluation_summary(report)
        
        return {
            "results_file": results_file,
            "report_file": report_file,
            "samples_evaluated": len(samples),
            "mean_score": report["executive_summary"]["overall_mean_score"]
        }
    
    def _print_evaluation_summary(self, report: Dict[str, Any]) -> None:
        """Print a formatted evaluation summary"""
        summary = report["executive_summary"]
        
        print("\n" + "="*60)
        print("BANGLA LLM EVALUATION FRAMEWORK - SUMMARY REPORT")
        print("="*60)
        print(f"Total Samples Evaluated: {summary['total_samples_evaluated']}")
        print(f"Overall Mean Score: {summary['overall_mean_score']:.3f}")
        print(f"Score Range: {summary['overall_min_score']:.3f} - {summary['overall_max_score']:.3f}")
        print(f"Samples Above Threshold (0.6): {summary['samples_above_threshold']}")
        print(f"Excellence Samples (0.8+): {summary['excellence_samples']}")
        
        print("\nDIMENSION PERFORMANCE:")
        print("-" * 40)
        for dimension, stats in report["dimension_performance"].items():
            print(f"{dimension.replace('_', ' ').title():.<30} {stats['mean']:.3f}")
        
        print("\nTOP RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(report["recommendations"][:3], 1):
            print(f"{i}. {rec}")
        
        print("\nFRAMEWORK EFFECTIVENESS:")
        print("-" * 40)
        effectiveness = report["framework_effectiveness"]
        print(f"Reliability: {effectiveness['framework_reliability']}")
        print(f"Distinguishability: {effectiveness['distinguishability']:.3f}")
        
        print("="*60)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Bangla LLM Evaluation Framework")
    parser.add_argument("--dataset-file", type=str, help="Path to dataset JSON file")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate new dataset")
    parser.add_argument("--run-evaluation", action="store_true", help="Run evaluation")
    parser.add_argument("--generate-report", action="store_true", help="Generate report")
    parser.add_argument("--complete-pipeline", action="store_true", help="Run complete pipeline")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = EvaluationOrchestrator(args.output_dir)
    
    try:
        if args.complete_pipeline:
            # Run the complete pipeline
            results = orchestrator.run_complete_evaluation_pipeline(args.dataset_file)
            print(f"\nPipeline completed successfully!")
            print(f"Results saved to: {results['results_file']}")
            print(f"Report saved to: {results['report_file']}")
            
        elif args.generate_dataset:
            # Generate dataset only
            dataset = orchestrator.generate_dataset()
            print(f"Generated dataset with {len(dataset)} samples")
            
        elif args.run_evaluation:
            # Run evaluation only
            if not args.dataset_file:
                print("Error: --dataset-file required for evaluation")
                return
            
            samples = orchestrator.load_dataset_from_json(args.dataset_file)
            results = orchestrator.run_evaluation(samples)
            results_file = orchestrator.save_results(results)
            print(f"Evaluation completed. Results saved to: {results_file}")
            
        elif args.generate_report:
            # Generate report from existing results
            # This would require loading results from a file
            print("Report generation from existing results not implemented in this demo")
            
        else:
            # Default: run complete pipeline
            results = orchestrator.run_complete_evaluation_pipeline(args.dataset_file)
            print(f"\nDefault pipeline completed!")
            print(f"Results: {results}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

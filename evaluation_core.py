from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import re
import statistics
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationDimension(Enum):
    """Enumeration of evaluation dimensions"""
    CONVERSATION_FLUENCY = "conversation_fluency"
    TOOL_CALLING_PERFORMANCE = "tool_calling_performance"
    GUARDRAILS_COMPLIANCE = "guardrails_compliance"
    EDGE_CASE_HANDLING = "edge_case_handling"
    SPECIAL_INSTRUCTION_ADHERENCE = "special_instruction_adherence"
    LANGUAGE_PROFICIENCY = "language_proficiency"
    TASK_EXECUTION_ACCURACY = "task_execution_accuracy"

@dataclass
class ConversationSample:
    """Represents a single conversation sample for evaluation"""
    id: str
    messages: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    expected_tools: Optional[List[str]] = None
    ground_truth: Optional[str] = None
    instruction: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_failure_case: bool = False

@dataclass
class EvaluationResult:
    """Stores evaluation results for a single dimension"""
    dimension: EvaluationDimension
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    reasoning: str
    confidence: float = 1.0

@dataclass
class ComprehensiveEvaluation:
    """Comprehensive evaluation results for a conversation sample"""
    sample_id: str
    dimension_scores: Dict[EvaluationDimension, EvaluationResult]
    overall_score: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "sample_id": self.sample_id,
            "dimension_scores": {
                dim.value: {
                    "score": result.score,
                    "details": result.details,
                    "reasoning": result.reasoning,
                    "confidence": result.confidence
                }
                for dim, result in self.dimension_scores.items()
            },
            "overall_score": self.overall_score,
            "timestamp": self.timestamp
        }

class BaseEvaluator(ABC):
    """Abstract base class for dimension-specific evaluators"""
    
    def __init__(self, dimension: EvaluationDimension):
        self.dimension = dimension
        
    @abstractmethod
    def evaluate(self, sample: ConversationSample) -> EvaluationResult:
        """Evaluate a conversation sample for this dimension"""
        pass
    
    def _extract_bangla_text(self, text: str) -> str:
        """Extract Bangla text from mixed content"""
        # Simple regex to identify Bangla characters
        bangla_pattern = r'[\u0980-\u09FF]+'
        bangla_matches = re.findall(bangla_pattern, text)
        return ' '.join(bangla_matches) if bangla_matches else text

class BanglaLanguageProcessor:
    """Utility class for Bangla language processing"""
    
    @staticmethod
    def is_bangla_text(text: str) -> bool:
        """Check if text contains Bangla characters"""
        bangla_pattern = r'[\u0980-\u09FF]'
        return bool(re.search(bangla_pattern, text))
    
    @staticmethod
    def count_bangla_words(text: str) -> int:
        """Count Bangla words in text"""
        bangla_words = re.findall(r'[\u0980-\u09FF]+', text)
        return len(bangla_words)
    
    @staticmethod
    def extract_bangla_sentences(text: str) -> List[str]:
        """Extract Bangla sentences from text"""
        # Split by common Bangla sentence terminators
        sentences = re.split(r'[।!?]', text)
        return [s.strip() for s in sentences if s.strip() and BanglaLanguageProcessor.is_bangla_text(s)]
    
    @staticmethod
    def detect_formality_level(text: str) -> str:
        """Detect formality level of Bangla text"""
        # Common formal markers in Bangla
        formal_markers = ['আপনি', 'আপনার', 'আপনাকে', 'দয়া করে', 'অনুগ্রহ করে']
        informal_markers = ['তুমি', 'তোমার', 'তোকে', 'তোমাকে']
        
        formal_count = sum(1 for marker in formal_markers if marker in text)
        informal_count = sum(1 for marker in informal_markers if marker in text)
        
        if formal_count > informal_count:
            return "formal"
        elif informal_count > formal_count:
            return "informal"
        else:
            return "neutral"

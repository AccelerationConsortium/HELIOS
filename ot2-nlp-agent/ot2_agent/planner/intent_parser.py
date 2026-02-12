"""
Intent Parser - Extract structured intent from natural language.

This module handles the conversion of free-form user input into
a structured Intent object.
"""

import re
from typing import Dict, List, Optional, Tuple

from ..ir import Intent, PlanningContext


class IntentParser:
    """
    Parses natural language input to extract user intent.

    Supports both English and Chinese input.
    """

    # Domain keywords for classification
    DOMAIN_KEYWORDS = {
        "electrochemistry": {
            "en": [
                "oer", "oxygen evolution", "her", "hydrogen evolution",
                "electrochemical", "electrolysis", "voltammetry", "lsv",
                "cv", "cyclic voltammetry", "eis", "impedance",
                "tafel", "overpotential", "current density", "electrode",
                "catalyst", "electrolyte", "potentiostat", "galvanostat"
            ],
            "zh": [
                "析氧", "产氧", "oer", "析氢", "her", "电化学",
                "电解", "伏安", "线性扫描", "循环伏安", "阻抗",
                "塔菲尔", "过电位", "电流密度", "电极", "催化剂",
                "电解液", "恒电位仪"
            ],
        },
        "liquid_handling": {
            "en": [
                "transfer", "pipette", "dispense", "aspirate", "dilution",
                "serial dilution", "mix", "distribute", "aliquot"
            ],
            "zh": [
                "转移", "移液", "分配", "吸取", "稀释",
                "梯度稀释", "混合", "分装"
            ],
        },
    }

    # Goal extraction patterns
    GOAL_PATTERNS = {
        "en": [
            (r"(?:i want to|i'd like to|please|can you)\s+(.+?)(?:\.|$)", 1),
            (r"(?:perform|do|run|execute|conduct)\s+(.+?)(?:\.|$)", 1),
            (r"(?:measure|test|characterize|analyze)\s+(.+?)(?:\.|$)", 1),
            (r"^(.+?)\s+(?:measurement|test|experiment|analysis)", 1),
        ],
        "zh": [
            (r"(?:我想|想要|请|帮我)\s*(.+?)(?:。|$)", 1),
            (r"(?:做|进行|执行|测)\s*(.+?)(?:。|$)", 1),
            (r"(?:测量|测试|表征|分析)\s*(.+?)(?:。|$)", 1),
        ],
    }

    # Target metric patterns
    METRIC_PATTERNS = {
        "en": {
            "overpotential": [r"overpotential", r"over-potential", r"eta"],
            "tafel_slope": [r"tafel\s*slope", r"tafel"],
            "current_density": [r"current\s*density", r"j\s*@"],
            "stability": [r"stability", r"durability", r"long[- ]term"],
            "impedance": [r"impedance", r"eis", r"charge\s*transfer"],
            "faradaic_efficiency": [r"farad(?:a)?ic\s*efficiency", r"fe"],
        },
        "zh": {
            "overpotential": [r"过电位", r"过电势"],
            "tafel_slope": [r"塔菲尔斜率", r"tafel斜率"],
            "current_density": [r"电流密度"],
            "stability": [r"稳定性", r"耐久性"],
            "impedance": [r"阻抗", r"电荷转移"],
            "faradaic_efficiency": [r"法拉第效率"],
        },
    }

    # Condition extraction patterns
    CONDITION_PATTERNS = {
        "electrode": {
            "en": [r"(?:using|with|on)\s+(\w+(?:\s+\w+)?)\s+(?:electrode|catalyst)"],
            "zh": [r"用(?:的是)?(.+?)(?:催化剂|电极|材料)"],
        },
        "electrolyte": {
            "en": [r"in\s+([\d.]+\s*[mM]\s*\w+)", r"(\w+)\s+(?:electrolyte|solution)"],
            "zh": [r"(?:在|用)([\d.]+\s*[mM]?\s*\w+)(?:电解液|溶液)?"],
        },
        "temperature": {
            "en": [r"at\s+(\d+)\s*(?:°?C|degrees?|celsius)"],
            "zh": [r"(?:在)?(\d+)\s*(?:°C|度|摄氏度)"],
        },
    }

    def __init__(self):
        """Initialize the intent parser."""
        pass

    def parse(self, text: str, context: PlanningContext = None) -> Intent:
        """
        Parse natural language text into an Intent.

        Args:
            text: User's natural language input
            context: Optional planning context with known conditions

        Returns:
            Intent object with extracted information
        """
        # Detect language
        language = self._detect_language(text)

        # Clean and normalize text
        text_clean = self._normalize_text(text)

        # Detect domain
        domain, domain_confidence = self._detect_domain(text_clean, language)

        # Extract goal
        goal = self._extract_goal(text_clean, language)

        # Extract target metrics
        target_metrics = self._extract_metrics(text_clean, language)

        # Extract known conditions
        known_conditions = self._extract_conditions(text_clean, language)

        # Merge with context
        if context:
            known_conditions.update(context.materials)
            known_conditions.update(context.constraints)

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            domain_confidence,
            bool(goal),
            len(target_metrics),
            len(known_conditions)
        )

        return Intent(
            goal=goal or self._infer_goal(domain, target_metrics),
            domain=domain,
            original_text=text,
            language=language,
            target_metrics=target_metrics,
            known_conditions=known_conditions,
            confidence=confidence,
        )

    def _detect_language(self, text: str) -> str:
        """Detect language (en or zh)."""
        # Count Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(" ", ""))

        if total_chars == 0:
            return "en"

        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.3 else "en"

    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        # Convert to lowercase for English
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _detect_domain(self, text: str, language: str) -> Tuple[str, float]:
        """
        Detect the domain of the experiment.

        Returns:
            Tuple of (domain, confidence)
        """
        scores = {}

        for domain, lang_keywords in self.DOMAIN_KEYWORDS.items():
            keywords = lang_keywords.get(language, []) + lang_keywords.get("en", [])
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[domain] = score

        if not scores:
            return "general", 0.3

        best_domain = max(scores, key=scores.get)
        max_score = scores[best_domain]
        confidence = min(0.5 + max_score * 0.1, 0.95)

        return best_domain, confidence

    def _extract_goal(self, text: str, language: str) -> Optional[str]:
        """Extract the main goal from the text."""
        patterns = self.GOAL_PATTERNS.get(language, [])

        for pattern, group_idx in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                goal = match.group(group_idx).strip()
                # Clean up the goal
                goal = re.sub(r'\s+', ' ', goal)
                return goal

        return None

    def _extract_metrics(self, text: str, language: str) -> List[str]:
        """Extract target metrics from the text."""
        metrics = []
        patterns = self.METRIC_PATTERNS.get(language, {})

        for metric_name, metric_patterns in patterns.items():
            for pattern in metric_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    metrics.append(metric_name)
                    break

        return list(set(metrics))

    def _extract_conditions(self, text: str, language: str) -> Dict[str, str]:
        """Extract known experimental conditions."""
        conditions = {}

        for condition_name, lang_patterns in self.CONDITION_PATTERNS.items():
            patterns = lang_patterns.get(language, [])

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    conditions[condition_name] = match.group(1).strip()
                    break

        return conditions

    def _infer_goal(self, domain: str, metrics: List[str]) -> str:
        """Infer a goal if none was explicitly stated."""
        if domain == "electrochemistry":
            if "overpotential" in metrics or "tafel_slope" in metrics:
                return "perform OER characterization"
            elif "stability" in metrics:
                return "perform stability test"
            else:
                return "perform electrochemical measurement"
        elif domain == "liquid_handling":
            return "perform liquid handling operation"
        else:
            return "perform experiment"

    def _calculate_confidence(
        self,
        domain_confidence: float,
        has_goal: bool,
        num_metrics: int,
        num_conditions: int
    ) -> float:
        """Calculate overall intent parsing confidence."""
        base = domain_confidence

        if has_goal:
            base += 0.1

        # More specific = more confident
        if num_metrics > 0:
            base += min(num_metrics * 0.05, 0.15)

        if num_conditions > 0:
            base += min(num_conditions * 0.05, 0.15)

        return min(base, 0.95)

"""
Constitutional config parser for Contemplative Constitutional AI.
Parses markdown-based constitutional principles into structured templates.
"""

import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ConstitutionalPrinciple:
    """Represents a single constitutional principle with its templates."""
    name: str
    critique_template: str
    revision_guideline: str
    example_applications: List[str]
    
    def create_critique_prompt(self, original_response: str, context: str = "") -> str:
        """
        Create a critique prompt using this principle.
        
        Args:
            original_response: The response to critique
            context: Optional context about the original prompt
            
        Returns:
            Formatted critique prompt
        """
        prompt = f"""Principle: {self.name}

Evaluation Criteria: {self.critique_template}

{f"Context: {context}" if context else ""}

Response to evaluate:
{original_response}

Please provide a thoughtful critique based on the {self.name} principle. Consider whether the response aligns with or violates this principle, and explain your reasoning.

Critique:"""
        return prompt
    
    def create_revision_prompt(self, original_response: str, critique: str, context: str = "") -> str:
        """
        Create a revision prompt using this principle.
        
        Args:
            original_response: The original response
            critique: The critique of the original response
            context: Optional context about the original prompt
            
        Returns:
            Formatted revision prompt
        """
        prompt = f"""Principle: {self.name}

Revision Guideline: {self.revision_guideline}

{f"Context: {context}" if context else ""}

Original Response:
{original_response}

Critique:
{critique}

Please revise the original response to better align with the {self.name} principle, addressing the issues raised in the critique.

Revised Response:"""
        return prompt


class ConstitutionalParser:
    """Parser for markdown-based constitutional configurations."""
    
    def __init__(self):
        self.principles: List[ConstitutionalPrinciple] = []
        self.meta_guidelines: List[str] = []
    
    def parse_markdown_principles(self, md_path: str) -> List[ConstitutionalPrinciple]:
        """
        Parse constitutional principles from a markdown file.
        
        Args:
            md_path: Path to the markdown file containing principles
            
        Returns:
            List of ConstitutionalPrinciple objects
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split content by principle sections (## headers)
        sections = re.split(r'\n## ([^#\n]+)', content)

        if not sections:
            return []

        # The first element (index 0) contains any preamble or meta principles
        preamble = sections[0].strip()
        self.meta_guidelines = self._extract_guidelines(preamble)

        principles = []

        # Remaining elements come in pairs: [name, content, name, content, ...]
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break

            principle_name = sections[i].strip()
            principle_content = sections[i + 1]

            principle = self._parse_principle_section(principle_name, principle_content)
            if principle:
                principles.append(principle)

        self.principles = principles
        return principles
    
    def _parse_principle_section(self, name: str, content: str) -> Optional[ConstitutionalPrinciple]:
        """
        Parse a single principle section.
        
        Args:
            name: Name of the principle
            content: Content of the principle section
            
        Returns:
            ConstitutionalPrinciple object or None if parsing fails
        """
        # Extract subsections using ### headers
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Check for subsection header
            if line.startswith('### '):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line[4:].strip()
                current_content = []
            elif current_section:
                # Add content to current section
                if line:  # Skip empty lines at the start
                    current_content.append(line)
                elif current_content:  # Keep empty lines in the middle
                    current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Extract required fields if explicitly defined
        critique_template = sections.get('Critique Template', '').strip()
        revision_guideline = sections.get('Revision Guideline', '').strip()

        # Extract example applications
        example_applications: List[str] = []
        example_content = sections.get('Example Application', '').strip()
        if example_content:
            # Split by bullet points or dashes
            examples = re.split(r'\n[-•*]\s*', example_content)
            example_applications = [ex.strip() for ex in examples if ex.strip()]

        # If critique/revision templates are missing, synthesize them from guidelines
        if not critique_template or not revision_guideline:
            guidelines = self._extract_guidelines(content)
            if not guidelines:
                print(f"Warning: Principle '{name}' missing structured directives")
                return None

            critique_template = self._build_default_critique_template(name, guidelines)
            revision_guideline = self._build_default_revision_guideline(name, guidelines)
            if not example_applications:
                example_applications = guidelines

        return ConstitutionalPrinciple(
            name=name,
            critique_template=critique_template,
            revision_guideline=revision_guideline,
            example_applications=example_applications
        )
    
    def get_principle_by_name(self, name: str) -> Optional[ConstitutionalPrinciple]:
        """
        Get a principle by its name.
        
        Args:
            name: Name of the principle to retrieve
            
        Returns:
            ConstitutionalPrinciple object or None if not found
        """
        for principle in self.principles:
            if principle.name.lower() == name.lower():
                return principle
        return None
    
    def get_all_principles(self) -> List[ConstitutionalPrinciple]:
        """
        Get all loaded principles.
        
        Returns:
            List of all ConstitutionalPrinciple objects
        """
        return self.principles.copy()
    
    def export_to_yaml(self, output_path: str) -> None:
        """
        Export loaded principles to a YAML file for inspection.
        
        Args:
            output_path: Path to save the YAML file
        """
        data = {
            'constitutional_principles': [
                {
                    'name': p.name,
                    'critique_template': p.critique_template,
                    'revision_guideline': p.revision_guideline,
                    'example_applications': p.example_applications
                }
                for p in self.principles
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def create_batch_critique_prompts(self, responses: List[str], principle_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Create critique prompts for multiple responses using specified principles.
        
        Args:
            responses: List of responses to critique
            principle_names: List of principle names to use (all if None)
            
        Returns:
            List of prompt dictionaries with metadata
        """
        if principle_names is None:
            principles_to_use = self.principles
        else:
            principles_to_use = [self.get_principle_by_name(name) for name in principle_names]
            principles_to_use = [p for p in principles_to_use if p is not None]
        
        prompts = []
        for i, response in enumerate(responses):
            for principle in principles_to_use:
                prompt_data = {
                    'prompt': principle.create_critique_prompt(response),
                    'response_index': i,
                    'principle_name': principle.name,
                    'original_response': response,
                    'type': 'critique'
                }
                prompts.append(prompt_data)
        
        return prompts

    def create_batch_revision_prompts(self, response_critique_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Create revision prompts for response-critique pairs.
        
        Args:
            response_critique_pairs: List of dicts with 'response', 'critique', and 'principle_name'
            
        Returns:
            List of prompt dictionaries with metadata
        """
        prompts = []
        for i, pair in enumerate(response_critique_pairs):
            principle = self.get_principle_by_name(pair['principle_name'])
            if principle:
                prompt_data = {
                    'prompt': principle.create_revision_prompt(
                        pair['response'], 
                        pair['critique']
                    ),
                    'pair_index': i,
                    'principle_name': principle.name,
                    'original_response': pair['response'],
                    'critique': pair['critique'],
                    'type': 'revision'
                }
                prompts.append(prompt_data)
        
        return prompts

    def _extract_guidelines(self, content: str) -> List[str]:
        """Extract guideline-style statements from markdown content."""
        if not content:
            return []

        guidelines: List[str] = []
        for raw_line in content.split('\n'):
            line = raw_line.strip()
            if not line:
                continue

            # Remove leading bullets or numbering (e.g., "-", "*", "E1.", "1.")
            line = re.sub(r'^[\-\*•\u2022]+\s*', '', line)
            line = re.sub(r'^[A-Za-z]+\d+\.?\s*', '', line)
            line = re.sub(r'^\d+\.?\s*', '', line)

            # Remove emphasis markers
            line = line.replace('**', '').replace('__', '').strip()
            if not line:
                continue

            guidelines.append(line)

        return guidelines

    def _build_default_critique_template(self, name: str, guidelines: List[str]) -> str:
        """Construct a critique template when none is provided explicitly."""
        lines = [
            f"Evaluate whether the assistant response aligns with the \"{name}\" directives.",
        ]

        if self.meta_guidelines:
            lines.append("Meta-principles to keep in mind:")
            for meta in self.meta_guidelines:
                lines.append(f"- {meta}")

        lines.append("Principle-specific directives:")
        for guideline in guidelines:
            lines.append(f"- {guideline}")

        lines.append("Explain where the response conforms or deviates from these expectations.")
        return '\n'.join(lines)

    def _build_default_revision_guideline(self, name: str, guidelines: List[str]) -> str:
        """Construct a revision guideline when none is provided explicitly."""
        lines = [
            f"Revise the response so it embodies the \"{name}\" directives listed below.",
        ]

        if self.meta_guidelines:
            lines.append("Also honor the overarching meta-principles:")
            for meta in self.meta_guidelines:
                lines.append(f"- {meta}")

        lines.append("Key directives to integrate:")
        for guideline in guidelines:
            lines.append(f"- {guideline}")

        lines.append("Address each critique point while maintaining factual accuracy and safety.")
        return '\n'.join(lines)


def load_constitutional_config(config_path: str) -> List[ConstitutionalPrinciple]:
    """
    Convenience function to load constitutional principles.
    
    Args:
        config_path: Path to the markdown configuration file
        
    Returns:
        List of ConstitutionalPrinciple objects
    """
    parser = ConstitutionalParser()
    return parser.parse_markdown_principles(config_path)


if __name__ == "__main__":
    # Test the parser with the contemplative principles
    config_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "constitutions"
        / "contemplative-constitution-1.md"
    )
    
    if config_path.exists():
        print("Testing constitutional config parser...")
        parser = ConstitutionalParser()
        principles = parser.parse_markdown_principles(str(config_path))
        
        print(f"Loaded {len(principles)} principles:")
        for principle in principles:
            print(f"- {principle.name}")
            print(f"  Critique template length: {len(principle.critique_template)}")
            print(f"  Revision guideline length: {len(principle.revision_guideline)}")
            print(f"  Example applications: {len(principle.example_applications)}")
            print()
        
        # Test prompt creation
        if principles:
            test_response = "I think this is the only correct way to think about this issue."
            principle = principles[0]
            
            print("=== Test Critique Prompt ===")
            critique_prompt = principle.create_critique_prompt(test_response)
            print(critique_prompt)
            print()
            
            print("=== Test Revision Prompt ===")
            test_critique = "This response presents a single perspective as absolute truth without acknowledging other viewpoints."
            revision_prompt = principle.create_revision_prompt(test_response, test_critique)
            print(revision_prompt)
        
        # Export to YAML for inspection
        output_path = Path(__file__).parent.parent.parent / "results" / "parsed_principles.yaml"
        output_path.parent.mkdir(exist_ok=True)
        parser.export_to_yaml(str(output_path))
        print(f"Exported principles to: {output_path}")
        
    else:
        print(f"Constitutional config file not found: {config_path}")

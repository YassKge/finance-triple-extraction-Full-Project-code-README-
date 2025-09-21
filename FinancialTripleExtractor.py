import json
import spacy
import re
from datetime import datetime
from typing import List, Tuple, Dict, Any
import pandas as pd

class FinancialTripleExtractor:
    """
    A comprehensive system for extracting knowledge triples from financial texts.
    Handles JSON data input and produces structured triples in the format:
    (Subject, Relation, Object)
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        # Try to load spaCy models in order of preference
        self.nlp = self._load_spacy_model()

        # Financial relation keywords
        self.financial_verbs = {
            "incur": "incurred",
            "use": "used",
            "allocate": "allocated_to",
            "report": "reported",
            "generate": "generated",
            "earn": "earned",
            "pay": "paid",
            "receive": "received",
            "invest": "invested",
            "spend": "spent",
            "record": "recorded",
            "recognize": "recognized",
            "distribute": "distributed",
            "issue": "issued",
            "repurchase": "repurchased",
            "acquire": "acquired",
            "sell": "sold",
            "increase": "increased",
            "decrease": "decreased",
            "maintain": "maintained",
            "hold": "held"
        }

        # Financial entity patterns
        self.money_pattern = re.compile(r'\$[\d,]+\.?\d*\s?(?:million|billion|thousand|M|B|K)?', re.IGNORECASE)
        self.percentage_pattern = re.compile(r'\d+\.?\d*\s?%')
        self.date_pattern = re.compile(r'(?:December|January|February|March|April|May|June|July|August|September|October|November)\s+\d{1,2},?\s+\d{4}')
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')

    def _load_spacy_model(self):
        """Try to load spaCy models in order of preference."""
        models_to_try = [
            "en_core_web_trf",
            "en_core_web_lg",
            "en_core_web_md",
            "en_core_web_sm"
        ]

        for model_name in models_to_try:
            try:
                nlp = spacy.load(model_name)
                if self.verbose:
                    print(f"✅ Successfully loaded {model_name}")
                return nlp
            except (OSError, ValueError):
                if self.verbose:
                    print(f"⚠️  Could not load {model_name}")
                continue

        # If all models fail, create a basic pipeline
        if self.verbose:
            print("⚠️  No pre-trained models available. Creating basic pipeline...")
        try:
            nlp = spacy.blank("en")
            if self.verbose:
                print("✅ Created basic spaCy pipeline")
            return nlp
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to create any spaCy pipeline: {e}")
                print("\n🔧 SOLUTION: Run these commands:")
                print("!pip install -U spacy")
                print("!python -m spacy download en_core_web_sm")
            raise Exception("Cannot initialize spaCy. Please install spaCy models.")

    def load_json_data(self, file_path: str) -> Dict[str, Any]:
        """Load JSON data from file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def load_text_file(self, file_path: str) -> str:
        """Load text data from file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def process_text_segments(self, text: str) -> List[Dict]:
        """Process text that's divided by --- separators."""
        # Split by --- separator
        segments = [segment.strip() for segment in text.split('---') if segment.strip()]

        results = []
        for i, segment in enumerate(segments):
            if self.verbose:
                print(f"📄 Processing segment {i+1}/{len(segments)}...")
            try:
                triples = self.extract_financial_triples(segment)
                for triple in triples:
                    results.append({
                        'segment_id': i + 1,
                        'subject': triple[0],
                        'relation': triple[1],
                        'object': triple[2],
                        'source_text': segment[:200] + "..." if len(segment) > 200 else segment
                    })
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Error in segment {i+1}: {e}")

        return results

    def extract_entities_regex(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns (fallback method)."""
        entities = {
            'MONEY': [],
            'DATE': [],
            'PERCENT': [],
            'YEAR': [],
            'ORG': []
        }

        # Money patterns
        money_matches = self.money_pattern.findall(text)
        entities['MONEY'] = list(set(money_matches))

        # Percentage patterns
        percent_matches = self.percentage_pattern.findall(text)
        entities['PERCENT'] = list(set(percent_matches))

        # Date patterns
        date_matches = self.date_pattern.findall(text)
        entities['DATE'] = list(set(date_matches))

        # Year patterns
        year_matches = self.year_pattern.findall(text)
        entities['YEAR'] = [match[0] + match[1] for match in year_matches]
        entities['YEAR'] = list(set(entities['YEAR']))

        # Common organization terms
        org_patterns = [
            r'\b(Company|Corporation|Corp|Inc|LLC|Ltd|Group|Holdings|Enterprises)\b',
            r'\b([A-Z][a-z]+ (?:Company|Corporation|Corp|Inc|LLC|Ltd))\b'
        ]

        for pattern in org_patterns:
            org_matches = re.findall(pattern, text, re.IGNORECASE)
            if isinstance(org_matches[0], tuple) if org_matches else False:
                entities['ORG'].extend([match[0] for match in org_matches])
            else:
                entities['ORG'].extend(org_matches)

        entities['ORG'] = list(set(entities['ORG']))

        return entities

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER and custom patterns."""
        entities = {
            'PERSON': [],
            'ORG': [],
            'MONEY': [],
            'DATE': [],
            'PERCENT': [],
            'CARDINAL': [],
            'GPE': []
        }

        try:
            # Try spaCy NER if available
            if hasattr(self.nlp, 'pipe_names') and any(pipe in ['ner', 'transformer'] for pipe in self.nlp.pipe_names):
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in entities:
                        entities[ent.label_].append(ent.text.strip())
        except Exception as e:
            if self.verbose:
                print(f"⚠️  spaCy NER failed: {e}. Using regex fallback...")

        # Always use regex patterns as backup/supplement
        regex_entities = self.extract_entities_regex(text)

        # Merge results
        for key in ['MONEY', 'DATE', 'PERCENT']:
            if key in regex_entities:
                entities[key].extend(regex_entities[key])

        entities['ORG'].extend(regex_entities.get('ORG', []))
        entities['DATE'].extend(regex_entities.get('YEAR', []))

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def extract_relations_pattern_based(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relations using enhanced pattern matching."""
        relations = []

        # Enhanced patterns for financial statements
        financial_patterns = [
            # Net loss/income patterns
            {
                'pattern': r'(.*?)\s+(?:incurred|reported|had|recorded)\s+(?:a\s+)?net\s+(loss|income|profit)\s+of\s+(\$[\d,]+\.?\d*\s?(?:million|billion)?)',
                'relation': lambda match: "incurred_net_loss" if "loss" in match.group(2).lower() else "reported_net_income"
            },

            # Cash usage patterns
            {
                'pattern': r'(.*?)\s+used\s+(\$[\d,]+\.?\d*\s?(?:million|billion)?)\s+(?:of\s+)?cash\s+(?:in\s+|for\s+)?operations?',
                'relation': lambda match: "used_cash_in_operations"
            },

            # Revenue patterns
            {
                'pattern': r'(.*?)\s+(?:generated|earned|reported|recorded)\s+(?:total\s+)?(?:revenue|revenues|sales)\s+of\s+(\$[\d,]+\.?\d*\s?(?:million|billion)?)',
                'relation': lambda match: "generated_revenue"
            },

            # Allocation patterns with years
            {
                'pattern': r'(.*?)\s+allocated\s+(?:to\s+.*?\s+)?(?:in\s+|for\s+)?(\d{4}).*?(?:were?|was)\s+(\$[\d,]+\.?\d*\s?(?:million|billion)?)',
                'relation': lambda match: "allocated_in"
            },

            # Earnings patterns
            {
                'pattern': r'(.*?)\s+(?:earnings|income)\s+(?:.*?\s+)?(?:in\s+|for\s+)?(\d{4}).*?(?:were?|was)\s+(\$[\d,]+\.?\d*\s?(?:million|billion)?)',
                'relation': lambda match: "earnings_in"
            }
        ]

        for pattern_info in financial_patterns:
            pattern = pattern_info['pattern']
            relation_func = pattern_info['relation']

            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    subject = self._clean_entity(match.group(1))
                    relation = relation_func(match)

                    if 'allocated_in' in relation or 'earnings_in' in relation:
                        # Special handling for year-based relations
                        year = match.group(2) if match.lastindex >= 2 else ""
                        amount = match.group(3) if match.lastindex >= 3 else match.group(2)
                        obj = f"{year} : {amount}" if year else amount
                    else:
                        obj = match.group(2) if match.lastindex >= 2 else ""

                    if subject and relation and obj:
                        relations.append((subject, relation, obj))
                except Exception as e:
                    continue

        return relations

    def extract_relations_rule_based(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relations using dependency parsing (if available) or pattern matching."""
        try:
            # Try dependency parsing if available
            if hasattr(self.nlp, 'pipe_names') and 'parser' in self.nlp.pipe_names:
                doc = self.nlp(text)
                relations = []

                for sent in doc.sents:
                    for token in sent:
                        if token.lemma_.lower() in self.financial_verbs:
                            # Find subject
                            subjects = []
                            for child in token.lefts:
                                if child.dep_ in ("nsubj", "nsubjpass"):
                                    subjects.append(child.text)

                            # Find objects
                            objects = []
                            for child in token.rights:
                                if child.dep_ in ("dobj", "attr", "pobj"):
                                    objects.append(child.text)

                            # Create relations
                            relation = self.financial_verbs[token.lemma_.lower()]
                            for subj in subjects:
                                for obj in objects:
                                    relations.append((subj, relation, obj))

                return relations
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Dependency parsing failed: {e}. Using pattern matching...")

        # Fallback to pattern-based extraction
        return self.extract_relations_pattern_based(text)

    def extract_financial_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract financial triples using combined approach."""
        entities = self.extract_entities(text)
        relations = self.extract_relations_rule_based(text)
        triples = []

        # Add rule-based/pattern-based relations
        triples.extend(relations)

        # Additional pattern-based extractions
        additional_patterns = [
            # Time period patterns
            (r'(?:during|for)\s+the\s+(?:year|period)\s+ended\s+(.*?),?\s+(\d{4})',
             lambda m: ("Company", "reporting_period", f"{m.group(1).strip()}, {m.group(2)}")),

            # Multiple year data patterns
            (r'(.*?)\s+in\s+(\d{4}),?\s+(\d{4})\s+and\s+(\d{4})\s+were?\s+(\$[\d,]+\.?\d*\s?(?:billion|million)?),?\s+(\$[\d,]+\.?\d*\s?(?:billion|million)?)\s+and\s+(\$[\d,]+\.?\d*\s?(?:billion|million)?)',
             lambda m: [
                 (self._clean_entity(m.group(1)), "amount_in", f"{m.group(2)} : {m.group(5)}"),
                 (self._clean_entity(m.group(1)), "amount_in", f"{m.group(3)} : {m.group(6)}"),
                 (self._clean_entity(m.group(1)), "amount_in", f"{m.group(4)} : {m.group(7)}")
             ])
        ]

        # Apply additional patterns
        for pattern, extractor in additional_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    result = extractor(match)
                    if isinstance(result, list):
                        triples.extend(result)
                    else:
                        triples.append(result)
                except Exception as e:
                    continue

        # Entity-relation matching for remaining entities
        if entities['ORG'] and entities['MONEY']:
            for org in entities['ORG']:
                for money in entities['MONEY']:
                    if entities['DATE'] or entities.get('YEAR', []):
                        dates = entities['DATE'] + entities.get('YEAR', [])
                        for date in dates[:3]:  # Limit to avoid too many combinations
                            triples.append((org, "financial_amount_on", f"{date} : {money}"))

        # Remove duplicates and clean up
        unique_triples = []
        seen = set()
        for triple in triples:
            triple_str = str(triple)
            if triple_str not in seen and all(str(x).strip() for x in triple):
                seen.add(triple_str)
                unique_triples.append(triple)

        return unique_triples

    def _clean_entity(self, entity: str) -> str:
        """Clean and normalize entity names."""
        if not entity:
            return "Company"

        entity = entity.strip()
        # Remove common prefixes
        prefixes = ["the", "The", "during", "During", "for", "For", "and", "And"]
        for prefix in prefixes:
            if entity.startswith(prefix + " "):
                entity = entity[len(prefix + " "):]

        # Remove trailing punctuation
        entity = re.sub(r'[,.:;]+$', '', entity)

        return entity.strip() or "Company"

    def process_json_file(self, json_file_path: str, text_fields: List[str] = None) -> List[Dict]:
        """Process JSON file and extract triples from specified text fields."""
        data = self.load_json_data(json_file_path)
        results = []

        if text_fields is None:
            text_fields = ['text', 'content', 'description', 'summary', 'statement', 'narrative', 'notes']

        def extract_from_dict(obj, parent_key=""):
            triples = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_key = f"{parent_key}.{key}" if parent_key else key
                    if isinstance(value, str) and len(value.strip()) > 10:
                        # Check if field name suggests it contains text to analyze
                        if any(field in key.lower() for field in text_fields) or len(value) > 50:
                            extracted_triples = self.extract_financial_triples(value)
                            for triple in extracted_triples:
                                triples.append({
                                    'source_field': current_key,
                                    'subject': triple[0],
                                    'relation': triple[1],
                                    'object': triple[2],
                                    'source_text': value[:200] + "..." if len(value) > 200 else value
                                })
                    elif isinstance(value, (dict, list)):
                        triples.extend(extract_from_dict(value, current_key))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                    triples.extend(extract_from_dict(item, current_key))
            return triples

        return extract_from_dict(data)

    def save_triples_to_csv(self, triples: List[Dict], output_file: str):
        """Save extracted triples to CSV file."""
        if not triples:
            if self.verbose:
                print("No triples to save.")
            return

        df = pd.DataFrame(triples)
        df.to_csv(output_file, index=False)
        if self.verbose:
            print(f"✅ {len(triples)} triples saved to {output_file}")

    def save_triples_to_json(self, triples: List[Dict], output_file: str):
        """Save extracted triples to JSON file."""
        if not triples:
            if self.verbose:
                print("No triples to save.")
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(triples, f, indent=2, ensure_ascii=False)
        if self.verbose:
            print(f"✅ {len(triples)} triples saved to {output_file}")

# Example usage and testing
def main(verbose=False):
    """Initialize the extractor with optional verbose output."""
    try:
        extractor = FinancialTripleExtractor(verbose=verbose)
        return extractor
    except Exception as e:
        if verbose:
            print(f"❌ Failed to initialize extractor: {e}")
        return None

def process_your_file(extractor, file_path, verbose=False):
    """Process your specific input file."""
    if verbose:
        print(f"\n=== Processing Your File: {file_path} ===")

    try:
        # Load your text file
        if file_path.endswith('.json'):
            triples = extractor.process_json_file(file_path)
        else:
            # Load text file
            text_content = extractor.load_text_file(file_path)
            triples = extractor.process_text_segments(text_content)

        if verbose:
            print(f"\n📊 Successfully extracted {len(triples)} triples from your file!")

            # Display first few triples as preview
            print("\n🔍 Preview of extracted triples:")
            for i, triple in enumerate(triples[:10], 1):  # Show first 10
                if isinstance(triple, dict) and 'segment_id' in triple:
                    print(f"\n{i}. Segment {triple['segment_id']}:")
                    print(f"   Triple: ({triple['subject']}, {triple['relation']}, {triple['object']})")
                    print(f"   Source: {triple['source_text'][:100]}...")
                else:
                    print(f"\n{i}. Field: {triple.get('source_field', 'N/A')}")
                    print(f"   Triple: ({triple['subject']}, {triple['relation']}, {triple['object']})")

            if len(triples) > 10:
                print(f"\n... and {len(triples) - 10} more triples")

        # Save results
        if triples:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = f'extracted_triples_{timestamp}.csv'
            json_file = f'extracted_triples_{timestamp}.json'

            extractor.save_triples_to_csv(triples, csv_file)
            extractor.save_triples_to_json(triples, json_file)
            
            if verbose:
                print(f"\n✅ Results saved to {csv_file} and {json_file}")

        return triples

    except FileNotFoundError:
        if verbose:
            print(f"❌ File not found: {file_path}")
            print("Make sure the file path is correct and the file exists.")
        return []
    except Exception as e:
        if verbose:
            print(f"❌ Error processing file: {e}")
        return []

# Clean usage example
if __name__ == "__main__":
    # Initialize extractor silently
    extractor = main(verbose=False)
    
    # Process file silently
    if extractor:
        triples = process_your_file(extractor, '/content/extracted_text_only.txt', verbose=False)
        
        # Only show final results
        if triples:
            print(f"Extracted {len(triples)} financial triples successfully.")
        else:
            print("No triples extracted.")

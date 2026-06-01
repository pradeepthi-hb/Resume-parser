import os
import json
import logging
import random
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

try:
    from src.training.correction_learning import get_correction_learning_store
    STRUCTURED_CORRECTION_DATA_AVAILABLE = True
except ImportError:
    STRUCTURED_CORRECTION_DATA_AVAILABLE = False
    get_correction_learning_store = None


@dataclass
class TrainingExample:
    input_text: str
    output_text: str
    field: str
    metadata: Dict[str, Any]
    source: str = "feedback"  
    verified: bool = False


@dataclass
class TrainingDataset:
    name: str
    field: str
    examples: List[TrainingExample]
    created_at: datetime
    version: str = "1.0"
    description: str = ""


class DataPreparator:
    
    
    DEFAULT_SPLIT = {
        "train": 0.7,
        "val": 0.15,
        "test": 0.15
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            data_dir = os.path.join(base_dir, 'training_data')
        
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def load_feedback_data(
        self,
        field_name: Optional[str] = None,
        min_quality: float = 0.5
    ) -> List[TrainingExample]:
        examples = []
        
        
        feedback_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'learning_data', 'feedback'
        )
        
        if not os.path.exists(feedback_dir):
            logger.warning(f"Feedback directory not found: {feedback_dir}")
            return examples
            
        for filename in os.listdir(feedback_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(feedback_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                
                if field_name and data.get('field_name') != field_name:
                    continue
                    
                
                if data.get('feedback_type') != 'correction':
                    continue
                    
                
                example = TrainingExample(
                    input_text=data.get('original_value', ''),
                    output_text=data.get('corrected_value', ''),
                    field=data.get('field_name', ''),
                    metadata={
                        "source_file": filename,
                        "timestamp": data.get('timestamp', ''),
                        "user_id": data.get('user_id', ''),
                        "comment": data.get('comment', '')
                    },
                    source="feedback",
                    verified=False
                )
                
                examples.append(example)
                
            except Exception as e:
                logger.error(f"Error loading feedback file {filename}: {e}")
                
        logger.info(f"Loaded {len(examples)} training examples from feedback")
        return examples
    
    def load_approved_corrections(
        self,
        field_name: Optional[str] = None
    ) -> List[TrainingExample]:
        # Legacy `sample_*.json` based correction storage has been retired.
        # Structured corrections from `structured_corrections.jsonl` are used instead.
        return []

    def load_structured_corrections(
        self,
        field_name: Optional[str] = None,
        only_changed: bool = True
    ) -> List[TrainingExample]:
        examples: List[TrainingExample] = []
        if not STRUCTURED_CORRECTION_DATA_AVAILABLE or get_correction_learning_store is None:
            return examples

        try:
            store = get_correction_learning_store()
            samples = store.load_samples(
                field_name=field_name,
                status="approved",
                only_changed=only_changed,
            )
            for sample in samples:
                examples.append(
                    TrainingExample(
                        input_text=sample.get("original_value", ""),
                        output_text=sample.get("corrected_value", ""),
                        field=sample.get("field_name", ""),
                        metadata={
                            "sample_id": sample.get("sample_id"),
                            "timestamp": sample.get("timestamp", ""),
                            "confidence_before": sample.get("confidence_before", 0.0),
                            "confidence_after": sample.get("confidence_after", 0.0),
                            "feedback_type": sample.get("feedback_type", ""),
                            "source": sample.get("source", ""),
                        },
                        source="feedback",
                        verified=True,
                    )
                )
        except Exception as e:
            logger.error(f"Error loading structured corrections: {e}")

        return examples
    
    def add_manual_example(
        self,
        input_text: str,
        output_text: str,
        field: str,
        verified: bool = True
    ) -> bool:
        try:
            example = TrainingExample(
                input_text=input_text,
                output_text=output_text,
                field=field,
                metadata={},
                source="manual",
                verified=verified
            )
            
            
            field_dir = os.path.join(self.data_dir, 'manual', field)
            os.makedirs(field_dir, exist_ok=True)
            
            filename = f"example_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            filepath = os.path.join(field_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(asdict(example), f, default=str)
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding manual example: {e}")
            return False
    
    def augment_data(
        self,
        examples: List[TrainingExample],
        augmentation_factor: float = 0.2
    ) -> List[TrainingExample]:
        augmented = []
        
        for example in examples:
            
            if len(example.input_text) < 10:
                continue
                
            
            variations = self._create_variations(example)
            num_augmented = max(1, int(len(examples) * augmentation_factor))
            
            for _ in range(min(num_augmented, len(variations))):
                augmented.append(random.choice(variations))
                
        logger.info(f"Created {len(augmented)} augmented examples")
        return examples + augmented
    
    def _create_variations(self, example: TrainingExample) -> List[TrainingExample]:
        variations = []
        
        input_text = example.input_text
        output_text = example.output_text
        
        
        if '  ' in input_text:
            variations.append(TrainingExample(
                input_text=' '.join(input_text.split()),
                output_text=output_text,
                field=example.field,
                metadata={**example.metadata, "variation": "normalize_whitespace"},
                source="generated",
                verified=False
            ))
            
        
        variations.append(TrainingExample(
            input_text=input_text.lower(),
            output_text=output_text,
            field=example.field,
            metadata={**example.metadata, "variation": "lowercase"},
            source="generated",
            verified=False
        ))
        
        
        import re
        cleaned = re.sub(r'[^\w\s]', '', input_text)
        if cleaned != input_text:
            variations.append(TrainingExample(
                input_text=cleaned,
                output_text=output_text,
                field=example.field,
                metadata={**example.metadata, "variation": "remove_special_chars"},
                source="generated",
                verified=False
            ))
            
        return variations
    
    def split_dataset(
        self,
        examples: List[TrainingExample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True
    ) -> Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Ratios must sum to 1.0")
            
        
        if shuffle:
            examples = examples.copy()
            random.shuffle(examples)
            
        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = examples[:train_end]
        val = examples[train_end:val_end]
        test = examples[val_end:]
        
        logger.info(f"Split dataset: train={len(train)}, val={len(val)}, test={len(test)}")
        
        return train, val, test
    
    def save_dataset(
        self,
        dataset: TrainingDataset,
        format: str = "json"
    ) -> str:
        field_dir = os.path.join(self.data_dir, dataset.field)
        os.makedirs(field_dir, exist_ok=True)
        
        filename = f"{dataset.name}_{dataset.version}.{format}"
        filepath = os.path.join(field_dir, filename)
        
        if format == "json":
            data = {
                "name": dataset.name,
                "field": dataset.field,
                "version": dataset.version,
                "description": dataset.description,
                "created_at": dataset.created_at.isoformat(),
                "examples": [asdict(ex) for ex in dataset.examples]
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format == "spacy":
            
            import spacy
            nlp = spacy.blank("en")
            train_data = []
            
            for ex in dataset.examples:
                doc = nlp.make_doc(ex.input_text)
                
                train_data.append((ex.input_text, {"cats": {ex.field: 1.0}}))
                
            
            filepath = filepath.replace('.spacy', '.jsonl')
            with open(filepath, 'w') as f:
                for item in train_data:
                    f.write(json.dumps(item) + '\n')
                    
        logger.info(f"Saved dataset to {filepath}")
        return filepath
    
    def load_dataset(
        self,
        name: str,
        field: str,
        version: str = "1.0"
    ) -> Optional[TrainingDataset]:
        filepath = os.path.join(
            self.data_dir, field, f"{name}_{version}.json"
        )
        
        if not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            examples = [
                TrainingExample(**ex) for ex in data.get('examples', [])
            ]
            
            return TrainingDataset(
                name=data.get('name', name),
                field=data.get('field', field),
                version=data.get('version', version),
                description=data.get('description', ''),
                examples=examples,
                created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
            )
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None
    
    def prepare_for_training(
        self,
        field: str,
        min_examples: int = 10,
        augment: bool = True,
        augmentation_factor: float = 0.2
    ) -> Optional[Tuple[List[TrainingExample], List[TrainingExample], List[TrainingExample]]]:
        
        examples = []
        
        
        examples.extend(self.load_feedback_data(field))
        
        examples.extend(self.load_structured_corrections(field, only_changed=True))
        
        
        manual_dir = os.path.join(self.data_dir, 'manual', field)
        if os.path.exists(manual_dir):
            for filename in os.listdir(manual_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(manual_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            examples.append(TrainingExample(**data))
                    except:
                        pass
        
        
        if len(examples) < min_examples:
            logger.warning(f"Insufficient examples for {field}: {len(examples)} < {min_examples}")
            return None
        
        
        if augment:
            examples = self.augment_data(examples, augmentation_factor)
        
        
        train, val, test = self.split_dataset(examples)
        
        return train, val, test
    
    def get_dataset_statistics(
        self,
        field: Optional[str] = None
    ) -> Dict[str, Any]:
        stats = {
            "total_examples": 0,
            "by_field": {},
            "by_source": {"feedback": 0, "manual": 0, "generated": 0},
            "verified": 0,
            "unverified": 0,
            "fields": []
        }
        
        
        sources = [
            self.load_feedback_data,
            self.load_structured_corrections,
        ]
        
        for source_func in sources:
            examples = source_func() if field is None else source_func(field)
            for ex in examples:
                stats['total_examples'] += 1
                stats['by_source'][ex.source] = stats['by_source'].get(ex.source, 0) + 1
                
                if ex.field not in stats['by_field']:
                    stats['by_field'][ex.field] = 0
                    stats['fields'].append(ex.field)
                stats['by_field'][ex.field] += 1
                
                if ex.verified:
                    stats['verified'] += 1
                else:
                    stats['unverified'] += 1
        
        return stats



_data_preparator = None


def get_data_preparator() -> DataPreparator:
    global _data_preparator
    if _data_preparator is None:
        _data_preparator = DataPreparator()
    return _data_preparator

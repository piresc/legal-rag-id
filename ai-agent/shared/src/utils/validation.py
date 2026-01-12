"""
Validation utilities for Indonesian Legal RAG System
"""

import re
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Validation result"""
    status: ValidationStatus
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    details: Optional[Dict[str, Any]] = None


class DocumentValidator:
    """Validator for Indonesian legal documents"""
    
    # Indonesian legal document patterns
    DOC_TYPE_PATTERNS = {
        'UU': r'(?:Undang-Undang|UU)\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})',
        'PP': r'(?:Peraturan\s+Pemerintah|PP)\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})',
        'Perpres': r'(?:Peraturan\s+Presiden|Perpres)\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})',
        'Keppres': r'(?:Keputusan\s+Presiden|Keppres)\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})',
        'Permen': r'(?:Peraturan\s+Menteri|Permen)\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})',
        'Perda': r'(?:Peraturan\s+Daerah|Perda)\s+Nomor\s+(\d+)\s+Tahun\s+(\d{4})'
    }
    
    # Pasal (Article) patterns
    PASAL_PATTERNS = [
        r'Pasal\s+(\d+)',
        r'Artikel\s+(\d+)',
        r'Pasal\s+(\d+)\s+Ayat\s+(\d+)',
        r'Pasal\s+(\d+)\s+[(]([a-zA-Z0-9]+)[)]'
    ]
    
    # Quality indicators
    MIN_CONTENT_LENGTH = 100
    MAX_CONTENT_LENGTH = 1000000  # 1MB
    REQUIRED_METADATA_FIELDS = ['title', 'doc_type']
    
    def __init__(self):
        self.compile_patterns()
    
    def compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self.compiled_doc_patterns = {
            doc_type: re.compile(pattern, re.IGNORECASE)
            for doc_type, pattern in self.DOC_TYPE_PATTERNS.items()
        }
        
        self.compiled_pasal_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.PASAL_PATTERNS
        ]
    
    def validate_document(self, document: Dict[str, Any]) -> List[ValidationResult]:
        """Validate a complete document"""
        results = []
        
        # Validate basic structure
        results.extend(self._validate_basic_structure(document))
        
        # Validate content
        if 'content' in document:
            results.extend(self._validate_content(document['content']))
        
        # Validate metadata
        if 'metadata' in document:
            results.extend(self._validate_metadata(document['metadata']))
        
        # Validate legal structure
        if 'content' in document:
            results.extend(self._validate_legal_structure(document['content']))
        
        return results
    
    def _validate_basic_structure(self, document: Dict[str, Any]) -> List[ValidationResult]:
        """Validate basic document structure"""
        results = []
        
        # Check required fields
        required_fields = ['doc_id', 'title', 'content']
        for field in required_fields:
            if field not in document:
                results.append(ValidationResult(
                    status=ValidationStatus.ERROR,
                    message=f"Missing required field: {field}",
                    field=field
                ))
            elif not document[field]:
                results.append(ValidationResult(
                    status=ValidationStatus.ERROR,
                    message=f"Required field cannot be empty: {field}",
                    field=field,
                    value=document[field]
                ))
        
        # Check doc_id format
        if 'doc_id' in document:
            doc_id = document['doc_id']
            if not isinstance(doc_id, str) or len(doc_id) < 3:
                results.append(ValidationResult(
                    status=ValidationStatus.ERROR,
                    message="Document ID must be a string with at least 3 characters",
                    field='doc_id',
                    value=doc_id
                ))
        
        return results
    
    def _validate_content(self, content: str) -> List[ValidationResult]:
        """Validate document content"""
        results = []
        
        if not isinstance(content, str):
            results.append(ValidationResult(
                status=ValidationStatus.ERROR,
                message="Content must be a string",
                field='content',
                value=type(content).__name__
            ))
            return results
        
        # Check content length
        if len(content) < self.MIN_CONTENT_LENGTH:
            results.append(ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Content too short: {len(content)} characters (minimum: {self.MIN_CONTENT_LENGTH})",
                field='content',
                value=len(content)
            ))
        
        if len(content) > self.MAX_CONTENT_LENGTH:
            results.append(ValidationResult(
                status=ValidationStatus.ERROR,
                message=f"Content too long: {len(content)} characters (maximum: {self.MAX_CONTENT_LENGTH})",
                field='content',
                value=len(content)
            ))
        
        # Check for Indonesian legal content indicators
        has_legal_terms = self._has_legal_indicators(content)
        if not has_legal_terms:
            results.append(ValidationResult(
                status=ValidationStatus.WARNING,
                message="Content may not be an Indonesian legal document (no legal indicators found)",
                field='content'
            ))
        
        return results
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> List[ValidationResult]:
        """Validate document metadata"""
        results = []
        
        # Check required metadata fields
        for field in self.REQUIRED_METADATA_FIELDS:
            if field not in metadata:
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Missing recommended metadata field: {field}",
                    field=f'metadata.{field}'
                ))
        
        # Validate document type
        if 'doc_type' in metadata:
            doc_type = metadata['doc_type']
            valid_types = list(self.DOC_TYPE_PATTERNS.keys())
            if doc_type not in valid_types:
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Unknown document type: {doc_type}. Valid types: {valid_types}",
                    field='metadata.doc_type',
                    value=doc_type
                ))
        
        # Validate year
        if 'year' in metadata:
            year = metadata['year']
            if not isinstance(year, int) or year < 1900 or year > 2030:
                results.append(ValidationResult(
                    status=ValidationStatus.ERROR,
                    message=f"Invalid year: {year}. Must be between 1900 and 2030",
                    field='metadata.year',
                    value=year
                ))
        
        # Validate number
        if 'number' in metadata:
            number = metadata['number']
            if not isinstance(number, (str, int)) or (isinstance(number, str) and not number.isdigit()):
                results.append(ValidationResult(
                    status=ValidationStatus.ERROR,
                    message=f"Invalid document number: {number}",
                    field='metadata.number',
                    value=number
                ))
        
        return results
    
    def _validate_legal_structure(self, content: str) -> List[ValidationResult]:
        """Validate legal document structure"""
        results = []
        
        # Check for document type identification
        doc_type_found = False
        for doc_type, pattern in self.compiled_doc_patterns.items():
            if pattern.search(content):
                doc_type_found = True
                break
        
        if not doc_type_found:
            results.append(ValidationResult(
                status=ValidationStatus.WARNING,
                message="No document type pattern found (UU, PP, Perpres, etc.)",
                field='content'
            ))
        
        # Check for Pasal (Article) structure
        pasal_matches = []
        for pattern in self.compiled_pasal_patterns:
            matches = pattern.findall(content)
            pasal_matches.extend(matches)
        
        if not pasal_matches:
            results.append(ValidationResult(
                status=ValidationStatus.WARNING,
                message="No Pasal (Article) structure found",
                field='content'
            ))
        else:
            results.append(ValidationResult(
                status=ValidationStatus.VALID,
                message=f"Found {len(pasal_matches)} Pasal references",
                field='content',
                details={'pasal_count': len(pasal_matches)}
            ))
        
        return results
    
    def _has_legal_indicators(self, content: str) -> bool:
        """Check if content has Indonesian legal document indicators"""
        legal_terms = [
            'pasal', 'ayat', 'huruf', 'angka', 'bab', 'bagian', 'paragraf',
            'undang-undang', 'peraturan pemerintah', 'peraturan presiden',
            'menetapkan', 'dengan persetujuan', 'sebagaimana dimaksud',
            'dipidana', 'denda', 'kurungan', 'penjara'
        ]
        
        content_lower = content.lower()
        return any(term in content_lower for term in legal_terms)


class QueryValidator:
    """Validator for user queries"""
    
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 1000
    FORBIDDEN_WORDS = ['password', 'secret', 'token', 'api_key']
    
    def validate_query(self, query: str) -> List[ValidationResult]:
        """Validate user query"""
        results = []
        
        if not isinstance(query, str):
            results.append(ValidationResult(
                status=ValidationStatus.ERROR,
                message="Query must be a string",
                field='query',
                value=type(query).__name__
            ))
            return results
        
        # Check length
        if len(query.strip()) < self.MIN_QUERY_LENGTH:
            results.append(ValidationResult(
                status=ValidationStatus.ERROR,
                message=f"Query too short: {len(query)} characters (minimum: {self.MIN_QUERY_LENGTH})",
                field='query',
                value=len(query)
            ))
        
        if len(query) > self.MAX_QUERY_LENGTH:
            results.append(ValidationResult(
                status=ValidationStatus.ERROR,
                message=f"Query too long: {len(query)} characters (maximum: {self.MAX_QUERY_LENGTH})",
                field='query',
                value=len(query)
            ))
        
        # Check for forbidden words
        query_lower = query.lower()
        for word in self.FORBIDDEN_WORDS:
            if word in query_lower:
                results.append(ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Query contains potentially sensitive word: {word}",
                    field='query'
                ))
        
        # Check for Indonesian language indicators
        has_indonesian = self._has_indonesian_indicators(query)
        if not has_indonesian:
            results.append(ValidationResult(
                status=ValidationStatus.WARNING,
                message="Query may not be in Indonesian language",
                field='query'
            ))
        
        return results
    
    def _has_indonesian_indicators(self, text: str) -> bool:
        """Check if text has Indonesian language indicators"""
        indonesian_words = [
            'apa', 'bagaimana', 'mengapa', 'kapan', 'dimana', 'siapa', 'berapa',
            'adalah', 'yaitu', 'yakni', 'dan', 'atau', 'tetapi', 'jika',
            'pasal', 'undang-undang', 'peraturan', 'hukum', 'sanksi', 'denda'
        ]
        
        text_lower = text.lower()
        return any(word in text_lower for word in indonesian_words)


def validate_document(document: Dict[str, Any]) -> List[ValidationResult]:
    """Validate document (convenience function)"""
    validator = DocumentValidator()
    return validator.validate_document(document)


def validate_query(query: str) -> List[ValidationResult]:
    """Validate query (convenience function)"""
    validator = QueryValidator()
    return validator.validate_query(query)


def is_valid_document(document: Dict[str, Any]) -> bool:
    """Check if document is valid (no errors)"""
    results = validate_document(document)
    return not any(r.status == ValidationStatus.ERROR for r in results)


def is_valid_query(query: str) -> bool:
    """Check if query is valid (no errors)"""
    results = validate_query(query)
    return not any(r.status == ValidationStatus.ERROR for r in results)


class QualityScorer:
    """Quality scoring for documents and chunks"""
    
    def score_document(self, document: Dict[str, Any]) -> float:
        """Score document quality (0.0 to 1.0)"""
        score = 0.0
        max_score = 0.0
        
        # Content quality (0.3)
        max_score += 0.3
        if 'content' in document:
            content = document['content']
            if len(content) > 1000:
                score += 0.1
            if len(content) > 10000:
                score += 0.1
            if self._has_legal_structure(content):
                score += 0.1
        
        # Metadata quality (0.3)
        max_score += 0.3
        if 'metadata' in document:
            metadata = document['metadata']
            if 'doc_type' in metadata and 'year' in metadata and 'number' in metadata:
                score += 0.2
            if 'issuer' in metadata or 'subject' in metadata:
                score += 0.1
        
        # Structure quality (0.2)
        max_score += 0.2
        if 'content' in document:
            content = document['content']
            pasal_count = len(re.findall(r'Pasal\s+\d+', content, re.IGNORECASE))
            if pasal_count > 5:
                score += 0.1
            if pasal_count > 20:
                score += 0.1
        
        # Uniqueness (0.2)
        max_score += 0.2
        if 'doc_id' in document and len(document['doc_id']) > 10:
            score += 0.2
        
        return score / max_score if max_score > 0 else 0.0
    
    def _has_legal_structure(self, content: str) -> bool:
        """Check if content has legal structure"""
        return bool(re.search(r'(Pasal|Bab|Bagian)\s+\w+', content, re.IGNORECASE))
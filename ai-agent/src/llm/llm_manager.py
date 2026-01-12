"""
LLM Manager for Indonesian Legal RAG System AI Agent Layer
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import time
from abc import ABC, abstractmethod

from shared.src.utils.config import Config, get_config
from shared.src.utils.logging import get_logger, log_async_performance
from shared.src.utils.metrics import track_performance, increment_counter, record_timing

logger = get_logger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GROQ = "groq"


@dataclass
class LLMResponse:
    """LLM response structure"""
    content: str
    provider: str
    model: str
    tokens_used: int
    processing_time: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: LLMProvider
    model: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30
    enabled: bool = True


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = config.provider.value
        self.model_name = config.model
        self.api_key = config.api_key
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.timeout = config.timeout
        
    @abstractmethod
    async def generate_response(self, query: str, context: str, **kwargs) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self._client = None
    
    async def _get_client(self):
        """Get OpenAI client"""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncClient(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError("openai package is required for OpenAI provider")
        return self._client
    
    @log_async_performance("openai_response_generation")
    async def generate_response(self, query: str, context: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI"""
        start_time = time.time()
        
        try:
            client = await self._get_client()
            
            # Build system prompt for Indonesian legal assistant
            system_prompt = self._build_system_prompt()
            
            # Build user prompt with context
            user_prompt = self._build_user_prompt(query, context)
            
            # Make API call
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            processing_time = time.time() - start_time
            
            # Extract response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            increment_counter("openai_requests_total")
            record_timing("openai_response_time", processing_time)
            
            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model_name,
                tokens_used=tokens_used,
                processing_time=processing_time,
                confidence=0.8,  # Default confidence for OpenAI
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            increment_counter("openai_errors_total")
            raise
    
    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        try:
            client = await self._get_client()
            response = await client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information"""
        return {
            'provider': self.provider_name,
            'model': self.model_name,
            'base_url': self.base_url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for Indonesian legal assistant"""
        return """Anda adalah asisten hukum Indonesia yang profesional dan berpengetahuan luas. 
Tugas Anda adalah menjawab pertanyaan hukum berdasarkan dokumen hukum Indonesia yang diberikan.

Pedoman jawaban:
1. Jawab dalam bahasa Indonesia yang jelas dan mudah dipahami
2. Berikan jawaban yang akurat berdasarkan informasi dari dokumen
3. Sertakan referensi ke pasal atau dokumen yang relevan
4. Jika informasi tidak cukup, jelaskan secara jelas
5. Berikan jawaban yang ringkas namun komprehensif
6. Hindari memberikan nasihat hukum formal, sebutkan bahwa ini adalah informasi umum

Format jawaban:
- Jawaban langsung pertanyaan
- Dasar hukum (pasal/dokumen)
- Penjelasan singkat jika diperlukan"""
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build user prompt with context"""
        return f"""Pertanyaan: {query}

{context}

Berdasarkan dokumen hukum di atas, jawab pertanyaan tersebut secara akurat dan profesional."""


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com"
        self._client = None
    
    async def _get_client(self):
        """Get Anthropic client"""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError("anthropic package is required for Anthropic provider")
        return self._client
    
    @log_async_performance("anthropic_response_generation")
    async def generate_response(self, query: str, context: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic Claude"""
        start_time = time.time()
        
        try:
            client = await self._get_client()
            
            # Build prompt
            prompt = self._build_prompt(query, context)
            
            # Make API call
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            processing_time = time.time() - start_time
            
            # Extract response
            content = response.content[0].text if response.content else ""
            tokens_used = response.usage.input_tokens + response.usage.output_tokens if response.usage else 0
            
            increment_counter("anthropic_requests_total")
            record_timing("anthropic_response_time", processing_time)
            
            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model_name,
                tokens_used=tokens_used,
                processing_time=processing_time,
                confidence=0.85,  # Default confidence for Claude
                metadata={
                    'stop_reason': response.stop_reason,
                    'input_tokens': response.usage.input_tokens if response.usage else 0,
                    'output_tokens': response.usage.output_tokens if response.usage else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            increment_counter("anthropic_errors_total")
            raise
    
    async def health_check(self) -> bool:
        """Check Anthropic API health"""
        try:
            client = await self._get_client()
            # Simple test message
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information"""
        return {
            'provider': self.provider_name,
            'model': self.model_name,
            'base_url': self.base_url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for Claude"""
        return f"""Anda adalah asisten hukum Indonesia yang profesional. Berdasarkan dokumen hukum berikut, jawab pertanyaan tersebut:

Dokumen Hukum:
{context}

Pertanyaan: {query}

Jawab dalam bahasa Indonesia dengan format:
1. Jawaban langsung
2. Dasar hukum (pasal/dokumen)
3. Penjelasan jika diperlukan

Berikan jawaban yang akurat dan profesional."""


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.deepseek.com"
    
    @log_async_performance("deepseek_response_generation")
    async def generate_response(self, query: str, context: str, **kwargs) -> LLMResponse:
        """Generate response using DeepSeek"""
        start_time = time.time()
        
        try:
            import aiohttp
            
            # Build request
            prompt = self._build_prompt(query, context)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "Anda adalah asisten hukum Indonesia yang profesional."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"DeepSeek API error: {response.status} - {error_text}")
                    
                    result = await response.json()
            
            processing_time = time.time() - start_time
            
            # Extract response
            content = result["choices"][0]["message"]["content"]
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            
            increment_counter("deepseek_requests_total")
            record_timing("deepseek_response_time", processing_time)
            
            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model_name,
                tokens_used=tokens_used,
                processing_time=processing_time,
                confidence=0.8,
                metadata={
                    'finish_reason': result["choices"][0].get("finish_reason")
                }
            )
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            increment_counter("deepseek_errors_total")
            raise
    
    async def health_check(self) -> bool:
        """Check DeepSeek API health"""
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
            
        except Exception as e:
            logger.error(f"DeepSeek health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get DeepSeek model information"""
        return {
            'provider': self.provider_name,
            'model': self.model_name,
            'base_url': self.base_url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for DeepSeek"""
        return f"""Sebagai asisten hukum Indonesia, jawab pertanyaan berikut berdasarkan dokumen hukum yang disediakan:

Dokumen Hukum:
{context}

Pertanyaan: {query}

Berikan jawaban yang akurat dalam bahasa Indonesia dengan menyertakan dasar hukumnya."""


class GroqProvider(BaseLLMProvider):
    """Groq provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.groq.com/openai/v1"
    
    @log_async_performance("groq_response_generation")
    async def generate_response(self, query: str, context: str, **kwargs) -> LLMResponse:
        """Generate response using Groq"""
        start_time = time.time()
        
        try:
            import aiohttp
            
            # Build request
            prompt = self._build_prompt(query, context)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "Anda adalah asisten hukum Indonesia yang profesional."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Groq API error: {response.status} - {error_text}")
                    
                    result = await response.json()
            
            processing_time = time.time() - start_time
            
            # Extract response
            content = result["choices"][0]["message"]["content"]
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            
            increment_counter("groq_requests_total")
            record_timing("groq_response_time", processing_time)
            
            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model_name,
                tokens_used=tokens_used,
                processing_time=processing_time,
                confidence=0.75,
                metadata={
                    'finish_reason': result["choices"][0].get("finish_reason")
                }
            )
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            increment_counter("groq_errors_total")
            raise
    
    async def health_check(self) -> bool:
        """Check Groq API health"""
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
            
        except Exception as e:
            logger.error(f"Groq health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Groq model information"""
        return {
            'provider': self.provider_name,
            'model': self.model_name,
            'base_url': self.base_url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature
        }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for Groq"""
        return f"""Sebagai asisten hukum Indonesia, jawab pertanyaan berikut:

Dokumen Hukum:
{context}

Pertanyaan: {query}

Jawab dalam bahasa Indonesia dengan akurat dan profesional."""


class LLMManager:
    """Main LLM manager for handling multiple providers"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.default_provider = LLMProvider(self.config.llm.default_provider)
        self.stats = {
            'total_requests': 0,
            'requests_by_provider': {},
            'average_response_time': 0.0,
            'total_tokens_used': 0
        }
        
    async def initialize(self):
        """Initialize LLM providers"""
        provider_configs = self._get_provider_configs()
        
        for provider_config in provider_configs:
            if provider_config.enabled and provider_config.api_key:
                try:
                    provider = self._create_provider(provider_config)
                    if provider:
                        self.providers[provider_config.provider] = provider
                        logger.info(f"Initialized {provider_config.provider.value} provider")
                except Exception as e:
                    logger.error(f"Failed to initialize {provider_config.provider.value} provider: {str(e)}")
        
        if not self.providers:
            logger.warning("No LLM providers initialized")
    
    def _get_provider_configs(self) -> List[LLMConfig]:
        """Get provider configurations from config"""
        configs = []
        
        # OpenAI
        if self.config.llm.openai_api_key:
            configs.append(LLMConfig(
                provider=LLMProvider.OPENAI,
                model=self.config.llm.default_model if self.config.llm.default_provider == "openai" else "gpt-3.5-turbo",
                api_key=self.config.llm.openai_api_key,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            ))
        
        # Anthropic
        if self.config.llm.anthropic_api_key:
            configs.append(LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-sonnet-20240229",
                api_key=self.config.llm.anthropic_api_key,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            ))
        
        # DeepSeek
        if self.config.llm.deepseek_api_key:
            configs.append(LLMConfig(
                provider=LLMProvider.DEEPSEEK,
                model=self.config.llm.default_model if self.config.llm.default_provider == "deepseek" else "deepseek-reasoner",
                api_key=self.config.llm.deepseek_api_key,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            ))
        
        # Groq
        if self.config.llm.groq_api_key:
            configs.append(LLMConfig(
                provider=LLMProvider.GROQ,
                model="llama3-70b-8192",
                api_key=self.config.llm.groq_api_key,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout
            ))
        
        return configs
    
    def _create_provider(self, config: LLMConfig) -> Optional[BaseLLMProvider]:
        """Create provider instance based on type"""
        provider_map = {
            LLMProvider.OPENAI: OpenAIProvider,
            LLMProvider.ANTHROPIC: AnthropicProvider,
            LLMProvider.DEEPSEEK: DeepSeekProvider,
            LLMProvider.GROQ: GroqProvider
        }
        
        provider_class = provider_map.get(config.provider)
        if provider_class:
            return provider_class(config)
        
        return None
    
    @track_performance("llm_response_generation")
    async def generate_response(self, query: str, context: str, 
                              provider: Optional[str] = None, 
                              model: Optional[str] = None) -> LLMResponse:
        """Generate response using specified or default provider"""
        if not self.providers:
            raise RuntimeError("No LLM providers available")
        
        # Select provider
        selected_provider = None
        
        if provider:
            try:
                provider_enum = LLMProvider(provider)
                selected_provider = self.providers.get(provider_enum)
            except ValueError:
                logger.warning(f"Unknown provider: {provider}, using default")
        
        if not selected_provider:
            selected_provider = self.providers.get(self.default_provider)
        
        if not selected_provider:
            # Fallback to any available provider
            selected_provider = next(iter(self.providers.values()))
        
        try:
            # Generate response
            response = await selected_provider.generate_response(query, context)
            
            # Update statistics
            self._update_stats(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response with {selected_provider.provider_name}: {str(e)}")
            
            # Try fallback provider
            if len(self.providers) > 1:
                for fallback_provider in self.providers.values():
                    if fallback_provider != selected_provider:
                        try:
                            logger.info(f"Trying fallback provider: {fallback_provider.provider_name}")
                            response = await fallback_provider.generate_response(query, context)
                            self._update_stats(response)
                            return response
                        except Exception as fallback_error:
                            logger.error(f"Fallback provider also failed: {str(fallback_error)}")
                            continue
            
            raise RuntimeError("All LLM providers failed")
    
    async def health_check(self) -> bool:
        """Check health of all providers"""
        if not self.providers:
            return False
        
        healthy_count = 0
        for provider in self.providers.values():
            try:
                if await provider.health_check():
                    healthy_count += 1
            except Exception as e:
                logger.error(f"Health check failed for {provider.provider_name}: {str(e)}")
        
        return healthy_count > 0
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {}
        
        for provider_enum, provider in self.providers.items():
            try:
                is_healthy = await provider.health_check()
                status[provider_enum.value] = {
                    'healthy': is_healthy,
                    'model_info': provider.get_model_info()
                }
            except Exception as e:
                status[provider_enum.value] = {
                    'healthy': False,
                    'error': str(e)
                }
        
        return status
    
    def _update_stats(self, response: LLMResponse):
        """Update provider statistics"""
        self.stats['total_requests'] += 1
        self.stats['total_tokens_used'] += response.tokens_used
        
        # Update provider-specific stats
        provider_key = response.provider
        if provider_key not in self.stats['requests_by_provider']:
            self.stats['requests_by_provider'][provider_key] = 0
        self.stats['requests_by_provider'][provider_key] += 1
        
        # Update average response time
        total_time = self.stats['average_response_time'] * (self.stats['total_requests'] - 1)
        self.stats['average_response_time'] = (total_time + response.processing_time) / self.stats['total_requests']
        
        # Record timing
        record_timing("llm_response_time", response.processing_time)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get LLM manager statistics"""
        return {
            'llm_stats': self.stats,
            'provider_count': len(self.providers),
            'providers': list(self.providers.keys()),
            'default_provider': self.default_provider.value
        }
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [provider.value for provider in self.providers.keys()]
    
    def set_default_provider(self, provider: str) -> bool:
        """Set default provider"""
        try:
            provider_enum = LLMProvider(provider)
            if provider_enum in self.providers:
                self.default_provider = provider_enum
                logger.info(f"Default provider set to: {provider}")
                return True
            else:
                logger.error(f"Provider {provider} not available")
                return False
        except ValueError:
            logger.error(f"Invalid provider: {provider}")
            return False
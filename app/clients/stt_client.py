"""
Philips AI Model Serving STT API Client with token caching.

This module provides a production-ready client for the Philips AI Model Serving
Speech-to-Text API with automatic JWT token management and caching.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import jwt
import time
import os
import logging
import threading
from typing import Optional, Dict, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class STTClientError(Exception):
    """Base exception for STT Client errors."""
    pass


class STTAuthenticationError(STTClientError):
    """Raised when authentication fails."""
    pass


class STTTranscriptionError(STTClientError):
    """Raised when transcription fails."""
    pass


class STTConfigurationError(STTClientError):
    """Raised when configuration is invalid."""
    pass


class STTClient:
    """
    Client for Philips AI Model Serving STT API with token caching.
    
    This client provides:
    - Automatic JWT token management with caching
    - Thread-safe token refresh
    - Configurable retry logic for transient failures
    - Request timeouts
    - Connection pooling
    
    Example:
        >>> from dotenv import load_dotenv
        >>> load_dotenv()
        >>> client = STTClient()
        >>> result = client.transcribe("audio.mp3")
        >>> print(result)
    """

    DEFAULT_TENANT_ID = "1a407a2d-7675-4d17-8692-b3ac285306e4"
    DEFAULT_SCOPE = "api://philips-ai-model-serving-api-non-prod/.default"
    DEFAULT_HOST_URL = "https://dev.api.it.philips.com"
    DEFAULT_TIMEOUT = 120  # seconds
    DEFAULT_TOKEN_BUFFER = 300  # 5 minutes before expiry
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
        scope: Optional[str] = None,
        host_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        token_buffer_seconds: int = DEFAULT_TOKEN_BUFFER
    ):
        """
        Initialize the STT Client.
        
        Args:
            client_id: Azure AD application client ID. Falls back to AZURE_CLIENT_ID env var.
            client_secret: Azure AD application client secret. Falls back to AZURE_CLIENT_SECRET env var.
            tenant_id: Azure AD tenant ID. Falls back to AZURE_TENANT_ID env var or default.
            scope: API scope. Falls back to STT_API_SCOPE env var or default.
            host_url: API host URL. Falls back to STT_API_HOST_URL env var or default.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for transient failures.
            token_buffer_seconds: Seconds before token expiry to refresh.
            
        Raises:
            STTConfigurationError: If required credentials are missing.
        """
        # Load from environment variables if not provided
        self.client_id = client_id or os.getenv("AZURE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("AZURE_CLIENT_SECRET")
        self.tenant_id = tenant_id or os.getenv("AZURE_TENANT_ID", self.DEFAULT_TENANT_ID)
        self.scope = scope or os.getenv("STT_API_SCOPE", self.DEFAULT_SCOPE)
        self.host_url = (host_url or os.getenv("STT_API_HOST_URL", self.DEFAULT_HOST_URL)).rstrip("/")
        self.timeout = timeout
        self.token_buffer_seconds = token_buffer_seconds

        # Validate required configuration
        if not self.client_id or not self.client_secret:
            raise STTConfigurationError(
                "client_id and client_secret are required. "
                "Set AZURE_CLIENT_ID and AZURE_CLIENT_SECRET environment variables or pass them as arguments."
            )

        self.token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        self.api_url = f"{self.host_url}/philips-ai-modelserving-api/v1/ai/openai/deployments/gpt-4o-mini-transcribe/audio/transcriptions"

        # Token cache with thread safety
        self._cached_token: Optional[str] = None
        self._token_expiry: Optional[float] = None
        self._token_lock = threading.Lock()

        # Create session with retry logic and connection pooling
        self._session = self._create_session(max_retries)
        
        logger.info("STT Client initialized for host: %s", self.host_url)

    def _create_session(self, max_retries: int) -> requests.Session:
        """Create a requests session with retry logic and connection pooling."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session

    def _is_token_valid(self) -> bool:
        """Check if cached token is still valid (with configurable buffer)."""
        if self._cached_token is None or self._token_expiry is None:
            return False
        return time.time() < (self._token_expiry - self.token_buffer_seconds)

    def _get_token(self) -> str:
        """
        Get JWT token, using cache if valid. Thread-safe.
        
        Returns:
            Valid JWT access token.
            
        Raises:
            STTAuthenticationError: If token acquisition fails.
        """
        with self._token_lock:
            if self._is_token_valid():
                logger.debug("Using cached token")
                return self._cached_token

            logger.info("Requesting new access token")
            
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": self.scope
            }

            try:
                response = self._session.post(
                    self.token_url,
                    data=data,
                    timeout=30
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error("Failed to acquire token: %s", str(e))
                raise STTAuthenticationError(f"Failed to acquire access token: {e}") from e

            try:
                token_data = response.json()
                self._cached_token = token_data["access_token"]
            except (KeyError, ValueError) as e:
                logger.error("Invalid token response: %s", str(e))
                raise STTAuthenticationError(f"Invalid token response: {e}") from e

            # Decode token to get expiry
            try:
                decoded = jwt.decode(self._cached_token, options={"verify_signature": False})
                self._token_expiry = decoded.get("exp", time.time() + 3600)
                logger.info("Token acquired, expires at %s", time.ctime(self._token_expiry))
            except jwt.DecodeError as e:
                logger.warning("Could not decode token expiry, using default: %s", str(e))
                self._token_expiry = time.time() + 3600

            return self._cached_token

    def transcribe(
        self,
        audio_file_path: str,
        correlation_id: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_file_path: Path to the audio file (mp3, wav, etc.)
            correlation_id: Optional correlation ID for request tracking.
            language: Optional language hint for transcription.

        Returns:
            Transcription response as dict containing the transcribed text.
            
        Raises:
            STTTranscriptionError: If transcription fails.
            FileNotFoundError: If audio file does not exist.
        """
        # Validate file exists
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        if not audio_path.is_file():
            raise STTTranscriptionError(f"Path is not a file: {audio_file_path}")

        token = self._get_token()
        
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = f"stt-{int(time.time() * 1000)}"

        headers = {
            "Authorization": f"Bearer {token}",
            "correlation-id": correlation_id,
            "Accept": "application/json"
        }

        params = {"api-version": "2024-02-01"}
        if language:
            params["language"] = language

        logger.info("Transcribing file: %s (correlation_id: %s)", audio_file_path, correlation_id)

        try:
            with open(audio_file_path, "rb") as f:
                files = {"file": (audio_path.name, f, self._get_mime_type(audio_path))}
                
                response = self._session.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    params=params,
                    timeout=self.timeout
                )
        except IOError as e:
            logger.error("Failed to read audio file: %s", str(e))
            raise STTTranscriptionError(f"Failed to read audio file: {e}") from e
        except requests.exceptions.Timeout as e:
            logger.error("Request timeout: %s", str(e))
            raise STTTranscriptionError(f"Request timeout after {self.timeout}s: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error("Request failed: %s", str(e))
            raise STTTranscriptionError(f"Transcription request failed: {e}") from e

        if response.status_code == 401:
            # Token might be invalid, clear cache and retry once
            logger.warning("Received 401, clearing token cache and retrying")
            self.clear_token_cache()
            return self.transcribe(audio_file_path, correlation_id, language)

        if not response.ok:
            error_msg = f"Transcription failed with status {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f": {error_detail}"
            except ValueError:
                error_msg += f": {response.text}"
            logger.error(error_msg)
            raise STTTranscriptionError(error_msg)

        result = response.json()
        logger.info("Transcription completed successfully")
        return result

    @staticmethod
    def _get_mime_type(file_path: Path) -> str:
        """Get MIME type based on file extension."""
        mime_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }
        return mime_types.get(file_path.suffix.lower(), "application/octet-stream")

    def clear_token_cache(self) -> None:
        """Clear cached token (force refresh on next request). Thread-safe."""
        with self._token_lock:
            self._cached_token = None
            self._token_expiry = None
            logger.debug("Token cache cleared")

    def health_check(self) -> bool:
        """
        Check if the client can authenticate successfully.
        
        Returns:
            True if authentication succeeds, False otherwise.
        """
        try:
            self._get_token()
            return True
        except STTAuthenticationError:
            return False

    def close(self) -> None:
        """Close the underlying session and release resources."""
        self._session.close()
        logger.debug("Session closed")

    def __enter__(self) -> "STTClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the client."""
        return f"STTClient(host_url={self.host_url!r}, tenant_id={self.tenant_id!r})"


"""
API executor for calling external REST/GraphQL APIs.

Examples:
    # Simple GET request
    executor = APIExecutor()
    result = await executor.execute(APIAction(
        name="get_status",
        url="https://api.example.com/status",
        method=HTTPMethod.GET
    ))

    # POST with JSON body
    result = await executor.execute(APIAction(
        name="create_job",
        url="https://api.example.com/jobs",
        method=HTTPMethod.POST,
        json_body={"type": "analysis", "params": {"temp": 150}},
        headers={"Authorization": "Bearer token123"}
    ))

    # With retry and authentication
    executor = APIExecutor(
        base_url="https://api.example.com",
        default_headers={"Authorization": "Bearer token"}
    )
    result = await executor.execute(APIAction(
        name="submit_data",
        url="/data/submit",  # Will be joined with base_url
        method=HTTPMethod.POST,
        json_body={"samples": [1, 2, 3]},
        retries=3,
        retry_on_status=[500, 502, 503]
    ))

    # GraphQL query
    result = await executor.execute(APIAction(
        name="query_experiments",
        url="https://api.example.com/graphql",
        method=HTTPMethod.POST,
        graphql_query=\"\"\"
            query GetExperiments($status: String!) {
                experiments(status: $status) {
                    id
                    name
                    status
                }
            }
        \"\"\",
        graphql_variables={"status": "running"}
    ))
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

from .base import (
    BaseAction,
    ExternalExecutor,
    ExecutionResult,
    ExecutionStatus,
    ExecutionError,
)


class HTTPMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass
class APIAction(BaseAction):
    """Action for API calls.

    Attributes:
        url: API endpoint URL (can be relative if base_url is set)
        method: HTTP method
        headers: Request headers
        query_params: URL query parameters
        json_body: JSON request body (will be serialized)
        form_data: Form data for form-encoded requests
        raw_body: Raw request body (bytes or string)
        graphql_query: GraphQL query string
        graphql_variables: GraphQL variables
        expected_status: Expected status codes for success (default: 2xx)
        retry_on_status: Status codes that should trigger retry
        follow_redirects: Whether to follow redirects
        verify_ssl: Whether to verify SSL certificates
    """
    url: str = ""
    method: HTTPMethod = HTTPMethod.GET
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    json_body: Optional[Dict[str, Any]] = None
    form_data: Optional[Dict[str, Any]] = None
    raw_body: Optional[Union[str, bytes]] = None
    graphql_query: Optional[str] = None
    graphql_variables: Optional[Dict[str, Any]] = None
    expected_status: List[int] = field(default_factory=lambda: list(range(200, 300)))
    retry_on_status: List[int] = field(default_factory=lambda: [500, 502, 503, 504])
    follow_redirects: bool = True
    verify_ssl: bool = True

    def validate(self) -> None:
        super().validate()
        if not self.url:
            raise ValueError("URL is required")

        # Check for conflicting body types
        body_count = sum([
            self.json_body is not None,
            self.form_data is not None,
            self.raw_body is not None,
            self.graphql_query is not None,
        ])
        if body_count > 1:
            raise ValueError("Cannot specify multiple body types")


@dataclass
class APIResponse:
    """Structured API response."""
    status_code: int
    headers: Dict[str, str]
    body: Any  # Parsed JSON or raw text
    raw_body: bytes
    elapsed_ms: float

    @property
    def json(self) -> Optional[Dict[str, Any]]:
        """Get body as JSON dict if possible."""
        if isinstance(self.body, dict):
            return self.body
        return None


class APIExecutor(ExternalExecutor):
    """Executor for external API calls.

    Features:
    - REST and GraphQL support
    - Automatic JSON serialization/deserialization
    - Configurable retry on specific status codes
    - Base URL and default headers
    - SSL verification control

    Security considerations:
    - Use verify_ssl=True in production
    - Store API keys in environment variables
    - Use allowed_domains to restrict API calls
    """

    def __init__(
        self,
        name: str = "api",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        allowed_domains: Optional[List[str]] = None,
        http_client: Optional[Any] = None,
    ):
        """Initialize API executor.

        Args:
            name: Executor name for logging
            base_url: Base URL for all requests
            default_headers: Default headers for all requests
            allowed_domains: Optional whitelist of allowed domains
            http_client: Optional pre-configured HTTP client
        """
        super().__init__(name)
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.allowed_domains = allowed_domains
        self._http_client = http_client

    def _get_full_url(self, url: str) -> str:
        """Get full URL by joining with base_url if needed."""
        if self.base_url and not url.startswith(("http://", "https://")):
            return urljoin(self.base_url, url)
        return url

    def _check_allowed_domain(self, url: str) -> None:
        """Check if URL domain is in allowed list."""
        if self.allowed_domains is None:
            return

        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc

        if domain not in self.allowed_domains:
            raise ExecutionError(
                f"Domain '{domain}' not in allowed list: {self.allowed_domains}"
            )

    def _build_request_body(self, action: APIAction) -> tuple:
        """Build request body and content type.

        Returns:
            Tuple of (body, content_type)
        """
        if action.graphql_query:
            # GraphQL request
            body = {
                "query": action.graphql_query,
            }
            if action.graphql_variables:
                body["variables"] = action.graphql_variables
            return json.dumps(body), "application/json"

        if action.json_body is not None:
            return json.dumps(action.json_body), "application/json"

        if action.form_data is not None:
            # URL-encoded form data
            from urllib.parse import urlencode
            return urlencode(action.form_data), "application/x-www-form-urlencoded"

        if action.raw_body is not None:
            if isinstance(action.raw_body, str):
                return action.raw_body, "text/plain"
            return action.raw_body, "application/octet-stream"

        return None, None

    async def _execute_impl(self, action: BaseAction) -> ExecutionResult:
        """Execute API request."""
        if not isinstance(action, APIAction):
            raise TypeError(f"Expected APIAction, got {type(action)}")

        # Build full URL
        full_url = self._get_full_url(action.url)
        self._check_allowed_domain(full_url)

        # Build headers
        headers = self.default_headers.copy()
        headers.update(action.headers)

        # Build body
        body, content_type = self._build_request_body(action)
        if content_type and "Content-Type" not in headers:
            headers["Content-Type"] = content_type

        try:
            # Try to use httpx (preferred) or fall back to aiohttp
            response = await self._make_request(
                method=action.method.value,
                url=full_url,
                headers=headers,
                params=action.query_params or None,
                body=body,
                timeout=action.timeout_seconds,
                follow_redirects=action.follow_redirects,
                verify_ssl=action.verify_ssl,
            )

            # Check if status code should trigger retry
            if response.status_code in action.retry_on_status:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    output=response.body,
                    error=f"Retryable status code: {response.status_code}",
                    exit_code=response.status_code,
                    metadata={
                        "url": full_url,
                        "method": action.method.value,
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                    }
                )

            # Check if status code is expected
            success = response.status_code in action.expected_status

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED,
                output=response.body,
                error=None if success else f"Unexpected status: {response.status_code}",
                exit_code=response.status_code,
                duration_ms=response.elapsed_ms,
                metadata={
                    "url": full_url,
                    "method": action.method.value,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                }
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                metadata={
                    "url": full_url,
                    "method": action.method.value,
                    "exception": type(e).__name__,
                }
            )

    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, str]],
        body: Optional[Union[str, bytes]],
        timeout: float,
        follow_redirects: bool,
        verify_ssl: bool,
    ) -> APIResponse:
        """Make HTTP request using available library."""
        import time
        start_time = time.time()

        try:
            # Try httpx first (modern, async-native)
            import httpx

            async with httpx.AsyncClient(
                follow_redirects=follow_redirects,
                verify=verify_ssl,
                timeout=timeout,
            ) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    content=body,
                )

                elapsed_ms = (time.time() - start_time) * 1000

                # Try to parse JSON response
                try:
                    parsed_body = response.json()
                except (json.JSONDecodeError, ValueError):
                    parsed_body = response.text

                return APIResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=parsed_body,
                    raw_body=response.content,
                    elapsed_ms=elapsed_ms,
                )

        except ImportError:
            pass

        try:
            # Fall back to aiohttp
            import aiohttp

            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            connector = aiohttp.TCPConnector(ssl=verify_ssl)

            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_obj
            ) as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    data=body,
                    allow_redirects=follow_redirects,
                ) as response:
                    elapsed_ms = (time.time() - start_time) * 1000
                    raw_body = await response.read()

                    # Try to parse JSON
                    try:
                        parsed_body = json.loads(raw_body)
                    except (json.JSONDecodeError, ValueError):
                        parsed_body = raw_body.decode("utf-8", errors="replace")

                    return APIResponse(
                        status_code=response.status,
                        headers=dict(response.headers),
                        body=parsed_body,
                        raw_body=raw_body,
                        elapsed_ms=elapsed_ms,
                    )

        except ImportError:
            pass

        # Fall back to urllib (sync, wrapped in executor)
        import asyncio
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError
        import ssl

        def make_sync_request():
            req = Request(url, method=method, headers=headers)
            if body:
                if isinstance(body, str):
                    req.data = body.encode()
                else:
                    req.data = body

            # Add query params to URL
            if params:
                from urllib.parse import urlencode
                sep = "&" if "?" in url else "?"
                req.full_url = url + sep + urlencode(params)

            context = None
            if not verify_ssl:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE

            try:
                with urlopen(req, timeout=timeout, context=context) as resp:
                    raw = resp.read()
                    return resp.status, dict(resp.headers), raw
            except HTTPError as e:
                raw = e.read() if hasattr(e, 'read') else b""
                return e.code, dict(e.headers) if hasattr(e, 'headers') else {}, raw

        loop = asyncio.get_event_loop()
        status, resp_headers, raw_body = await loop.run_in_executor(
            None, make_sync_request
        )

        elapsed_ms = (time.time() - start_time) * 1000

        try:
            parsed_body = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError):
            parsed_body = raw_body.decode("utf-8", errors="replace")

        return APIResponse(
            status_code=status,
            headers=resp_headers,
            body=parsed_body,
            raw_body=raw_body,
            elapsed_ms=elapsed_ms,
        )


# Convenience functions
async def get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    **kwargs
) -> ExecutionResult:
    """Make a GET request."""
    executor = APIExecutor()
    action = APIAction(
        name="get",
        url=url,
        method=HTTPMethod.GET,
        headers=headers or {},
        query_params=params or {},
        timeout_seconds=timeout,
        **kwargs
    )
    return await executor.execute(action)


async def post(
    url: str,
    json_body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    **kwargs
) -> ExecutionResult:
    """Make a POST request with JSON body."""
    executor = APIExecutor()
    action = APIAction(
        name="post",
        url=url,
        method=HTTPMethod.POST,
        headers=headers or {},
        json_body=json_body,
        timeout_seconds=timeout,
        **kwargs
    )
    return await executor.execute(action)


async def graphql(
    url: str,
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    **kwargs
) -> ExecutionResult:
    """Make a GraphQL request."""
    executor = APIExecutor()
    action = APIAction(
        name="graphql",
        url=url,
        method=HTTPMethod.POST,
        headers=headers or {},
        graphql_query=query,
        graphql_variables=variables,
        timeout_seconds=timeout,
        **kwargs
    )
    return await executor.execute(action)

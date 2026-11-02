class DSSRAPIError(Exception):
    """Base exception for all DSSR API errors."""

    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response


class DSSRTimeoutError(DSSRAPIError):
    """Raised when a request to the DSSR API times out."""


class DSSRConnectionError(DSSRAPIError):
    """Raised when unable to connect to the DSSR API."""


class DSSRResponseError(DSSRAPIError):
    """Raised when the API returns an error or unexpected response."""

    def __init__(self, message, status_code=None, response=None):
        super().__init__(message, response=response)
        self.status_code = status_code

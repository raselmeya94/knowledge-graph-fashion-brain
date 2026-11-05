"""Application errors mapped to API responses without stack traces."""


class FashionGPTError(Exception):
    status_code = 400
    code = "error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ValidationFailed(FashionGPTError):
    status_code = 422
    code = "validation_failed"


class PermissionDenied(FashionGPTError):
    status_code = 403
    code = "permission_denied"


class Unauthorized(FashionGPTError):
    status_code = 401
    code = "unauthorized"


class RequestTimeout(FashionGPTError):
    status_code = 504
    code = "timeout"


class NotFoundError(FashionGPTError):
    status_code = 404
    code = "not_found"


class ConflictError(FashionGPTError):
    status_code = 409
    code = "conflict"


class GraphBackendUnavailable(FashionGPTError):
    status_code = 503
    code = "graph_unavailable"


class ModelUnavailableError(FashionGPTError):
    status_code = 503
    code = "model_unavailable"


class ProductBackendUnavailable(FashionGPTError):
    status_code = 503
    code = "product_backend_unavailable"


class ParserNotAvailable(FashionGPTError):
    status_code = 422
    code = "parser_not_available"


class RelationshipValidationError(FashionGPTError):
    status_code = 422
    code = "relationship_rejected"


class UnknownSourceType(FashionGPTError):
    status_code = 422
    code = "unknown_source_type"

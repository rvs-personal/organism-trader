"""Conservative, bounded recovery of a failed Cursor episode, not a trading day."""
import re


class EpisodeRetriesExhausted(RuntimeError):
    """Do not cascade into the day-level retry loop and repeat completed research."""


def retry_reason(backend, error):
    if getattr(backend, 'name', None) != 'cursor':
        return None
    message = str(error).strip().lower()
    if isinstance(error, RuntimeError) and message == 'cursor episode exceeded the configured tool-call budget.':
        return 'tool_budget_exceeded'
    # Rate limits use the existing concurrency backoff. Quota, billing, auth,
    # invalid requests and research budgets must never trigger a blind retry.
    if any(s in message for s in ('429', 'rate limit', 'ratelimit', 'rate_limit',
            'quota', 'usage limit', 'spending limit', 'insufficient credits',
            'paid charge', 'billing', '401', '403', 'unauthorized', 'forbidden',
            'authentication', 'api key', 'permission', 'context length',
            'context window', 'token limit', 'tool-call budget', 'invalid request')):
        return None
    if isinstance(error, RuntimeError) and message in {'cursor run error:', 'cursor run error: none'}:
        return 'cursor_empty_error'
    if isinstance(error, (TimeoutError, ConnectionError)) or any(s in message for s in
            ('timed out', 'timeout', 'connection reset', 'connection refused',
             'connection aborted', 'server disconnected', 'remoteprotocolerror')):
        return 'transport_failure'
    if re.search(r'\b(502|503|504)\b', message) or 'overloaded' in message:
        return 'provider_unavailable'
    return None

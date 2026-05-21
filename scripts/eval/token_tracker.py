"""Token tracking module for API usage monitoring.

This module provides a TokenTracker class to track token usage across API calls.
It can be imported by multiple scripts to ensure consistent tracking.
"""


class TokenTracker:
    """Track token usage across API calls."""

    def __init__(self):
        self.total_queries = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def update(self, usage):
        """Update tracker with usage from a single API call."""
        self.total_queries += 1
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens

    def report(self):
        """Return formatted token usage report."""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        return {
            "total_queries": self.total_queries,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": total_tokens
        }

    def print_report(self):
        """Print formatted token usage report."""
        report = self.report()
        print("\nToken Usage Report:")
        print(f"  Total API calls: {report['total_queries']}")
        print(f"  Input tokens: {report['input_tokens']:,}")
        print(f"  Output tokens: {report['output_tokens']:,}")
        print(f"  Total tokens: {report['total_tokens']:,}")

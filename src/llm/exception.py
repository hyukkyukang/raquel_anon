class TooMuchThinkingError(Exception):
    """Exception raised when the model thinks too much."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class SQLParsingError(Exception):
    """Exception raised when SQL parsing fails.

    Attributes:
        sql (str): The SQL that failed to parse
        message (str): Explanation of the error
    """

    def __init__(self, sql: str, message: str = "SQL parsing failed"):
        self.sql = sql
        self.message = f"{message}: {sql}"
        super().__init__(self.message)

import string
import random
from .database import Database

__all__ = ['Database', 'Utils']

class Utils:
    """
    Utility class for common utility functions.
    """
    
    @staticmethod
    def generate_verification_code(size: int = 6) -> str:
        """
        Generate a verification code.

        Parameters:
        size (int): Length of the verification code.

        Returns:
        str: The generated verification code.
        """
        CHARS = string.ascii_letters + string.digits
        return ''.join(random.choice(CHARS) for _ in range(size))

import abc
import inspect
import os
from dataclasses import dataclass
from functools import cached_property
from typing import *

import dspy
import hkkang_utils.pattern as pattern_utils

# Constants for prompt file names
SYSTEM_INSTRUCTION_FILE_NAME = "system_instruction.txt"
USER_INSTRUCTION_FILE_NAME = "user_instruction.txt"


class PromptTemplate(metaclass=pattern_utils.SingletonMetaWithArgs):
    """
    A singleton class that manages prompt templates by loading system and user instructions from text files.
    Uses SingletonMetaWithArgs to ensure unique instances per directory path.
    """

    def __init__(self, dir_path: str):
        """
        Initialize the PromptTemplate with a directory path.

        Args:
            dir_path (str): Path to the directory containing prompt instruction files
        """
        self.dir_path = dir_path

    @cached_property
    def system_instruction(self) -> str:
        """
        Loads and returns the system instruction from a text file.

        The system instruction file should be named 'system_instruction.txt' and
        located in the specified directory path.

        Returns:
            str: The contents of the system instruction file.

        Raises:
            FileNotFoundError: If the system instruction file cannot be found.
        """
        with open(os.path.join(self.dir_path, SYSTEM_INSTRUCTION_FILE_NAME), "r") as f:
            return f.read()

    @cached_property
    def user_instruction(self) -> str:
        """
        Loads and returns the user instruction from a text file.

        The user instruction file should be named 'user_instruction.txt' and
        located in the specified directory path.

        Returns:
            str: The contents of the user instruction file.

        Raises:
            FileNotFoundError: If the user instruction file cannot be found.
        """
        with open(os.path.join(self.dir_path, USER_INSTRUCTION_FILE_NAME), "r") as f:
            return f.read()


class Prompt(metaclass=abc.ABCMeta):
    """
    Abstract base class for handling system and user prompts.

    This class provides functionality to read system instructions and user prompts
    from text files located in the same directory as the concrete subclass.
    The class uses cached properties to efficiently load and store the prompt contents.
    Child classes must implement:
    - __str__: to provide a string representation of the prompt
    - get_user_prompt: to generate the user prompt with given arguments
    - signature: to provide a DSPy signature for the prompt
    """

    @cached_property
    def prompt_template(self) -> PromptTemplate:
        """
        Creates and returns a PromptTemplate instance for the current class's directory.

        Returns:
            PromptTemplate: A singleton instance of PromptTemplate for the current class's directory.
        """
        # Find the file in which the concrete subclass is defined
        subclass_file = inspect.getfile(self.__class__)
        base_dir = os.path.dirname(subclass_file)
        return PromptTemplate(base_dir)

    @property
    def system_instruction(self) -> str:
        """
        Returns the system instruction from the prompt template.

        Returns:
            str: The system instruction text.
        """
        return self.prompt_template.system_instruction

    @property
    def user_instruction(self) -> str:
        """
        Returns the user instruction from the prompt template.

        Returns:
            str: The user instruction text.
        """
        return self.prompt_template.user_instruction

    @abc.abstractmethod
    def get_user_prompt(self, *args, **kwargs) -> str:
        """
        Generates a user prompt with the given arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The generated user prompt.

        Raises:
            NotImplementedError: If the child class does not implement this method.
        """
        raise NotImplementedError("Implement get_user_prompt method in child class")

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the prompt.

        This method must be implemented by child classes to provide
        a meaningful string representation of the prompt.

        Returns:
            str: A string representation of the prompt.

        Raises:
            NotImplementedError: If the child class does not implement this method.
        """
        raise NotImplementedError("Implement __str__ method in child class")

    @classmethod
    @abc.abstractmethod
    def signature(cls) -> Type[dspy.Signature]:
        """
        Returns a DSPy signature for the prompt.

        This method must be implemented by child classes to provide
        a DSPy signature that defines the input and output types
        for the prompt.

        Returns:
            dspy.Signature: A DSPy signature object defining the prompt's interface.

        Raises:
            NotImplementedError: If the child class does not implement this method.
        """
        raise NotImplementedError("Implement signature method in child class")

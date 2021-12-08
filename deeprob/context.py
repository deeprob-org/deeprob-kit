# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala

import contextlib
import contextvars

#: Thread-safe context variables, i.e. each thread will have its own flags assignments
_context_variables = contextvars.ContextVar(
    'context_variables',
    default={
        'check_dtype': True,
        'check_spn': True
    }
)


def is_check_dtype_enabled() -> bool:
    """Returns whether the context flag 'check_dtype' is enabled."""
    return _context_variables.get()['check_dtype']


def is_check_spn_enabled() -> bool:
    """Returns whether the context flag 'check_spn' is enabled."""
    return _context_variables.get()['check_spn']


class ContextState(contextlib.ContextDecorator):
    def __init__(self, **kwargs):
        """
        Thread-safe Context State that disables some flags during execution.

        Current supported flags are the following:
        - check_dtype: bool = True, Whether to check (and cast when needed) Numpy arrays data types.
        - check_spn: bool = True, Whether to check the SPNs structure properties.
        """
        self.__token = None
        self.__state = _context_variables.get().copy()
        for flag, value in kwargs.items():
            if flag not in self.__state:
                raise ValueError("Cannot set an unknown flag called '{}', suitable flags are: {}".format(
                    flag, ', '.join(self.__state.keys())
                ))
            self.__state[flag] = value

    def __enter__(self):
        self.__token = _context_variables.set(self.__state)

    def __exit__(self, *exc):
        _context_variables.reset(self.__token)

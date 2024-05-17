import sys
from src.logger import logging


def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name= exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] at line number [{1}] with error message [{2}]"
    file_name, exc_tb.tb_lineno, str(error)

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        self.message = error_message
        self.error_detail = error_message_detail(self, error_detail)
        super().__init__(self.message)

    def __str__(self):
        return self.error_detail

    def __repr__(self):
        return self.error_detail
    
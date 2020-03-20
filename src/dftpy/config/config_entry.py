from json import JSONEncoder
from numexpr import evaluate
import re

def format_bool(expression):
    s = expression.lower()[0]
    if s in ["n", "f", "0"]:
        return False
    else:
        return True

def format_float(expression):
    return evaluate(expression).item()

def format_str(expression):
    return expression

def format_cstr(expression):
    return expression.capitalize()

def format_intlist(expression):
    return list(map(int, expression.split()))

def format_floatlist(expression):
    return list(map(format_float, expression.split()))

def format_strlist(expression):
    return expression.split()

def format_cstrlist(expression):
    return expression.capitalize().split()

def format_direction(expression):
    direct_dict = {
        "x": 0,
        "y": 1,
        "z": 2
    }
    if expression in direct_dict:
        return direct_dict[expression]
    else:
        return int(expression)

class ConfigEntry(object):


    def __init__(self, type, default=None, comment='', options='', **kwargs):
        self.type = type
        self.default = default
        self.comment = comment
        self.options = options
        if self.type == 'bool' and self.options == '':
            self.options = 'True, False'

    @classmethod
    def fromdict(cls, datadict):
        type = datadict['type']
        if 'default' in datadict:
            default = datadict['default']
        else:
            default = None
        if 'comment' in datadict:
            comment = datadict['comment']
        else:
            comment = ''
        if 'options' in datadict:
            options = datadict['options']
        else:
            options = ''
        return cls(**datadict)

    def format(self, string):
        format_dict = {
            "bool": format_bool,
            "int": int,
            "float": format_float,
            "str": format_str,
            "cstr": format_cstr,
            "intlist": format_intlist,
            "floatlist": format_floatlist,
            "strlist": format_strlist,
            "cstrlist": format_cstrlist,
            "direction": format_direction,
        }
        expression = re.split('#|!', string)[0]
        return format_dict[self.type](expression)



class ConfigEntryEncoder(JSONEncoder):

    def default(self, o):
        return o.__dict__

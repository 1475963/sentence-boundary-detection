#!/bin/bash

find scripts srcs tests -type f -iname "*.py" ! -iname "*__.py" | xargs pylint

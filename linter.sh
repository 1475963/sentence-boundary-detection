#!/bin/bash

find scripts srcs -type f -iname "*.py" ! -iname "*__.py" | xargs pylint

#!/bin/sh

python -m zipapp nogood_parsing -p "/usr/bin/env python3" -o nogood_parsing.pyz
chmod +x nogood_parsing.pyz
# Copyright (c) 2023 Ira Shokar

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.#

import os, sys

# Use modules from src directory in notebooks
path = os.path.dirname(os.path.realpath(__file__)) + '/../src'
sys.path.insert(1, path)
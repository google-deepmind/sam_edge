# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

virtualenv .
source ./bin/activate

pip3 install dm-haiku
pip3 install jax
pip3 install flax
pip3 install tensorflow
pip3 install tensorflow-datasets
pip3 install matplotlib

python3 -m image_classification --rho=0.1 --time_limit_in_hours=0.03 --hessian_check_gap=0.001 --mlp_width=100

## mnist-vit-pl


**环境搭建**
```
# 安装虚拟环境 transformers
VERSION_ALIAS="transformers"  pyenv install miniconda3-4.7.12
pyenv global transformers

# 切换 python 版本到 3.9.12
conda install python="3.9.12"

# 安装 pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 安装 pytorch-lightning
pip install --trusted-host mirrors.aliyun.com -i  http://mirrors.aliyun.com/pypi/simple/ pytorch-lightning

# 安装 torchmetircs
pip install --trusted-host mirrors.aliyun.com -i  http://mirrors.aliyun.com/pypi/simple/ torchmetrics

# 安装 transformers
pip install --trusted-host mirrors.aliyun.com -i  http://mirrors.aliyun.com/pypi/simple/ transformers==4.18

# 安装 jupyter
pip install --trusted-host mirrors.aliyun.com -i  http://mirrors.aliyun.com/pypi/simple/ jupyter
```

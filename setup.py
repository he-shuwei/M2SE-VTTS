from setuptools import setup, find_packages

setup(
    name='m2se-vtts',
    version='1.0.0',
    description='M2SE-VTTS: Multi-modal and Multi-scale Spatial Environment Understanding for Immersive Visual Text-to-Speech',
    author='Rui Liu, Shuwei He, Yifan Hu, Haizhou Li',
    url='https://github.com/he-shuwei/M2SE-VTTS',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchaudio>=2.0.0',
        'numpy>=1.24.0',
        'librosa>=0.10.0',
        'scipy>=1.10.0',
        'soundfile>=0.12.0',
        'matplotlib>=3.7.0',
        'Pillow>=9.0.0',
        'pyyaml>=6.0',
        'tqdm>=4.60.0',
        'tensorboard>=2.14.0',
        'transformers>=4.24.0',
        'speechbrain>=1.0.0',
        'x-transformers>=1.30.0',
        'TextGrid>=1.6.0',
    ],
)

from setuptools import setup, find_packages

setup(
    version="1.0",
    name="auto_subtitle",
    packages=find_packages(),
    py_modules=["auto_subtitle"],
    author="Miguel Piedrafita",
    install_requires=[
        'openai-whisper',
        'ffmpeg-python',
        'deep-translator',
        'openai',
        'python-dotenv',
        'fastapi',
        'uvicorn[standard]',
        'python-multipart',
        'transformers',
    ],
    description="Automatically generate and embed subtitles into your videos",
    entry_points={
        'console_scripts': [
            'auto_subtitle=auto_subtitle.cli:main',
            'auto_sub=auto_subtitle.cli:simple_main',
            'drakonsub-web=auto_subtitle.web:main',
        ],
    },
    package_data={'auto_subtitle': ['static/*']},
)

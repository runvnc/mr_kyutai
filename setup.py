from setuptools import setup, find_packages

setup(
    name="mr_kyutai",
    version="0.1.0",
    description="Kyutai streaming TTS plugin for MindRoot (drop-in for mr_eleven_stream)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "python-dotenv",
        "websockets",
        "msgpack",
    ],
    extras_require={
        # Heavy deps only needed for local inference mode (no KYUTAI_REMOTE).
        "local": [
            "moshi>=0.2.11",
            "torch",
        ],
        "dev": [
            "pytest",
            "pytest-asyncio",
        ],
    },
    python_requires=">=3.10",
)

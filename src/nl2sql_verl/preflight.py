"""Legacy AutoModel config checks; native configs are tested by test_dapo_16gpu_config.py."""
from .legacy.automodel.preflight import load_config, validate, main

if __name__ == '__main__':
    main()

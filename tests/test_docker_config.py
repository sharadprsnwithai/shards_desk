import os
import yaml

def test_docker_compose_structure():
    compose_path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
    assert os.path.exists(compose_path)
    with open(compose_path, "r", encoding="utf-8") as f:
        dc = yaml.safe_load(f)
    assert "services" in dc
    services = dc["services"]
    assert "kronos-scheduler" in services
    assert "kronos-monthly" in services
    assert "kronos-intraday" in services

    # Verify environment file loading as requested by user
    for s_name in ["kronos-scheduler", "kronos-monthly", "kronos-intraday"]:
        assert dc["services"][s_name].get("env_file") == ".env"

    # Verify model cache volume persistence
    volumes = dc.get("volumes", {})
    assert "hf_cache" in volumes

def test_dockerfile_exists():
    dockerfile_path = os.path.join(os.path.dirname(__file__), "..", "Dockerfile")
    assert os.path.exists(dockerfile_path)
    with open(dockerfile_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "python:3.11" in content
    assert "Asia/Kolkata" in content

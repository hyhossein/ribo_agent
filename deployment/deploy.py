from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model, Environment, CodeConfiguration
from azure.identity import DefaultAzureCredential
import json, argparse

def get_client():
    return MLClient(DefaultAzureCredential(), subscription_id="your-sub-id", resource_group_name="rg-ribo-agent", workspace_name="ws-ribo-agent")

def register_model(c):
    m = Model(path="./model_artifacts/", name="ribo-qwen-qlora", version="1", description="Qwen 2.5 7B + QLoRA for RIBO L1", properties={"accuracy":"0.6568","base":"Qwen2.5-7B-4bit","traces":"253"})
    c.models.create_or_update(m)
    print("Model registered")

def create_endpoint(c):
    e = ManagedOnlineEndpoint(name="ribo-agent-endpoint", description="RIBO L1 exam agent", auth_mode="key")
    c.online_endpoints.begin_create_or_update(e).result()
    print("Endpoint created")

def deploy_model(c, model_name="ribo-qwen-qlora:1", dep_name="qlora-v1"):
    env = Environment(image="mcr.microsoft.com/azureml/mlflow-ubuntu22.04-py311-cpu-inference:latest", conda_file="deployment/environment.yml")
    d = ManagedOnlineDeployment(name=dep_name, endpoint_name="ribo-agent-endpoint", model=model_name, code_configuration=CodeConfiguration(code="deployment/", scoring_script="score.py"), environment=env, instance_type="Standard_DS3_v2", instance_count=1)
    c.online_deployments.begin_create_or_update(d).result()
    e = c.online_endpoints.get("ribo-agent-endpoint")
    e.traffic = {dep_name: 100}
    c.online_endpoints.begin_create_or_update(e).result()
    print(f"Deployed {dep_name} with 100% traffic")

def blue_green(c, new_ver, new_name):
    deploy_model(c, f"ribo-qwen-qlora:{new_ver}", new_name)
    e = c.online_endpoints.get("ribo-agent-endpoint")
    old = [k for k in e.traffic if k != new_name][0]
    e.traffic = {old: 90, new_name: 10}
    c.online_endpoints.begin_create_or_update(e).result()
    print(f"Canary: 90/{old} + 10/{new_name}")
    input("Press Enter to promote...")
    e.traffic = {new_name: 100}
    c.online_endpoints.begin_create_or_update(e).result()
    c.online_deployments.begin_delete(name=old, endpoint_name="ribo-agent-endpoint").result()
    print(f"Promoted {new_name}, deleted {old}")

def test(c):
    r = c.online_endpoints.invoke(endpoint_name="ribo-agent-endpoint", request_file=json.dumps({"question":"Under the RIB Act, what penalty for operating without licence?","options":{"A":"25000","B":"50000","C":"100000","D":"250000"},"agent":"v9_qlora"}))
    print(json.loads(r))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--action", choices=["register","deploy","upgrade","test"], required=True)
    p.add_argument("--model-version", default="1")
    p.add_argument("--deployment-name", default="qlora-v1")
    a = p.parse_args()
    c = get_client()
    if a.action == "register": register_model(c)
    elif a.action == "deploy": create_endpoint(c); deploy_model(c)
    elif a.action == "upgrade": blue_green(c, a.model_version, a.deployment_name)
    elif a.action == "test": test(c)
